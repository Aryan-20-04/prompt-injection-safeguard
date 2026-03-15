import argparse
import json
import yaml
from pathlib import Path
from tqdm import tqdm

from bench.datasets.hf_loader import HuggingFaceLoader
from bench.evaluator import Evaluator
from bench.leaderboard import LeaderboardGenerator
from bench.models.registry import load_from_yaml, get as get_model
from bench.utils.reproducibility import capture_snapshot


class BenchmarkRunner:
    def __init__(self):
        self.evaluator = Evaluator()

    def run_from_config_path(self, config_path: str, output_dir: str = None):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        output_dir = output_dir or config["run"]["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "resolved_config.yaml").write_text(
            yaml.dump(config, default_flow_style=False)
        )

        seed = config["run"].get("seed", 42)
        snapshot = capture_snapshot(config, seed=seed)
        (Path(output_dir) / "run_metadata.json").write_text(
            json.dumps(snapshot, indent=2)
        )

        device = config["run"].get("device", "cpu")
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        batch_size = config["run"].get("batch_size", 32)
        labels = config["labels"]

        load_from_yaml("configs/model_registry.yaml")

        dataset_cfgs = config.get("datasets", [])
        model_cfgs = config.get("models", [])

        datasets = []
        for dcfg in dataset_cfgs:
            loader = HuggingFaceLoader({
                "hf_name": "Smooth-3/llm-prompt-injection-attacks",
                "split": dcfg.get("split", "validation"),
                "max_samples": dcfg.get("max_samples", None),
                "revision": dcfg.get("revision", None),
            })
            dataset = loader.load()
            datasets.append(dataset)

        snapshot["dataset_hashes"] = {ds.name: ds.content_hash for ds in datasets}
        (Path(output_dir) / "run_metadata.json").write_text(
            json.dumps(snapshot, indent=2)
        )

        metrics_dir = Path(output_dir) / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        raw_dir = Path(output_dir) / "raw_predictions"
        raw_dir.mkdir(exist_ok=True)

        all_results = []

        for mcfg in model_cfgs:
            model_id = mcfg["id"]
            plugin = get_model(model_id)
            plugin.load({
                "labels": labels,
                "device": device,
                "model_name": mcfg.get("model_name", model_id),
                "max_len": mcfg.get("max_len", 256),
            })

            for dataset in datasets:
                predictions = self._run_inference(
                    plugin, dataset, batch_size
                )

                raw_path = raw_dir / f"{model_id}__{dataset.name.replace('/', '_')}.jsonl"
                with open(raw_path, "w") as f:
                    for sample, pred in zip(dataset.samples, predictions):
                        f.write(json.dumps({
                            "sample_id": sample.sample_id,
                            "text": sample.text,
                            "ground_truth": sample.ground_truth_labels,
                            "predicted": pred.predicted_labels,
                            "probabilities": pred.label_probabilities,
                            "inference_time_ms": pred.inference_time_ms,
                        }) + "\n")

                result = self.evaluator.evaluate(dataset, predictions, plugin.metadata)
                all_results.append(result)

                metrics_path = metrics_dir / f"{model_id}__{dataset.name.replace('/', '_')}.json"
                metrics_path.write_text(json.dumps({
                    "model_name": result.model_name,
                    "dataset_name": result.dataset_name,
                    "label_schema": result.label_schema,
                    "global_metrics": {
                        "micro_f1": result.global_metrics.micro_f1,
                        "macro_f1": result.global_metrics.macro_f1,
                        "precision": result.global_metrics.precision,
                        "recall": result.global_metrics.recall,
                        "per_class_f1": result.global_metrics.per_class_f1,
                        "confusion_matrix": result.global_metrics.confusion_matrix,
                        "sample_count": result.global_metrics.sample_count,
                    },
                    "per_attack_metrics": {
                        atype: {
                            "micro_f1": ms.micro_f1,
                            "macro_f1": ms.macro_f1,
                            "precision": ms.precision,
                            "recall": ms.recall,
                            "per_class_f1": ms.per_class_f1,
                            "sample_count": ms.sample_count,
                        }
                        for atype, ms in result.per_attack_metrics.items()
                    },
                    "latency_p50_ms": result.latency_p50_ms,
                    "latency_p95_ms": result.latency_p95_ms,
                    "latency_p99_ms": result.latency_p99_ms,
                }, indent=2))

                print(
                    f"[done] {model_id} on {dataset.name} — "
                    f"micro_f1={result.global_metrics.micro_f1:.4f} "
                    f"macro_f1={result.global_metrics.macro_f1:.4f} "
                    f"p50={result.latency_p50_ms:.1f}ms"
                )

        LeaderboardGenerator().generate(output_dir, output_dir)
        print(f"\nRun complete. Results written to: {output_dir}")
        return all_results

    def _run_inference(self, plugin, dataset, batch_size):
        predictions = []
        samples = dataset.samples
        batches = [
            samples[i:i + batch_size]
            for i in range(0, len(samples), batch_size)
        ]
        for batch in tqdm(batches, desc=f"Inferring {plugin.metadata.name}", unit="batch"):
            texts = [s.text for s in batch]
            predictions.extend(plugin.predict(texts))
        return predictions


def cli_entry():
    p = argparse.ArgumentParser(description="Run the prompt injection benchmark.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--output", default=None, help="Override output directory.")
    args = p.parse_args()
    BenchmarkRunner().run_from_config_path(args.config, args.output)


if __name__ == "__main__":
    cli_entry()