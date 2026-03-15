import argparse
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
from models.discovery import discover_models
from data.hf_loader import HuggingFaceLoader
from evaluation.evaluator import Evaluator
from leaderboard.generator import LeaderboardGenerator
from models.model_registry import load_from_yaml, get as get_model
from reproducibility.snapshot import capture_snapshot

discover_models()

class BenchmarkRunner:

    def __init__(self):
        self.evaluator = Evaluator()

    def run_from_config_path(self, config_path: str, output_dir: str | None = None):

        with open(config_path) as f:
            config = yaml.safe_load(f)

        output_dir = output_dir or config["run"]["output_dir"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # save resolved config
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True)
        )

        # reproducibility snapshot
        seed = config["run"].get("seed", 42)
        snapshot = capture_snapshot(config, seed=seed)

        # device selection
        device = config["run"].get("device", "auto")

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        batch_size = config["run"].get("batch_size", 32)
        labels = config["labels"]

        # load model registry
        registry_path = Path(__file__).parent.parent / "configs/model_registry.yaml"
        load_from_yaml(registry_path)
        from data.suites import SUITES
        if "suite" in config:
            dataset_cfgs = SUITES[config["suite"]]
        else:
            dataset_cfgs = config.get("datasets", [])
        model_cfgs = config.get("models", [])

        datasets = []

        for dcfg in dataset_cfgs:

            loader = HuggingFaceLoader({
                "hf_name": dcfg["hf_name"],
                "split": dcfg.get("split", "validation"),
                "max_samples": dcfg.get("max_samples"),
                "revision": dcfg.get("revision"),
                "text_field": dcfg.get("text_field", "text"),
                "label_field": dcfg.get("label_field", "labels"),
            })

            dataset = loader.load()
            datasets.append(dataset)

        snapshot["dataset_hashes"] = {
            ds.name: ds.content_hash for ds in datasets
        }

        (output_dir / "run_metadata.json").write_text(
            json.dumps(snapshot, indent=2)
        )

        metrics_dir = output_dir / "metrics"
        raw_dir = output_dir / "raw_predictions"

        metrics_dir.mkdir(exist_ok=True)
        raw_dir.mkdir(exist_ok=True)

        all_results = []

        for mcfg in model_cfgs:

            model_id = mcfg["id"]

            try:

                plugin = get_model(model_id)

                if plugin is None:
                    raise ValueError(f"Model '{model_id}' not found in registry")

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

                            probs = pred.label_probabilities

                            if hasattr(probs, "detach"):
                                probs = probs.detach().cpu().tolist()
                            elif hasattr(probs, "tolist"):
                                probs = probs.tolist()

                            f.write(json.dumps({
                                "sample_id": sample.sample_id,
                                "text": sample.text,
                                "ground_truth": sample.ground_truth_labels,
                                "predicted": pred.predicted_labels,
                                "probabilities": probs,
                                "inference_time_ms": pred.inference_time_ms,
                            }) + "\n")

                    result = self.evaluator.evaluate(
                        dataset,
                        predictions,
                        plugin.metadata
                    )

                    all_results.append(result)

                    metrics_path = metrics_dir / f"{model_id}__{dataset.name.replace('/', '_')}.json"

                    metrics_path.write_text(json.dumps({
                        "model_name": result.model_name,
                        "dataset_name": result.dataset_name,
                        "label_schema": result.label_schema,
                        "global_metrics": result.global_metrics.to_dict(),
                        "per_attack_metrics": {
                            k: v.to_dict()
                            for k, v in result.per_attack_metrics.items()
                        },
                        "latency_p50_ms": result.latency_p50_ms,
                        "latency_p95_ms": result.latency_p95_ms,
                        "latency_p99_ms": result.latency_p99_ms,
                    }, indent=2))

                    print(
                        f"[done] {model_id} on {dataset.name} "
                        f"micro_f1={result.global_metrics.micro_f1:.4f} "
                        f"macro_f1={result.global_metrics.macro_f1:.4f}"
                    )

            except Exception as e:

                print(f"[error] {model_id} — {e}")

        LeaderboardGenerator().generate(
            metrics_dir,
            output_dir / "leaderboard"
        )

        print(f"\nRun complete → {output_dir}")

        return all_results


    def _run_inference(self, plugin, dataset, batch_size):

        predictions = []

        samples = dataset.samples

        model_name = getattr(plugin.metadata, "name", "model")

        for i in tqdm(
            range(0, len(samples), batch_size),
            desc=f"Inferring {model_name}",
            unit="batch"
        ):

            batch = samples[i:i + batch_size]

            texts = [s.text for s in batch]

            batch_preds = plugin.predict(texts)

            if len(batch_preds) != len(batch):

                raise ValueError(
                    f"Plugin {plugin.metadata.name} returned "
                    f"{len(batch_preds)} predictions for batch size {len(batch)}"
                )

            predictions.extend(batch_preds)

        return predictions


def cli_entry():

    parser = argparse.ArgumentParser(
        description="Run the prompt injection benchmark."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Override output directory."
    )

    args = parser.parse_args()

    BenchmarkRunner().run_from_config_path(
        args.config,
        args.output
    )


if __name__ == "__main__":
    cli_entry()