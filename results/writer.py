import json
from pathlib import Path


class ResultsWriter:

    def __init__(self, output_dir: Path):

        self.output_dir = Path(output_dir)

        self.metrics_dir = self.output_dir / "metrics"
        self.pred_dir = self.output_dir / "raw_predictions"

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(parents=True, exist_ok=True)

    def write_predictions(self, model_id, dataset_name, dataset, predictions):

        path = self.pred_dir / f"{model_id}__{dataset_name.replace('/', '_')}.jsonl"

        with open(path, "w") as f:

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

        return path

    def write_metrics(self, model_id, dataset_name, result):

        path = self.metrics_dir / f"{model_id}__{dataset_name.replace('/', '_')}.json"

        path.write_text(json.dumps({
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

        return path