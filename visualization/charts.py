import json
from pathlib import Path
import matplotlib.pyplot as plt


class ChartGenerator:

    def __init__(self, metrics_dir: Path, output_dir: Path):

        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self):

        data = self._load_metrics()

        if not data:
            print("[charts] no metric files found")
            return

        self._model_comparison(data)
        self._latency_chart(data)

    def _load_metrics(self):

        metrics = []

        for f in self.metrics_dir.glob("*.json"):
            metrics.append(json.loads(f.read_text()))

        return metrics

    def _model_comparison(self, data):

        models = [d["model_name"] for d in data]
        scores = [d["global_metrics"]["micro_f1"] for d in data]

        plt.figure(figsize=(10, 6))

        plt.bar(models, scores)

        plt.title("Model Micro F1 Comparison")
        plt.ylabel("Micro F1")
        plt.xticks(rotation=30)

        path = self.output_dir / "model_comparison.png"

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        print(f"[charts] saved → {path}")

    def _latency_chart(self, data):

        models = [d["model_name"] for d in data]
        latency = [d["latency_p50_ms"] for d in data]

        plt.figure(figsize=(10, 6))

        plt.bar(models, latency)

        plt.title("Median Inference Latency")
        plt.ylabel("Latency (ms)")
        plt.xticks(rotation=30)

        path = self.output_dir / "latency_comparison.png"

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        print(f"[charts] saved → {path}")