"""
ResultsWriter — persists raw predictions and metric JSON files.

Output layout inside run output_dir:
    metrics/
        {model_id}__{dataset_safe_name}.json
    raw_predictions/
        {model_id}__{dataset_safe_name}.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    """Convert dataset name (may contain / :) to a filesystem-safe string."""
    return name.replace("/", "_").replace(":", "_")


class ResultsWriter:

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.pred_dir = self.output_dir / "raw_predictions"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Raw predictions
    # ------------------------------------------------------------------

    def write_predictions(self, model_id: str, dataset_name: str, dataset, predictions) -> Path:
        path = self.pred_dir / f"{model_id}__{_safe_name(dataset_name)}.jsonl"

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

        logger.debug("Predictions written → %s", path)
        return path

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def write_metrics(self, model_id: str, dataset_name: str, result) -> Path:
        path = self.metrics_dir / f"{model_id}__{_safe_name(dataset_name)}.json"

        payload = {
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "label_schema": result.label_schema,
            "evaluation_mode": result.evaluation_mode,
            "global_metrics": result.global_metrics.to_dict(),
            "per_attack_metrics": {
                k: v.to_dict() for k, v in result.per_attack_metrics.items()
            },
            "latency_p50_ms": result.latency_p50_ms,
            "latency_p95_ms": result.latency_p95_ms,
            "latency_p99_ms": result.latency_p99_ms,
        }

        path.write_text(json.dumps(payload, indent=2))
        logger.debug("Metrics written → %s", path)
        return path
