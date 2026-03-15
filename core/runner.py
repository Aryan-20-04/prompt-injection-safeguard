"""
BenchmarkRunner — orchestrates the full evaluation pipeline.

Flow
----
1. Load & validate config
2. Expand suite → dataset config list
3. Capture reproducibility snapshot
4. Load all datasets
5. For each model × dataset:
   a. Load model
   b. Run batched inference (optionally parallel)
   c. Evaluate metrics
   d. Write raw predictions + metrics JSON
6. Generate leaderboard
7. Build HTML dashboard
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List

import torch
import yaml

from core.config_loader import ConfigLoader
from data.hf_loader import HuggingFaceLoader
from data.suites import get_suite, load_suites_from_yaml
from evaluation.evaluator import Evaluator
from inference.engine import InferenceEngine
from inference.prediction_types import Prediction
from leaderboard.generator import LeaderboardGenerator
from models.discovery import discover_models
from models.model_registry import get as get_model, load_from_yaml
from reproducibility.snapshot import capture_snapshot
from results.writer import ResultsWriter
from visualization.charts import ChartGenerator
from visualization.dashboard import DashboardBuilder

logger = logging.getLogger(__name__)

# Auto-discover adapters at import time
discover_models()
# Merge any additional suites defined in configs/datasets.yaml
load_suites_from_yaml()


class BenchmarkRunner:

    def __init__(self, eval_mode: str = "binary") -> None:
        """
        Parameters
        ----------
        eval_mode : "binary" | "multi_label"
            Passed through to the Evaluator.
        """
        self.evaluator = Evaluator(mode=eval_mode)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_from_config_path(
        self,
        config_path: str,
        output_dir: str | None = None,
    ) -> List:
        config = ConfigLoader().load(config_path)

        run_output_dir = Path(output_dir or config["run"]["output_dir"])
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Persist resolved config for reproducibility
        (run_output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True)
        )

        # ── Device ──────────────────────────────────────────────────────
        device = self._resolve_device(config["run"].get("device", "auto"))
        batch_size = config["run"].get("batch_size", 32)
        parallel_workers = config["run"].get("parallel_workers", 1)
        seed = config["run"].get("seed", 42)
        labels = config["labels"]
        eval_mode = config["run"].get("eval_mode", "binary")
        self.evaluator = Evaluator(mode=eval_mode)

        # ── Reproducibility ─────────────────────────────────────────────
        snapshot = capture_snapshot(config, seed=seed)

        # ── Model registry ───────────────────────────────────────────────
        registry_path = Path(__file__).parent.parent / "configs" / "model_registry.yaml"
        if registry_path.exists():
            load_from_yaml(registry_path)

        # ── Datasets ─────────────────────────────────────────────────────
        if "suite" in config:
            dataset_cfgs = get_suite(config["suite"])
        else:
            dataset_cfgs = config["datasets"]

        datasets = self._load_datasets(dataset_cfgs)
        snapshot["dataset_hashes"] = {ds.name: ds.content_hash for ds in datasets}

        (run_output_dir / "run_metadata.json").write_text(
            json.dumps(snapshot, indent=2, default=str)
        )

        # ── Inference + Evaluation ───────────────────────────────────────
        writer = ResultsWriter(run_output_dir)
        engine = InferenceEngine(num_workers=parallel_workers)
        model_cfgs = config.get("models", [])
        all_results = []

        for mcfg in model_cfgs:
            model_id = mcfg["id"]
            try:
                plugin = get_model(model_id)
                plugin.load({
                    "labels": labels,
                    "device": device,
                    "model_name": mcfg.get("model_name", model_id),
                    "max_len": mcfg.get("max_len", 512),
                    "api_key": mcfg.get("api_key"),
                })

                snapshot["model_versions"][model_id] = getattr(
                    plugin.metadata, "version", None
                ) or mcfg.get("model_name", model_id)

                for dataset in datasets:
                    predictions = engine.run(plugin, dataset, batch_size)

                    writer.write_predictions(model_id, dataset.name, dataset, predictions)

                    result = self.evaluator.evaluate(dataset, predictions, plugin.metadata)
                    all_results.append(result)

                    writer.write_metrics(model_id, dataset.name, result)

                    logger.info(
                        "[done] %s on %s | micro_f1=%.4f macro_f1=%.4f "
                        "fpr=%.4f adr=%.4f latency_p50=%.1fms",
                        model_id, dataset.name,
                        result.global_metrics.micro_f1,
                        result.global_metrics.macro_f1,
                        result.global_metrics.false_positive_rate,
                        result.global_metrics.attack_detection_rate,
                        result.latency_p50_ms,
                    )
                    print(
                        f"[done] {model_id} on {dataset.name} "
                        f"micro_f1={result.global_metrics.micro_f1:.4f} "
                        f"macro_f1={result.global_metrics.macro_f1:.4f} "
                        f"fpr={result.global_metrics.false_positive_rate:.4f} "
                        f"adr={result.global_metrics.attack_detection_rate:.4f}"
                    )

            except Exception as exc:
                logger.error("[error] %s — %s", model_id, exc, exc_info=True)
                print(f"[error] {model_id} — {exc}")

        # ── Leaderboard ───────────────────────────────────────────────────
        leaderboard_dir = run_output_dir / "leaderboard"
        LeaderboardGenerator().generate(writer.metrics_dir, leaderboard_dir)

        # ── Charts + Dashboard ────────────────────────────────────────────
        charts_dir = run_output_dir / "charts"
        ChartGenerator(writer.metrics_dir, charts_dir).generate_all()

        dashboard_path = run_output_dir / "dashboard"
        DashboardBuilder().build(
            leaderboard_dir / "leaderboard.json",
            writer.metrics_dir,
            charts_dir,
            dashboard_path,
        )

        # Persist updated snapshot (now includes model_versions)
        (run_output_dir / "run_metadata.json").write_text(
            json.dumps(snapshot, indent=2, default=str)
        )

        print(f"\nRun complete → {run_output_dir}")
        return all_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    @staticmethod
    def _load_datasets(dataset_cfgs: list):
        datasets = []
        for dcfg in dataset_cfgs:
            loader = HuggingFaceLoader({
                "hf_name": dcfg["hf_name"],
                "split": dcfg.get("split", "validation"),
                "max_samples": dcfg.get("max_samples"),
                "revision": dcfg.get("revision"),
                "text_field": dcfg.get("text_field", "text"),
                "label_field": dcfg.get("label_field", "label"),
                "label_map": dcfg.get("label_map", {}),
            })
            datasets.append(loader.load())
        return datasets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_entry() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the prompt injection benchmark.")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config.")
    parser.add_argument("--output", default=None, help="Override output directory.")
    parser.add_argument(
        "--eval-mode",
        default="binary",
        choices=["binary", "multi_label"],
        help="Evaluation mode (default: binary).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    BenchmarkRunner(eval_mode=args.eval_mode).run_from_config_path(
        args.config, args.output
    )


if __name__ == "__main__":
    cli_entry()
