"""
Leaderboard generator.

Reads all per-model-per-dataset metric JSON files from metrics_dir
and produces a ranked leaderboard.json sorted by micro_f1.

Also appends latency columns and per-attack F1 columns so the
dashboard can render full comparison tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Primary ranking metric (can be overridden in generate())
DEFAULT_RANK_METRIC = "micro_f1"


class LeaderboardGenerator:

    def generate(
        self,
        metrics_dir: Path,
        output_dir: Path,
        rank_by: str = DEFAULT_RANK_METRIC,
    ) -> List[Dict[str, Any]]:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metric_files = sorted(Path(metrics_dir).glob("*.json"))

        if not metric_files:
            logger.warning("[leaderboard] No metric files found in %s", metrics_dir)
            return []

        rows: List[Dict[str, Any]] = []

        for f in metric_files:
            try:
                data = json.loads(f.read_text())
                gm = data.get("global_metrics", {})

                row: Dict[str, Any] = {
                    "model": data.get("model_name", "unknown"),
                    "dataset": data.get("dataset_name", "unknown"),
                    "eval_mode": data.get("evaluation_mode", "binary"),
                    # Core metrics
                    "micro_f1": gm.get("micro_f1", 0.0),
                    "macro_f1": gm.get("macro_f1", 0.0),
                    "precision": gm.get("precision", 0.0),
                    "recall": gm.get("recall", 0.0),
                    "false_positive_rate": gm.get("false_positive_rate", 0.0),
                    "attack_detection_rate": gm.get("attack_detection_rate", 0.0),
                    "sample_count": gm.get("sample_count", 0),
                    # Latency
                    "latency_p50_ms": data.get("latency_p50_ms", 0.0),
                    "latency_p95_ms": data.get("latency_p95_ms", 0.0),
                    "latency_p99_ms": data.get("latency_p99_ms", 0.0),
                    # Per-attack F1 (flattened)
                    "per_attack_f1": {
                        k: v.get("micro_f1", 0.0)
                        for k, v in data.get("per_attack_metrics", {}).items()
                    },
                    # Per-class F1
                    "per_class_f1": gm.get("per_class_f1", {}),
                }
                rows.append(row)
            except Exception as exc:
                logger.warning("Could not parse metrics file %s: %s", f, exc)

        # Sort and rank
        rows.sort(key=lambda r: r.get(rank_by, 0.0), reverse=True)
        for i, row in enumerate(rows, start=1):
            row["rank"] = i

        leaderboard_path = output_dir / "leaderboard.json"
        leaderboard_path.write_text(json.dumps(rows, indent=2))
        logger.info("[leaderboard] Written → %s (%d entries)", leaderboard_path, len(rows))
        print(f"[leaderboard] written → {leaderboard_path}")

        return rows
