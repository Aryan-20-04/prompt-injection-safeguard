"""
Chart generator — produces PNG charts from metrics JSON files.

Charts produced
---------------
1. model_comparison.png      — Micro F1 bar chart across all models × datasets
2. latency_comparison.png    — Median (P50) latency bar chart
3. macro_vs_micro.png        — Scatter: macro F1 vs micro F1 per model
4. per_attack_heatmap.png    — Heatmap: model × attack-type F1
5. fpr_adr.png               — Grouped bar: FPR vs ADR per model
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ChartGenerator:

    def __init__(self, metrics_dir: Path, output_dir: Path) -> None:
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> None:
        data = self._load_metrics()
        if not data:
            logger.warning("[charts] No metric files found in %s", self.metrics_dir)
            return

        self._model_comparison(data)
        self._latency_chart(data)
        self._macro_vs_micro(data)
        self._per_attack_heatmap(data)
        self._fpr_adr(data)

    # ------------------------------------------------------------------
    # Individual charts
    # ------------------------------------------------------------------

    def _model_comparison(self, data: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        labels_set = sorted({d["dataset_name"] for d in data})
        models = sorted({d["model_name"] for d in data})
        x = np.arange(len(models))
        width = 0.8 / max(len(labels_set), 1)

        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.4), 6))
        for i, ds in enumerate(labels_set):
            scores = [
                next(
                    (d["global_metrics"]["micro_f1"] for d in data
                     if d["model_name"] == m and d["dataset_name"] == ds),
                    0.0,
                )
                for m in models
            ]
            ax.bar(x + i * width, scores, width, label=_short(ds))

        ax.set_xticks(x + width * (len(labels_set) - 1) / 2)
        ax.set_xticklabels([_short(m) for m in models], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Micro F1")
        ax.set_title("Model Micro F1 — by Dataset")
        ax.legend(title="Dataset", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        self._save(fig, "model_comparison.png")

    def _latency_chart(self, data: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        models = [d["model_name"] for d in data]
        p50 = [d.get("latency_p50_ms", 0) for d in data]
        p95 = [d.get("latency_p95_ms", 0) for d in data]

        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.4), 6))
        ax.bar(x - 0.2, p50, 0.4, label="P50", color="#4c72b0")
        ax.bar(x + 0.2, p95, 0.4, label="P95", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in models], rotation=30, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency per Sample (P50 / P95)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        self._save(fig, "latency_comparison.png")

    def _macro_vs_micro(self, data: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        for d in data:
            gm = d["global_metrics"]
            ax.scatter(gm["micro_f1"], gm["macro_f1"], s=80, zorder=3)
            ax.annotate(
                _short(d["model_name"]),
                (gm["micro_f1"], gm["macro_f1"]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=8,
            )

        ax.set_xlabel("Micro F1")
        ax.set_ylabel("Macro F1")
        ax.set_title("Macro vs Micro F1 per Model")
        lim = [0, 1.05]
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.grid(alpha=0.3)

        self._save(fig, "macro_vs_micro.png")

    def _per_attack_heatmap(self, data: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        # Collect all attack types across all results
        attack_types: set = set()
        for d in data:
            attack_types.update(d.get("per_attack_metrics", {}).keys())

        if not attack_types:
            return

        attack_types = sorted(attack_types)
        models = [d["model_name"] for d in data]
        matrix = np.zeros((len(models), len(attack_types)))

        for i, d in enumerate(data):
            pam = d.get("per_attack_metrics", {})
            for j, at in enumerate(attack_types):
                if at in pam:
                    matrix[i, j] = pam[at].get("micro_f1", 0.0)

        fig, ax = plt.subplots(
            figsize=(max(8, len(attack_types) * 1.2), max(4, len(models) * 0.8))
        )
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Micro F1")

        ax.set_xticks(range(len(attack_types)))
        ax.set_xticklabels([_short(a) for a in attack_types], rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([_short(m) for m in models])
        ax.set_title("Per-Attack-Type F1 Heatmap")

        for i in range(len(models)):
            for j in range(len(attack_types)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

        self._save(fig, "per_attack_heatmap.png")

    def _fpr_adr(self, data: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        models = [d["model_name"] for d in data]
        fpr = [d["global_metrics"].get("false_positive_rate", 0) for d in data]
        adr = [d["global_metrics"].get("attack_detection_rate", 0) for d in data]

        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.4), 6))
        ax.bar(x - 0.2, adr, 0.4, label="Attack Detection Rate (↑)", color="#2ca02c")
        ax.bar(x + 0.2, fpr, 0.4, label="False Positive Rate (↓)", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in models], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title("Attack Detection Rate vs False Positive Rate")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        self._save(fig, "fpr_adr.png")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_metrics(self) -> List[Dict[str, Any]]:
        results = []
        for f in sorted(self.metrics_dir.glob("*.json")):
            try:
                results.append(json.loads(f.read_text()))
            except Exception as exc:
                logger.warning("Could not parse %s: %s", f, exc)
        return results

    def _save(self, fig, filename: str) -> None:
        import matplotlib.pyplot as plt

        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[charts] saved → %s", path)
        print(f"[charts] saved → {path}")


def _short(name: str, maxlen: int = 22) -> str:
    """Truncate long names for chart axis labels."""
    return name if len(name) <= maxlen else name[:maxlen - 1] + "…"
