import matplotlib
matplotlib.use("Agg")           # headless — works in CI with no display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class ChartGenerator:

    def generate_all(self, result, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.confusion_matrix(result, output_dir)
        self.per_attack_radar(result, output_dir)
        self.latency_chart(result, output_dir)

    def confusion_matrix(self, result, output_dir: str):
        cm   = np.array(result.global_metrics.confusion_matrix)
        schema = ...   # from dataset
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=schema, yticklabels=schema, ax=ax)
        ax.set_title(f"Confusion Matrix — {result.model_name}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(f"{output_dir}/{result.model_name}_confusion.png", dpi=150)
        plt.close(fig)
        
    def per_attack_radar(self, result, output_dir: str):
        """Radar chart: each spoke = attack type, value = macro-F1"""
        attacks = list(result.per_attack_metrics.keys())
        values  = [result.per_attack_metrics[a].macro_f1 for a in attacks]
        values += values[:1]      # close the polygon
        angles  = [n/float(len(attacks))*2*np.pi for n in range(len(attacks))]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=1.5, linestyle="solid", color="#1A56DB")
        ax.fill(angles, values, alpha=0.15, color="#1A56DB")
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(attacks, size=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"Per-Attack F1 — {result.model_name}", pad=20)
        fig.savefig(f"{output_dir}/{result.model_name}_radar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
