# bench/leaderboard.py
import json, csv, os
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class LeaderboardRow:
    model: str; dataset: str; micro_f1: float; macro_f1: float
    precision: float; recall: float; latency_p50: float; latency_p95: float
    run_date: str; config_hash: str

class LeaderboardGenerator:
    def generate(self, results_dir: str, output_dir: str):
        rows = self._collect_rows(results_dir)
        rows.sort(key=lambda r: r.micro_f1, reverse=True)
        self._write_csv(rows, output_dir)
        self._write_markdown(rows, output_dir)
        self._write_html(rows, output_dir)
        print(f"Leaderboard written to {output_dir} ({len(rows)} entries)")
        
    def _collect_rows(self, results_dir: str) -> List[LeaderboardRow]:
        rows = []
        for path in Path(results_dir).rglob("metrics/*.json"):
            with open(path) as f:
                data = json.load(f)
            meta_path = path.parent.parent / "run_metadata.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            gm = data["global_metrics"]
            rows.append(LeaderboardRow(
                model=data["model_name"], dataset=data["dataset_name"],
                micro_f1=gm["micro_f1"], macro_f1=gm["macro_f1"],
                precision=gm["precision"], recall=gm["recall"],
                latency_p50=data["latency_p50_ms"], latency_p95=data["latency_p95_ms"],
                run_date=meta.get("timestamp",""),
                config_hash=meta.get("config_hash",""),
            ))
        return rows
    
    def _write_markdown(self, rows, output_dir):
        header = "| Rank | Model | Dataset | Micro-F1 | Macro-F1 | P50 (ms) |"
        sep    = "|------|-------|---------|----------|----------|----------|"
        lines  = [header, sep]
        for i, r in enumerate(rows, 1):
            lines.append(
                f"| {i} | {r.model} | {r.dataset} | "
                f"{r.micro_f1:.4f} | {r.macro_f1:.4f} | {r.latency_p50:.1f} |"
            )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir)/"leaderboard.md").write_text("\n".join(lines))