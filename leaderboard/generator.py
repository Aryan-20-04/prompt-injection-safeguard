import json
from pathlib import Path
from .ranking import rank_models


class LeaderboardGenerator:

    def generate(self, metrics_dir: Path, output_dir: Path):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metric_files = list(Path(metrics_dir).glob("*.json"))

        results = []

        for f in metric_files:

            data = json.loads(f.read_text())

            results.append(data)

        leaderboard = sorted(
            results,
            key=lambda r: r["global_metrics"]["micro_f1"],
            reverse=True
        )

        table = []

        for i, r in enumerate(leaderboard, start=1):

            table.append({
                "rank": i,
                "model": r["model_name"],
                "dataset": r["dataset_name"],
                "micro_f1": r["global_metrics"]["micro_f1"],
                "macro_f1": r["global_metrics"]["macro_f1"],
                "precision": r["global_metrics"]["precision"],
                "recall": r["global_metrics"]["recall"],
            })

        leaderboard_path = output_dir / "leaderboard.json"

        leaderboard_path.write_text(json.dumps(table, indent=2))

        print(f"[leaderboard] written → {leaderboard_path}")