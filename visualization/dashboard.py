import json
from pathlib import Path


class DashboardBuilder:

    def build(self, leaderboard_path: Path, charts_dir: Path, output_dir: Path):

        leaderboard = json.loads(Path(leaderboard_path).read_text())

        html = self._render_html(leaderboard, charts_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "dashboard.html"

        path.write_text(html)

        print(f"[dashboard] written → {path}")

    def _render_html(self, leaderboard, charts_dir):

        rows = ""

        for r in leaderboard:

            rows += f"""
            <tr>
                <td>{r['rank']}</td>
                <td>{r['model']}</td>
                <td>{r['dataset']}</td>
                <td>{r['micro_f1']:.4f}</td>
                <td>{r['macro_f1']:.4f}</td>
                <td>{r['precision']:.4f}</td>
                <td>{r['recall']:.4f}</td>
            </tr>
            """

        return f"""
        <html>
        <head>
            <title>Benchmark Dashboard</title>
            <style>
                body {{ font-family: Arial; margin:40px }}
                table {{ border-collapse: collapse }}
                th, td {{ border:1px solid #ccc; padding:8px }}
                th {{ background:#eee }}
            </style>
        </head>
        <body>

        <h1>Benchmark Leaderboard</h1>

        <table>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Dataset</th>
                <th>Micro F1</th>
                <th>Macro F1</th>
                <th>Precision</th>
                <th>Recall</th>
            </tr>

            {rows}

        </table>

        <h2>Charts</h2>

        <img src="../charts/model_comparison.png" width="700">
        <img src="../charts/latency_comparison.png" width="700">

        </body>
        </html>
        """