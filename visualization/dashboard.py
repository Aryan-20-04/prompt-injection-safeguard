"""
Static HTML dashboard builder.

Reads leaderboard.json + metrics files and produces a single self-contained
dashboard.html with:
  • Sortable model leaderboard table
  • Latency vs Performance scatter (inline Chart.js)
  • Per-attack-type detection chart
  • Per-dataset comparison table
  • Embedded PNG charts (base64)
  • Run metadata summary
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DashboardBuilder:

    def build(
        self,
        leaderboard_path: Path,
        metrics_dir: Path,
        charts_dir: Path,
        output_dir: Path,
    ) -> Path:
        leaderboard: List[Dict[str, Any]] = json.loads(
            Path(leaderboard_path).read_text()
        )
        metrics = self._load_metrics(metrics_dir)
        chart_embeds = self._embed_charts(charts_dir)

        html = self._render(leaderboard, metrics, chart_embeds)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "dashboard.html"
        path.write_text(html, encoding="utf-8")

        logger.info("[dashboard] written → %s", path)
        print(f"[dashboard] written → {path}")
        return path

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        leaderboard: List[Dict],
        metrics: List[Dict],
        charts: Dict[str, str],
    ) -> str:

        leaderboard_rows = self._render_leaderboard_rows(leaderboard)
        per_attack_rows  = self._render_per_attack_rows(metrics)
        chart_sections   = self._render_chart_sections(charts)
        js_data          = self._render_js_data(leaderboard)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prompt Injection Benchmark Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2e3347;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #6366f1;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; color: var(--muted); margin: 2rem 0 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #252836; color: var(--muted); font-weight: 600; padding: 0.6rem 0.8rem; text-align: left; cursor: pointer; user-select: none; white-space: nowrap; }}
  th:hover {{ color: var(--text); }}
  td {{ padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #252836; }}
  .rank {{ font-weight: 700; color: var(--accent); }}
  .pill {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
  .pill-green {{ background: #14532d; color: var(--green); }}
  .pill-red   {{ background: #450a0a; color: var(--red); }}
  .pill-yellow{{ background: #451a03; color: var(--yellow); }}
  .bar-cell {{ display: flex; align-items: center; gap: 0.5rem; }}
  .bar {{ height: 8px; background: var(--accent); border-radius: 4px; min-width: 2px; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; }}
  .chart-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; }}
  .chart-card img {{ width: 100%; border-radius: 8px; }}
  .chart-card h3 {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 0.75rem; }}
  canvas {{ max-height: 320px; }}
  .badge-binary {{ background: #1e3a5f; color: #93c5fd; }}
  .badge-multi  {{ background: #3b0764; color: #d8b4fe; }}
</style>
</head>
<body>

<h1>🛡️ Prompt Injection Benchmark</h1>
<p class="subtitle">Research-grade evaluation framework — model leaderboard &amp; analysis</p>

<h2>📊 Model Leaderboard</h2>
<div class="card">
<table id="leaderboard">
  <thead>
    <tr>
      <th onclick="sortTable(0)">#</th>
      <th onclick="sortTable(1)">Model</th>
      <th onclick="sortTable(2)">Dataset</th>
      <th onclick="sortTable(3)">Mode</th>
      <th onclick="sortTable(4)">Micro F1 ↑</th>
      <th onclick="sortTable(5)">Macro F1 ↑</th>
      <th onclick="sortTable(6)">Precision</th>
      <th onclick="sortTable(7)">Recall</th>
      <th onclick="sortTable(8)">ADR ↑</th>
      <th onclick="sortTable(9)">FPR ↓</th>
      <th onclick="sortTable(10)">P50 (ms)</th>
      <th onclick="sortTable(11)">Samples</th>
    </tr>
  </thead>
  <tbody>
{leaderboard_rows}
  </tbody>
</table>
</div>

<h2>📈 Latency vs Performance</h2>
<div class="card" style="max-width:700px">
  <canvas id="latencyChart"></canvas>
</div>

<h2>🎯 Per-Attack Detection</h2>
<div class="card">
<table>
  <thead>
    <tr><th>Model</th><th>Dataset</th><th>Attack Type</th><th>Micro F1</th><th>FPR</th><th>ADR</th><th>Samples</th></tr>
  </thead>
  <tbody>
{per_attack_rows}
  </tbody>
</table>
</div>

<h2>📉 Charts</h2>
<div class="charts-grid">
{chart_sections}
</div>

<script>
{js_data}

// Latency vs Performance scatter
const ctx = document.getElementById('latencyChart').getContext('2d');
new Chart(ctx, {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'Models',
      data: chartData.map(d => ({{ x: d.latency_p50_ms, y: d.micro_f1, label: d.model }})),
      backgroundColor: '#6366f1',
      pointRadius: 7,
      pointHoverRadius: 10,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      tooltip: {{
        callbacks: {{
          label: ctx => `${{ctx.raw.label}} — P50: ${{ctx.raw.x.toFixed(1)}}ms | F1: ${{ctx.raw.y.toFixed(4)}}`
        }}
      }},
      legend: {{ display: false }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Median Latency (ms)', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#2e3347' }} }},
      y: {{ title: {{ display: true, text: 'Micro F1', color: '#94a3b8' }}, min: 0, max: 1, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#2e3347' }} }}
    }}
  }}
}});

// Sortable table
function sortTable(col) {{
  const table = document.getElementById('leaderboard');
  const rows = Array.from(table.querySelectorAll('tbody tr'));
  const asc = table.dataset.sortCol == col && table.dataset.sortDir === 'asc';
  rows.sort((a, b) => {{
    const va = a.cells[col].dataset.val ?? a.cells[col].textContent.trim();
    const vb = b.cells[col].dataset.val ?? b.cells[col].textContent.trim();
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => table.querySelector('tbody').appendChild(r));
  table.dataset.sortCol = col;
  table.dataset.sortDir = asc ? 'desc' : 'asc';
}}
</script>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Sub-renderers
    # ------------------------------------------------------------------

    def _render_leaderboard_rows(self, leaderboard: List[Dict]) -> str:
        rows = ""
        for r in leaderboard:
            micro = r.get("micro_f1", 0)
            fpr   = r.get("false_positive_rate", 0)
            adr   = r.get("attack_detection_rate", 0)
            mode  = r.get("eval_mode", "binary")
            mode_pill = (
                f'<span class="pill badge-binary">binary</span>' if mode == "binary"
                else f'<span class="pill badge-multi">multi</span>'
            )
            fpr_cls = "pill-green" if fpr < 0.05 else ("pill-yellow" if fpr < 0.15 else "pill-red")
            adr_cls = "pill-green" if adr > 0.90 else ("pill-yellow" if adr > 0.70 else "pill-red")

            rows += f"""    <tr>
      <td class="rank" data-val="{r.get('rank',0)}">{r.get('rank','')}</td>
      <td><strong>{r.get('model','')}</strong></td>
      <td style="color:var(--muted);font-size:0.8rem">{r.get('dataset','')}</td>
      <td>{mode_pill}</td>
      <td data-val="{micro:.6f}">
        <div class="bar-cell"><div class="bar" style="width:{micro*160:.0f}px"></div>{micro:.4f}</div>
      </td>
      <td data-val="{r.get('macro_f1',0):.6f}">{r.get('macro_f1',0):.4f}</td>
      <td>{r.get('precision',0):.4f}</td>
      <td>{r.get('recall',0):.4f}</td>
      <td data-val="{adr:.6f}"><span class="pill {adr_cls}">{adr:.3f}</span></td>
      <td data-val="{fpr:.6f}"><span class="pill {fpr_cls}">{fpr:.3f}</span></td>
      <td>{r.get('latency_p50_ms',0):.1f}</td>
      <td style="color:var(--muted)">{r.get('sample_count',0)}</td>
    </tr>\n"""
        return rows

    def _render_per_attack_rows(self, metrics: List[Dict]) -> str:
        rows = ""
        for d in metrics:
            model = d.get("model_name", "")
            dataset = d.get("dataset_name", "")
            for attack, m in d.get("per_attack_metrics", {}).items():
                f1  = m.get("micro_f1", 0)
                fpr = m.get("false_positive_rate", 0)
                adr = m.get("attack_detection_rate", 0)
                n   = m.get("sample_count", 0)
                rows += f"""    <tr>
      <td><strong>{model}</strong></td>
      <td style="color:var(--muted);font-size:0.8rem">{dataset}</td>
      <td><span class="pill pill-yellow">{attack}</span></td>
      <td>{f1:.4f}</td>
      <td>{fpr:.3f}</td>
      <td>{adr:.3f}</td>
      <td style="color:var(--muted)">{n}</td>
    </tr>\n"""
        return rows

    def _render_chart_sections(self, charts: Dict[str, str]) -> str:
        titles = {
            "model_comparison.png": "Micro F1 by Model & Dataset",
            "latency_comparison.png": "Inference Latency (P50 / P95)",
            "macro_vs_micro.png": "Macro vs Micro F1 Scatter",
            "per_attack_heatmap.png": "Per-Attack-Type F1 Heatmap",
            "fpr_adr.png": "Attack Detection Rate vs False Positive Rate",
        }
        html = ""
        for filename, b64 in charts.items():
            title = titles.get(filename, filename)
            html += f"""  <div class="chart-card">
    <h3>{title}</h3>
    <img src="data:image/png;base64,{b64}" alt="{title}">
  </div>\n"""
        return html

    def _render_js_data(self, leaderboard: List[Dict]) -> str:
        simplified = [
            {
                "model": r.get("model", ""),
                "micro_f1": r.get("micro_f1", 0),
                "latency_p50_ms": r.get("latency_p50_ms", 0),
            }
            for r in leaderboard
        ]
        return f"const chartData = {json.dumps(simplified)};"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_metrics(metrics_dir: Path) -> List[Dict]:
        results = []
        for f in sorted(Path(metrics_dir).glob("*.json")):
            try:
                results.append(json.loads(f.read_text()))
            except Exception:
                pass
        return results

    @staticmethod
    def _embed_charts(charts_dir: Path) -> Dict[str, str]:
        """Read PNG files and return base64-encoded strings keyed by filename."""
        charts: Dict[str, str] = {}
        charts_dir = Path(charts_dir)
        if not charts_dir.exists():
            return charts
        for png in sorted(charts_dir.glob("*.png")):
            with open(png, "rb") as f:
                charts[png.name] = base64.b64encode(f.read()).decode("ascii")
        return charts
