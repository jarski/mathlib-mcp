"""Render the headline chart (solved rate by condition and model) as light/dark SVGs.

    uv run python -m bench.harness.chart bench/results/eval-v1 --out docs

GitHub picks the variant with ``<picture>`` + ``prefers-color-scheme``. Colors are the
validated reference palette (categorical slots 1-2 per mode; both pass the dataviz
validator); text uses ink tokens, never the series color.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

from bench.harness.config import CONDITIONS

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "ring": "rgba(11,11,11,0.10)",
              "series": ["#2a78d6", "#eb6834"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "ring": "rgba(255,255,255,0.10)",
             "series": ["#3987e5", "#d95926"]},
}
CONDITION_NAMES = {"none": ("No help", "model knowledge only"),
                   "docs": ("Docs in prompt", "llms-full.txt, 101–134k tokens"),
                   "mcp": ("MCP tools", "mathlib-mcp, on demand")}
MODEL_NAMES = {"haiku": "Claude Haiku 4.5", "sonnet": "Claude Sonnet 5.5"}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

W, H = 720, 400
LEFT, RIGHT, TOP, BOTTOM = 64, 24, 92, 72
BAR, GAP = 24, 14  # bars of a pair do not touch, so their value labels fit


def column(x: float, y: float, height: float, color: str) -> str:
    """A column with a 4px rounded data-end and a square baseline."""
    r = min(4.0, height)
    if height <= 0:
        return ""
    return (f'<path d="M{x:.1f},{y + height:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},'
            f'{y:.1f} H{x + BAR - r:.1f} Q{x + BAR:.1f},{y:.1f} {x + BAR:.1f},{y + r:.1f} '
            f'V{y + height:.1f} Z" fill="{color}"/>')


def render(summary: dict, theme: str, samples: int, tasks: int) -> str:
    t = THEMES[theme]
    models = [m for m in ("haiku", "sonnet") if m in summary["models"]]
    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM
    base = TOP + plot_h

    def y_of(value: float) -> float:
        return TOP + plot_h * (1 - value)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc" '
             f'font-family=\'{FONT}\'>']
    desc = "; ".join(
        f"{MODEL_NAMES[m]} {CONDITION_NAMES[c][0]}: "
        f"{100 * summary['models'][m][c]['solved'][0]:.0f}% "
        f"(95% CI {100 * summary['models'][m][c]['solved'][1]:.0f}–"
        f"{100 * summary['models'][m][c]['solved'][2]:.0f}%)"
        for m in models for c in CONDITIONS if c in summary["models"][m])
    parts += [f'<title id="title">Solved rate on {tasks} GPU tasks by condition and model'
              f'</title>', f'<desc id="desc">{escape(desc)}</desc>',
              f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" '
              f'fill="{t["surface"]}" stroke="{t["ring"]}"/>']
    parts.append(f'<text x="{LEFT - 40}" y="34" font-size="16" font-weight="600" '
                 f'fill="{t["ink"]}">Solved rate on {tasks} GPU tasks</text>')
    parts.append(f'<text x="{LEFT - 40}" y="56" font-size="12" fill="{t["ink2"]}">'
                 f'Passes all checks on a Tesla T4 · {samples} '
                 f'{"sample" if samples == 1 else "samples"} per task · whiskers: '
                 f'95% CI (bootstrap over tasks)</text>')
    # legend (always present for two series)
    lx = W - RIGHT - 300
    for i, model in enumerate(models):
        x = lx + i * 150
        parts.append(f'<rect x="{x}" y="25" width="10" height="10" rx="2" '
                     f'fill="{t["series"][i]}"/>')
        parts.append(f'<text x="{x + 16}" y="34" font-size="12" fill="{t["ink2"]}">'
                     f'{MODEL_NAMES[model]}</text>')
    # grid + y ticks
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(tick)
        color = t["axis"] if tick == 0 else t["grid"]
        parts.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="{color}" stroke-width="1"/>')
        parts.append(f'<text x="{LEFT - 10}" y="{y + 4:.1f}" font-size="12" '
                     f'text-anchor="end" fill="{t["muted"]}" '
                     f'style="font-variant-numeric: tabular-nums">{int(tick * 100)}%</text>')
    # groups
    band = plot_w / len(CONDITIONS)
    pair = len(models) * BAR + (len(models) - 1) * GAP
    for g, condition in enumerate(CONDITIONS):
        cx = LEFT + band * (g + 0.5)
        name, detail = CONDITION_NAMES[condition]
        parts.append(f'<text x="{cx:.1f}" y="{base + 24:.1f}" font-size="13" '
                     f'text-anchor="middle" fill="{t["ink"]}">{name}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{base + 42:.1f}" font-size="11" '
                     f'text-anchor="middle" fill="{t["muted"]}">{escape(detail)}</text>')
        for i, model in enumerate(models):
            stats = summary["models"][model].get(condition)
            if stats is None:
                continue
            mean, low, high = stats["solved"]
            x = cx - pair / 2 + i * (BAR + GAP)
            parts.append(column(x, y_of(mean), base - y_of(mean), t["series"][i]))
            mid = x + BAR / 2
            parts.append(f'<line x1="{mid:.1f}" x2="{mid:.1f}" y1="{y_of(high):.1f}" '
                         f'y2="{y_of(low):.1f}" stroke="{t["ink2"]}" stroke-width="1"/>')
            for v in (low, high):
                parts.append(f'<line x1="{mid - 4:.1f}" x2="{mid + 4:.1f}" '
                             f'y1="{y_of(v):.1f}" y2="{y_of(v):.1f}" stroke="{t["ink2"]}" '
                             f'stroke-width="1"/>')
            parts.append(f'<text x="{mid:.1f}" y="{y_of(high) - 6:.1f}" font-size="11" '
                         f'font-weight="600" text-anchor="middle" fill="{t["ink"]}">'
                         f'{100 * mean:.0f}%</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    config = json.loads((args.run / "run.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        svg = render(summary, theme, config["reps"], len(config["tasks"]))
        (args.out / f"results-{theme}.svg").write_text(svg)
    print(f"wrote {args.out}/results-light.svg and results-dark.svg")


if __name__ == "__main__":
    main()
