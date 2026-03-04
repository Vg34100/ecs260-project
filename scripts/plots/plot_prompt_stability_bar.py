"""Bar chart: average stability metrics per prompt variant, one subplot per metric.

Reads stability CSVs produced by compute_stability.py and groups rows by
(model_name, prompt_file).  For each metric the plot shows one bar group per
prompt variant, with one bar per model.  Missing metric values (empty string)
are skipped so that partially-filled CSVs still render correctly.

Output: figures/prompt_stability_bar.png
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_STABILITY_FILES = [
    "metrics/ollama_llama32_3b_stability.csv",
    "metrics/ollama_mistral_7b_stability.csv",
]

_METRICS = [
    ("exact_match_rate", "Exact Match Rate"),
    ("ast_jaccard_mean", "AST Jaccard"),
    ("behavior_consistency", "Behavior Consistency"),
]


def load_stability_rows(paths: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  Skipping missing file: {path}")
            continue
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    return rows


def model_label(row: Dict[str, str]) -> str:
    """Prefer model_name (e.g. 'llama3.2:3b') over generic backend name."""
    return row.get("model_name") or row.get("model") or "unknown"


def aggregate(
    rows: List[Dict[str, str]],
) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    """Return {(model_label, prompt_file): {metric: [values]}}."""
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {m: [] for m, _ in _METRICS}
    )
    for row in rows:
        key = (model_label(row), row.get("prompt_file", ""))
        for metric, _ in _METRICS:
            raw = row.get(metric, "")
            if raw != "":
                try:
                    grouped[key][metric].append(float(raw))
                except ValueError:
                    pass
    return grouped


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main() -> None:
    rows = load_stability_rows(_STABILITY_FILES)
    if not rows:
        print("No stability data found; skipping plot_prompt_stability_bar.")
        return

    grouped = aggregate(rows)

    models = sorted({m for (m, _) in grouped})
    prompts = sorted({p for (_, p) in grouped})

    if not models or not prompts:
        print("No data to plot.")
        return

    n_metrics = len(_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    bar_width = 0.8 / max(len(models), 1)
    x = np.arange(len(prompts))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (metric_key, metric_label) in zip(axes, _METRICS):
        for idx, model in enumerate(models):
            means = []
            for prompt in prompts:
                vals = grouped.get((model, prompt), {}).get(metric_key, [])
                means.append(sum(vals) / len(vals) if vals else 0.0)
            offset = (idx - (len(models) - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                means,
                width=bar_width,
                label=model,
                color=colors[idx % len(colors)],
                alpha=0.85,
            )

        ax.set_title(metric_label)
        ax.set_xticks(x)
        # Strip directory prefix and .txt suffix for readability.
        short_prompts = [Path(p).stem for p in prompts]
        ax.set_xticklabels(short_prompts, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Avg value" if ax is axes[0] else "")
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle("Avg Stability per Prompt Variant", fontsize=12)
    plt.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    out = "figures/prompt_stability_bar.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
