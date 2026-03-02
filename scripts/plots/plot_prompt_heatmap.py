"""Prompt-level stability heatmap (averaged across tasks)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main() -> None:
    rows = load("metrics/ollama_llama32_3b_stability.csv") + load(
        "metrics/ollama_mistral_7b_stability.csv"
    )

    grouped = defaultdict(list)
    for r in rows:
        key = (r.get("model_name") or r.get("model"), r.get("prompt_file"))
        grouped[key].append(float(r.get("ast_jaccard_mean", 0.0)))

    models = sorted({m for (m, _) in grouped.keys()})
    prompts = sorted({p for (_, p) in grouped.keys()})
    m_index = {m: i for i, m in enumerate(models)}
    p_index = {p: i for i, p in enumerate(prompts)}

    mat = np.zeros((len(models), len(prompts)))
    for (model, prompt), vals in grouped.items():
        mat[m_index[model], p_index[prompt]] = sum(vals) / len(vals)

    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Prompt-Level Stability (AST Jaccard)")
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    fig.colorbar(im, ax=ax, shrink=0.9, label="AST Jaccard")
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/prompt_stability_heatmap.png", dpi=200)


if __name__ == "__main__":
    main()
