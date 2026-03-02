"""Stability atlas heatmap (static PNG)."""
from __future__ import annotations

import csv
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


def build_matrix(rows, metric: str):
    tasks = sorted({r["task_id"] for r in rows})
    prompts = sorted({r["prompt_file"] for r in rows})
    t_index = {t: i for i, t in enumerate(tasks)}
    p_index = {p: i for i, p in enumerate(prompts)}
    mat = np.zeros((len(tasks), len(prompts)))
    for r in rows:
        i = t_index[r["task_id"]]
        j = p_index[r["prompt_file"]]
        mat[i, j] = float(r.get(metric, 0.0))
    return tasks, prompts, mat


def plot(ax, rows, title: str, metric: str):
    tasks, prompts, mat = build_matrix(rows, metric)
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, rotation=45, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=6)
    return im


def main() -> None:
    llama = load("metrics/ollama_llama32_3b_stability.csv")
    mistral = load("metrics/ollama_mistral_7b_stability.csv")
    metric = "ast_jaccard_mean"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, constrained_layout=True)
    im1 = plot(axes[0], llama, "Llama3.2:3b", metric)
    im2 = plot(axes[1], mistral, "Mistral:7b", metric)
    fig.colorbar(
        im2,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        pad=0.04,
        fraction=0.05,
        label="AST Jaccard",
    )
    fig.suptitle("Stability Atlas (AST Jaccard)")
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/stability_atlas.png", dpi=200)


if __name__ == "__main__":
    main()
