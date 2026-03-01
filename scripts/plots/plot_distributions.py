"""Distribution plots for stability metrics (violin + jitter)."""
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


def extract(rows, key: str):
    return [float(r.get(key, 0.0)) for r in rows]


def jitter(n, scale=0.02):
    return np.random.uniform(-scale, scale, size=n)


def main() -> None:
    llama = load("metrics/ollama_llama32_3b_stability.csv")
    mistral = load("metrics/ollama_mistral_7b_stability.csv")

    metrics = [
        ("exact_match_rate", "Exact Match"),
        ("ast_jaccard_mean", "AST Jaccard"),
        ("behavior_consistency", "Behavior Consistency"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (key, label) in zip(axes, metrics):
        x1 = extract(llama, key)
        x2 = extract(mistral, key)

        parts = ax.violinplot([x1, x2], positions=[1, 2], showmeans=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.5)

        ax.scatter(1 + jitter(len(x1)), x1, s=10, alpha=0.6)
        ax.scatter(2 + jitter(len(x2)), x2, s=10, alpha=0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Llama3.2:3b", "Mistral:7b"], rotation=15)
        ax.set_title(label)
        ax.set_ylim(0, 1)

    plt.suptitle("Stability Metric Distributions")
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/stability_distributions.png", dpi=200)


if __name__ == "__main__":
    main()
