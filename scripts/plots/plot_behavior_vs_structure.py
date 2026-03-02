"""Scatter: AST similarity vs behavior consistency by model."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main() -> None:
    llama = load("metrics/ollama_llama32_3b_stability.csv")
    mistral = load("metrics/ollama_mistral_7b_stability.csv")

    def points(rows):
        x = [float(r["ast_jaccard_mean"]) for r in rows]
        y = [float(r["behavior_consistency"]) for r in rows]
        return x, y

    x1, y1 = points(llama)
    x2, y2 = points(mistral)

    plt.figure(figsize=(7, 5))
    plt.scatter(x1, y1, label="Llama3.2:3b", alpha=0.8)
    plt.scatter(x2, y2, label="Mistral:7b", alpha=0.8)
    plt.xlabel("AST Jaccard Similarity")
    plt.ylabel("Behavior Consistency")
    plt.title("Behavior vs Structure")
    plt.legend()
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/behavior_vs_structure.png", dpi=200)


if __name__ == "__main__":
    main()
