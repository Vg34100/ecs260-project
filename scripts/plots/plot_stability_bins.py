"""Binned stability distributions by model."""
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


def bin_counts(values):
    bins = [0, 0, 0]
    for v in values:
        if v < 0.34:
            bins[0] += 1
        elif v < 0.67:
            bins[1] += 1
        else:
            bins[2] += 1
    total = sum(bins) or 1
    return [b / total for b in bins]


def main() -> None:
    llama = load("metrics/ollama_llama32_3b_stability.csv")
    mistral = load("metrics/ollama_mistral_7b_stability.csv")

    def extract(rows, key):
        return [float(r.get(key, 0.0)) for r in rows]

    llama_vals = extract(llama, "exact_match_rate")
    mistral_vals = extract(mistral, "exact_match_rate")

    llama_bins = bin_counts(llama_vals)
    mistral_bins = bin_counts(mistral_vals)

    labels = ["Low (<0.34)", "Mid (0.34-0.67)", "High (>=0.67)"]
    x = range(len(labels))

    plt.figure(figsize=(6, 4))
    plt.bar([i - 0.2 for i in x], llama_bins, width=0.4, label="Llama3.2:3b")
    plt.bar([i + 0.2 for i in x], mistral_bins, width=0.4, label="Mistral:7b")
    plt.xticks(list(x), labels, rotation=15, ha="right")
    plt.ylabel("Share of Tasks")
    plt.title("Exact-Match Stability Bins")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/stability_bins.png", dpi=200)


if __name__ == "__main__":
    main()
