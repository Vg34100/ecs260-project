"""Bar plot of task-feature correlations."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    path = Path("metrics/task_feature_analysis.csv")
    rows = list(csv.DictReader(path.open()))
    labels = [r["metric"] for r in rows]
    values = [float(r["pearson_r"]) for r in rows]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Pearson r")
    plt.title("Task Feature Correlations")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/task_feature_correlations.png", dpi=200)


if __name__ == "__main__":
    main()
