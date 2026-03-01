"""Plots for summarization, bug detection, and test generation tasks."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: str):
    return list(csv.DictReader(Path(path).open()))


def plot_summarization():
    xsum = read_csv("metrics/xsum_llama32_3b_eval.csv")
    csn = read_csv("metrics/codesearchnet_llama32_3b_eval.csv")
    xsum_vals = [float(r["similarity"]) for r in xsum]
    csn_vals = [float(r["similarity"]) for r in csn]

    plt.figure(figsize=(6, 4))
    plt.boxplot([xsum_vals, csn_vals], tick_labels=["XSum", "CodeSearchNet"], showfliers=False)
    plt.ylim(0, 1)
    plt.ylabel("Embedding Similarity")
    plt.title("Summarization Similarity Distributions")
    plt.tight_layout()
    plt.savefig("figures/summarization_similarity.png", dpi=200)


def plot_testgen():
    def pass_rate(path: str):
        rows = read_csv(path)
        passed = sum(int(r["passed"]) for r in rows)
        total = len(rows)
        return passed / total if total else 0.0

    llama = pass_rate("metrics/mbpp_testgen_llama32_3b_eval.csv")
    mistral = pass_rate("metrics/mbpp_testgen_mistral_7b_eval.csv")

    plt.figure(figsize=(5, 4))
    plt.bar(["Llama3.2:3b", "Mistral:7b"], [llama, mistral])
    plt.ylim(0, 1)
    plt.ylabel("Pass Rate")
    plt.title("MBPP Test Generation")
    plt.tight_layout()
    plt.savefig("figures/mbpp_testgen_compare.png", dpi=200)


def plot_bug_detection_bias():
    rows = read_csv("metrics/defect_llama32_3b_balanced_eval.csv")
    pred_buggy = sum(1 for r in rows if r["pred"] == "buggy")
    pred_clean = sum(1 for r in rows if r["pred"] == "clean")
    total = len(rows)

    plt.figure(figsize=(5, 4))
    plt.bar(["Pred buggy", "Pred clean"], [pred_buggy / total, pred_clean / total])
    plt.ylim(0, 1)
    plt.ylabel("Prediction Share")
    plt.title("Bug Detection Prediction Bias")
    plt.tight_layout()
    plt.savefig("figures/bug_detection_bias.png", dpi=200)


def main() -> None:
    Path("figures").mkdir(exist_ok=True)
    plot_summarization()
    plot_bug_detection_bias()
    plot_testgen()


if __name__ == "__main__":
    main()
