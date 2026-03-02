"""Stacked error breakdown for test generation."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def summarize(path: str):
    rows = list(csv.DictReader(Path(path).open()))
    total = len(rows) or 1
    passed = sum(int(r["passed"]) for r in rows)
    errors = Counter(r["error"] for r in rows if r.get("error"))
    no_assert = errors.get("no asserts found", 0)
    timeout = errors.get("timeout", 0)
    invalid = total - passed
    return {
        "passed": passed / total,
        "invalid": invalid / total,
        "no_assert": no_assert / total,
        "timeout": timeout / total,
    }


def main() -> None:
    llama = summarize("metrics/mbpp_testgen_llama32_3b_eval.csv")
    mistral = summarize("metrics/mbpp_testgen_mistral_7b_eval.csv")

    labels = ["Passed", "Invalid", "No asserts", "Timeout"]
    llama_vals = [llama["passed"], llama["invalid"], llama["no_assert"], llama["timeout"]]
    mistral_vals = [mistral["passed"], mistral["invalid"], mistral["no_assert"], mistral["timeout"]]

    x = range(len(labels))
    plt.figure(figsize=(7, 4))
    plt.bar([i - 0.2 for i in x], llama_vals, width=0.4, label="Llama3.2:3b")
    plt.bar([i + 0.2 for i in x], mistral_vals, width=0.4, label="Mistral:7b")
    plt.xticks(list(x), labels, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Share of Outputs")
    plt.title("MBPP Test Generation Error Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/mbpp_testgen_errors.png", dpi=200)


if __name__ == "__main__":
    main()
