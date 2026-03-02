"""Compare MBPP test generation pass rates."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def pass_rate(path: str) -> float:
    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        return 0.0
    passed = sum(int(r["passed"]) for r in rows)
    return passed / len(rows)


def main() -> None:
    llama = pass_rate("metrics/mbpp_testgen_llama32_3b_eval.csv")
    mistral = pass_rate("metrics/mbpp_testgen_mistral_7b_eval.csv")

    plt.figure(figsize=(5, 4))
    plt.bar(["Llama3.2:3b", "Mistral:7b"], [llama, mistral])
    plt.ylim(0, 1)
    plt.ylabel("Pass Rate")
    plt.title("MBPP Test Generation")
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/mbpp_testgen_compare.png", dpi=200)


if __name__ == "__main__":
    main()
