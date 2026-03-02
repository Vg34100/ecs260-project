"""Analyze correlations between task features and outcomes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson(x: List[float], y: List[float]) -> float:
    if not x or not y or len(x) != len(y):
        return 0.0
    mx = mean(x)
    my = mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x) ** 0.5
    deny = sum((b - my) ** 2 for b in y) ** 0.5
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def load_rows(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze task feature correlations")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="metrics/task_feature_analysis.csv")
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    for input_path in args.inputs:
        rows.extend(load_rows(input_path))

    prompt_len = [float(r.get("prompt_len_tokens", 0)) for r in rows]
    comp_len = [float(r.get("completion_len_tokens", 0)) for r in rows]
    syntax = [float(r.get("syntax_valid", 0)) for r in rows]
    passed = [float(r.get("passed", 0)) for r in rows]

    results = {
        "prompt_len_vs_pass": pearson(prompt_len, passed),
        "completion_len_vs_pass": pearson(comp_len, passed),
        "syntax_valid_vs_pass": pearson(syntax, passed),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["metric", "pearson_r"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.3f}"])

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
