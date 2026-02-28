"""Aggregate paraphrase sensitivity by model and dataset."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paraphrase sensitivity")
    parser.add_argument("--paraphrase", default="metrics/paraphrase_sensitivity.csv")
    parser.add_argument("--out", default="metrics/paraphrase_summary.csv")
    args = parser.parse_args()

    rows = []
    with Path(args.paraphrase).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (r.get("dataset_source", ""), r.get("model", ""))
        grouped[key].append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        fieldnames = [
            "dataset_source",
            "model",
            "tasks",
            "mean_exact_match",
            "mean_ast_jaccard",
            "mean_behavior_consistency",
            "mean_pass_rate",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for key, items in grouped.items():
            dataset_source, model = key
            exact = [float(r.get("mean_exact_match", 0.0)) for r in items]
            ast = [float(r.get("mean_ast_jaccard", 0.0)) for r in items]
            beh = [float(r.get("mean_behavior_consistency", 0.0)) for r in items]
            pr = [float(r.get("mean_pass_rate", 0.0)) for r in items]

            writer.writerow(
                {
                    "dataset_source": dataset_source,
                    "model": model,
                    "tasks": len(items),
                    "mean_exact_match": f"{mean(exact):.3f}",
                    "mean_ast_jaccard": f"{mean(ast):.3f}",
                    "mean_behavior_consistency": f"{mean(beh):.3f}",
                    "mean_pass_rate": f"{mean(pr):.3f}",
                }
            )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
