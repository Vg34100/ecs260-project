"""Aggregate model-level stability metrics across multiple CSV files."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model stability metrics")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="metrics/model_comparison.csv")
    args = parser.parse_args()

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)

    for input_path in args.inputs:
        path = Path(input_path)
        rows = read_rows(path)
        for r in rows:
            model_name = r.get("model_name", "") or r.get("model", "")
            key = (r.get("dataset_source", ""), model_name)
            grouped[key].append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        fieldnames = [
            "dataset_source",
            "model",
            "rows",
            "mean_exact_match",
            "mean_ast_jaccard",
            "mean_ast_tree_edit",
            "mean_behavior_consistency",
            "mean_pass_rate",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for key, items in grouped.items():
            dataset_source, model_name = key
            exact = [float(r.get("exact_match_rate", 0.0)) for r in items]
            ast = [float(r.get("ast_jaccard_mean", 0.0)) for r in items]
            tree = [float(r.get("ast_tree_edit_mean", 0.0)) for r in items if r.get("ast_tree_edit_mean")]
            beh = [float(r.get("behavior_consistency", 0.0)) for r in items]
            pr = [float(r.get("pass_rate", 0.0)) for r in items]

            writer.writerow(
                {
                    "dataset_source": dataset_source,
                    "model": model_name,
                    "rows": len(items),
                    "mean_exact_match": f"{mean(exact):.3f}",
                    "mean_ast_jaccard": f"{mean(ast):.3f}",
                    "mean_ast_tree_edit": f"{mean(tree):.3f}" if tree else "",
                    "mean_behavior_consistency": f"{mean(beh):.3f}",
                    "mean_pass_rate": f"{mean(pr):.3f}",
                }
            )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
