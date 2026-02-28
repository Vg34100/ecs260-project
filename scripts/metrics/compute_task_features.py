"""Compute task/prompt features for RQ4 analysis."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.ast_utils import parse_python


def load_runs(path: str) -> Iterable[Dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_eval(path: str) -> Dict[Tuple[str, str, str, str, int], int]:
    results: Dict[Tuple[str, str, str, str, int], int] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get("dataset_source", ""),
                row.get("task_id", ""),
                row.get("prompt_file", ""),
                row.get("model_name", "") or row.get("model", ""),
                int(row.get("repeat", 0)),
            )
            results[key] = int(row.get("passed", 0))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RQ4 feature table")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", default="metrics/rq4_features.csv")
    args = parser.parse_args()

    eval_map = load_eval(args.eval)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        fieldnames = [
            "dataset_source",
            "task_id",
            "prompt_file",
            "model_name",
            "repeat",
            "prompt_len_chars",
            "prompt_len_tokens",
            "completion_len_chars",
            "completion_len_tokens",
            "syntax_valid",
            "passed",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in load_runs(args.runs):
            dataset_source = str(rec.get("dataset_source", ""))
            task_id = str(rec.get("task_id", ""))
            prompt_file = str(rec.get("prompt_file", ""))
            model_name = str(rec.get("model_name", "")) or str(rec.get("model", ""))
            repeat = int(rec.get("repeat", 0))
            prompt = str(rec.get("prompt", ""))
            completion = str(rec.get("completion", ""))

            prompt_tokens = prompt.split()
            completion_tokens = completion.split()

            syntax_valid = 1 if parse_python(completion) is not None else 0
            passed = eval_map.get(
                (dataset_source, task_id, prompt_file, model_name, repeat), 0
            )

            writer.writerow(
                {
                    "dataset_source": dataset_source,
                    "task_id": task_id,
                    "prompt_file": prompt_file,
                    "model_name": model_name,
                    "repeat": repeat,
                    "prompt_len_chars": len(prompt),
                    "prompt_len_tokens": len(prompt_tokens),
                    "completion_len_chars": len(completion),
                    "completion_len_tokens": len(completion_tokens),
                    "syntax_valid": syntax_valid,
                    "passed": passed,
                }
            )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
