#!/usr/bin/env python3
"""
Run codegen with canonical solutions to verify the pipeline.
This should produce 100% accuracy when evaluated.
"""
import argparse
import json
import time
from pathlib import Path

from humaneval import load_humaneval


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate outputs using canonical solutions")
    parser.add_argument("--dataset", default="datasets/nominal/HumanEval_py.jsonl")
    parser.add_argument("--prompt", default="prompts/codegen_base.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="runs/canonical.jsonl")
    args = parser.parse_args()

    prompt_prefix = Path(args.prompt).read_text(encoding="utf-8")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for idx, item in enumerate(load_humaneval(args.dataset, limit=args.limit), start=1):
            task_id = item.get("task_id")
            base_prompt = item.get("prompt", "")
            full_prompt = prompt_prefix + base_prompt
            
            # Use the canonical solution from the dataset
            completion = item.get("canonical_solution", "\n    pass\n")

            print(f"Task {idx} - {task_id}")

            record = {
                "task_id": task_id,
                "prompt_file": Path(args.prompt).name,
                "dataset_source": item.get("dataset_source", args.dataset),
                "perturbation_name": Path(item.get("dataset_source", args.dataset)).stem,
                "prompt": full_prompt,
                "completion": completion,
                "model": "canonical",
                "model_path": None,
                "repeat": 0,
                "timestamp": time.time(),
            }

            out.write(json.dumps(record) + "\n")
            out.flush()

    print(f"Done! Output written to {args.out}")


if __name__ == "__main__":
    main()
