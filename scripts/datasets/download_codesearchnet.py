"""Download CodeSearchNet (Python) and write a JSONL subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CodeSearchNet Python to JSONL")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="datasets/codesearchnet_py_subset.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("code_search_net", "python", split=args.split)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for i, item in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            record = {
                "task_id": f"codesearchnet/python/{args.split}/{i}",
                "input": item.get("func_code_string", ""),
                "target": item.get("func_documentation_string", ""),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {args.limit} rows to {out_path}")


if __name__ == "__main__":
    main()
