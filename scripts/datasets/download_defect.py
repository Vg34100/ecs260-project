"""Download CodeXGLUE defect detection and write a JSONL subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download defect detection to JSONL")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--balanced", action="store_true", help="balance clean/buggy")
    parser.add_argument("--out", default="datasets/defect_subset.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("code_x_glue_cc_defect_detection", split=args.split)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.balanced:
        with out_path.open("w", encoding="utf-8") as out:
            for i, item in enumerate(ds):
                if args.limit and i >= args.limit:
                    break
                label = 1 if bool(item.get("target", 0)) else 0
                record = {
                    "task_id": f"defect/{args.split}/{i}",
                    "input": item.get("func", ""),
                    "target": label,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {args.limit} rows to {out_path}")
        return

    target_per_class = args.limit // 2 if args.limit else 50
    counts = {"0": 0, "1": 0}
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for i, item in enumerate(ds):
            label = 1 if bool(item.get("target", 0)) else 0
            label = str(label)
            if label not in counts:
                continue
            if counts[label] >= target_per_class:
                continue
            record = {
                "task_id": f"defect/{args.split}/{i}",
                "input": item.get("func", ""),
                "target": int(label),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[label] += 1
            written += 1
            if counts["0"] >= target_per_class and counts["1"] >= target_per_class:
                break

    print(f"Wrote {written} rows to {out_path} (balanced)")


if __name__ == "__main__":
    main()
