"""Evaluate bug detection outputs (buggy vs clean)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalize_label(text: str) -> str:
    text = (text or "").strip().lower()
    if "bug" in text:
        return "buggy"
    if "clean" in text or "no bug" in text or "not buggy" in text:
        return "clean"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bug detection")
    parser.add_argument("--runs", default="runs/defect_runs.jsonl")
    parser.add_argument("--out", default="metrics/defect_eval.csv")
    args = parser.parse_args()

    rows = []
    with Path(args.runs).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    unknown = 0

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        fieldnames = ["task_id", "repeat", "model", "pred", "target", "correct"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in rows:
            pred = normalize_label(rec.get("completion", ""))
            target = "buggy" if str(rec.get("target", "0")) == "1" else "clean"
            is_correct = int(pred == target)
            if pred == "unknown":
                unknown += 1
            total += 1
            correct += is_correct
            writer.writerow(
                {
                    "task_id": rec.get("task_id", ""),
                    "repeat": rec.get("repeat", 0),
                    "model": rec.get("model", ""),
                    "pred": pred,
                    "target": target,
                    "correct": is_correct,
                }
            )

    acc = (correct / total) if total else 0.0
    print(f"Total: {total} Correct: {correct} Unknown: {unknown} Acc: {acc:.3f}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
