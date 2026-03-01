"""Evaluate summarization outputs with embedding cosine similarity."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import List


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate summarization outputs")
    parser.add_argument("--runs", default="runs/xsum_runs.jsonl")
    parser.add_argument("--out", default="metrics/xsum_eval.csv")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception as e:
        raise SystemExit("sentence-transformers is required") from e

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(args.model, device=device)

    rows = []
    with Path(args.runs).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        fieldnames = [
            "task_id",
            "repeat",
            "model",
            "similarity",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in rows:
            target = rec.get("target", "")
            completion = rec.get("completion", "")
            if not target or not completion:
                sim = 0.0
            else:
                emb = model.encode(
                    [target, completion],
                    normalize_embeddings=True,
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                )
                sim = cosine(emb[0], emb[1])
            writer.writerow(
                {
                    "task_id": rec.get("task_id", ""),
                    "repeat": rec.get("repeat", 0),
                    "model": rec.get("model", ""),
                    "similarity": f"{sim:.3f}",
                }
            )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
