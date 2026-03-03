"""Paraphrase audit tool: sample tasks from run logs and export a CSV for human labeling.

Usage
-----
# Basic: random sample of 20 tasks
python3 scripts/paraphrase_audit.py \
    --runs runs/ollama_llama32_3b.jsonl \
    --out audit/paraphrase_audit.csv

# With pass/fail labels from eval
python3 scripts/paraphrase_audit.py \
    --runs runs/ollama_llama32_3b.jsonl \
    --eval-csv metrics/ollama_llama32_3b_eval.csv \
    --out audit/paraphrase_audit.csv

# Prioritize the most unstable tasks (needs stability CSV)
python3 scripts/paraphrase_audit.py \
    --runs runs/ollama_llama32_3b.jsonl \
    --eval-csv metrics/ollama_llama32_3b_eval.csv \
    --stability-csv metrics/ollama_llama32_3b_stability.csv \
    --prioritize-unstable \
    --n 30 \
    --out audit/paraphrase_audit.csv

Output CSV columns
------------------
task_id, dataset_source, perturbation_name, model, prompt_file, repeat,
passed, completion_preview, human_label, notes

- human_label: annotator fills this in (e.g. "correct", "incorrect", "partial")
- notes: annotator free-text remarks
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_runs(path: str) -> List[Dict[str, Any]]:
    """Load all records from a JSONL run log produced by run_codegen.py."""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_eval(path: str) -> Dict[Tuple[str, str, str, str, int], int]:
    """Load eval CSV into a lookup dict keyed by (dataset_source, task_id, prompt_file, model, repeat)."""
    results: Dict[Tuple[str, str, str, str, int], int] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get("dataset_source", ""),
                row.get("task_id", ""),
                row.get("prompt_file", ""),
                row.get("model", ""),
                int(row.get("repeat", 0)),
            )
            results[key] = int(row.get("passed", 0))
    return results


def load_stability(path: str) -> List[Dict[str, str]]:
    """Load stability CSV rows."""
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _unique_task_ids(records: List[Dict[str, Any]]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for r in records:
        tid = str(r.get("task_id", ""))
        if tid and tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return ordered


def sample_task_ids(
    records: List[Dict[str, Any]],
    n: int,
    seed: Optional[int],
    prioritize_unstable: bool,
    stability_rows: Optional[List[Dict[str, str]]],
) -> List[str]:
    """Return up to n task_id strings chosen from records.

    When prioritize_unstable is True and stability_rows is provided, tasks are
    ranked by exact_match_rate ascending so that the most variable tasks come
    first.  The remaining slots are filled with a random sample.
    """
    all_ids = _unique_task_ids(records)
    all_ids_set = set(all_ids)
    rng = random.Random(seed)

    if prioritize_unstable and stability_rows:
        scored: List[Tuple[float, str]] = []
        seen_in_stability: Set[str] = set()
        for row in stability_rows:
            tid = str(row.get("task_id", ""))
            if tid not in all_ids_set:
                continue
            emr = float(row.get("exact_match_rate", 1.0))
            scored.append((emr, tid))
            seen_in_stability.add(tid)
        # Sort ascending: most unstable (lowest exact_match_rate) first.
        scored.sort(key=lambda x: x[0])
        selected = [tid for _, tid in scored][:n]
        # Fill remaining slots with random tasks not yet selected.
        if len(selected) < n:
            remaining = [tid for tid in all_ids if tid not in set(selected)]
            rng.shuffle(remaining)
            selected.extend(remaining[: n - len(selected)])
        return selected[:n]

    # Default: uniform random sample.
    return rng.sample(all_ids, min(n, len(all_ids)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample tasks from run logs and export a CSV for paraphrase human labeling"
    )
    parser.add_argument(
        "--runs", required=True,
        help="JSONL run log produced by run_codegen.py"
    )
    parser.add_argument(
        "--eval-csv", default=None,
        help="Optional eval CSV (from eval_codegen.py) to include pass/fail per row"
    )
    parser.add_argument(
        "--stability-csv", default=None,
        help="Optional stability CSV (from compute_stability.py) used when --prioritize-unstable is set"
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of distinct tasks to sample (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    parser.add_argument(
        "--prioritize-unstable", action="store_true",
        help="Rank tasks by lowest exact_match_rate first; requires --stability-csv"
    )
    parser.add_argument(
        "--completion-chars", type=int, default=300,
        help="Max characters of completion shown in completion_preview (default: 300)"
    )
    parser.add_argument(
        "--out", default="audit/paraphrase_audit.csv",
        help="Output CSV path (default: audit/paraphrase_audit.csv)"
    )
    args = parser.parse_args()

    # Load run records.
    records = load_runs(args.runs)
    if not records:
        raise SystemExit(f"No records found in: {args.runs}")
    print(f"Loaded {len(records)} run records from {args.runs}")

    # Optionally load eval pass/fail.
    eval_map: Dict[Tuple[str, str, str, str, int], int] = {}
    if args.eval_csv:
        if not Path(args.eval_csv).exists():
            print(f"Warning: --eval-csv not found, pass/fail column will be empty: {args.eval_csv}")
        else:
            eval_map = load_eval(args.eval_csv)
            print(f"Loaded {len(eval_map)} eval entries from {args.eval_csv}")

    # Optionally load stability rows for guided sampling.
    stability_rows: Optional[List[Dict[str, str]]] = None
    if args.stability_csv:
        if not Path(args.stability_csv).exists():
            print(f"Warning: --stability-csv not found, falling back to random sampling: {args.stability_csv}")
        else:
            stability_rows = load_stability(args.stability_csv)
            print(f"Loaded {len(stability_rows)} stability rows from {args.stability_csv}")

    # Sample task IDs.
    sampled_ids: Set[str] = set(
        sample_task_ids(
            records,
            n=args.n,
            seed=args.seed,
            prioritize_unstable=args.prioritize_unstable,
            stability_rows=stability_rows,
        )
    )
    print(f"Sampled {len(sampled_ids)} task IDs")

    # Build output rows: one row per (task_id, prompt_file, repeat) in sampled set.
    fieldnames = [
        "task_id",
        "dataset_source",
        "perturbation_name",
        "model",
        "prompt_file",
        "repeat",
        "passed",
        "completion_preview",
        "human_label",
        "notes",
    ]

    rows_out: List[Dict[str, Any]] = []
    for rec in records:
        tid = str(rec.get("task_id", ""))
        if tid not in sampled_ids:
            continue

        dataset_source = str(rec.get("dataset_source", ""))
        prompt_file = str(rec.get("prompt_file", ""))
        model = str(rec.get("model", ""))
        repeat = int(rec.get("repeat", 0))
        completion = str(rec.get("completion", ""))

        # Inline newlines so the cell stays on one spreadsheet line.
        preview = completion[: args.completion_chars].replace("\n", "\\n")

        eval_key = (dataset_source, tid, prompt_file, model, repeat)
        passed_val: Any = eval_map.get(eval_key, "") if eval_map else ""

        rows_out.append(
            {
                "task_id": tid,
                "dataset_source": dataset_source,
                "perturbation_name": str(rec.get("perturbation_name", "")),
                "model": model,
                "prompt_file": prompt_file,
                "repeat": repeat,
                "passed": passed_val,
                "completion_preview": preview,
                "human_label": "",
                "notes": "",
            }
        )

    # Sort for readability: group by task_id -> prompt_file -> repeat.
    rows_out.sort(key=lambda r: (r["task_id"], r["prompt_file"], r["repeat"]))

    # Write CSV.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows ({len(sampled_ids)} tasks) -> {out_path}")


if __name__ == "__main__":
    main()
