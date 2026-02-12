import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    # Exact-match consistency across repeated runs per task.
    # 1.0 means all repeats are identical; lower means drift across repeats.
    parser = argparse.ArgumentParser(description="Compute exact-match consistency")
    parser.add_argument("--runs", default="runs/codegen.jsonl")
    parser.add_argument("--out", default="metrics/consistency.csv")
    args = parser.parse_args()

    by_task = defaultdict(list)
    # Group completions by task_id.
    with Path(args.runs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_task[rec["task_id"]].append(rec["completion"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    total_consistency = 0.0
    # Write per-task consistency plus an overall average row.
    with out_path.open("w", encoding="utf-8") as out:
        out.write("task_id,unique_outputs,total_runs,consistency\n")
        for task_id, completions in sorted(by_task.items()):
            total = len(completions)
            unique = len(set(completions))
            consistency = 1.0 - ((unique - 1) / total) if total else 0.0
            total_tasks += 1
            total_consistency += consistency
            out.write(f"{task_id},{unique},{total},{consistency:.3f}\n")

        avg = total_consistency / total_tasks if total_tasks else 0.0
        out.write(f"ALL,{''},{''},{avg:.3f}\n")

    print(f"Tasks: {total_tasks} Avg consistency: {avg:.3f}")


if __name__ == "__main__":
    main()
