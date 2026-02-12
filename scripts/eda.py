import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_runs(path: Path) -> pd.DataFrame:
    # Load JSONL runs into a DataFrame and add simple length features.
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append(
                {
                    "task_id": rec["task_id"],
                    "repeat": rec["repeat"],
                    "prompt_len": len(rec["prompt"]),
                    "completion_len": len(rec["completion"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA tables and plots")
    parser.add_argument("--runs", default="runs/codegen_repeats.jsonl")
    parser.add_argument("--summary", default="metrics/summary_repeats.csv")
    parser.add_argument("--consistency", default="metrics/consistency.csv")
    parser.add_argument("--metrics-out", default="metrics")
    parser.add_argument("--figures-out", default="figures")
    args = parser.parse_args()

    runs_df = load_runs(Path(args.runs))
    # Summary CSV can include commas inside the error field, so parse manually.
    summary_rows = []
    with Path(args.summary).open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",", 3)
            if len(parts) < 3:
                continue
            summary_rows.append(
                {
                    "task_id": parts[0],
                    "repeat": int(parts[1]),
                    "passed": int(parts[2]),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    consistency_df = pd.read_csv(args.consistency)

    # Merge pass/fail into runs.
    merged = runs_df.merge(summary_df, on=["task_id", "repeat"], how="left")
    merged["passed"] = merged["passed"].fillna(0).astype(int)

    metrics_dir = Path(args.metrics_out)
    figures_dir = Path(args.figures_out)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: pass rate by task.
    pass_rate_by_task = (
        merged.groupby("task_id")["passed"].mean().reset_index().rename(columns={"passed": "pass_rate"})
    )
    pass_rate_by_task.to_csv(metrics_dir / "eda_pass_rate_by_task.csv", index=False)

    # Table 2: consistency by task (from previous script).
    consistency_df.to_csv(metrics_dir / "eda_consistency_by_task.csv", index=False)

    # Table 3: completion length five-number summary.
    lengths = merged["completion_len"].to_numpy()
    q1, med, q3 = np.percentile(lengths, [25, 50, 75])
    length_stats = pd.DataFrame(
        [
            {
                "min": int(np.min(lengths)),
                "q1": int(q1),
                "median": int(med),
                "q3": int(q3),
                "max": int(np.max(lengths)),
            }
        ]
    )
    length_stats.to_csv(metrics_dir / "eda_completion_len_fivenum.csv", index=False)

    # Plot 1: histogram of completion length.
    plt.figure(figsize=(6, 4))
    plt.hist(merged["completion_len"], bins=20, color="#4c72b0", edgecolor="white")
    plt.title("Completion Length Distribution")
    plt.xlabel("Completion Length (chars)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_completion_len_hist.png", dpi=200)
    plt.close()

    # Plot 2: pass rate by task (bar chart).
    plt.figure(figsize=(8, 4))
    plt.bar(pass_rate_by_task["task_id"], pass_rate_by_task["pass_rate"], color="#55a868")
    plt.xticks(rotation=90, fontsize=7)
    plt.ylim(0, 1)
    plt.title("Pass Rate by Task")
    plt.ylabel("Pass Rate")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_pass_rate_by_task.png", dpi=200)
    plt.close()

    # Plot 3: boxplot of completion length by pass/fail.
    plt.figure(figsize=(6, 4))
    merged.boxplot(column="completion_len", by="passed")
    plt.title("Completion Length by Pass/Fail")
    plt.suptitle("")
    plt.xlabel("Passed (0=fail, 1=pass)")
    plt.ylabel("Completion Length (chars)")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_completion_len_by_pass.png", dpi=200)
    plt.close()

    print("EDA tables written to", metrics_dir)
    print("EDA plots written to", figures_dir)


if __name__ == "__main__":
    main()
