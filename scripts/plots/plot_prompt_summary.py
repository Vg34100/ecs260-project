from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "pass_rate",
    "behavior_consistency",
    "exact_match_rate",
    "ast_jaccard_mean",
]


def available_metrics(df: pd.DataFrame, requested: List[str]) -> List[str]:
    out = []
    for col in requested:
        if col in df.columns and df[col].notna().any():
            out.append(col)
    return out


def sanitize_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(description="Create prompt-level summary CSV and plots from stability CSV")
    ap.add_argument("--stability", required=True, help="Path to stability CSV")
    ap.add_argument("--out-csv", required=True, help="Output prompt-level summary CSV")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory to save prompt-level plots (one PNG per metric)",
    )
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to aggregate and plot (default: pass_rate behavior_consistency exact_match_rate ast_jaccard_mean)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.stability)

    if "prompt_file" not in df.columns:
        raise ValueError("Expected 'prompt_file' column in stability CSV")

    metrics = available_metrics(df, args.metrics)
    if not metrics:
        raise ValueError("None of the requested metrics exist with non-NaN values")

    # Group by prompt and average selected metrics
    summary = df.groupby("prompt_file", as_index=False)[metrics].mean()

    # Save summary CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"[OK] Wrote summary CSV: {out_csv}")

    # Save one plot per metric
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        plot_df = summary.sort_values(metric, ascending=False)

        plt.figure(figsize=(10, 5))
        plt.bar(plot_df["prompt_file"], plot_df[metric])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Prompt variant")
        plt.ylabel(metric)
        plt.title(f"Prompt-level average {metric}")
        plt.tight_layout()

        out_path = out_dir / f"prompt_summary_{sanitize_filename(metric)}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[OK] Wrote plot: {out_path}")

    print(f"[INFO] Metrics plotted: {metrics}")


if __name__ == "__main__":
    main()