"""Build all plots."""
from __future__ import annotations

import subprocess


PLOTS = [
    "scripts/plots/plot_behavior_vs_structure.py",
    "scripts/plots/plot_task_feature_correlations.py",
    "scripts/plots/plot_stability_atlas.py",
    "scripts/plots/plot_distributions.py",
    "scripts/plots/plot_task_summaries.py",
    "scripts/plots/plot_stability_bins.py",
    "scripts/plots/plot_mbpp_testgen_errors.py",
    "scripts/plots/plot_prompt_heatmap.py",
    "scripts/plots/plot_prompt_stability_bar.py",
    "scripts/plots/plot_bug_confusion.py",
]


def main() -> None:
    for plot in PLOTS:
        print(f"Running {plot}")
        subprocess.run(["python3", plot], check=True)


if __name__ == "__main__":
    main()
