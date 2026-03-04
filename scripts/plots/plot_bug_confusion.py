"""Confusion-matrix plot for bug detection (predicted vs actual label).

Reads one or more defect eval CSVs produced by eval_bug_detection.py.
Each CSV must have columns: task_id, repeat, model, pred, target, correct.
Labels are "buggy" / "clean" / "unknown".  "unknown" predictions are tallied
separately in the title rather than polluting the 2x2 matrix.

One subplot is drawn per model found in the data.

Output: figures/bug_detection_confusion.png
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


_EVAL_FILES = [
    "metrics/defect_llama32_3b_balanced_eval.csv",
    "metrics/defect_mistral_7b_balanced_eval.csv",
]

_CLASSES = ["buggy", "clean"]   # row/col order for the matrix


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_rows(paths: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  Skipping missing file: {path}")
            continue
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    return rows


def build_matrices(
    rows: List[Dict[str, str]],
) -> Dict[str, Tuple[np.ndarray, int]]:
    """Return {model: (confusion_matrix_2x2, unknown_count)}.

    Matrix axes: mat[actual_idx][pred_idx].
    """
    # Group by model.
    by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row.get("model", "unknown")].append(row)

    result: Dict[str, Tuple[np.ndarray, int]] = {}
    idx = {c: i for i, c in enumerate(_CLASSES)}

    for model, model_rows in by_model.items():
        mat = np.zeros((len(_CLASSES), len(_CLASSES)), dtype=int)
        unknown = 0
        for row in model_rows:
            pred = row.get("pred", "unknown")
            target = row.get("target", "unknown")
            if pred == "unknown" or target not in idx:
                unknown += 1
                continue
            if pred not in idx:
                unknown += 1
                continue
            mat[idx[target], idx[pred]] += 1
        result[model] = (mat, unknown)

    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _draw_matrix(ax: plt.Axes, mat: np.ndarray, unknown: int, title: str) -> None:
    total = mat.sum() + unknown
    # Normalize row-wise for background color; keep raw counts for text.
    row_sums = mat.sum(axis=1, keepdims=True)
    normed = np.where(row_sums > 0, mat / row_sums, 0.0)

    ax.imshow(normed, vmin=0, vmax=1, cmap="Blues", aspect="equal")

    for i in range(len(_CLASSES)):
        for j in range(len(_CLASSES)):
            count = mat[i, j]
            pct = 100 * count / total if total else 0.0
            text = f"{count}\n({pct:.1f}%)"
            color = "white" if normed[i, j] > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(_CLASSES)))
    ax.set_xticklabels([f"Pred: {c}" for c in _CLASSES])
    ax.set_yticks(range(len(_CLASSES)))
    ax.set_yticklabels([f"Actual: {c}" for c in _CLASSES])
    ax.set_title(f"{title}\n(unknown={unknown}, n={total})", fontsize=9)


def main() -> None:
    rows = load_eval_rows(_EVAL_FILES)
    if not rows:
        print("No bug detection eval data found; skipping plot_bug_confusion.")
        return

    matrices = build_matrices(rows)
    models = sorted(matrices)

    if not models:
        print("No models found in data.")
        return

    fig, axes = plt.subplots(
        1, len(models),
        figsize=(4.5 * len(models), 4),
        squeeze=False,
    )

    for ax, model in zip(axes[0], models):
        mat, unknown = matrices[model]
        _draw_matrix(ax, mat, unknown, model)

    fig.suptitle("Bug Detection Confusion Matrix (predicted vs actual)", fontsize=11)
    plt.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    out = "figures/bug_detection_confusion.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
