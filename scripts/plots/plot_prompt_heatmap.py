"""Prompt-level stability heatmap (averaged across tasks)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import plotly.express as px


def load(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main() -> None:
    rows = load("metrics/ollama_llama32_3b_stability.csv") + load(
        "metrics/ollama_mistral_7b_stability.csv"
    )

    grouped = defaultdict(list)
    for r in rows:
        key = (r.get("model_name") or r.get("model"), r.get("prompt_file"))
        grouped[key].append(float(r.get("ast_jaccard_mean", 0.0)))

    data = []
    for (model, prompt), vals in grouped.items():
        data.append({"model": model, "prompt_file": prompt, "value": sum(vals) / len(vals)})

    fig = px.density_heatmap(
        data,
        x="prompt_file",
        y="model",
        z="value",
        color_continuous_scale="Viridis",
        title="Prompt-Level Stability (AST Jaccard)",
    )
    Path("figures").mkdir(exist_ok=True)
    fig.write_html("figures/prompt_stability_heatmap.html")


if __name__ == "__main__":
    main()
