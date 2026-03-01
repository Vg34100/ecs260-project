"""Interactive stability atlas heatmap."""
from __future__ import annotations

import csv
from pathlib import Path

import plotly.express as px


def load(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def make_df(rows, metric: str):
    data = []
    for r in rows:
        data.append(
            {
                "task_id": r["task_id"],
                "prompt_file": r["prompt_file"],
                "value": float(r.get(metric, 0.0)),
                "model": r.get("model_name") or r.get("model"),
            }
        )
    return data


def main() -> None:
    llama = load("metrics/ollama_llama32_3b_stability.csv")
    mistral = load("metrics/ollama_mistral_7b_stability.csv")

    metric = "ast_jaccard_mean"
    rows = make_df(llama, metric) + make_df(mistral, metric)

    fig = px.density_heatmap(
        rows,
        x="prompt_file",
        y="task_id",
        z="value",
        facet_col="model",
        color_continuous_scale="Viridis",
        title="Stability Atlas (AST Jaccard)",
    )
    Path("figures").mkdir(exist_ok=True)
    fig.write_html("figures/stability_atlas.html")


if __name__ == "__main__":
    main()
