"""Compute stability metrics from run logs and evaluation results."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.ast_utils import ast_tokens, ast_to_tree, normalize_ast, parse_python
from scripts.utils.semantics import embed_texts, load_codebert_model


try:
    import zss
except Exception:
    zss = None


Record = Dict[str, object]


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def mean_pairwise(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def exact_match_rate(completions: List[str]) -> float:
    if not completions:
        return 0.0
    total = len(completions)
    counts: Dict[str, int] = defaultdict(int)
    for c in completions:
        counts[c] += 1
    max_count = max(counts.values())
    return max_count / total


def behavioral_consistency(passed: List[int]) -> float:
    if not passed:
        return 0.0
    total = len(passed)
    ones = sum(passed)
    zeros = total - ones
    return max(ones, zeros) / total


def ast_jaccard_mean(completions: List[str]) -> float:
    tokens = [ast_tokens(c) for c in completions]
    valid = [t for t in tokens if t]
    if len(valid) < 2:
        return 0.0
    sims = []
    for a, b in combinations(valid, 2):
        sims.append(jaccard_similarity(a, b))
    return mean_pairwise(sims)


def ast_tree_edit_mean(completions: List[str]) -> Optional[float]:
    if zss is None:
        return None
    trees = []
    for c in completions:
        tree = parse_python(c)
        if not tree:
            continue
        tree = normalize_ast(tree)
        trees.append(ast_to_tree(tree))
    if len(trees) < 2:
        return 0.0

    def get_children(node):
        return node.children

    def get_label(node):
        return node.label

    dists = []
    for a, b in combinations(trees, 2):
        dist = zss.simple_distance(a, b, get_children=get_children, get_label=get_label)
        max_nodes = max(_count_nodes(a), _count_nodes(b))
        if max_nodes == 0:
            dists.append(0.0)
        else:
            dists.append(1.0 - (dist / max_nodes))
    return mean_pairwise(dists)


def _count_nodes(node) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def semantic_similarity_mean(
    completions: List[str],
    model,
    batch_size: int,
    max_length: int,
) -> Optional[float]:
    if model is None:
        return None
    if len(completions) < 2:
        return 0.0
    embeddings = embed_texts(
        model,
        completions,
        batch_size=batch_size,
        max_length=max_length,
    )
    sims = []
    for i, j in combinations(range(len(embeddings)), 2):
        sims.append(_cosine(embeddings[i], embeddings[j]))
    return mean_pairwise(sims)


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_runs(path: str) -> Iterable[Record]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_eval(path: str) -> Dict[Tuple[str, str, str, str, int], int]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stability metrics")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", default="metrics/stability_by_prompt.csv")
    parser.add_argument("--with-semantic", action="store_true", help="enable CodeBERT similarity")
    parser.add_argument("--semantic-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--semantic-batch-size", type=int, default=16)
    parser.add_argument("--semantic-max-length", type=int, default=256)
    parser.add_argument("--with-tree-edit", action="store_true", help="enable AST tree-edit similarity")
    args = parser.parse_args()

    eval_map = load_eval(args.eval)

    semantic_model = None
    if args.with_semantic:
        semantic_model = load_codebert_model(device=args.semantic_device)

    grouped: Dict[Tuple[str, str, str, str, str, str], List[Record]] = defaultdict(list)
    for rec in load_runs(args.runs):
        key = (
            str(rec.get("dataset_source", "")),
            str(rec.get("task_id", "")),
            str(rec.get("prompt_file", "")),
            str(rec.get("model", "")),
            str(rec.get("model_name", "")),
            str(rec.get("perturbation_name", "")),
        )
        grouped[key].append(rec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        fieldnames = [
            "dataset_source",
            "task_id",
            "prompt_file",
            "model",
            "model_name",
            "perturbation_name",
            "repeats",
            "exact_match_rate",
            "ast_jaccard_mean",
            "ast_tree_edit_mean",
            "behavior_consistency",
            "pass_rate",
            "semantic_similarity_mean",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for key, records in grouped.items():
            dataset_source, task_id, prompt_file, model, model_name, perturbation_name = key
            completions = [str(r.get("completion", "")) for r in records]
            repeats = len(completions)
            passed = []
            for r in records:
                eval_key = (
                    dataset_source,
                    task_id,
                    prompt_file,
                    model,
                    int(r.get("repeat", 0)),
                )
                passed.append(eval_map.get(eval_key, 0))

            row = {
                "dataset_source": dataset_source,
                "task_id": task_id,
                "prompt_file": prompt_file,
                "model": model,
                "model_name": model_name,
                "perturbation_name": perturbation_name,
                "repeats": repeats,
                "exact_match_rate": f"{exact_match_rate(completions):.3f}",
                "ast_jaccard_mean": f"{ast_jaccard_mean(completions):.3f}",
                "ast_tree_edit_mean": "",
                "behavior_consistency": f"{behavioral_consistency(passed):.3f}",
                "pass_rate": f"{(sum(passed) / repeats) if repeats else 0.0:.3f}",
                "semantic_similarity_mean": "",
            }

            if args.with_tree_edit:
                tree_edit = ast_tree_edit_mean(completions)
                if tree_edit is not None:
                    row["ast_tree_edit_mean"] = f"{tree_edit:.3f}"

            if args.with_semantic:
                semantic = semantic_similarity_mean(
                    completions,
                    model=semantic_model,
                    batch_size=args.semantic_batch_size,
                    max_length=args.semantic_max_length,
                )
                if semantic is not None:
                    row["semantic_similarity_mean"] = f"{semantic:.3f}"

            writer.writerow(row)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
