"""Run the full analysis suite in order."""
from __future__ import annotations

import argparse
import subprocess


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full suite")
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--llama-model", default="llama3.2:3b")
    parser.add_argument("--mistral-model", default="mistral:7b")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-summarization", action="store_true")
    parser.add_argument("--include-bug", action="store_true")
    parser.add_argument("--include-testgen", action="store_true")
    args = parser.parse_args()

    # Codegen (Llama)
    run([
        "python3", "scripts/run_codegen.py",
        "--dataset", "datasets/nominal/HumanEval_py.jsonl",
        "--model", "ollama",
        "--ollama-model", args.llama_model,
        "--ollama-url", args.ollama_url,
        "--limit", str(args.limit),
        "--repeats", str(args.repeats),
        "--out", "runs/ollama_llama32_3b.jsonl",
    ])
    run([
        "python3", "scripts/eval_codegen.py",
        "--dataset", "datasets/nominal/HumanEval_py.jsonl",
        "--runs", "runs/ollama_llama32_3b.jsonl",
        "--out", "metrics/ollama_llama32_3b_eval.csv",
    ])
    run([
        "python3", "scripts/metrics/compute_stability.py",
        "--runs", "runs/ollama_llama32_3b.jsonl",
        "--eval", "metrics/ollama_llama32_3b_eval.csv",
        "--out", "metrics/ollama_llama32_3b_stability.csv",
        "--with-tree-edit",
    ])

    # Codegen (Mistral)
    run([
        "python3", "scripts/run_codegen.py",
        "--dataset", "datasets/nominal/HumanEval_py.jsonl",
        "--model", "ollama",
        "--ollama-model", args.mistral_model,
        "--ollama-url", args.ollama_url,
        "--limit", str(args.limit),
        "--repeats", str(args.repeats),
        "--out", "runs/ollama_mistral_7b.jsonl",
    ])
    run([
        "python3", "scripts/eval_codegen.py",
        "--dataset", "datasets/nominal/HumanEval_py.jsonl",
        "--runs", "runs/ollama_mistral_7b.jsonl",
        "--out", "metrics/ollama_mistral_7b_eval.csv",
    ])
    run([
        "python3", "scripts/metrics/compute_stability.py",
        "--runs", "runs/ollama_mistral_7b.jsonl",
        "--eval", "metrics/ollama_mistral_7b_eval.csv",
        "--out", "metrics/ollama_mistral_7b_stability.csv",
        "--with-tree-edit",
    ])

    # Paraphrase sensitivity (Llama)
    run([
        "python3", "scripts/metrics/compute_paraphrase_sensitivity.py",
        "--stability", "metrics/ollama_llama32_3b_stability.csv",
        "--out", "metrics/ollama_llama32_3b_paraphrase.csv",
    ])
    run([
        "python3", "scripts/metrics/aggregate_paraphrase_sensitivity.py",
        "--paraphrase", "metrics/ollama_llama32_3b_paraphrase.csv",
        "--out", "metrics/ollama_llama32_3b_paraphrase_summary.csv",
    ])

    # Model comparison
    run([
        "python3", "scripts/metrics/compare_models.py",
        "--inputs",
        "metrics/ollama_llama32_3b_stability.csv",
        "metrics/ollama_mistral_7b_stability.csv",
        "--out", "metrics/model_comparison.csv",
    ])

    # Task feature analysis
    run([
        "python3", "scripts/metrics/compute_task_features.py",
        "--runs", "runs/ollama_llama32_3b.jsonl",
        "--eval", "metrics/ollama_llama32_3b_eval.csv",
        "--out", "metrics/task_features_llama32_3b.csv",
    ])
    run([
        "python3", "scripts/metrics/compute_task_features.py",
        "--runs", "runs/ollama_mistral_7b.jsonl",
        "--eval", "metrics/ollama_mistral_7b_eval.csv",
        "--out", "metrics/task_features_mistral_7b.csv",
    ])
    run([
        "python3", "scripts/metrics/analyze_task_features.py",
        "--inputs",
        "metrics/task_features_llama32_3b.csv",
        "metrics/task_features_mistral_7b.csv",
        "--out", "metrics/task_feature_analysis.csv",
    ])

    if args.include_summarization:
        run([
            "python3", "scripts/tasks/summarization/run_summarization.py",
            "--dataset", "datasets/xsum_subset.jsonl",
            "--model", "ollama",
            "--ollama-model", args.llama_model,
            "--ollama-url", args.ollama_url,
            "--limit", str(min(args.limit, 20)),
            "--repeats", str(args.repeats),
            "--out", "runs/xsum_llama32_3b.jsonl",
        ])
        run([
            "python3", "scripts/tasks/summarization/eval_summarization.py",
            "--runs", "runs/xsum_llama32_3b.jsonl",
            "--out", "metrics/xsum_llama32_3b_eval.csv",
            "--device", "cuda",
        ])

        run([
            "python3", "scripts/tasks/code_summarization/run_code_summarization.py",
            "--dataset", "datasets/codesearchnet_py_subset.jsonl",
            "--model", "ollama",
            "--ollama-model", args.llama_model,
            "--ollama-url", args.ollama_url,
            "--limit", str(min(args.limit, 20)),
            "--repeats", str(args.repeats),
            "--out", "runs/codesearchnet_llama32_3b.jsonl",
        ])
        run([
            "python3", "scripts/tasks/code_summarization/eval_code_summarization.py",
            "--runs", "runs/codesearchnet_llama32_3b.jsonl",
            "--out", "metrics/codesearchnet_llama32_3b_eval.csv",
            "--device", "cuda",
        ])

    if args.include_bug:
        run([
            "python3", "scripts/tasks/bug_detection/run_bug_detection.py",
            "--dataset", "datasets/defect_subset_balanced.jsonl",
            "--model", "ollama",
            "--ollama-model", args.llama_model,
            "--ollama-url", args.ollama_url,
            "--limit", str(min(args.limit, 100)),
            "--repeats", str(args.repeats),
            "--out", "runs/defect_llama32_3b_balanced.jsonl",
        ])
        run([
            "python3", "scripts/tasks/bug_detection/eval_bug_detection.py",
            "--runs", "runs/defect_llama32_3b_balanced.jsonl",
            "--out", "metrics/defect_llama32_3b_balanced_eval.csv",
        ])

    if args.include_testgen:
        run([
            "python3", "scripts/tasks/test_generation/run_test_generation.py",
            "--dataset", "datasets/nominal/mbpp_wtest.jsonl",
            "--model", "ollama",
            "--ollama-model", args.llama_model,
            "--ollama-url", args.ollama_url,
            "--limit", str(min(args.limit, 20)),
            "--repeats", str(args.repeats),
            "--out", "runs/mbpp_testgen_llama32_3b.jsonl",
        ])
        run([
            "python3", "scripts/tasks/test_generation/eval_test_generation.py",
            "--dataset", "datasets/nominal/mbpp_wtest.jsonl",
            "--runs", "runs/mbpp_testgen_llama32_3b.jsonl",
            "--out", "metrics/mbpp_testgen_llama32_3b_eval.csv",
        ])

    # Plots
    run(["python3", "scripts/plots/build_all.py"])


if __name__ == "__main__":
    main()
