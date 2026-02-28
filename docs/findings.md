# Findings Log

## 2026-02-28
- Llama3.2 3B (Ollama) on HumanEval_py (limit=10, repeats=3, 6 prompts)
  - Pass rate: 0.750 (135/180)
  - exact_match_rate: 0.333–1.000 (mean ~0.439)
  - ast_jaccard_mean: 0.69–1.00 (mean ~0.897)
  - behavior_consistency: 0.667–1.000 (mean ~0.950)
  - Interpretation: non-trivial drift across repeats with high structural similarity.
  - Commit: `feat: add stability metrics and ollama backend`
  - Commands:
    - `python3 scripts/run_codegen.py --dataset datasets/nominal/HumanEval_py.jsonl --model ollama --ollama-model llama3.2:3b --ollama-url http://100.121.2.67:11434 --limit 10 --repeats 3 --out runs/ollama_llama32_3b.jsonl`
    - `python3 scripts/eval_codegen.py --dataset datasets/nominal/HumanEval_py.jsonl --runs runs/ollama_llama32_3b.jsonl --out metrics/ollama_llama32_3b_eval.csv`
    - `python3 scripts/metrics/compute_stability.py --runs runs/ollama_llama32_3b.jsonl --eval metrics/ollama_llama32_3b_eval.csv --out metrics/ollama_llama32_3b_stability.csv`

- Qwen3 8B (Ollama) on HumanEval_py (limit=10, repeats=3, 6 prompts)
  - Pass rate: 0.000 (0/180)
  - Errors dominated by AssertionError (wrong outputs)
  - Interpretation: model outputs are incorrect on this task; not useful for accuracy comparison.
  - Commit: `feat: add stability metrics and ollama backend`
  - Commands:
    - `python3 scripts/run_codegen.py --dataset datasets/nominal/HumanEval_py.jsonl --model ollama --ollama-model qwen3:8b --ollama-url http://100.121.2.67:11434 --limit 10 --repeats 3 --out runs/ollama_qwen3_8b.jsonl`
    - `python3 scripts/eval_codegen.py --dataset datasets/nominal/HumanEval_py.jsonl --runs runs/ollama_qwen3_8b.jsonl --out metrics/ollama_qwen3_8b_eval.csv`
    - `python3 scripts/metrics/compute_stability.py --runs runs/ollama_qwen3_8b.jsonl --eval metrics/ollama_qwen3_8b_eval.csv --out metrics/ollama_qwen3_8b_stability.csv`

- Qwen2.5-Coder-1.5B (Transformers) on HumanEval_py (limit=10, repeats=3, 6 prompts)
  - Pass rate: 0.350 (63/180)
  - exact_match_rate: 1.000 across repeats
  - Interpretation: deterministic outputs with low correctness.
  - Commit: `feat: add stability metrics and ollama backend`
  - Commands:
    - `python3 scripts/run_codegen.py --dataset datasets/nominal/HumanEval_py.jsonl --model transformers --model-path models/qwen2.5-coder-1.5b-instruct --device cuda --limit 10 --repeats 3 --out runs/repeat_smoke.jsonl`
    - `python3 scripts/eval_codegen.py --dataset datasets/nominal/HumanEval_py.jsonl --runs runs/repeat_smoke.jsonl --out metrics/repeat_smoke_eval.csv`
    - `python3 scripts/metrics/compute_stability.py --runs runs/repeat_smoke.jsonl --eval metrics/repeat_smoke_eval.csv --out metrics/repeat_smoke_stability.csv`

Notes:
- Llama3.2 3B is the best current candidate for stability analysis with observable drift.
- Qwen3 8B appears unsuitable for HumanEval_py under current prompts/settings.

- XSum summarization (Llama3.2 3B, 20 tasks x 3 repeats)
  - Embedding similarity to reference summaries: mean ~0.52 (min 0.21, max 0.81)
  - Interpretation: summaries are often semantically related but far from identical, which is expected for abstractive summarization. The spread shows repeat-level variability in meaning even under deterministic settings, making it a good non-code task for drift analysis.
  - Commit: `feat: add non-code summarization pipeline`
  - Commands:
    - `python3 scripts/datasets/download_xsum.py --split train --limit 200 --out datasets/xsum_subset.jsonl`
    - `python3 scripts/tasks/summarization/run_summarization.py --dataset datasets/xsum_subset.jsonl --model ollama --ollama-model llama3.2:3b --ollama-url http://100.121.2.67:11434 --limit 20 --repeats 3 --out runs/xsum_llama32_3b.jsonl`
    - `python3 scripts/tasks/summarization/eval_summarization.py --runs runs/xsum_llama32_3b.jsonl --out metrics/xsum_llama32_3b_eval.csv --device cuda`

- CodeSearchNet code summarization (Llama3.2 3B, 20 tasks x 3 repeats)
  - Embedding similarity to reference docstrings: mean ~0.66 (min 0.21, max 0.91)
  - Interpretation: the model often captures core intent in code summaries, but outputs vary across repeats. This suggests higher semantic alignment than the non-code summarization task, yet still shows measurable drift, which supports RQ2 and the stability analysis.
  - Commit: `feat: add code summarization pipeline`
  - Commands:
    - `python3 scripts/datasets/download_codesearchnet.py --split train --limit 200 --out datasets/codesearchnet_py_subset.jsonl`
    - `python3 scripts/tasks/code_summarization/run_code_summarization.py --dataset datasets/codesearchnet_py_subset.jsonl --model ollama --ollama-model llama3.2:3b --ollama-url http://100.121.2.67:11434 --limit 20 --repeats 3 --out runs/codesearchnet_llama32_3b.jsonl`
    - `python3 scripts/tasks/code_summarization/eval_code_summarization.py --runs runs/codesearchnet_llama32_3b.jsonl --out metrics/codesearchnet_llama32_3b_eval.csv --device cuda`

- CodeXGLUE defect detection (balanced subset, 100 tasks x 3 repeats)
  - Accuracy: 0.52 (156/300), Unknown: 1
  - Targets are balanced (150 clean, 150 buggy), but the model heavily favors 'buggy' predictions.
  - Interpretation: the task is harder for this model and prompt format; results are near chance with a strong class bias, which is still useful for stability analysis but weak for correctness. We should consider prompt tuning or a smaller classification-focused model if accuracy matters.
  - Commit: `feat: add bug detection pipeline`
  - Commands:
    - `python3 scripts/datasets/download_defect.py --split train --limit 100 --balanced --out datasets/defect_subset_balanced.jsonl`
    - `python3 scripts/tasks/bug_detection/run_bug_detection.py --dataset datasets/defect_subset_balanced.jsonl --model ollama --ollama-model llama3.2:3b --ollama-url http://100.121.2.67:11434 --limit 100 --repeats 3 --out runs/defect_llama32_3b_balanced.jsonl`
    - `python3 scripts/tasks/bug_detection/eval_bug_detection.py --runs runs/defect_llama32_3b_balanced.jsonl --out metrics/defect_llama32_3b_balanced_eval.csv`

- Paraphrase sensitivity summary (HumanEval_py, Llama3.2 3B, 10 tasks, 6 prompt variants)
  - mean_exact_match ~0.44, mean_ast_jaccard ~0.90, mean_behavior_consistency ~0.95, mean_pass_rate ~0.75
  - Interpretation: paraphrasing causes noticeable surface-form changes, but code structure and correctness remain mostly stable. This supports RQ2 by showing sensitivity in outputs without large behavioral instability.
  - Commit: `feat: add paraphrase sensitivity analysis`
  - Commands:
    - `python3 scripts/metrics/compute_paraphrase_sensitivity.py --stability metrics/ollama_llama32_3b_stability.csv --out metrics/ollama_llama32_3b_paraphrase.csv`
    - `python3 scripts/metrics/aggregate_paraphrase_sensitivity.py --paraphrase metrics/ollama_llama32_3b_paraphrase.csv --out metrics/ollama_llama32_3b_paraphrase_summary.csv`

- Model comparison (HumanEval_py, deterministic runs, 10 tasks x 3 repeats x 6 prompts)
  - Llama3.2:3b shows higher stability and accuracy than Mistral:7b on this setup.
  - Despite Mistral being larger, it is less stable here, which suggests parameter count alone does not guarantee stability.
  - Differences may come from architecture, training data, or alignment, not just size.
  - Commit: `feat: add model comparison and model_name support`
  - Commands:
    - `python3 scripts/metrics/compute_stability.py --runs runs/ollama_llama32_3b_tagged.jsonl --eval metrics/ollama_llama32_3b_eval.csv --out metrics/ollama_llama32_3b_stability.csv --with-tree-edit`
    - `python3 scripts/metrics/compute_stability.py --runs runs/ollama_mistral_7b_tagged.jsonl --eval metrics/ollama_mistral_7b_eval.csv --out metrics/ollama_mistral_7b_stability.csv --with-tree-edit`
    - `python3 scripts/metrics/compare_models.py --inputs metrics/ollama_llama32_3b_stability.csv metrics/ollama_mistral_7b_stability.csv --out metrics/model_comparison.csv`

- Task-feature tables (HumanEval_py, Llama3.2:3b and Mistral:7b)
  - Feature logs include prompt length, completion length, syntax validity, and pass/fail per run.
  - Interpretation: this enables RQ4 analysis by correlating stability and correctness with prompt/completion length and syntax validity. Initial inspection shows model-dependent completion length differences (e.g., Llama3.2 outputs longer code than Mistral on the same prompt).
  - Commit: `feat: add rq4 feature table`
  - Commands:
    - `python3 scripts/metrics/compute_task_features.py --runs runs/ollama_llama32_3b_tagged.jsonl --eval metrics/ollama_llama32_3b_eval.csv --out metrics/task_features_llama32_3b.csv`
    - `python3 scripts/metrics/compute_task_features.py --runs runs/ollama_mistral_7b_tagged.jsonl --eval metrics/ollama_mistral_7b_eval.csv --out metrics/task_features_mistral_7b.csv`

- MBPP test generation (20 tasks x 3 repeats)
  - Llama3.2:3b pass rate: 0.383 (23/60). Frequent invalid tests or missing asserts.
  - Mistral:7b pass rate: 0.633 (38/60). Fewer invalid tests; more usable assertions.
  - Interpretation: test-generation quality is model-dependent; Mistral produces more executable tests under the same prompt, suggesting better reliability for this task.
  - Commit: `feat: add mbpp test generation task`
  - Commands:
    - `python3 scripts/tasks/test_generation/run_test_generation.py --dataset datasets/nominal/mbpp_wtest.jsonl --model ollama --ollama-model llama3.2:3b --ollama-url http://100.121.2.67:11434 --limit 20 --repeats 3 --out runs/mbpp_testgen_llama32_3b.jsonl`
    - `python3 scripts/tasks/test_generation/eval_test_generation.py --dataset datasets/nominal/mbpp_wtest.jsonl --runs runs/mbpp_testgen_llama32_3b.jsonl --out metrics/mbpp_testgen_llama32_3b_eval.csv`
    - `python3 scripts/tasks/test_generation/run_test_generation.py --dataset datasets/nominal/mbpp_wtest.jsonl --model ollama --ollama-model mistral:7b --ollama-url http://100.121.2.67:11434 --limit 20 --repeats 3 --out runs/mbpp_testgen_mistral_7b.jsonl`
    - `python3 scripts/tasks/test_generation/eval_test_generation.py --dataset datasets/nominal/mbpp_wtest.jsonl --runs runs/mbpp_testgen_mistral_7b.jsonl --out metrics/mbpp_testgen_mistral_7b_eval.csv`

- Task-feature correlation analysis (Llama3.2:3b + Mistral:7b)
  - prompt_len_vs_pass r = -0.294; completion_len_vs_pass r = -0.174; syntax_valid_vs_pass r = 0.261.
  - Interpretation: longer prompts and longer completions are weakly associated with lower pass rates in this small sample, while syntax validity is positively associated with correctness. These are modest effects (not causal), but they support RQ4 by identifying prompt/completion length and syntax validity as measurable factors linked to performance.
  - Commit: `feat: add task feature analysis`
  - Commands:
    - `python3 scripts/metrics/analyze_task_features.py --inputs metrics/task_features_llama32_3b.csv metrics/task_features_mistral_7b.csv --out metrics/task_feature_analysis.csv`
