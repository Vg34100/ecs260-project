# Findings Log

## 2026-02-28
- Llama3.2 3B (Ollama) on HumanEval_py (limit=10, repeats=3, 6 prompts)
  - Pass rate: 0.750 (135/180)
  - exact_match_rate: 0.333–1.000 (mean ~0.439)
  - ast_jaccard_mean: 0.69–1.00 (mean ~0.897)
  - behavior_consistency: 0.667–1.000 (mean ~0.950)
  - Interpretation: non-trivial drift across repeats with high structural similarity.

- Qwen3 8B (Ollama) on HumanEval_py (limit=10, repeats=3, 6 prompts)
  - Pass rate: 0.000 (0/180)
  - Errors dominated by AssertionError (wrong outputs)
  - Interpretation: model outputs are incorrect on this task; not useful for accuracy comparison.

- Qwen2.5-Coder-1.5B (Transformers) on HumanEval_py (limit=10, repeats=3, 6 prompts)
  - Pass rate: 0.350 (63/180)
  - exact_match_rate: 1.000 across repeats
  - Interpretation: deterministic outputs with low correctness.

Notes:
- Llama3.2 3B is the best current candidate for stability analysis with observable drift.
- Qwen3 8B appears unsuitable for HumanEval_py under current prompts/settings.

- XSum summarization (Llama3.2 3B, 20 tasks x 3 repeats)
  - Embedding similarity to reference summaries: mean ~0.52 (min 0.21, max 0.81)
  - Interpretation: summaries are often semantically related but far from identical, which is expected for abstractive summarization. The spread shows repeat-level variability in meaning even under deterministic settings, making it a good non-code task for drift analysis.

- CodeSearchNet code summarization (Llama3.2 3B, 20 tasks x 3 repeats)
  - Embedding similarity to reference docstrings: mean ~0.66 (min 0.21, max 0.91)
  - Interpretation: the model often captures core intent in code summaries, but outputs vary across repeats. This suggests higher semantic alignment than the non-code summarization task, yet still shows measurable drift, which supports RQ2 and the stability analysis.
