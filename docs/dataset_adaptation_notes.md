# Dataset Adaptation Notes (One Step at a Time)

## Current Step (Completed)
Changed only `scripts/eval_codegen.py`.

### What changed
Evaluator now resolves dataset rows strictly by:
- `(dataset_source, task_id)`

Evaluation CSV now includes perturbation provenance fields:
- `dataset_source`
- `perturbation_name`
- plus `prompt_file` and `model` for easier grouping in Python report tables.

### Why this step
This prevents wrong test matching when many perturbation files share the same `task_id`,
and makes perturbation-level analysis straightforward without extra joins.

## Next Steps (Not implemented yet)
1. Add a consistency aggregation grouped by `perturbation_name`.
2. Add language-aware evaluators for JS/Java/C++/Go.

## New Step (Completed)
Added output cleanup in `scripts/run_codegen.py` via `cleanup_completion()`.

### What changed
- Strip leaked FIM tokens: `<|fim_*|>` and `<|cursor|>`.
- Strip markdown code fences: ```` ```python ```` and ```` ``` ````.
- Extract fenced code body when fenced blocks are present.
- Trim trailing non-solution content after markers like:
  - `# Test cases`
  - `assert `
  - `print(`
  - `import unittest`
  - explanation text headings

### Why this step
- Evaluation failures were dominated by syntax issues caused by output artifacts rather than core logic errors.
- Observed artifacts in runs/eval included:
  - FIM token leakage (e.g., `<|fim_prefix|>`, `<|cursor|>`)
  - markdown fence leakage (e.g., ` ```python `)
  - appended test/explanation text that breaks executable function-only outputs

## New Step (Completed)
Added `scripts/summarize_by_perturbation.py`.

### What it does
- Reads evaluation CSV from `scripts/eval_codegen.py` (default: `metrics/summary.csv`).
- Groups by `perturbation_name`.
- Writes aggregated pass counts and pass rate CSV (default: `metrics/passrate_by_perturbation.csv`).

### Example
```bash
python3 scripts/summarize_by_perturbation.py \
  --eval-csv metrics/summary.csv \
  --out metrics/summarize_by_perturbation.csv
```

## Utility Added
Added `scripts/download_instruct_model.py` to download an instruct model locally.

Example:

```bash
python3 scripts/download_instruct_model.py \
  --repo-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --out models/qwen2.5-coder-1.5b-instruct
```

## Example usage after current step
You can now point existing scripts to a dataset directory or glob:

```bash
python3 scripts/run_codegen.py \
  --dataset "datasets/perturbed/humanevaljs/full/**/*.jsonl" \
  --model dummy \
  --limit 20 \
  --out runs/smoke.jsonl
```

Note: run records already include `dataset_source` and `perturbation_name`.
Prompt handling is unchanged.
For non-Python datasets (JS/Java/C++/Go), `eval_codegen.py` still needs language-specific executors.
