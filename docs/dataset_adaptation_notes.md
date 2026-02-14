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
1. Add one small aggregation script for pass rate grouped by `perturbation_name` (Python only).
2. Add a consistency aggregation grouped by `perturbation_name`.
3. Add language-aware evaluators for JS/Java/C++/Go.

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
