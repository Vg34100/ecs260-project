# Dataset Adaptation Notes (One Step at a Time)

## Current Step (Completed)
Changed only `scripts/eval_codegen.py`.

### What changed
Evaluator now resolves dataset rows by:
- `(dataset_source, task_id)` first
- fallback to `task_id` for older run files

### Why this step
This prevents wrong test matching when many perturbation files share the same `task_id`.

## Next Steps (Not implemented yet)
1. Add summary outputs grouped by `perturbation_name` for report tables.
2. Add aggregation script for per-perturbation pass rate and consistency.
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
Prompt handling and CSV schema are unchanged.
For non-Python datasets (JS/Java/C++/Go), `eval_codegen.py` still needs language-specific executors.
