# Dataset Adaptation Notes (One Step at a Time)

## Current Step (Completed)
Changed only `scripts/humaneval.py`.

### What changed
`load_humaneval(path, limit)` now accepts:
- a single `.jsonl` file path
- a directory (recursively loads all `.jsonl` files)
- a glob pattern (e.g. `dataset-release/perturbed_finalized/humaneval/full/**/*.jsonl`)

### Why this first
This is the smallest safe change that unlocks running on perturbed dataset collections without touching run/eval logic yet.

## Next Steps (Not implemented yet)
1. Add optional metadata in run outputs: `dataset_source` and `perturbation_name`.
2. Update evaluator to disambiguate tasks when the same `task_id` appears across many files.
3. Add aggregation script for per-perturbation report tables.

## Example usage after current step
You can now point existing scripts to a dataset directory or glob:

```bash
python3 scripts/run_codegen.py \
  --dataset "dataset-release/perturbed_finalized/humaneval/full/**/*.jsonl" \
  --model dummy \
  --limit 20 \
  --out runs/smoke.jsonl
```

Note: this step only changes dataset loading. Prompt handling and evaluation schema are unchanged.
