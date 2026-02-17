# Next Codex Handoff

## Project Goal (Current Scope)
- Validate Python codegen-eval pipeline on nominal + perturbed HumanEval-style datasets.
- Prepare reproducible runs and perturbation-level summaries for report work.

## Current Pipeline Status
- `scripts/humaneval.py`
  - Loader supports file, directory, or glob input.
  - Emits `dataset_source` per row.
- `scripts/run_codegen.py`
  - Run JSONL records include:
    - `task_id`
    - `dataset_source`
    - `perturbation_name`
    - `prompt_file`
    - `model`, `model_path`, `repeat`, `timestamp`
- `scripts/eval_codegen.py`
  - Evaluates with strict key match: `(dataset_source, task_id)`.
  - Output CSV includes:
    - `task_id, repeat, passed, dataset_source, perturbation_name, prompt_file, model, error`
- `scripts/download_instruct_model.py`
  - Downloads model from HF.
  - Reads token from `HFTOKEN` (env or `.env`).

## Known Behavior from Latest Model Smoke
- `metrics/model_smoke_py_eval.csv` shows many failures.
- Frequent error types:
  - Syntax errors from truncated/garbled output.
  - Markdown fence leakage (e.g., ```python).
  - FIM special token leakage (`<|fim_*|>`).
  - Occasional `NameError: List is not defined`.
- Interpretation:
  - Pipeline wiring works.
  - Main blocker is generation output formatting/post-processing quality.

## Setup on New Machine
1. Clone repo and enter it.
2. Create env:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub sentencepiece accelerate
```
3. Put HF token in `.env`:
```bash
HFTOKEN=hf_xxx
```

## Download Model
```bash
python3 scripts/download_instruct_model.py \
  --repo-id Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --out models/qwen2.5-coder-1.5b-instruct
```

## Minimal Smoke Commands (Python Nominal)
```bash
python3 scripts/run_codegen.py \
  --dataset datasets/nominal/HumanEval_py.jsonl \
  --model transformers \
  --model-path models/qwen2.5-coder-1.5b-instruct \
  --device cpu \
  --limit 5 \
  --repeats 1 \
  --out runs/model_smoke_py.jsonl
```

```bash
python3 scripts/eval_codegen.py \
  --dataset datasets/nominal/HumanEval_py.jsonl \
  --runs runs/model_smoke_py.jsonl \
  --out metrics/model_smoke_py_eval.csv
```

## Perturbed Python Smoke (All Subfolders)
```bash
python3 scripts/run_codegen.py \
  --dataset "datasets/perturbed/humanevalpy/full/**/*.jsonl" \
  --model transformers \
  --model-path models/qwen2.5-coder-1.5b-instruct \
  --device cpu \
  --limit 20 \
  --repeats 1 \
  --out runs/smoke_py.jsonl
```

```bash
python3 scripts/eval_codegen.py \
  --dataset "datasets/perturbed/humanevalpy/full/**/*.jsonl" \
  --runs runs/smoke_py.jsonl \
  --out metrics/eval_py_smoke.csv
```

## Next Tasks (Priority Order)
1. Add output cleanup in `scripts/run_codegen.py` (single focused patch):
   - Strip markdown fences.
   - Strip FIM tokens (`<|fim_prefix|>`, etc.).
   - Trim obvious trailing junk after function body.
2. Re-run nominal smoke (`limit=5`) and verify syntax errors drop.
3. Scale nominal smoke (`limit=20`), then set `repeats=10`.
4. Add perturbation-level consistency summary (after pass-rate summary).

## Completed Update
- Added `scripts/summarize_by_perturbation.py`.
- Input: eval CSV (`--eval-csv`, default `metrics/summary.csv`)
- Output columns: `perturbation_name,total,passed,pass_rate`
- Default output: `metrics/summarize_by_perturbation.csv`

## Quick Validation Checks
```bash
python3 -m py_compile scripts/humaneval.py scripts/run_codegen.py scripts/eval_codegen.py scripts/download_instruct_model.py
```

```bash
sed -n '1,20p' metrics/model_smoke_py_eval.csv
```

```bash
python3 - <<'PY'
import csv
from collections import Counter
rows=list(csv.DictReader(open("metrics/model_smoke_py_eval.csv")))
print("rows", len(rows), "passed", sum(int(r["passed"]) for r in rows))
errs=Counter(r["error"] for r in rows if r["error"])
print("top_errors")
for e,c in errs.most_common(10):
    print(c, e[:160])
PY
```

## Notes
- `--limit` applies across all matched files in sorted order. With globs, this may bias early subfolders.
- Current evaluator is Python-execution based; non-Python language execution is not implemented yet.
