ECS260 - Software Engineering Project
-
Prompt Drift as a Reliability Risk: Measuring Stability of LLM-Based SE Pipelines
Ethan Chen (918004305), Khoi Nguyen (919833517), Pei-Yu Lin (925570136), Pablo Rodriguez (925562115), Shuang Ma (924922662)

Basic MVP (codegen)

Setup
- Place HumanEval JSONL at `data/human-eval-v2-20210705.jsonl` (or update `--dataset`).
- Optional: have a local transformers model on disk.

Run (dummy model)
- `python3 scripts/run_codegen.py --dataset data/HumanEval.jsonl --limit 20 --out runs/codegen.jsonl`

Run (local transformers)
- `python3 scripts/run_codegen.py --dataset data/human-eval-v2-20210705.jsonl --model transformers --model-path /path/to/model --limit 20`

Evaluate
- `python3 scripts/eval_codegen.py --dataset data/human-eval-v2-20210705.jsonl --runs runs/codegen.jsonl --out metrics/summary.csv`

Repeat runs + consistency (small subset)
- `python3 scripts/run_codegen.py --dataset data/human-eval-v2-20210705.jsonl --model transformers --model-path /path/to/model --limit 20 --repeats 3 --out runs/codegen_repeats.jsonl`
- `python3 scripts/eval_codegen.py --dataset data/human-eval-v2-20210705.jsonl --runs runs/codegen_repeats.jsonl --out metrics/summary_repeats.csv`
- `python3 scripts/compute_consistency.py --runs runs/codegen_repeats.jsonl --out metrics/consistency.csv`
- `python3 scripts/summarize_metrics.py --summary metrics/summary_repeats.csv --consistency metrics/consistency.csv --out metrics/table.csv`
