ECS260 - Software Engineering Project
-
Prompt Drift as a Reliability Risk: Measuring Stability of LLM-Based SE Pipelines
Ethan Chen (918004305), Khoi Nguyen (919833517), Pei-Yu Lin (925570136), Pablo Rodriguez (925562115), Shuang Ma (924922662)

Basic MVP (codegen)

Setup
- Place HumanEval JSONL at `data/HumanEval.jsonl`.
- Optional: have a local transformers model on disk.

Run (dummy model)
- `python3 scripts/run_codegen.py --dataset data/HumanEval.jsonl --limit 20 --out runs/codegen.jsonl`

Run (local transformers)
- `python3 scripts/run_codegen.py --model transformers --model-path /path/to/model --limit 20`

Evaluate
- `python3 scripts/eval_codegen.py --dataset data/HumanEval.jsonl --runs runs/codegen.jsonl --out metrics/summary.csv`
