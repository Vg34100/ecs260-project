ECS260 - Software Engineering Project
Prompt Drift as a Reliability Risk: Measuring Stability of LLM-Based SE Pipelines
Ethan Chen (918004305), Khoi Nguyen (919833517), Pei-Yu Lin (925570136), Pablo Rodriguez (925562115), Shuang Ma (924922662)

Overview
This repo measures prompt drift and run-to-run stability of LLM-based software engineering pipelines across code generation, code summarization, bug detection, test generation, and non-code summarization. The core pipeline produces run logs, evaluation results, stability metrics, and plots. The latest findings are in `docs/findings.md`.

Requirements
- Python 3.10+
- Virtual environment (recommended)
- For Ollama runs: Ollama server running on Windows/macOS/Linux
- For plots: `plotly`, `matplotlib`, `numpy`

Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install datasets sentence-transformers plotly zss
```

Datasets (download small subsets)
```bash
python3 scripts/datasets/download_xsum.py --split train --limit 200 --out datasets/xsum_subset.jsonl
python3 scripts/datasets/download_codesearchnet.py --split train --limit 200 --out datasets/codesearchnet_py_subset.jsonl
python3 scripts/datasets/download_defect.py --split train --limit 100 --balanced --out datasets/defect_subset_balanced.jsonl
```

Transformers (local model)
```bash
python3 scripts/run_codegen.py \
  --dataset datasets/nominal/HumanEval_py.jsonl \
  --model transformers \
  --model-path models/qwen2.5-coder-1.5b-instruct \
  --device cuda \
  --limit 30 \
  --repeats 10 \
  --out runs/qwen25_codegen.jsonl
```

Ollama (Windows + WSL2)
If Ollama runs on Windows, use the Windows host IP. Example from this machine:
`http://100.121.2.67:11434`. Update as needed. If `grep nameserver /etc/resolv.conf`
does not work, validate with:
```bash
curl -s http://<windows-ip>:11434/api/tags
```

Core Codegen Run (Ollama)
```bash
python3 scripts/run_codegen.py \
  --dataset datasets/nominal/HumanEval_py.jsonl \
  --model ollama \
  --ollama-model llama3.2:3b \
  --ollama-url http://100.121.2.67:11434 \
  --limit 30 \
  --repeats 10 \
  --out runs/ollama_llama32_3b.jsonl
```

Evaluate + Stability
```bash
python3 scripts/eval_codegen.py \
  --dataset datasets/nominal/HumanEval_py.jsonl \
  --runs runs/ollama_llama32_3b.jsonl \
  --out metrics/ollama_llama32_3b_eval.csv

python3 scripts/metrics/compute_stability.py \
  --runs runs/ollama_llama32_3b.jsonl \
  --eval metrics/ollama_llama32_3b_eval.csv \
  --out metrics/ollama_llama32_3b_stability.csv \
  --with-tree-edit
```

Full Suite Runner
```bash
python3 scripts/run_full_suite.py \
  --ollama-url http://100.121.2.67:11434 \
  --include-summarization --include-bug --include-testgen \
  --limit 30 --repeats 10
```

Plots
```bash
python3 scripts/plots/build_all.py
```

Reproducibility Notes
- Runs are deterministic (temperature=0) unless otherwise noted.
- Outputs are written to `runs/` and metrics to `metrics/`.
- Visualizations are written to `figures/`.
