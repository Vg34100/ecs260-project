# Teammate Changes Since Your Last Handoff

## Comparison baseline
- Baseline used: `fee98d1` (`2026-02-12`, "Add comprehensive documentation for the Codex handoff process and pipeline status").
- Compared range: `fee98d1..HEAD` (up to `4e78a83`, merged `2026-03-02`).
- Net repo delta in this range: **44 files changed, 3770 insertions, 15 deletions**.

## High-level summary
Your handoff was focused on the core HumanEval codegen/eval pipeline plus perturbation-aware bookkeeping.  
Your teammate extended this into a broader experimental framework with:
- model-to-model stability analysis,
- multiple new SE/NLP task pipelines,
- automated feature/correlation analysis,
- a plotting suite,
- and one-command full-suite orchestration.

## What was added after your handoff

### 1) Core pipeline hardening and model support
- Added output validity checking utility: `scripts/validaity.py` (`b08078e`).
- Improved output cleanup in `scripts/run_codegen.py` (extra trailing-pattern cleanup + cleanup ordering fixes): `19190ed`, `d72c3c2`.
- Added Ollama backend and stability instrumentation in `scripts/run_codegen.py` with new utilities:
  - `scripts/utils/ast_utils.py`
  - `scripts/utils/semantics.py`
  - `scripts/metrics/compute_stability.py`
  (`752811b`).
- Updated `scripts/eval_codegen.py` + stability code to support `model_name` and model-comparison workflows (`f0ace7d`).

### 2) New datasets and loaders (non-HumanEval tasks)
- Added dataset downloaders:
  - `scripts/datasets/download_xsum.py`
  - `scripts/datasets/download_codesearchnet.py`
  - `scripts/datasets/download_defect.py`
  (`4454938`, `968a219`).
- Added checked-in subsets:
  - `datasets/xsum_subset.jsonl`
  - `datasets/codesearchnet_py_subset.jsonl`
  - `datasets/defect_subset.jsonl`
  - `datasets/defect_subset_balanced.jsonl`

### 3) New task pipelines
- Non-code summarization (XSum):
  - `scripts/tasks/summarization/run_summarization.py`
  - `scripts/tasks/summarization/eval_summarization.py`
  (`b87b3af`).
- Code summarization (CodeSearchNet):
  - `scripts/tasks/code_summarization/run_code_summarization.py`
  - `scripts/tasks/code_summarization/eval_code_summarization.py`
  (`8224da8`).
- Bug detection (CodeXGLUE defect):
  - `scripts/tasks/bug_detection/run_bug_detection.py`
  - `scripts/tasks/bug_detection/eval_bug_detection.py`
  (`968a219`).
- MBPP test generation:
  - `scripts/tasks/test_generation/run_test_generation.py`
  - `scripts/tasks/test_generation/eval_test_generation.py`
  (`460bec0`, later update `9c3479e`).

### 4) New analysis/metrics layer
- Paraphrase sensitivity:
  - `scripts/metrics/compute_paraphrase_sensitivity.py`
  - `scripts/metrics/aggregate_paraphrase_sensitivity.py`
  (`846513c`).
- Model comparison:
  - `scripts/metrics/compare_models.py`
  (`f0ace7d`).
- RQ4-style feature engineering + correlation analysis:
  - `scripts/metrics/compute_task_features.py`
  - `scripts/metrics/analyze_task_features.py`
  (`d349692`, `4c61e5e`).

### 5) Visualization suite added
- New plotting package under `scripts/plots/` (`1a5a22c`, with fixes `0af63b2`, `f4c06ba`):
  - `build_all.py`
  - stability bins/distributions/atlas
  - behavior-vs-structure scatter
  - prompt heatmap
  - task-feature correlation plot
  - MBPP comparison/error plots
  - task summary plots

### 6) End-to-end orchestration
- Added `scripts/run_full_suite.py` to run codegen, eval, stability, comparison, feature analysis, optional extra tasks, and plots in one command (`9c3479e`).

### 7) Documentation/results updates
- Added and continuously expanded `docs/findings.md` with experiment results, interpretations, commands, and commit references (multiple commits starting `752811b`, updates through `9c3479e`).
- Updated `README.md` to reflect the new multi-task + full-suite workflow (`3449953`, `965bccc`).

## Practical difference from your handoff state
- **Before (your handoff):** stable base for HumanEval codegen/eval + perturbation provenance.
- **Now:** a multi-task research pipeline with comparative stability analytics, reproducible full-suite execution, and publication-ready visual outputs.

## New Pull Update (After `4e78a83`, up to `3a8b7c8`)
- New commits detected after the previous summary baseline:
  - `f17341c` (`2026-03-02`) - Paraphrase audit tool
  - `0392270` (`2026-03-04`) - fix (follow-up changes to audit tool)
  - `7111f14` (`2026-03-04`) - more plots
  - Merged by `e7da3c1` (PR #8) and `3a8b7c8` (PR #9)

### Files added/updated in this new pull

#### 1) New human-labeling audit utility
- Added `scripts/paraphrase_audit.py`.
- Purpose:
  - samples task IDs from run logs (`--runs`),
  - can attach pass/fail from eval CSV (`--eval-csv`),
  - can prioritize unstable tasks using stability CSV (`--stability-csv --prioritize-unstable`),
  - exports an annotation-ready CSV with `human_label` and `notes` fields.
- Practical impact: this adds a manual-review workflow for validating paraphrase-drift patterns qualitatively, not just numerically.

#### 2) New bug detection confusion-matrix plot
- Added `scripts/plots/plot_bug_confusion.py`.
- Purpose:
  - reads bug-detection eval CSV(s),
  - builds predicted-vs-actual confusion matrices per model,
  - tracks `unknown` predictions separately,
  - writes `figures/bug_detection_confusion.png`.

#### 3) New prompt-level stability bar plot
- Added `scripts/plots/plot_prompt_stability_bar.py`.
- Purpose:
  - aggregates stability metrics by `(model_name, prompt_file)`,
  - visualizes average `exact_match_rate`, `ast_jaccard_mean`, `behavior_consistency` by prompt variant,
  - writes `figures/prompt_stability_bar.png`.

#### 4) Plot runner updated
- Modified `scripts/plots/build_all.py`.
- Change: includes the two new plot scripts in the `PLOTS` list, so they run automatically with `build_all.py`.
