import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_runs(path: Path):
    # Load JSONL runs into simple arrays for analysis.
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append(
                {
                    "task_id": rec["task_id"],
                    "repeat": rec["repeat"],
                    "prompt_len": len(rec["prompt"]),
                    "completion_len": len(rec["completion"]),
                }
            )
    return rows


def load_summary(path: Path):
    # Summary CSV can include commas in error strings, so parse manually.
    rows = []
    with path.open("r", encoding="utf-8") as f:
        _header = f.readline()
        for line in f:
            parts = line.strip().split(",", 3)
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "task_id": parts[0],
                    "repeat": int(parts[1]),
                    "passed": int(parts[2]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirmatory data analysis (CDA)")
    parser.add_argument("--runs", default="runs/codegen_repeats.jsonl")
    parser.add_argument("--summary", default="metrics/summary_repeats.csv")
    parser.add_argument("--out-dir", default="metrics")
    args = parser.parse_args()

    runs = load_runs(Path(args.runs))
    summary = load_summary(Path(args.summary))

    # Join runs with pass/fail by (task_id, repeat).
    passed_map = {(r["task_id"], r["repeat"]): r["passed"] for r in summary}
    joined = []
    for r in runs:
        key = (r["task_id"], r["repeat"])
        if key in passed_map:
            r = dict(r)
            r["passed"] = passed_map[key]
            joined.append(r)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CDA Option B1: t-test on completion length by pass/fail.
    pass_lens = np.array([r["completion_len"] for r in joined if r["passed"] == 1])
    fail_lens = np.array([r["completion_len"] for r in joined if r["passed"] == 0])
    t_stat, p_val = stats.ttest_ind(pass_lens, fail_lens, equal_var=False)
    pass_mean = float(pass_lens.mean()) if len(pass_lens) else 0.0
    fail_mean = float(fail_lens.mean()) if len(fail_lens) else 0.0
    pass_std = float(pass_lens.std(ddof=1)) if len(pass_lens) > 1 else 0.0
    fail_std = float(fail_lens.std(ddof=1)) if len(fail_lens) > 1 else 0.0
    # Cohen's d for effect size (pooled std).
    if len(pass_lens) > 1 and len(fail_lens) > 1:
        pooled = np.sqrt(
            ((len(pass_lens) - 1) * pass_std**2 + (len(fail_lens) - 1) * fail_std**2)
            / (len(pass_lens) + len(fail_lens) - 2)
        )
        cohens_d = (pass_mean - fail_mean) / pooled if pooled > 0 else 0.0
    else:
        cohens_d = 0.0

    with (out_dir / "cda_ttest.csv").open("w", encoding="utf-8") as f:
        f.write("group,n,mean_completion_len,std_completion_len\n")
        f.write(f"pass,{len(pass_lens)},{pass_mean:.3f},{pass_std:.3f}\n")
        f.write(f"fail,{len(fail_lens)},{fail_mean:.3f},{fail_std:.3f}\n")
        f.write(f"t_stat,,{t_stat:.4f},\n")
        f.write(f"p_value,,{p_val:.3e},\n")
        f.write(f"cohens_d,,{cohens_d:.4f},\n")

    # CDA Option B2: chi-square on short/long completion vs pass/fail.
    median_len = np.median([r["completion_len"] for r in joined])
    # 2x2 contingency table: rows=short/long, cols=fail/pass
    short_fail = sum(1 for r in joined if r["completion_len"] <= median_len and r["passed"] == 0)
    short_pass = sum(1 for r in joined if r["completion_len"] <= median_len and r["passed"] == 1)
    long_fail = sum(1 for r in joined if r["completion_len"] > median_len and r["passed"] == 0)
    long_pass = sum(1 for r in joined if r["completion_len"] > median_len and r["passed"] == 1)
    chi2, p_chi, _, _ = stats.chi2_contingency([[short_fail, short_pass], [long_fail, long_pass]])

    with (out_dir / "cda_chi_square.csv").open("w", encoding="utf-8") as f:
        f.write("group,fail,pass\n")
        f.write(f"short,{short_fail},{short_pass}\n")
        f.write(f"long,{long_fail},{long_pass}\n")
        f.write(f"chi2,{chi2:.4f},p={p_chi:.3e}\n")

    # CDA Option A: simple linear regression (pass as numeric outcome).
    x = np.array([r["completion_len"] for r in joined], dtype=float)
    y = np.array([r["passed"] for r in joined], dtype=float)
    slope, intercept, r_value, p_reg, _stderr = stats.linregress(x, y)
    r_squared = r_value**2

    with (out_dir / "cda_regression.csv").open("w", encoding="utf-8") as f:
        f.write("slope,intercept,r_value,r_squared,p_value\n")
        f.write(f"{slope:.6f},{intercept:.6f},{r_value:.4f},{r_squared:.4f},{p_reg:.3e}\n")

    print("CDA outputs written to", out_dir)
    print("t-test p-value:", f"{p_val:.3e}")
    print("chi-square p-value:", f"{p_chi:.3e}")
    print("regression p-value:", f"{p_reg:.3e}")


if __name__ == "__main__":
    main()
