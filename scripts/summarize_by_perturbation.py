import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize eval results by perturbation_name"
    )
    parser.add_argument("--eval-csv", default="metrics/summary.csv")
    parser.add_argument("--out", default="metrics/summarize_by_perturbation.csv")
    args = parser.parse_args()

    grouped = defaultdict(lambda: {"total": 0, "passed": 0})

    with Path(args.eval_csv).open("r", encoding="utf-8", newline="") as in_f:
        reader = csv.DictReader(in_f)
        for row in reader:
            name = row.get("perturbation_name", "") or "unknown"
            grouped[name]["total"] += 1
            grouped[name]["passed"] += int(row.get("passed", 0))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["perturbation_name", "total", "passed", "pass_rate"],
        )
        writer.writeheader()
        for name in sorted(grouped):
            total = grouped[name]["total"]
            passed = grouped[name]["passed"]
            pass_rate = (passed / total) if total else 0.0
            writer.writerow(
                {
                    "perturbation_name": name,
                    "total": total,
                    "passed": passed,
                    "pass_rate": f"{pass_rate:.3f}",
                }
            )

    print(f"Wrote: {out_path} ({len(grouped)} groups)")


if __name__ == "__main__":
    main()
