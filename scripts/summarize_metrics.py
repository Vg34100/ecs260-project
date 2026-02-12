import argparse
import csv
from pathlib import Path


def read_pass_rate(path: Path):
    # Summary CSV is one row per run with a passed flag.
    total = 0
    passed = 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            passed += int(row["passed"])
    return total, (passed / total) if total else 0.0


def read_avg_consistency(path: Path):
    # Consistency CSV includes a final ALL row with the average.
    avg = 0.0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["task_id"] == "ALL":
                avg = float(row["consistency"])
                break
    return avg


def main() -> None:
    # Produce a tiny report table for the progress report.
    parser = argparse.ArgumentParser(description="Make small summary table")
    parser.add_argument("--summary", default="metrics/summary.csv")
    parser.add_argument("--consistency", default="metrics/consistency.csv")
    parser.add_argument("--out", default="metrics/table.csv")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    cons_path = Path(args.consistency)
    out_path = Path(args.out)

    total, pass_rate = read_pass_rate(summary_path)
    avg_consistency = read_avg_consistency(cons_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        out.write("total_runs,pass_rate,avg_consistency\n")
        out.write(f"{total},{pass_rate:.3f},{avg_consistency:.3f}\n")

    print(f"Total runs: {total} Pass rate: {pass_rate:.3f} Avg consistency: {avg_consistency:.3f}")


if __name__ == "__main__":
    main()
