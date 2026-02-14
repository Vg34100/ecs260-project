import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Tuple

from humaneval import load_humaneval


def _worker(program: str, test_code: str, entry_point: str, q: mp.Queue) -> None:
    try:
        ns = {}
        exec(program, ns)
        if entry_point not in ns:
            raise RuntimeError("entry point missing")
        ns["candidate"] = ns[entry_point]
        exec(test_code, ns)
        q.put((True, ""))
    except Exception as e:
        q.put((False, repr(e)))


def run_test(program: str, test_code: str, entry_point: str, timeout_s: int) -> Tuple[bool, str]:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker, args=(program, test_code, entry_point, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "timeout"
    if q.empty():
        return False, "no result"
    return q.get()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HumanEval codegen outputs")
    parser.add_argument("--dataset", default="data/HumanEval.jsonl")
    parser.add_argument("--runs", default="runs/codegen.jsonl")
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--out", default="metrics/summary.csv")
    args = parser.parse_args()

    dataset_by_source_and_task: Dict[Tuple[str, str], Dict[str, object]] = {}
    for item in load_humaneval(args.dataset):
        task_id = str(item["task_id"])
        source = str(item.get("dataset_source", ""))
        dataset_by_source_and_task[(source, task_id)] = item
    total = 0
    passed = 0
    failures = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        out.write("task_id,repeat,passed,error\n")
        with Path(args.runs).open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                task_id = rec["task_id"]
                source = str(rec.get("dataset_source", ""))
                item = dataset_by_source_and_task.get((source, task_id))
                if not item:
                    continue

                prompt = rec["prompt"]
                completion = rec["completion"]
                program = prompt + completion
                ok, err = run_test(program, item["test"], item["entry_point"], args.timeout)

                total += 1
                if ok:
                    passed += 1
                else:
                    failures += 1

                out.write(f"{task_id},{rec['repeat']},{int(ok)},{err}\n")

    pass_rate = (passed / total) if total else 0.0
    print(f"Total: {total} Passed: {passed} Failed: {failures} Pass rate: {pass_rate:.3f}")


if __name__ == "__main__":
    main()
