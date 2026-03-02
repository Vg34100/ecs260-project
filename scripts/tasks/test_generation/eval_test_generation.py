"""Evaluate generated tests by running them against canonical solutions."""
from __future__ import annotations

import argparse
import json
import signal
import re
import textwrap
from pathlib import Path
from typing import Dict


def load_mbpp(path: str) -> Dict[str, Dict[str, str]]:
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data[item.get("task_id")] = item
    return data


def extract_test_code(text: str) -> str:
    if "```" in text:
        parts = re.split(r"```(?:python)?", text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            text = parts[1]
    text = text.replace("```", "")
    text = textwrap.dedent(text).strip()
    return text


def filter_asserts(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("assert "):
            lines.append(line)
    return "\n".join(lines).strip()


def include_setup_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            lines.append(stripped)
        elif " = " in stripped and "(" not in stripped and ")" not in stripped:
            # likely a simple constant assignment, keep it
            lines.append(stripped)
        elif stripped.startswith("cost =") or stripped.startswith("lst ="):
            lines.append(stripped)
    return "\n".join(lines).strip()


def drop_incomplete_lines(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        if line.count("(") != line.count(")"):
            continue
        if line.count("[") != line.count("]"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate test generation")
    parser.add_argument("--dataset", default="datasets/nominal/mbpp_wtest.jsonl")
    parser.add_argument("--runs", default="runs/mbpp_testgen.jsonl")
    parser.add_argument("--out", default="metrics/mbpp_testgen_eval.csv")
    args = parser.parse_args()

    dataset = load_mbpp(args.dataset)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    passed = 0
    invalid = 0

    with Path(args.runs).open("r", encoding="utf-8") as f, out_path.open(
        "w", encoding="utf-8"
    ) as out:
        out.write("task_id,repeat,passed,invalid,error\n")
        for line in f:
            rec = json.loads(line)
            task_id = rec.get("task_id")
            item = dataset.get(task_id)
            if not item:
                continue

            completion = rec.get("completion", "")
            prompt = item.get("prompt", "")
            canonical = item.get("canonical_solution", "")
            solution = f"{prompt.lstrip()}\n{canonical}".rstrip()
            entry = item.get("entry_point", "")
            test_code = extract_test_code(completion)
            test_code = include_setup_lines(test_code)
            test_code = drop_incomplete_lines(test_code)

            code = f"{solution}\n\n{test_code}\n"
            local_env = {}
            try:
                def _timeout_handler(signum, frame):
                    raise TimeoutError("timeout")

                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(1)

                if not test_code:
                    raise RuntimeError("no asserts found")
                exec(code, local_env)
                func = local_env.get(entry)
                if func is None:
                    raise RuntimeError("entry point missing")
                total += 1
                passed += 1
                out.write(f"{task_id},{rec.get('repeat',0)},1,0,\n")
            except Exception as e:
                total += 1
                invalid += 1
                err = str(e).replace("\n", " ")
                out.write(f"{task_id},{rec.get('repeat',0)},0,1,{err}\n")
            finally:
                signal.alarm(0)

    rate = passed / total if total else 0.0
    print(f"Total: {total} Passed: {passed} Invalid: {invalid} Pass rate: {rate:.3f}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
