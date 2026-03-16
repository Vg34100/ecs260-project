#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REQUIRED_FIELDS = {
    "task_id": str,
    "prompt_file": str,
    "prompt": str,
    "completion": str,
    "model": str,
    "repeat": int,
    "timestamp": (int, float),
}

OPTIONAL_FIELDS = {
    "dataset_source": (str, type(None)),
    "perturbation_name": (str, type(None)),
    "model_name": (str, type(None)),
    "model_path": (str, type(None)),
}

DUP_KEY_FIELDS = ("task_id", "prompt_file", "model", "repeat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate JSONL run files and catch malformed records early."
    )
    parser.add_argument(
        "--runs",
        required=True,
        nargs="+",
        help="One or more JSONL files to validate.",
    )
    parser.add_argument(
        "--allow-empty-completion",
        action="store_true",
        help="Allow empty completion strings.",
    )
    parser.add_argument(
        "--strict-extra-fields",
        action="store_true",
        help="Warn about fields not in the expected schema.",
    )
    return parser.parse_args()


def is_instance_of(value: Any, expected_type: Any) -> bool:
    return isinstance(value, expected_type)


def validate_required_fields(
    record: Dict[str, Any],
    line_no: int,
    file_path: Path,
    allow_empty_completion: bool,
) -> List[str]:
    errors: List[str] = []

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f"{file_path}:{line_no}: missing required field '{field}'")
            continue

        value = record[field]
        if not is_instance_of(value, expected_type):
            errors.append(
                f"{file_path}:{line_no}: field '{field}' has wrong type "
                f"(got {type(value).__name__}, expected {expected_type})"
            )
            continue

        if isinstance(value, str):
            if field != "completion" and value.strip() == "":
                errors.append(f"{file_path}:{line_no}: field '{field}' is empty")
            if field == "completion" and not allow_empty_completion and value == "":
                errors.append(f"{file_path}:{line_no}: field 'completion' is empty")

    return errors


def validate_optional_fields(
    record: Dict[str, Any],
    line_no: int,
    file_path: Path,
) -> List[str]:
    errors: List[str] = []

    for field, expected_type in OPTIONAL_FIELDS.items():
        if field in record and not is_instance_of(record[field], expected_type):
            errors.append(
                f"{file_path}:{line_no}: optional field '{field}' has wrong type "
                f"(got {type(record[field]).__name__}, expected {expected_type})"
            )

    return errors


def validate_content(
    record: Dict[str, Any],
    line_no: int,
    file_path: Path,
) -> List[str]:
    errors: List[str] = []

    prompt_file = record.get("prompt_file")
    if isinstance(prompt_file, str) and prompt_file and not prompt_file.endswith(".txt"):
        errors.append(
            f"{file_path}:{line_no}: prompt_file does not end with .txt: '{prompt_file}'"
        )

    repeat = record.get("repeat")
    if isinstance(repeat, int) and repeat < 0:
        errors.append(f"{file_path}:{line_no}: repeat must be >= 0")

    timestamp = record.get("timestamp")
    if isinstance(timestamp, (int, float)) and timestamp <= 0:
        errors.append(f"{file_path}:{line_no}: timestamp must be > 0")

    return errors


def validate_extra_fields(
    record: Dict[str, Any],
    line_no: int,
    file_path: Path,
) -> List[str]:
    expected = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)
    extra = sorted(set(record) - expected)
    if not extra:
        return []
    return [
        f"{file_path}:{line_no}: unexpected extra field(s): {', '.join(extra)}"
    ]


def iter_jsonl(path: Path) -> Iterable[Tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            yield idx, line.rstrip("\n")


def main() -> int:
    args = parse_args()

    total_records = 0
    total_errors = 0
    total_warnings = 0
    

    all_errors: List[str] = []
    all_warnings: List[str] = []

    for run_path_str in args.runs:
        path = Path(run_path_str)
        file_duplicate_keys: Counter[Tuple[Any, ...]] = Counter()
        if not path.exists():
            all_errors.append(f"{path}: file does not exist")
            continue

        if not path.is_file():
            all_errors.append(f"{path}: not a file")
            continue

        if path.suffix != ".jsonl":
            all_warnings.append(f"{path}: file does not end with .jsonl")
            total_warnings += 1

        file_record_count = 0

        for line_no, raw_line in iter_jsonl(path):
            if raw_line.strip() == "":
                all_errors.append(f"{path}:{line_no}: empty line")
                total_errors += 1
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as e:
                all_errors.append(f"{path}:{line_no}: invalid JSON ({e.msg})")
                total_errors += 1
                continue

            if not isinstance(record, dict):
                all_errors.append(f"{path}:{line_no}: JSON root must be an object")
                total_errors += 1
                continue

            file_record_count += 1
            total_records += 1

            record_errors: List[str] = []
            record_errors.extend(
                validate_required_fields(
                    record,
                    line_no,
                    path,
                    allow_empty_completion=args.allow_empty_completion,
                )
            )
            record_errors.extend(validate_optional_fields(record, line_no, path))
            record_errors.extend(validate_content(record, line_no, path))

            if record_errors:
                all_errors.extend(record_errors)
                total_errors += len(record_errors)

            if args.strict_extra_fields:
                warnings = validate_extra_fields(record, line_no, path)
                all_warnings.extend(warnings)
                total_warnings += len(warnings)

            if all(field in record for field in DUP_KEY_FIELDS):
                dup_key = tuple(record[field] for field in DUP_KEY_FIELDS)
                file_duplicate_keys[dup_key] += 1

        print(f"[OK] Scanned {path} ({file_record_count} record(s))")

    duplicate_error_count = 0
    for dup_key, count in file_duplicate_keys.items():
        if count > 1:
            key_str = ", ".join(f"{k}={v!r}" for k, v in zip(DUP_KEY_FIELDS, dup_key))
            all_errors.append(
                f"{path}: duplicate record key ({key_str}) appears {count} times"
            )
            total_errors += 1

    total_errors += duplicate_error_count

    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"  - {warning}")

    if all_errors:
        print("\nErrors:")
        for err in all_errors:
            print(f"  - {err}")

    print("\nSummary:")
    print(f"  Records scanned : {total_records}")
    print(f"  Warnings        : {total_warnings}")
    print(f"  Errors          : {total_errors}")

    if total_errors > 0:
        print("\nValidation FAILED.")
        return 1

    print("\nValidation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())