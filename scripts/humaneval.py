import json
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, List


def _resolve_dataset_files(path: str) -> List[Path]:
    p = Path(path)
    if p.exists() and p.is_file():
        return [p]
    if p.exists() and p.is_dir():
        return sorted(p.rglob("*.jsonl"))
    # Support glob patterns when path does not exist as a literal path.
    files = sorted(Path(".").glob(path))
    return [f for f in files if f.is_file() and f.suffix == ".jsonl"]


def load_humaneval(path: str, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Load HumanEval-style JSONL from file, directory, or glob pattern."""
    files = _resolve_dataset_files(path)
    if not files:
        raise FileNotFoundError(f"Dataset not found: {path}")

    count = 0
    for p in files:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                yield item
                count += 1
                if limit is not None and count >= limit:
                    return
