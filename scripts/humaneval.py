import json
from pathlib import Path
from typing import Iterable, Dict, Any, Optional


def load_humaneval(path: str, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Load HumanEval-style JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    count = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            yield item
            count += 1
            if limit is not None and count >= limit:
                break
