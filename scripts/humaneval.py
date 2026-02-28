import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _resolve_dataset_files(path: str) -> List[Path]:
    p = Path(path)
    if p.exists() and p.is_file():
        return [p]
    if p.exists() and p.is_dir():
        return sorted(p.rglob("*.jsonl"))
    # Support glob patterns when path does not exist as a literal path.
    files = sorted(Path(".").glob(path))
    return [f for f in files if f.is_file() and f.suffix == ".jsonl"]


def load_codegen_dataset(path: str, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Load codegen JSONL data from file, directory, or glob pattern.

    This is dataset-agnostic and works for multi-language folders
    (e.g., humanevalpy/humanevaljs/humanevalcpp/humanevaljava/humanevalgo)
    as long as rows contain the expected task fields used by the pipeline.
    """
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
                item["dataset_source"] = str(p)
                yield item
                count += 1
                if limit is not None and count >= limit:
                    return


def load_humaneval(path: str, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Backward-compatible alias for older imports."""
    yield from load_codegen_dataset(path, limit=limit)
