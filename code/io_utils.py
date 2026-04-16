from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union


PathLike = Union[str, Path]


def load_dataset(path: PathLike) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")

    suf = p.suffix.lower()
    if suf == ".jsonl":
        return list(iter_jsonl(p))

    if suf == ".json":
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            if not all(isinstance(x, dict) for x in obj):
                raise ValueError("JSON list dataset must contain dict items.")
            return obj

        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            data = obj["data"]
            if not all(isinstance(x, dict) for x in data):
                raise ValueError("JSON dict dataset's 'data' list must contain dict items.")
            return data

        raise ValueError("JSON dataset must be a list, or a dict containing list at key 'data'.")

    raise ValueError(f"Only .json or .jsonl supported. Got: {p.suffix}")


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {lineno} must be an object/dict.")
            yield obj


def save_json(path: PathLike, obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def save_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("save_jsonl expects an iterable of dict rows.")
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
