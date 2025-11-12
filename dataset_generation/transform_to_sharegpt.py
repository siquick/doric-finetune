"""
transform_to_sharegpt.py

Transform Doric dataset JSONL rows from:

{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {...}
}

to ShareGPT-shaped rows:

{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt",   "value": "..."}
  ],
  "meta": {...}  # optional
}

Usage:
  uv run python transform_to_sharegpt.py --input doric_synth.jsonl --output doric_sharegpt.jsonl
  # optionally drop meta:
  uv run python transform_to_sharegpt.py --input doric_synth.jsonl --output doric_sharegpt.jsonl --drop-meta
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, Iterable, List, Optional

try:  # optional faster JSON
    import orjson as _orjson  # type: ignore

    def dumps(obj: Any) -> bytes:
        return _orjson.dumps(obj)
except Exception:  # pragma: no cover

    def dumps(obj: Any) -> bytes:  # type: ignore
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")


ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
}


def transform_row(row: Dict[str, Any], keep_meta: bool) -> Optional[Dict[str, Any]]:
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        logging.warning("row missing 'messages'; skipping")
        return None

    convs: List[Dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).lower()
        content = m.get("content", "")
        mapped = ROLE_MAP.get(role)
        if mapped is None:
            # Skip unknown roles (e.g., system). ShareGPT expects human/gpt pairs.
            continue
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)
        convs.append({"from": mapped, "value": content})

    out: Dict[str, Any] = {"conversations": convs}
    if keep_meta and "meta" in row and isinstance(row["meta"], dict):
        out["meta"] = row["meta"]
    return out


def load_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logging.error("invalid JSON at line %d: %s", lineno, exc)


def write_rows(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "wb") as wf:
        for row in rows:
            wf.write(dumps(row))
            wf.write(b"\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transform dataset to ShareGPT shape")
    p.add_argument("--input", required=True, help="Input JSONL path")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument(
        "--drop-meta", action="store_true", help="Do not include input 'meta' in output"
    )
    p.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    keep_meta = not args.drop_meta
    transformed: List[Dict[str, Any]] = []
    count_in = 0
    count_out = 0

    for row in load_rows(args.input):
        count_in += 1
        out = transform_row(row, keep_meta=keep_meta)
        if out is None:
            continue
        transformed.append(out)
        count_out += 1

    write_rows(args.output, transformed)
    logging.info("transformed %d/%d rows -> %s", count_out, count_in, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

