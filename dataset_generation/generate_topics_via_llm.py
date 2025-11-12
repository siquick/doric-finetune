"""
generate_topics_via_llm.py

Use an LLM to expand grouped topics in topics.json without repeating
existing entries. Produces an updated topics JSON with appended topics.

Default model: gpt-4.1 (OpenAI-compatible /v1/chat/completions).

Examples:
  uv run python generate_topics_via_llm.py --input topics.json --output topics_augmented.json --per-group 40
  # In-place update
  uv run python generate_topics_via_llm.py --input topics.json --in-place --per-group 30

Env:
  OPENAI_API_KEY (required for API mode)
  OPENAI_BASE_URL (optional, defaults to https://api.openai.com)
  MODEL (optional, defaults to gpt-4.1)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # optional faster JSON
    import orjson as _orjson  # type: ignore

    def dumps(obj: Any) -> bytes:
        return _orjson.dumps(obj)
except Exception:  # pragma: no cover

    def dumps(obj: Any) -> bytes:  # type: ignore
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")


# OpenAI SDK
try:  # type: ignore
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# dotenv for env loading
try:  # type: ignore
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_topic(s: str) -> str:
    s = clean_text(s).lower()
    # Keep only alphanumerics and whitespace; collapse spaces.
    kept_chars = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            kept_chars.append(ch)
        else:
            kept_chars.append(" ")
    s2 = "".join(kept_chars)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def token_overlap(a: str, b: str) -> float:
    a_set = set(normalize_topic(a).split())
    b_set = set(normalize_topic(b).split())
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def load_topics_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    groups: Dict[str, List[str]] = {}
    if isinstance(data, dict) and "groups" in data and isinstance(data["groups"], dict):
        for g, arr in data["groups"].items():
            groups[str(g)] = [
                clean_text(x) for x in arr if isinstance(x, str) and x.strip()
            ]
        return groups
    if isinstance(data, dict):
        for g, arr in data.items():
            if isinstance(arr, list):
                groups[str(g)] = [
                    clean_text(x) for x in arr if isinstance(x, str) and x.strip()
                ]
        if groups:
            return groups
    if isinstance(data, list):
        # single unnamed group
        groups["general"] = [
            clean_text(x) for x in data if isinstance(x, str) and x.strip()
        ]
        return groups
    raise ValueError("Unsupported topics.json shape")


def write_topics_json(path: str, groups: Mapping[str, Sequence[str]]) -> None:
    payload = {"groups": {g: list(v) for g, v in groups.items()}}
    with open(path, "wb") as wf:
        wf.write(dumps(payload))
        wf.write(b"\n")


SYSTEM_MSG = (
    "You help curate concise, practical topics for a Doric/English dataset."
    " Return JSON only: an array of short topic strings."
)


PROMPT_TEMPLATE = (
    "Generate {n} new short topics for the group '{group}'.\n"
    "Style: mix English and Doric phrasing (nae ower much Doric), natural everyday wording.\n"
    "Avoid repeating or paraphrasing near-duplicates of the given examples.\n"
    "Do not start lines with: How to/How tae, A quick guide, Tips for, Nae bother, A wee blether, Getting started, Plain talk, Fit ye need.\n"
    "Keep each topic a single line, 5–12 words, no trailing punctuation.\n"
    "Return strictly JSON array of strings, nothing else.\n\n"
    "Examples (style hints, do not copy):\n{examples}\n"
)


GROUP_HINTS: Dict[str, str] = {
    # Summaries guide the model; group names match high-level coverage.
    "daily_life": "Food, travel, housekeeping, finance, health, relationships",
    "education": "Studying, explaining school topics, language learning, exams",
    "work": "CVs, interviews, workplace comms, teamwork, remote work",
    "technology": "Beginner computer help, programming, debugging, AI, tools",
    "science": "Physics, biology, chemistry, earth science, space",
    "society_culture": "History, politics, philosophy, religion, geography",
    "arts_creativity": "Music, literature, film, drawing, photography, writing",
    "business_economics": "Markets, startups, marketing, investing, finance",
    "diy_trades": "Carpentry, plumbing, electrics, car maintenance, farming, sewing",
    "emotional_personal": "Mental health, coping, decisions, morality, empathy",
    "extras": "Jokes, storytelling, hypotheticals, debates, roleplay, multilingual",
}


def few_shot_examples(
    group: str, existing: Dict[str, List[str]], k: int = 8
) -> List[str]:
    pool = existing.get(group, [])
    rnd = random.Random(1337)
    pool = list(pool)
    rnd.shuffle(pool)
    return pool[:k]


def build_prompt(group: str, n: int, examples: Sequence[str]) -> str:
    ex_lines = "\n".join(f"- {e}" for e in examples)
    return PROMPT_TEMPLATE.format(n=n, group=group, examples=ex_lines)


def parse_json_array(text: str) -> Optional[List[str]]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            out = [clean_text(str(x)) for x in data if isinstance(x, (str, int, float))]
            return [x for x in out if x]
    except Exception:
        pass
    return None


def api_generate_topics(
    client: "OpenAI",
    model: str,
    group: str,
    n: int,
    examples: Sequence[str],
) -> List[str]:
    system = SYSTEM_MSG
    user = build_prompt(group, n, examples)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=800,
    )
    content = resp.choices[0].message.content or ""
    arr = parse_json_array(content)
    if arr is None:
        # best-effort: try to extract lines starting with dash
        lines = [
            clean_text(l[1:].strip())
            for l in content.splitlines()
            if l.strip().startswith("-")
        ]
        arr = [l for l in lines if l]
    return arr or []


def fallback_generate_topics(group: str, n: int, examples: Sequence[str]) -> List[str]:
    # To avoid junk, do not generate when no API is available.
    logging.warning("No OPENAI_API_KEY set; skipping generation for group '%s'", group)
    return []


def starts_with_banned(topic: str) -> bool:
    t = topic.strip().lower()
    banned = (
        "how to",
        "how tae",
        "a quick guide",
        "tips for",
        "nae bother",
        "a wee blether",
        "getting started",
        "plain talk",
        "fit ye need",
    )
    return any(t.startswith(b) for b in banned)


def validate_topics(candidates: Sequence[str]) -> List[str]:
    out: List[str] = []
    for t in candidates:
        t = clean_text(t)
        if not t:
            continue
        if starts_with_banned(t):
            continue
        toks = t.split()
        if not (5 <= len(toks) <= 12):
            continue
        letters = sum(ch.isalpha() for ch in t)
        if letters < 8:
            continue
        uniq = {w.lower() for w in toks if any(c.isalpha() for c in w)}
        if len(uniq) < 3:
            continue
        # avoid heavy repetition (same token > 2 times)
        freqs: Dict[str, int] = {}
        ok = True
        for w in toks:
            w2 = w.lower()
            freqs[w2] = freqs.get(w2, 0) + 1
            if freqs[w2] > 2:
                ok = False
                break
        if not ok:
            continue
        out.append(t)
    return out


def deduplicate(
    proposed: Sequence[str],
    existing_norm: Sequence[str],
    threshold: float = 0.8,
) -> List[str]:
    out: List[str] = []
    seen_norm = set(existing_norm)
    for t in proposed:
        nrm = normalize_topic(t)
        if not nrm:
            continue
        if nrm in seen_norm:
            continue
        # Check near duplicates vs existing and vs out
        is_near = any(token_overlap(t, ex) >= threshold for ex in existing_norm) or any(
            token_overlap(t, p) >= threshold for p in out
        )
        if is_near:
            continue
        out.append(clean_text(t))
        seen_norm.add(nrm)
    return out


def expand_groups(
    groups: Dict[str, List[str]],
    per_group: int,
    model: str,
    api_key: Optional[str],
) -> Dict[str, List[str]]:
    # establish client if API available
    client: Optional["OpenAI"] = None
    if api_key and OpenAI is not None:
        client = OpenAI(api_key=api_key)

    existing_all_norm = [normalize_topic(t) for arr in groups.values() for t in arr]
    for gname in list(groups.keys()):
        examples = few_shot_examples(gname, groups, k=8)
        want = per_group
        try:
            if client is not None:
                raw = api_generate_topics(client, model, gname, want * 2, examples)
            else:
                raw = fallback_generate_topics(gname, want * 2, examples)
        except Exception as exc:  # pragma: no cover
            logging.warning("API error for group %s: %s; skipping group", gname, exc)
            raw = []

        raw = validate_topics(raw)
        filtered = deduplicate(raw, existing_all_norm, threshold=0.7)
        additions = filtered[:per_group]
        if not additions:
            logging.info("No additions for %s", gname)
            continue
        groups[gname].extend(additions)
        existing_all_norm.extend(normalize_topic(t) for t in additions)
        logging.info("Added %d topics to %s", len(additions), gname)

    # OpenAI client does not require explicit close
    return groups


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate additional topics via LLM and update topics.json"
    )
    p.add_argument("--input", required=True, help="Path to topics.json")
    p.add_argument("--output", help="Output path (defaults to input when --in-place)")
    p.add_argument(
        "--in-place", action="store_true", help="Write updates back to --input path"
    )
    p.add_argument(
        "--per-group", type=int, default=30, help="New topics to add per group"
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL") or "gpt-4.1",
        help="Model for /v1/chat/completions",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Load .env (dotenv preferred)
    if load_dotenv is not None:
        load_dotenv(override=True)

    if not args.in_place and not args.output:
        logging.error("Specify --output or use --in-place")
        return 2
    out_path = args.input if args.in_place else args.output

    groups = load_topics_json(args.input)
    api_key = os.environ.get("OPENAI_API_KEY")
    updated = expand_groups(
        groups,
        per_group=int(args.per_group),
        model=args.model,
        api_key=api_key,
    )
    write_topics_json(out_path, updated)
    logging.info("Wrote updated topics to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
