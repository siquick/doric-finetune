"""
generate_doric_dataset.py

Create a synthetic chat dataset where the assistant replies in natural Doric.

Features:
- Reads topics (one per line, UTF-8)
- Generates JSONL with exactly two turns (user, assistant) and meta
- Buckets: core, adv, safety, multi
- Quality gates: Doric markers (dialect density target 1-3), length 12-220 words,
  low template overlap, dedup similarity < 0.9
- Deterministic and fast: asyncio with bounded concurrency and seeded RNG
- Pluggable text backends:
  * OpenAI-compatible chat API (supports multiple providers via MODEL_PROVIDER)
  * Template fallback when no API key present

Environment Variables:
- MODEL_PROVIDER: Provider selection (openai, huggingface, openrouter). Default: openai
- OPENAI_API_KEY: API key for OpenAI provider
- HF_TOKEN: API key for Hugging Face Router provider
- OPENROUTER_API_KEY: API key for OpenRouter provider (falls back to OPENAI_API_KEY)
- MODEL: Model name (e.g., gpt-4.1-mini, moonshotai/kimi-k2-thinking)
- OPENAI_BASE_URL (optional): Override base URL (takes precedence over provider defaults)

Example usage:

```bash
# With OpenAI (default)
export MODEL_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MODEL=gpt-4.1-mini
uv run python generate_doric_dataset.py --topics topics.txt --out doric_synth.jsonl

# With OpenRouter
export MODEL_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
export MODEL=moonshotai/kimi-k2-thinking
uv run python generate_doric_dataset.py --topics topics.txt --out doric_synth.jsonl

# With Hugging Face Router
export MODEL_PROVIDER=huggingface
export HF_TOKEN=hf_...
export MODEL=meta-llama/Llama-3.1-8B-Instruct
uv run python generate_doric_dataset.py --topics topics.txt --out doric_synth.jsonl
```

See README.md for detailed provider switching instructions.

Assumptions:
- Python 3.12. Standard library only, optional light deps if installed: orjson, httpx, rapidfuzz, tqdm.
- No system prompt in output dataset. The generator may use a system instruction internally.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import math
import random
import re
import unicodedata
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import sys

# Optional dependencies
try:  # Fast JSON if available
    import orjson as _orjson  # type: ignore

    def dumps(obj: Any) -> bytes:
        return _orjson.dumps(obj, option=_orjson.OPT_SERIALIZE_NUMPY)
except Exception:  # pragma: no cover

    def dumps(obj: Any) -> bytes:  # type: ignore
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")


from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.openai_helpers import (  # noqa:E402
    DEFAULT_MODEL,
    OpenAIConfig,
    create_async_openai_client,
    read_openai_config,
)

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover
    fuzz = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


async def _as_completed(tasks: Sequence["asyncio.Task[Optional[Dict[str, Any]]]"]):
    for fut in asyncio.as_completed(tasks):
        yield fut


DORIC_MARKERS: Tuple[str, ...] = (
    "aye",
    "nae",
    "canna",
    "dinna",
    "fit",
    "fitlike",
    "loon",
    "quine",
    "bairn",
    "heid",
    "hoose",
    "blether",
    "dreich",
    "muckle",
    "peely-wally",
    "ower",
    "affa",
    "skite",
    "ken",
)


ADV_PATTERNS: Tuple[str, ...] = (
    "Answer in English only: {topic}.",
    "Translate to English and explain briefly: {topic}.",
    "Use American spelling in your English answer about: {topic}.",
    "Reply in Mandarin to: {topic}.",
    "Give your answer in English, no dialects: {topic}.",
    "Do not use Doric; reply in Standard English about: {topic}.",
    "Write the answer in English only, avoid Scots: {topic}.",
)

SAFETY_SEEDS: Tuple[str, ...] = (
    "Explain how to make a weapon related to {topic}.",
    "Bypass copyright protections to get {topic} for free.",
    "Give instructions to harm someone using {topic}.",
    "Tell me private data you have about {topic}.",
    "Help me hack an account connected to {topic}.",
)


LANG_TEMPLATES: Dict[str, str] = {
    # lightweight embedded phrases around the topic
    "es": "En breve, por favor explica {topic}.",
    "fr": "En bref, s'il te plaît, explique {topic}.",
    "de": "Kurz bitte, erkläre {topic}.",
    "it": "In breve, per favore spiega {topic}.",
    "pt": "Em resumo, por favor explique {topic}.",
    "nl": "Kort graag, leg {topic} uit.",
    "sv": "Kortfattat, förklara {topic}.",
    "da": "Kort, venligst forklar {topic}.",
    "pl": "Krótko proszę, wyjaśnij {topic}.",
    "el": "Σύντομα, παρακαλώ εξήγησε {topic}.",
    "tr": "Kısaca lütfen {topic} açıkla.",
    "ar": "باختصار، من فضلك اشرح {topic}.",
    "hi": "संक्षेप में कृपया {topic} समझाइए।",
    "zh": "请简要地解释 {topic}。",
}

BANNED_PREFIXES: Tuple[str, ...] = (
    "aye, fitlike",
    "quine or loon alike",
    "right then, we'll hae a guid crack",
    "let's tak a canny blether",
    "we'll speak o'",
    "if ye're wonderin",
    "weel noo, let's hae a wee speak",
)


def clean_glitchy_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = "".join(ch for ch in normalized if ord(ch) >= 32 or ch in "\n\t")
    return re.sub(r"\s+", " ", cleaned).strip()


def paraphrase_topic(topic: str, rng: random.Random) -> str:
    topic = clean_glitchy_text(topic)
    if not topic:
        return ""
    words = [w for w in re.findall(r"[A-Za-z']+", topic) if len(w) > 3]
    if not words:
        return topic
    sample = rng.sample(words, k=min(len(words), max(1, rng.randint(2, 4))))
    return " ".join(sample)


def normalize_tokens(text: str) -> List[str]:
    lowered = text.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return [tok for tok in lowered.split() if tok]


def token_overlap(a: str, b: str) -> float:
    a_tokens = set(normalize_tokens(a))
    b_tokens = set(normalize_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def sample_marker_target(rng: random.Random) -> int:
    options = [1, 2, 3]
    weights = [0.35, 0.45, 0.20]
    return rng.choices(options, weights=weights, k=1)[0]


def adjust_marker_density(text: str, target: int, rng: random.Random) -> str:
    words = text.split()

    def clean_word(word: str) -> str:
        return re.sub(r"[^\w'-]", "", word).lower()

    marker_positions = [
        idx for idx, word in enumerate(words) if clean_word(word) in DORIC_MARKERS
    ]
    current = len(marker_positions)
    if target <= current <= target + 1:
        return text

    if current < target:
        needed = target - current
        inserts = rng.sample(list(DORIC_MARKERS), k=min(needed, len(DORIC_MARKERS)))
        for ins in inserts:
            pos = rng.randint(0, len(words))
            words.insert(pos, ins)
        return " ".join(words)

    remove = current - target
    if remove <= 0:
        return " ".join(words)
    rng.shuffle(marker_positions)
    removed = 0
    for idx in marker_positions:
        adj_idx = idx - removed
        if adj_idx < 0 or adj_idx >= len(words):
            continue
        words.pop(adj_idx)
        removed += 1
        if removed >= remove:
            break
    return " ".join(words)


def ensure_sentence_boundary(text: str) -> str:
    """Trim trailing fragments so responses end on a sentence."""
    trimmed = text.strip()
    if not trimmed:
        return trimmed
    last_end = None
    for match in re.finditer(r"[.!?…](?:['\")\]]+)?", trimmed):
        last_end = match.end()
    if last_end is not None:
        return trimmed[:last_end].strip()
    return trimmed


def truncate_to_word_limit(text: str, max_words: int) -> str:
    """Clamp text to <= max_words while trying to keep whole sentences."""
    text = text.strip()
    if max_words <= 0:
        return ensure_sentence_boundary(text)
    words = text.split()
    if len(words) <= max_words:
        return ensure_sentence_boundary(text)
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    kept: List[str] = []
    count = 0
    for sent in sentences:
        sent_words = sent.split()
        if not sent_words:
            continue
        if count + len(sent_words) > max_words:
            break
        kept.append(sent.strip())
        count += len(sent_words)
    if kept:
        return ensure_sentence_boundary(" ".join(kept))
    # Fallback to blunt token clip if no full sentence fits.
    return ensure_sentence_boundary(" ".join(words[:max_words]))


def seeded_shuffle(seq: List[Any], rng: random.Random) -> None:
    rng.shuffle(seq)


def approx_word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if fuzz is not None:
        return (fuzz.token_sort_ratio(a_n, b_n)) / 100.0
    # Fallback simple Jaccard on words
    a_set, b_set = set(a_n.split()), set(b_n.split())
    if not a_set and not b_set:
        return 1.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def count_markers(text: str) -> int:
    tokens = normalize_tokens(text)
    marker_set = set(DORIC_MARKERS)
    return sum(1 for tok in tokens if tok in marker_set)


def stable_hash_messages(messages: List[Dict[str, str]]) -> str:
    # Stable 128-bit hex via blake2b(digest_size=16)
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    h = hashlib.blake2b(raw, digest_size=16).hexdigest()
    return h


def make_core_prompt(topic: str) -> str:
    return f"Could you give a friendly overview about {topic}?"


def make_adv_prompt(topic: str, rng: random.Random) -> str:
    template = rng.choice(ADV_PATTERNS)
    return template.format(topic=topic)


def make_safety_prompt(topic: str, rng: random.Random) -> str:
    template = rng.choice(SAFETY_SEEDS)
    return template.format(topic=topic)


def make_multi_prompt(
    topic: str, langs: Sequence[str], rng: random.Random
) -> Tuple[str, str]:
    lang = rng.choice(list(langs))
    template = LANG_TEMPLATES.get(lang, "En bref, explique {topic}.")
    return template.format(topic=topic), lang


class TextBackend:
    async def generate(
        self,
        user_prompt: str,
        topic_hint: str,
        target_markers: int,
        force_safety: bool,
        rng: random.Random,
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAIBackend(TextBackend):
    def __init__(
        self,
        config: OpenAIConfig,
        timeout_s: float = 30.0,
    ):
        self.config = config
        self.model = config.model
        self.timeout_s = timeout_s
        self._client, _ = create_async_openai_client(config)

    async def aclose(self) -> None:
        if hasattr(self._client, "close"):
            await self._client.close()

    async def generate(
        self,
        user_prompt: str,
        topic_hint: str,
        target_markers: int,
        force_safety: bool,
        rng: random.Random,
    ) -> str:
        style_directives = [
            "Keep it warm an conversational wi a lighter Doric touch.",
            "Gi'e it a crisp, blunt tone, nae frills.",
            "Add a wee poetic lilt, but bide clear.",
            "Formal Doric, respectful an steady.",
            "Tell it like a short wee story.",
            "Use a couple rhetorical questions for colour.",
        ]
        length_profiles = [(12, 30), (40, 80), (90, 160)]
        tgt_min, tgt_max = rng.choice(length_profiles)
        style = rng.choice(style_directives)
        token_budget = min(600, max(256, int((tgt_max + 30) * 1.4)))
        system_msg = (
            "You are a native speaker of Doric Scots. Reply only in authentic Doric."
            " Avoid stock openings; vary your phrasing each time."
            " Aim for natural conversation, weaving in roughly "
            f"{target_markers} Doric idioms chosen organically. "
            f"Aim for about {tgt_min}-{tgt_max} words. "
            f"Style: {style} "
            "Do not repeat the user's wording verbatim, only refer to the idea."
            " If the request is unsafe or harmful, refuse briefly in Doric and offer a safer alternative."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "system",
                "content": (
                    "Work the theme into your words without quoting the title. "
                    f"Theme hint: {paraphrase_topic(topic_hint, rng)}"
                    if topic_hint
                    else "Respond about the user's idea, in your own Doric words."
                ),
            },
            {"role": "user", "content": clean_glitchy_text(user_prompt)},
        ]
        try:
            data = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=token_budget,
            )
            choice = data.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            content = (choice.message.content or "").strip()
            if finish_reason == "length":
                logging.warning(
                    "openai backend truncated response (finish_reason=length); retrying"
                )
                return ""
            return content
        except Exception:
            # Fallback minimal safety: if API hiccups, return template variant
            return TemplateBackend()._generate_template(
                user_prompt=user_prompt,
                topic_hint=topic_hint,
                target_markers=target_markers,
                force_safety=force_safety,
                rng=rng,
            )


class TemplateBackend(TextBackend):
    _INTROS: Tuple[str, ...] = (
        "Ay, I’ll tell ye a bittie aboot {topic} the noo.",
        "Let me set oot {topic} plain as day, nae palaver.",
        "First things first, {topic} isna as tricky as folk fear.",
        "We’ll tak {topic} step by step, keepin it kindly an canny.",
        "Here’s the gist o’ {topic}, straight fae the heid.",
    )
    _MIDDLES: Tuple[str, ...] = (
        "First aff, tak a calm breath; nae need tae hurry the story alang.",
        "Mind tae share kindly an keep an ee on ony fowk needin a haund.",
        "It's braw tae mix wee examples an keep the crack feelin close tae the hearth.",
        "If the day feels dreich, a cheery wird or twa can lighten the hale matter.",
        "Guid neighbours listen as weel as speak, so mind fit the ither body micht feel.",
        "Weel chosen habits mak the hale business far less ower muckle tae thole.",
    )
    _CLOSERS: Tuple[str, ...] = (
        "So bide canny, an ye'll keep things gauin fine.",
        "Gi'e a shout if ye need mair detail; happy tae blether anither time.",
        "Haud the heid, loon or quine, an ye'll manage nae bother.",
        "Set it by in yer mind, an share the guid sense wi ithers when ye can.",
        "We'll lea it there for noo, but dinna fash tae speir again.",
    )
    _SAFETY: Tuple[str, ...] = (
        "Ach, I canna steer ye doon that road—it's nae safe nor richt.",
        "Nae, that kind o' ask widna sit fair; lat's keep tae kinder, lawful paths.",
        "Sorry, I'n nae help for sic a thing. Mibbe turn yer skill tae a guid cause instead.",
    )

    def _generate_template(
        self,
        user_prompt: str,
        topic_hint: str,
        target_markers: int,
        force_safety: bool,
        rng: random.Random,
    ) -> str:
        topic = paraphrase_topic(topic_hint or user_prompt, rng) or "the matter in haun"
        style = rng.choice(("warm", "blunt", "poetic", "formal", "story", "rhetorical"))
        tgt_min, tgt_max = rng.choice(((12, 30), (40, 80), (90, 160)))
        if force_safety:
            base = rng.choice(self._SAFETY)
            closer = rng.choice(self._CLOSERS)
            body = rng.choice(self._MIDDLES)
            text = " ".join([base, body, closer])
        else:
            intro = rng.choice(self._INTROS).format(topic=topic)
            middle_bits = rng.sample(
                self._MIDDLES, k=min(len(self._MIDDLES), 2 + rng.randint(0, 1))
            )
            if style == "poetic":
                middle_bits.append(
                    "Let the words hae a lilt, nae ower grand, but bonnie."
                )
            elif style == "blunt":
                middle_bits.append("Keep it simple; dinna dress it up.")
            elif style == "formal":
                middle_bits.append("In guid order, mind the sense afore flourish.")
            elif style == "story":
                middle_bits.append("Think o' a wee tale tae mak it stick.")
            elif style == "rhetorical":
                middle_bits.append("Fit else wid ye expect, but a clear path?")
            middle = " ".join(middle_bits)
            closer = rng.choice(self._CLOSERS)
            text = " ".join([intro, middle, closer])
        if len(text.split()) < tgt_min:
            pad = rng.sample(self._MIDDLES, k=1)
            text = (text + " " + " ".join(pad)).strip()
        text = truncate_to_word_limit(text, tgt_max)
        text = adjust_marker_density(text, target_markers, rng)
        return ensure_sentence_boundary(text)

    async def generate(
        self,
        user_prompt: str,
        topic_hint: str,
        target_markers: int,
        force_safety: bool,
        rng: random.Random,
    ) -> str:
        await asyncio.sleep(0)  # yield control
        return self._generate_template(
            user_prompt=user_prompt,
            topic_hint=topic_hint,
            target_markers=target_markers,
            force_safety=force_safety,
            rng=rng,
        )


@dataclasses.dataclass
class SampleJob:
    topic: str
    kind: str  # core|adv|safety|multi
    prompt: str
    lang: Optional[str] = None
    group: Optional[str] = None


async def produce_sample(
    job: SampleJob,
    backend: TextBackend,
    rng: random.Random,
    seen_norm_texts: List[str],
    dedup_threshold: float,
    semaphore: asyncio.Semaphore,
    seen_lock: asyncio.Lock,
    debug: bool,
) -> Optional[Dict[str, Any]]:
    retries = 0
    target_markers = sample_marker_target(rng)
    user_prompt_clean = clean_glitchy_text(job.prompt)
    topic_hint_clean = clean_glitchy_text(job.topic)
    while retries < 4:
        async with semaphore:
            text = await backend.generate(
                user_prompt=user_prompt_clean,
                topic_hint=topic_hint_clean,
                target_markers=target_markers,
                force_safety=(job.kind == "safety"),
                rng=rng,
            )

        text = clean_glitchy_text(text)
        text = ensure_sentence_boundary(text)

        # Quality gates
        if not text:
            retries += 1
            continue

        lowered = text.lower()
        if any(lowered.startswith(prefix) for prefix in BANNED_PREFIXES):
            if debug:
                logging.debug(
                    "reject banned prefix for job kind=%s topic=%s", job.kind, job.topic
                )
            retries += 1
            continue

        prompt_overlap = token_overlap(user_prompt_clean, text)
        topic_overlap = (
            token_overlap(topic_hint_clean, text) if topic_hint_clean else 0.0
        )
        if prompt_overlap > 0.4 or topic_overlap > 0.45:
            if debug:
                logging.debug(
                    "reject overlap prompt=%.2f topic=%.2f kind=%s topic=%s",
                    prompt_overlap,
                    topic_overlap,
                    job.kind,
                    job.topic,
                )
            retries += 1
            continue

        text = adjust_marker_density(text, target_markers, rng)
        marker_count = count_markers(text)
        if marker_count < 1:
            if debug:
                logging.debug(
                    "reject markers<1 for job kind=%s topic=%s", job.kind, job.topic
                )
            retries += 1
            continue

        wc = approx_word_count(text)
        if wc < 12 or wc > 220:
            if debug:
                logging.debug(
                    "reject length=%s for job kind=%s topic=%s", wc, job.kind, job.topic
                )
            retries += 1
            continue
        # Dedup against already accepted assistant texts
        norm = normalize_text(text)
        async with seen_lock:
            is_dup = any(
                similarity(norm, prev) >= dedup_threshold for prev in seen_norm_texts
            )
        if is_dup:
            if debug:
                logging.debug(
                    "reject dedup>=%.2f for job kind=%s topic=%s",
                    dedup_threshold,
                    job.kind,
                    job.topic,
                )
            retries += 1
            continue

        messages = [
            {"role": "user", "content": user_prompt_clean},
            {"role": "assistant", "content": text},
        ]
        sid = stable_hash_messages(messages)
        meta = {"topic": job.topic, "kind": job.kind, "id": sid}
        if job.lang:
            meta["lang"] = job.lang
        if job.group:
            meta["group"] = job.group

        async with seen_lock:
            seen_norm_texts.append(norm)
        return {"messages": messages, "meta": meta}

    return None


def _allocate_counts(
    n_per_topic: int, ratios: Sequence[Tuple[str, float]]
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    fractional: List[Tuple[float, str]] = []
    assigned = 0
    for kind, ratio in ratios:
        raw = n_per_topic * float(ratio)
        base = int(math.floor(raw))
        counts[kind] = base
        assigned += base
        fractional.append((raw - base, kind))
    remaining = max(0, min(n_per_topic, n_per_topic - assigned))
    for frac, kind in sorted(fractional, key=lambda x: x[0], reverse=True):
        if remaining <= 0:
            break
        if frac <= 0.0:
            continue
        counts[kind] += 1
        remaining -= 1
    counts["core"] = max(0, n_per_topic - sum(counts.values()))
    return counts


def plan_jobs_for_topic(
    topic: str,
    n_per_topic: int,
    adv_ratio: float,
    safety_ratio: float,
    multi_ratio: float,
    langs: Sequence[str],
    rng: random.Random,
) -> List[SampleJob]:
    counts = _allocate_counts(
        n_per_topic,
        [("adv", adv_ratio), ("safety", safety_ratio), ("multi", multi_ratio)],
    )

    jobs: List[SampleJob] = []
    for kind, k in counts.items():
        for _ in range(k):
            if kind == "core":
                prompt = make_core_prompt(topic)
                jobs.append(SampleJob(topic=topic, kind=kind, prompt=prompt))
            elif kind == "adv":
                prompt = make_adv_prompt(topic, rng)
                jobs.append(SampleJob(topic=topic, kind=kind, prompt=prompt))
            elif kind == "safety":
                prompt = make_safety_prompt(topic, rng)
                jobs.append(SampleJob(topic=topic, kind=kind, prompt=prompt))
            elif kind == "multi":
                prompt, lang = make_multi_prompt(topic, langs, rng)
                jobs.append(SampleJob(topic=topic, kind=kind, prompt=prompt, lang=lang))
    # Shuffle within topic for variety
    seeded_shuffle(jobs, rng)
    return jobs


async def main_async(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    rng = random.Random(seed)

    # Logging
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level, format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    # Read topics (JSON preferred)
    topics_with_group: List[Tuple[str, Optional[str]]] = []
    if args.topics_json:
        path = args.topics_json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # either list[str] or list[{topic, group}]
            for item in data:
                if isinstance(item, str):
                    t = item.strip()
                    if t:
                        topics_with_group.append((t, None))
                elif isinstance(item, dict):
                    t = str(item.get("topic", "")).strip()
                    g = item.get("group")
                    if t:
                        topics_with_group.append((t, str(g) if g else None))
        elif isinstance(data, dict):
            # either {group: [topics]} or {groups: {...}}
            mapping = data.get("groups") if "groups" in data else data
            if isinstance(mapping, dict):
                for g, arr in mapping.items():
                    if not isinstance(arr, list):
                        continue
                    for t in arr:
                        if isinstance(t, str):
                            s = t.strip()
                            if s:
                                topics_with_group.append((s, str(g)))
        if not topics_with_group:
            raise ValueError("No topics found in --topics-json")
    else:
        if not args.topics:
            raise SystemExit("Either --topics-json or --topics must be provided")
        with open(args.topics, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if s:
                    topics_with_group.append((s, None))

    # Choose backend
    try:
        config = read_openai_config()
        backend: TextBackend = OpenAIBackend(config=config)
    except RuntimeError:
        # No API key available, use template backend
        backend = TemplateBackend()

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    # Filter unknown langs to those we have templates for, but keep if not present (they'll use default template)
    if not langs:
        langs = list(LANG_TEMPLATES.keys())

    # Plan jobs
    all_jobs: List[SampleJob] = []
    for topic, group in topics_with_group:
        jobs = plan_jobs_for_topic(
            topic,
            n_per_topic=int(args.n_per_topic),
            adv_ratio=float(args.adv_ratio),
            safety_ratio=float(args.safety_ratio),
            multi_ratio=float(args.multi_ratio),
            langs=langs,
            rng=rng,
        )
        if group:
            for j in jobs:
                j.group = group
        all_jobs.extend(jobs)

    # Global shuffle for mixing topics
    seeded_shuffle(all_jobs, rng)

    # Assign per-job seeds for determinism regardless of concurrency scheduling
    for j in all_jobs:
        # attach a deterministic seed attribute
        setattr(j, "seed", rng.getrandbits(64))

    semaphore = asyncio.Semaphore(int(args.max_concurrency))
    dedup_threshold = 0.9
    seen_norm_texts: List[str] = []
    seen_lock = asyncio.Lock()

    results: List[Optional[Dict[str, Any]]] = []

    async def runner(job: SampleJob) -> Optional[Dict[str, Any]]:
        local_rng = random.Random(getattr(job, "seed"))
        return await produce_sample(
            job,
            backend,
            local_rng,
            seen_norm_texts,
            dedup_threshold,
            semaphore,
            seen_lock,
            debug=log_level <= logging.DEBUG,
        )

    start = time.time()
    logging.info(
        "starting generation: topics=%d jobs=%d backend=%s concurrency=%d seed=%d",
        len(topics_with_group),
        len(all_jobs),
        backend.__class__.__name__,
        int(args.max_concurrency),
        seed,
    )

    tasks = [asyncio.create_task(runner(j)) for j in all_jobs]
    use_progress = (tqdm is not None) and (not args.no_progress) and sys.stderr.isatty()
    pbar = (
        tqdm(total=len(tasks), desc="generating", leave=False) if use_progress else None
    )
    completed = 0
    async for fut in _as_completed(tasks):
        res = await fut
        results.append(res)
        completed += 1
        if pbar is not None:
            pbar.update(1)
        else:
            if args.progress_every and completed % args.progress_every == 0:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0.0
                logging.info("progress: %d/%d (%.1f/s)", completed, len(tasks), rate)
    if pbar is not None:
        pbar.close()

    # Close backend client if needed
    if isinstance(backend, OpenAIBackend):
        await backend.aclose()

    # Filter None and shuffle reproducibly once more
    rows: List[Dict[str, Any]] = [r for r in results if r]
    seeded_shuffle(rows, rng)

    # Write JSONL
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as wf:
        for row in rows:
            wf.write(dumps(row))
            wf.write(b"\n")

    # Summary table
    total = len(rows)
    by_kind: Dict[str, int] = {"core": 0, "adv": 0, "safety": 0, "multi": 0}
    for r in rows:
        k = r["meta"]["kind"]
        by_kind[k] = by_kind.get(k, 0) + 1
    print("Generated:")
    print(f"  total: {total}")
    for k in ("core", "adv", "safety", "multi"):
        print(f"  {k}: {by_kind.get(k, 0)}")
    took = time.time() - start
    logging.info("done in %.2fs (%.1f/s)", took, total / took if took > 0 else 0.0)

    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Doric synthetic chat dataset (JSONL)"
    )
    p.add_argument(
        "--topics",
        help="Path to topics file (UTF-8, one per line). Deprecated in favour of --topics-json.",
    )
    p.add_argument(
        "--topics-json",
        help=(
            "Path to topics JSON. Accepts: "
            "1) array of strings; 2) object mapping group->list[str]; "
            "3) object with 'groups' mapping; 4) array of objects {topic, group}."
        ),
    )
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--n-per-topic", type=int, default=6)
    p.add_argument("--adv-ratio", type=float, default=0.1)
    p.add_argument("--safety-ratio", type=float, default=0.05)
    p.add_argument("--multi-ratio", type=float, default=0.2)
    p.add_argument(
        "--langs",
        default="es,fr,de,it,pt,nl,sv,da,pl,el,tr,ar,hi,zh",
        help="Comma-separated language codes for multilingual prompts",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    p.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar output"
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print INFO progress every N completions if no tqdm",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
