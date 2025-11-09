# Doric Dataset Generator

This repo builds a synthetic chat dataset for fine‑tuning models to answer naturally in Doric Scots. Each example is a simple two‑turn chat (`user` → `assistant`) with metadata. The generator can use an OpenAI‑compatible API for high‑quality responses or a local heuristic fallback.

## Goals

- Produce fluent, varied Doric replies without templated openings
- Keep the training data system‑free (only `user`/`assistant` turns)
- Cover core prompts, adversarial “answer in English only” cases, multilingual user prompts, and safety/refusal examples — all answered in Doric
- Control dialect density (1–3 idiomatic markers) so outputs don’t feel cartoonish

## What’s Generated

Each JSONL row:

```
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {"topic": "...", "kind": "core|adv|safety|multi", "id": "...", "lang": "..."}
}
```

Kinds:
- `core`: regular queries in English/Doric
- `adv`: adversarial cues (e.g., “Answer in English only”) — assistant still replies in Doric
- `safety`: harmful/illegal requests — assistant refuses in Doric, offers safer guidance
- `multi`: multilingual user prompts (sv, de, fr, zh, ar, etc.) — assistant replies in Doric

Quality gates include:
- Dialect density target (1–3 idioms) with soft adjustment
- Reject stock intros (banned prefixes)
- Low overlap with user prompt and topic title
- Word count 12–220
- Deduplication on assistant text
- Unicode cleanup (NFKC + control char removal)

## Requirements

- Python 3.12+
- Optional packages (auto‑detected): `orjson`, `httpx`, `rapidfuzz`, `tqdm`
- For API generation: set `OPENAI_API_KEY`; optionally set `MODEL` and `OPENAI_BASE_URL`

## Quick Start

1) Prepare topics

- Preferred: grouped JSON `topics.json` (see `topics.json` in repo) with either:
  - an object `{ "groups": { "group": ["topic", ...], ... } }`, or
  - an object mapping `{ "group": ["topic", ...] }`, or
  - an array of `{ "topic": "...", "group": "..." }` objects, or
  - a plain array of strings `["topic", ...]`.

2) Generate dataset:

```
uv run python generate_doric_dataset.py --topics-json topics.json --out doric_synth.jsonl --max-concurrency 50
```

Useful flags:

- `--n-per-topic 6` — samples per topic
- `--adv-ratio 0.1 --safety-ratio 0.05 --multi-ratio 0.2` — bucket shares
- `--max-concurrency 8` — parallelism
- `--log-level DEBUG` — more detail

If `OPENAI_API_KEY` is set, the generator uses the OpenAI‑compatible backend. Otherwise, it uses a heuristic fallback (good for smoke tests, but lower quality). For backwards‑compat, you can still pass `--topics topics.txt` (one topic per line), but `--topics-json` is recommended.

Env loading:
- Both the dataset generator and topic expander auto‑load `.env` from the repo root and set variables (e.g., `OPENAI_API_KEY`, `MODEL`, `OPENAI_BASE_URL`). This happens on every run and overrides existing values.

## Validation Checklist (post‑gen)

- A few samples do not share the same opening phrasing
- Assistant replies are fluent Doric and do not parrot the prompt title
- Markers density varies (some light, some richer), roughly 1–3 idioms
- Safety prompts produce short refusals in Doric with safer alternatives
- Multilingual prompts still get Doric replies

## Fine‑Tuning Tips

- Supervised fine‑tuning (SFT), responses‑only masking
- 2–3 epochs on this dataset size works well as a first pass
- No system prompts in training examples

## Troubleshooting

- “Required arguments: --topics, --out” → pass both flags as shown above
- Too many similar openings → increase topics diversity and rerun (seed controls shuffling)
- Over‑dialected outputs → decrease the marker target or re‑run (the generator samples 1–3 markers per row)

## License

No license included by default. Add one if you plan to publish the dataset.
