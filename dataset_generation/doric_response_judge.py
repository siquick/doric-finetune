"""
doric_response_judge.py

Review an existing Doric chat dataset using batched/parallel LLM judging.
Each API call handles 10 conversations, and up to 50 calls run concurrently.
Every processed record is flushed incrementally into a new JSONL file.

Supports multiple OpenAI-compatible providers via MODEL_PROVIDER env var:
- openai (default): Uses OPENAI_API_KEY
- huggingface: Uses HF_TOKEN, base_url=https://router.huggingface.co/v1
- openrouter: Uses OPENROUTER_API_KEY (or OPENAI_API_KEY), base_url=https://openrouter.ai/api/v1

See README.md for provider switching details.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # Optional faster JSON writer
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

from instructor import AsyncInstructor, Mode, from_openai  # type: ignore
from pydantic import BaseModel, Field

SYSTEM_PROMPT = (
    "You are a native Doric editor who ensures responses sound authentically Doric "
    "while staying faithful to the source meaning and tone."
)

JUDGE_INSTRUCTIONS = (
    "You will review multiple assistant replies. For EACH entry:\n"
    "1. Decide if the reply is true Doric (dialectal vocabulary, grammar, idioms).\n"
    "2. If any part lapses into another dialect, rewrite the ENTIRE reply into natural Doric.\n"
    "3. Always produce JSON ONLY, no prose.\n"
    "4. CRITICAL: Copy entry_id EXACTLY as provided - do not modify, truncate, or typo it.\n"
    "Output an array whose length equals the number of entries provided, same order. "
    "Each object must include:\n"
    "  entry_id (string) - copy EXACTLY as shown, character-for-character.\n"
    "  is_true_doric (bool).\n"
    "  notes (string, <=40 words) - short justification.\n"
    "  corrected_response (string) - rewritten Doric reply; if already Doric, copy the original verbatim."
)


@dataclass
class EntryPayload:
    seq: int
    line_number: int
    entry_id: str
    entry: Dict[str, Any]
    assistant_index: int
    assistant_text: str


class PartialResponseError(Exception):
    """Raised when a batch response is incomplete but contains some valid results."""

    def __init__(
        self,
        message: str,
        results: Dict[str, Dict[str, Any]],
        missing_payloads: List[EntryPayload],
    ):
        super().__init__(message)
        self.results = results
        self.missing_payloads = missing_payloads


class DoricJudgeResult(BaseModel):
    entry_id: str = Field(..., description="Identifier for the dataset entry.")
    is_true_doric: bool = Field(
        ..., description="True only if the assistant reply already reads as Doric."
    )
    notes: str = Field(
        ...,
        description="Short justification (<=40 words) describing the judgement.",
    )
    corrected_response: str = Field(
        ..., description="Doric rewrite of the assistant reply."
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether assistant responses are true Doric and fix them when not."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to source JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the reviewed JSONL dataset.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Override OpenAI model name (default env MODEL or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Process at most this many records (useful for smoke tests).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between API calls to respect rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per record when the LLM response is malformed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing to an existing output file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of conversations to send per LLM request (default: 10).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=50,
        help="Maximum concurrent API calls (default: 50).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def load_dataset(path: str) -> Iterable[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield idx, line


def format_conversation(conversations: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in conversations:
        role = msg.get("from") or msg.get("role") or "unknown"
        value = (msg.get("value") or msg.get("content") or "").strip()
        parts.append(f"{role.upper()}: {value}")
    return "\n".join(parts)


def build_batch_user_message(batch: Sequence[EntryPayload]) -> str:
    header = [
        f"You will review {len(batch)} assistant replies for Doric authenticity.",
        JUDGE_INSTRUCTIONS,
    ]
    body: List[str] = []
    for idx, payload in enumerate(batch, start=1):
        meta = payload.entry.get("meta") or {}
        transcript = format_conversation(payload.entry.get("conversations") or [])
        # Make entry_id very prominent and clear to reduce typos
        body.extend(
            [
                f"\n{'=' * 60}",
                f"Entry {idx} of {len(batch)}",
                f"entry_id: {payload.entry_id}",
                f"{'=' * 60}",
                f"Metadata: topic={meta.get('topic')}, kind={meta.get('kind')}, group={meta.get('group')}",
                "Conversation transcript (assistant reply is the final turn shown):",
                transcript,
                "Assistant reply under review:",
                f"<<<{payload.assistant_text}>>>",
            ]
        )
    return "\n".join(header + body)


async def call_llm_judge_batch(
    instructor_client: AsyncInstructor,
    model: str,
    batch: Sequence[EntryPayload],
    max_retries: int,
    post_delay: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    user_message = build_batch_user_message(batch)
    # Calculate max_tokens dynamically: ~500 tokens per entry (for corrected_response + metadata)
    # Add buffer for JSON structure overhead and safety margin
    # Some entries may have longer corrected responses, so use a higher multiplier
    max_tokens = max(3000, len(batch) * 500)
    try:
        response: List[
            DoricJudgeResult
        ] = await instructor_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            response_model=List[DoricJudgeResult],  # type: ignore[arg-type]
            max_retries=max_retries,
        )
    except Exception as exc:  # pragma: no cover - depends on API env
        logging.warning(
            "Failed to judge batch of %s entries: %s",
            len(batch),
            exc,
        )
        raise

    # Handle partial responses gracefully
    if len(response) != len(batch):
        response_ids = {item.entry_id for item in response}
        batch_ids = {p.entry_id for p in batch}
        missing_ids = batch_ids - response_ids

        logging.warning(
            "Partial response: expected %d items, got %d. Missing entry_ids: %s",
            len(batch),
            len(response),
            list(missing_ids),
        )

        # If we got some results, process them and raise a special exception
        # that includes the missing entries for retry
        if len(response) > 0:
            results: Dict[str, Dict[str, Any]] = {}
            for item in response:
                results[item.entry_id] = item.model_dump()

            # Create a special exception that includes partial results
            missing_payloads = [p for p in batch if p.entry_id in missing_ids]
            raise PartialResponseError(
                f"Got {len(response)}/{len(batch)} responses",
                results=results,
                missing_payloads=missing_payloads,
            )
        else:
            # No results at all - this is a real failure
            raise ValueError(
                f"Model response must match batch size (expected {len(batch)}, got {len(response)})"
            )

    # Match by position since we require same order - ignore entry_id typos from model
    if len(response) != len(batch):
        raise ValueError(
            f"Response length mismatch: expected {len(batch)}, got {len(response)}"
        )

    results: Dict[str, Dict[str, Any]] = {}
    for payload, item in zip(batch, response):
        # Use batch's entry_id (ignore what model returned to avoid typo issues)
        entry_id = payload.entry_id
        result = item.model_dump()
        result["entry_id"] = entry_id  # Ensure correct entry_id
        results[entry_id] = result

    if post_delay > 0:
        await asyncio.sleep(post_delay)
    return results


def locate_assistant_turn(entry: Dict[str, Any]) -> Tuple[int, str]:
    conversations = entry.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Entry missing 'conversations' list.")
    if not conversations:
        raise ValueError("Entry has empty 'conversations' list.")

    # Search from the end (most recent messages first)
    for idx in reversed(range(len(conversations))):
        msg = conversations[idx]
        if not isinstance(msg, dict):
            continue
        role = msg.get("from") or msg.get("role")
        if not isinstance(role, str):
            continue
        if role.lower() in {"assistant", "gpt"}:
            text = msg.get("value") or msg.get("content")
            if not isinstance(text, str) or not text.strip():
                continue
            return idx, text
    raise ValueError("No assistant/gpt turn found in entry.")


def ensure_output_path(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"Output file '{path}' already exists. Pass --overwrite to replace it."
        )


async def process_dataset_async(
    args: argparse.Namespace, config: OpenAIConfig, out_f
) -> Tuple[int, int]:
    client, resolved = create_async_openai_client(config)
    instructor_client = from_openai(client, mode=Mode.JSON)
    model_name = resolved.model or DEFAULT_MODEL
    batch_size = max(1, args.batch_size)
    parallel_limit = max(1, args.parallel)
    tasks: Set[asyncio.Task] = set()
    current_batch: List[EntryPayload] = []
    pending_outputs: Dict[int, bytes] = {}
    next_seq_to_write = 0
    total_processed = 0
    corrected = 0
    enqueued = 0
    seq_counter = 0

    def flush_ready() -> None:
        nonlocal next_seq_to_write
        while next_seq_to_write in pending_outputs:
            out_f.write(pending_outputs.pop(next_seq_to_write))
            out_f.flush()
            next_seq_to_write += 1

    async def handle_done(done_tasks: Sequence[asyncio.Task]) -> None:
        nonlocal total_processed, corrected
        for task in done_tasks:
            try:
                batch, results = task.result()
            except Exception as exc:
                batch_size = len(getattr(task, "batch", []))
                logging.error(
                    "Task failed for batch of %d entries: %s",
                    batch_size,
                    exc,
                )
                # For failed batches, mark all entries as failed but keep original text
                # This allows processing to continue instead of stopping completely
                failed_batch = getattr(task, "batch", [])
                for payload in failed_batch:
                    # Keep original text, no metadata added
                    pending_outputs[payload.seq] = dumps(payload.entry) + b"\n"
                    total_processed += 1
                    logging.warning(
                        "Marked entry %s as failed due to batch error, keeping original text",
                        payload.entry_id,
                    )
                flush_ready()
                continue  # Continue processing other tasks

            # Process successful results
            for payload in batch:
                judge = results.get(payload.entry_id)
                if not judge:
                    logging.error(
                        "Judge output missing entry_id=%s in batch results. Available: %s",
                        payload.entry_id,
                        list(results.keys()),
                    )
                    # Create fallback result
                    judge = {
                        "entry_id": payload.entry_id,
                        "is_true_doric": True,
                        "notes": "Judge output missing from results",
                        "corrected_response": payload.assistant_text,
                    }

                if not judge["is_true_doric"]:
                    payload.entry["conversations"][payload.assistant_index]["value"] = (
                        judge["corrected_response"]
                    )
                    corrected += 1
                # No metadata added - just write the entry
                pending_outputs[payload.seq] = dumps(payload.entry) + b"\n"
                total_processed += 1
                if total_processed % 10 == 0:
                    logging.info(
                        "Processed %s records (%s corrected)...",
                        total_processed,
                        corrected,
                    )
            flush_ready()

    async def wait_for_completion(first_only: bool) -> None:
        nonlocal tasks
        if not tasks:
            return
        return_when = asyncio.FIRST_COMPLETED if first_only else asyncio.ALL_COMPLETED
        done, pending = await asyncio.wait(tasks, return_when=return_when)
        tasks = pending
        await handle_done(done)

    async def submit_batch(batch: List[EntryPayload]) -> None:
        async def runner(entries: List[EntryPayload]):
            try:
                results = await call_llm_judge_batch(
                    instructor_client=instructor_client,
                    model=model_name,
                    batch=entries,
                    max_retries=args.max_retries,
                    post_delay=args.sleep,
                )
                return entries, results
            except PartialResponseError as exc:
                # Handle partial responses: process what we got, retry missing entries
                logging.info(
                    "Partial response received: %d/%d entries processed. Retrying %d missing entries individually.",
                    len(exc.results),
                    len(entries),
                    len(exc.missing_payloads),
                )
                # Process the partial results
                processed_entries = [e for e in entries if e.entry_id in exc.results]
                # Retry missing entries individually
                for missing in exc.missing_payloads:
                    try:
                        single_results = await call_llm_judge_batch(
                            instructor_client=instructor_client,
                            model=model_name,
                            batch=[missing],
                            max_retries=args.max_retries,
                            post_delay=args.sleep,
                        )
                        exc.results.update(single_results)
                        processed_entries.append(missing)
                    except Exception as single_exc:
                        logging.error(
                            "Failed to process individual entry %s: %s",
                            missing.entry_id,
                            single_exc,
                        )
                        # Skip this entry - mark as failed but keep original text
                        exc.results[missing.entry_id] = {
                            "entry_id": missing.entry_id,
                            "is_true_doric": True,  # Assume it's fine to avoid breaking the pipeline
                            "notes": f"Failed to judge: {single_exc}",
                            "corrected_response": missing.assistant_text,
                        }
                        processed_entries.append(missing)

                return processed_entries, exc.results
            except Exception as exc:
                # Attach batch info to exception for better error reporting
                logging.error(
                    "Failed to process batch of %d entries (entry_ids: %s): %s",
                    len(entries),
                    [e.entry_id for e in entries],
                    exc,
                )
                raise

        task = asyncio.create_task(runner(batch))
        # Store batch reference in task for error handling
        task.batch = batch  # type: ignore
        tasks.add(task)

    try:
        for idx, raw_line in load_dataset(args.input):
            if args.max_records is not None and enqueued >= args.max_records:
                break
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as err:
                logging.error("Skipping record %s: Invalid JSON - %s", idx, err)
                continue

            if not isinstance(entry, dict):
                logging.error("Skipping record %s: Entry is not a dictionary", idx)
                continue

            try:
                msg_idx, assistant_text = locate_assistant_turn(entry)
            except ValueError as err:
                logging.error("Skipping record %s: %s", idx, err)
                continue

            # Validate assistant_text is not empty
            if not assistant_text or not assistant_text.strip():
                logging.warning(
                    "Skipping record %s (entry_id=%s): Empty assistant text",
                    idx,
                    str(entry.get("meta", {}).get("id") or f"line-{idx}"),
                )
                continue

            meta = entry.get("meta") or {}
            entry_id = str(meta.get("id") or f"line-{idx}")
            payload = EntryPayload(
                seq=seq_counter,
                line_number=idx,
                entry_id=entry_id,
                entry=entry,
                assistant_index=msg_idx,
                assistant_text=assistant_text,
            )
            seq_counter += 1
            enqueued += 1
            current_batch.append(payload)
            if len(current_batch) >= batch_size:
                batch_to_submit = current_batch
                current_batch = []
                await submit_batch(batch_to_submit)
                if len(tasks) >= parallel_limit:
                    await wait_for_completion(first_only=True)

        if current_batch:
            await submit_batch(current_batch)

        await wait_for_completion(first_only=False)
        return total_processed, corrected
    finally:
        await client.close()


def process_dataset(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ensure_output_path(args.output, args.overwrite)
    base_config = read_openai_config()
    resolved_config = OpenAIConfig(
        api_key=base_config.api_key,
        model=args.model or base_config.model or DEFAULT_MODEL,
        base_url=base_config.base_url,
    )
    with open(args.output, "wb") as out_f:
        total, corrected = asyncio.run(
            process_dataset_async(args, resolved_config, out_f)
        )
    logging.info(
        "Finished. Total records processed: %s, corrected: %s", total, corrected
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    process_dataset(args)


if __name__ == "__main__":
    main()
