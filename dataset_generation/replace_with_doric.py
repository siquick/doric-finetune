#!/usr/bin/env python3
"""
Replace English words/phrases in JSONL conversations with Doric equivalents.

Reads a dictionary mapping English -> Doric and replaces occurrences in conversation text.
Matches longer phrases first to avoid partial replacements.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

try:  # optional faster JSON
    import orjson as _orjson  # type: ignore

    def dumps(obj: Any) -> bytes:
        return _orjson.dumps(obj)
except Exception:  # pragma: no cover

    def dumps(obj: Any) -> bytes:  # type: ignore
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def load_dictionary(dict_path: str) -> Dict[str, str]:
    """
    Load the Doric dictionary from JSON file.
    
    Returns a dict mapping English (lowercase) -> Doric.
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    # Create mapping: English -> Doric
    # Use lowercase English as key for case-insensitive matching
    dictionary = {}
    for entry in entries:
        english = entry.get("english", "").strip()
        doric = entry.get("doric", "").strip()
        
        if english and doric:
            # Store original English for exact matching, and lowercase for lookup
            english_lower = english.lower()
            # If multiple entries have same English, keep the first one
            # (or we could keep the longest Doric, but first is simpler)
            if english_lower not in dictionary:
                dictionary[english_lower] = {
                    "original_english": english,
                    "doric": doric
                }
    
    return dictionary


def create_replacement_patterns(dictionary: Dict[str, Dict[str, str]]) -> List[tuple[str, str]]:
    """
    Create regex patterns for replacement, sorted by length (longest first).
    
    Returns list of (pattern, replacement) tuples.
    """
    patterns = []
    
    # Sort by English length (longest first) to match phrases before words
    sorted_entries = sorted(
        dictionary.items(),
        key=lambda x: len(x[1]["original_english"]),
        reverse=True
    )
    
    for english_lower, entry in sorted_entries:
        original_english = entry["original_english"]
        doric = entry["doric"]
        
        # Check if it's a single word or phrase
        words = original_english.split()
        
        if len(words) == 1:
            # Single word: use word boundaries
            # Escape special regex characters in the word
            escaped = re.escape(original_english)
            pattern = r"\b" + escaped + r"\b"
        else:
            # Multi-word phrase: match as phrase (with word boundaries at edges)
            # Escape each word and join with whitespace pattern
            escaped_words = [re.escape(w) for w in words]
            pattern = r"\b" + r"\s+".join(escaped_words) + r"\b"
        
        patterns.append((pattern, doric))
    
    return patterns


def replace_text(text: str, patterns: List[tuple[str, str]], case_sensitive: bool = False) -> str:
    """
    Replace English words/phrases in text with Doric equivalents.
    
    Args:
        text: Input text to process
        patterns: List of (pattern, replacement) tuples
        case_sensitive: Whether to do case-sensitive matching
    
    Returns:
        Text with replacements made
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    
    result = text
    
    for pattern, replacement in patterns:
        # Use a function to preserve original case if needed
        def replacer(match):
            matched_text = match.group(0)
            
            if not replacement:
                return matched_text
            
            # Preserve capitalization pattern
            if matched_text.isupper():
                # All caps -> all caps
                return replacement.upper()
            elif matched_text[0].isupper() and len(matched_text) > 1 and matched_text[1:].islower():
                # Title case -> title case
                return replacement[0].upper() + replacement[1:] if len(replacement) > 1 else replacement.upper()
            elif matched_text[0].isupper():
                # First letter capitalized -> capitalize first letter
                return replacement[0].upper() + replacement[1:] if len(replacement) > 1 else replacement.upper()
            else:
                # Lowercase -> lowercase
                return replacement.lower()
        
        result = re.sub(pattern, replacer, result, flags=flags)
    
    return result


def process_conversation(conversation: Dict[str, Any], patterns: List[tuple[str, str]]) -> Dict[str, Any]:
    """
    Process a single conversation, replacing English with Doric in all message values.
    
    Args:
        conversation: Conversation dict with "conversations" list
        patterns: Replacement patterns
    
    Returns:
        Modified conversation dict
    """
    if "conversations" not in conversation:
        return conversation
    
    result = conversation.copy()
    result["conversations"] = []
    
    for msg in conversation["conversations"]:
        msg_copy = msg.copy()
        
        if "value" in msg_copy and isinstance(msg_copy["value"], str):
            msg_copy["value"] = replace_text(msg_copy["value"], patterns)
        
        result["conversations"].append(msg_copy)
    
    return result


def load_jsonl(input_path: str):
    """Load JSONL file line by line."""
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue


def write_jsonl(output_path: str, rows):
    """Write rows to JSONL file."""
    with open(output_path, "wb") as f:
        for row in rows:
            f.write(dumps(row))
            f.write(b"\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replace English words/phrases with Doric equivalents in JSONL conversations"
    )
    parser.add_argument(
        "input_file",
        help="Input JSONL file path (e.g., doric_conversations_sharegpt.jsonl)"
    )
    parser.add_argument(
        "--dictionary",
        default="datasets/doric_dictionary.json",
        help="Path to Doric dictionary JSON file (default: datasets/doric_dictionary.json)"
    )
    parser.add_argument(
        "--output",
        help="Output JSONL file path (default: input_file with '_doric' suffix)"
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive matching (default: case-insensitive)"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    input_path = Path(args.input_file)
    if args.output:
        output_path = Path(args.output)
    else:
        # Add '_doric' suffix before .jsonl extension
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_doric{suffix}"
    
    # Load dictionary
    print(f"Loading dictionary from {args.dictionary}...")
    dictionary = load_dictionary(args.dictionary)
    print(f"Loaded {len(dictionary)} dictionary entries")
    
    # Create replacement patterns
    print("Creating replacement patterns...")
    patterns = create_replacement_patterns(dictionary)
    print(f"Created {len(patterns)} replacement patterns")
    
    # Process conversations
    print(f"Processing conversations from {args.input_file}...")
    processed_count = 0
    total_count = 0
    
    processed_rows = []
    for row in load_jsonl(args.input_file):
        total_count += 1
        processed = process_conversation(row, patterns)
        processed_rows.append(processed)
        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"  Processed {processed_count} conversations...")
    
    # Write output
    print(f"Writing {processed_count} processed conversations to {output_path}...")
    write_jsonl(str(output_path), processed_rows)
    
    print(f"Done! Processed {processed_count}/{total_count} conversations")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()

