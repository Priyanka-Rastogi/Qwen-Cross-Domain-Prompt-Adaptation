"""Single shared prompt-construction function used by every condition (A-D).

Truncation policy (documented, fixed before any experiment is run):
  - Every raw text inserted into a prompt (a demonstration's text OR the query's
    text) is independently truncated, by TOKEN COUNT, to at most
    PER_EXAMPLE_MAX_TOKENS tokens, keeping the FIRST PER_EXAMPLE_MAX_TOKENS tokens
    (head truncation) and dropping the rest.
  - This rule is identical across conditions A, B, C, D and across source/target
    domains; nothing about the method or domain changes how a given piece of text
    is truncated.
  - Truncation is applied only to the copy of the text placed in the prompt. The
    CSV files always store the full, untruncated original text.
  - After assembling the full prompt (instruction + demonstrations + query +
    chat template overhead), we additionally assert the total token count is under
    MAX_TOTAL_PROMPT_TOKENS as a sanity bound, and log a warning (not a silent
    change in behavior) if a prompt is unusually long.
  - Every truncation event (per example, per role) is logged: original token
    length, truncated length, whether truncation occurred.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PER_EXAMPLE_MAX_TOKENS = 256
MAX_TOTAL_PROMPT_TOKENS = 3072
OUTPUT_RESERVE_TOKENS = 8

CLASSIFICATION_REQUEST = "Respond with exactly one word: positive or negative."


@dataclass
class TruncationRecord:
    role: str  # "demo" or "query"
    original_tokens: int
    final_tokens: int
    truncated: bool


@dataclass
class BuiltPrompt:
    user_text: str
    messages: list
    truncation_log: list[TruncationRecord] = field(default_factory=list)
    prompt_token_count: int | None = None
    prompt_str: str | None = None


def truncate_to_tokens(tokenizer, text: str, max_tokens: int, role: str) -> tuple[str, TruncationRecord]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    original_len = len(ids)
    if original_len <= max_tokens:
        return text, TruncationRecord(role, original_len, original_len, False)
    truncated_ids = ids[:max_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text, TruncationRecord(role, original_len, len(truncated_ids), True)


def build_prompt(tokenizer, instruction: str, demonstrations: list[dict], query_text: str) -> BuiltPrompt:
    """demonstrations: list of {"text": str, "label_name": "positive"|"negative"}, in
    the exact order they should appear (ordering is fixed by a documented rule
    upstream; this function never reorders)."""
    log: list[TruncationRecord] = []
    blocks = [instruction.strip(), ""]

    for demo in demonstrations:
        demo_text, rec = truncate_to_tokens(tokenizer, demo["text"], PER_EXAMPLE_MAX_TOKENS, "demo")
        log.append(rec)
        blocks.append(f"Text: {demo_text}\nSentiment: {demo['label_name']}\n")

    query_trunc, rec = truncate_to_tokens(tokenizer, query_text, PER_EXAMPLE_MAX_TOKENS, "query")
    log.append(rec)
    blocks.append(f"Text: {query_trunc}\nSentiment:")

    user_text = "\n".join(blocks)
    messages = [{"role": "user", "content": user_text}]

    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_token_count = len(tokenizer.encode(prompt_str, add_special_tokens=False))

    built = BuiltPrompt(user_text=user_text, messages=messages, truncation_log=log,
                         prompt_token_count=prompt_token_count, prompt_str=prompt_str)

    if prompt_token_count + OUTPUT_RESERVE_TOKENS > MAX_TOTAL_PROMPT_TOKENS:
        raise ValueError(
            f"Assembled prompt ({prompt_token_count} tokens) exceeds "
            f"MAX_TOTAL_PROMPT_TOKENS={MAX_TOTAL_PROMPT_TOKENS} even after per-example "
            f"truncation to {PER_EXAMPLE_MAX_TOKENS} tokens; refusing silent truncation "
            f"at the full-prompt level."
        )
    return built


