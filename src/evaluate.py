"""Strict parsing + batched classification evaluation shared by every condition."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from cache import EvalCache, make_cache_key
from metrics import INVALID_PRED, compute_metrics
from prompt_templates import OUTPUT_RESERVE_TOKENS, build_prompt
from text_utils import text_hash

LABEL_TO_ID = {"positive": 1, "negative": 0}

CLASSIFY_DECODING_CONFIG = {
    "do_sample": False,
    "max_new_tokens": OUTPUT_RESERVE_TOKENS,
    "temperature": None,
    "top_p": None,
}

PREPROCESSING_CONFIG = {
    "per_example_max_tokens": 256,
    "truncation_rule": "head truncation (keep first N tokens) by tokenizer token count",
}


def strict_parse(raw_output: str) -> int:
    normalized = raw_output.strip().lower()
    if normalized == "positive":
        return 1
    if normalized == "negative":
        return 0
    return INVALID_PRED


@dataclass
class PhaseCost:
    phase: str
    wall_clock_seconds: float = 0.0
    model_invocations: int = 0
    examples_classified: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0

    def as_dict(self):
        return dict(
            phase=self.phase, wall_clock_seconds=self.wall_clock_seconds,
            model_invocations=self.model_invocations, examples_classified=self.examples_classified,
            input_tokens=self.input_tokens, output_tokens=self.output_tokens,
            cache_hits=self.cache_hits, cache_misses=self.cache_misses,
            peak_allocated_bytes=self.peak_allocated_bytes, peak_reserved_bytes=self.peak_reserved_bytes,
        )


def classify_examples(backend, cache: EvalCache, examples: list[dict], instruction: str,
                       demonstrations: list[dict], phase: str, batch_size: int = 8) -> tuple[list[dict], PhaseCost]:
    """examples: list of {"id","text","label"}. Returns (per-example result dicts,
    PhaseCost). Uses/updates `cache`; a cache hit costs no model invocation."""
    cost = PhaseCost(phase=phase)
    software_config = backend.software_config()
    results: list[dict | None] = [None] * len(examples)

    pending_idx = []
    pending_prompt_str = []
    pending_keys = []

    for i, ex in enumerate(examples):
        built = build_prompt(backend.tokenizer, instruction, demonstrations, ex["text"])
        key = make_cache_key(
            model_revision=backend.model_revision,
            tokenizer_revision=backend.model_revision,
            prompt_str=built.prompt_str,
            input_content_hash=text_hash(ex["text"]),
            preprocessing_config=PREPROCESSING_CONFIG,
            decoding_config=CLASSIFY_DECODING_CONFIG,
            software_config=software_config,
        )
        cached = cache.get(key)
        if cached is not None:
            cost.cache_hits += 1
            pred = cached["pred"]
            results[i] = {
                "id": ex["id"], "label": ex["label"], "pred": pred,
                "raw_output": cached["raw_output"], "valid": pred != INVALID_PRED,
                "prompt_tokens": cached["prompt_tokens"], "output_tokens": cached["output_tokens"],
                "truncated_demo": cached.get("truncated_demo", False),
                "truncated_query": cached.get("truncated_query", False),
                "from_cache": True,
            }
        else:
            cost.cache_misses += 1
            pending_idx.append(i)
            pending_prompt_str.append(built.prompt_str)
            pending_keys.append((key, built))

    for start in range(0, len(pending_idx), batch_size):
        batch_slice = slice(start, start + batch_size)
        idxs = pending_idx[batch_slice]
        prompts = pending_prompt_str[batch_slice]
        key_builts = pending_keys[batch_slice]

        gen = backend.generate_batch(prompts, max_new_tokens=OUTPUT_RESERVE_TOKENS, do_sample=False)
        cost.model_invocations += 1
        cost.wall_clock_seconds += gen.wall_clock_seconds
        cost.input_tokens += sum(gen.input_token_counts)
        cost.output_tokens += sum(gen.output_token_counts)
        cost.peak_allocated_bytes = max(cost.peak_allocated_bytes, gen.peak_allocated_bytes)
        cost.peak_reserved_bytes = max(cost.peak_reserved_bytes, gen.peak_reserved_bytes)

        for j, i in enumerate(idxs):
            ex = examples[i]
            key, built = key_builts[j]
            raw_output = gen.raw_outputs[j]
            pred = strict_parse(raw_output)
            truncated_demo = any(r.truncated and r.role == "demo" for r in built.truncation_log)
            truncated_query = any(r.truncated and r.role == "query" for r in built.truncation_log)
            value = {
                "pred": pred, "raw_output": raw_output,
                "prompt_tokens": gen.input_token_counts[j], "output_tokens": gen.output_token_counts[j],
                "truncated_demo": truncated_demo, "truncated_query": truncated_query,
            }
            cache.put(key, value)
            results[i] = {
                "id": ex["id"], "label": ex["label"], "pred": pred, "raw_output": raw_output,
                "valid": pred != INVALID_PRED, "prompt_tokens": value["prompt_tokens"],
                "output_tokens": value["output_tokens"], "truncated_demo": truncated_demo,
                "truncated_query": truncated_query, "from_cache": False,
            }
    # Logical count (includes cache hits), not raw model invocations — see PhaseCost docs.
    cost.examples_classified = len(examples)
    assert all(r is not None for r in results)
    return results, cost


def evaluate_condition(backend, cache: EvalCache, examples: list[dict], instruction: str,
                        demonstrations: list[dict], phase: str, batch_size: int = 8) -> dict:
    results, cost = classify_examples(backend, cache, examples, instruction, demonstrations, phase, batch_size)
    y_true = [r["label"] for r in results]
    y_pred = [r["pred"] for r in results]
    metrics = compute_metrics(y_true, y_pred)
    return {"metrics": metrics, "cost": cost.as_dict(), "predictions": results}
