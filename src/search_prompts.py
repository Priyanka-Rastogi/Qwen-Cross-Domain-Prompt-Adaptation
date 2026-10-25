"""Demonstration sampling, selection tie-break, and the bounded prompt search
used by conditions B (instruction-only), C (fixed+random demos), and D
(source-selected instruction+demonstration combination).

Ordering rule (fixed, documented, independent of development outcomes):
  within a balanced 4-example demonstration set, sort negatives by id ascending,
  sort positives by id ascending, then interleave as [neg, pos, neg, pos].

D is a bounded random-search joint prompt baseline (NOT MIPRO, NOT EASE, NOT a
gradient-based method): instructions come from the same source-only proposal
pool as B; demonstration sets are i.i.d. balanced random draws from the
demonstration-candidate pool; every (instruction, demo-set) pair is scored on
source-development macro-F1 and the best is kept.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from evaluate import evaluate_condition

N_DEMOS_PER_CLASS = 2  # condition C/D: 2 positive + 2 negative
MAX_DEMO_SETS_FOR_D = 10
MAX_COMBINATIONS_FOR_D = 80


def order_demonstrations(demo_rows: list[dict]) -> list[dict]:
    neg = sorted([d for d in demo_rows if d["label"] == 0], key=lambda d: d["id"])
    pos = sorted([d for d in demo_rows if d["label"] == 1], key=lambda d: d["id"])
    assert len(neg) == len(pos) == N_DEMOS_PER_CLASS
    ordered = []
    for n, p in zip(neg, pos):
        ordered.append(n)
        ordered.append(p)
    return ordered


def sample_balanced_demo_set(pool_df, rng: random.Random) -> list[dict]:
    neg_ids = pool_df[pool_df["label"] == 0]["id"].tolist()
    pos_ids = pool_df[pool_df["label"] == 1]["id"].tolist()
    chosen_neg = rng.sample(neg_ids, N_DEMOS_PER_CLASS)
    chosen_pos = rng.sample(pos_ids, N_DEMOS_PER_CLASS)
    chosen_ids = set(chosen_neg) | set(chosen_pos)
    rows = pool_df[pool_df["id"].isin(chosen_ids)].to_dict("records")
    return order_demonstrations(rows)


def condition_c_demo_set(pool_df, experiment_seed: int) -> list[dict]:
    rng = random.Random(f"condition_c_{experiment_seed}")
    return sample_balanced_demo_set(pool_df, rng)


def condition_d_demo_sets(pool_df, experiment_seed: int, c_demo_set: list[dict],
                           max_demo_sets: int = MAX_DEMO_SETS_FOR_D) -> list[list[dict]]:
    """Up to max_demo_sets unique balanced demo sets, guaranteed to include
    condition C's set as one of them."""
    def ids_key(demo_set):
        return frozenset(d["id"] for d in demo_set)

    sets: list[list[dict]] = [c_demo_set]
    seen = {ids_key(c_demo_set)}

    rng = random.Random(f"condition_d_{experiment_seed}")
    attempts = 0
    max_attempts = max_demo_sets * 20
    while len(sets) < max_demo_sets and attempts < max_attempts:
        attempts += 1
        candidate = sample_balanced_demo_set(pool_df, rng)
        key = ids_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        sets.append(candidate)
    return sets


def demo_row_for_prompt(d: dict) -> dict:
    return {"id": d["id"], "text": d["text"], "label_name": d["label_name"]}


@dataclass
class SelectionCandidate:
    candidate_id: str
    instruction: str
    demonstrations: list[dict]  # already ordered, prompt-ready rows
    dev_result: dict | None = None


def select_best(candidates: list[SelectionCandidate], backend, cache, dev_examples: list[dict],
                 phase: str, batch_size: int) -> tuple[SelectionCandidate, list[dict]]:
    """Evaluates every candidate on dev_examples, applies the documented
    tie-break (macro-F1 desc, accuracy desc, avg prompt tokens asc, candidate_id
    asc), and returns (best_candidate, all_scored_candidates_summary)."""
    scored = []
    for cand in candidates:
        result = evaluate_condition(
            backend, cache, dev_examples, cand.instruction, cand.demonstrations,
            phase=phase, batch_size=batch_size,
        )
        cand.dev_result = result
        avg_prompt_tokens = sum(p["prompt_tokens"] for p in result["predictions"]) / len(result["predictions"])
        scored.append({
            "candidate_id": cand.candidate_id,
            "macro_f1": result["metrics"]["macro_f1"],
            "accuracy": result["metrics"]["accuracy"],
            "avg_prompt_tokens": avg_prompt_tokens,
            "cost": result["cost"],
        })

    def sort_key(s):
        return (-s["macro_f1"], -s["accuracy"], s["avg_prompt_tokens"], s["candidate_id"])

    scored_sorted = sorted(scored, key=sort_key)
    best_id = scored_sorted[0]["candidate_id"]
    best_candidate = next(c for c in candidates if c.candidate_id == best_id)
    return best_candidate, scored_sorted
