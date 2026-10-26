"""Conditional-likelihood binary-label scoring: an alternative PREDICTION
INTERFACE to free-generation decoding (src/evaluate.py), evaluated on the
exact same frozen prompts (instruction + demonstrations) already selected by
the original pilot run. The model's weights, chat template, and
instruction/demonstration selection are completely unchanged; only how a
predicted label is extracted from the frozen model changes. No fine-tuning,
no gradient updates, no new datasets.

Scoring rule (fixed BEFORE any evaluation; never tuned after seeing results):
  For a built prompt (identical to src/prompt_templates.build_prompt, with
  add_generation_prompt=True), compute, for each of the two candidate labels
  {"negative", "positive"}, the model's TEACHER-FORCED joint log-likelihood of
  producing that label's own token sequence as the immediate continuation (no
  sampling; this is an exact likelihood computation, not a generation):

      score(label) = sum_{k=0}^{len(label_tokens)-1}
                     log P(label_tokens[k] | prompt, label_tokens[:k])

  The predicted label is the argmax of the two scores:
      pred = "positive" if score(positive) > score(negative) else "negative"

  Tie-break (deterministic, fixed in advance, pre-registered here before any
  evaluation was run): if score(positive) == score(negative) EXACTLY (bit-for-
  bit equal after casting logits to float32), predict TIE_BREAK_LABEL =
  "negative". This is logged per-example (`tie` column) for audit; with
  continuous log-probabilities an exact tie is expected to be vanishingly
  rare.

Tokenization verification (never assumed to be one token, per project
requirement): for each prompt, independently for each label, we verify how
the label text actually tokenizes in the LIVE chat-template context by
encoding `prompt_str + candidate_form` and checking that its first
len(encode(prompt_str)) tokens exactly match encode(prompt_str) alone --
i.e. that appending the label text does not retroactively change how the
prompt itself tokenizes at the boundary (a known BPE merge-boundary hazard).
We try the label word with no leading space first, then with a leading
space, and use whichever form passes this exact-prefix check. The resulting
continuation may be one or more tokens; every token of it is scored.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

LABELS = ("negative", "positive")
LABEL_TO_ID = {"negative": 0, "positive": 1}
TIE_BREAK_LABEL = "negative"  # fixed in advance; see module docstring

SCORING_METHOD_CONFIG = {
    "method": "conditional_log_likelihood_full_sequence",
    "labels": list(LABELS),
    "tie_break_label": TIE_BREAK_LABEL,
    "teacher_forced": True,
    "sampling": "none (exact teacher-forced log-likelihood, not greedy decoding)",
}


@dataclass
class LabelTokenization:
    label: str
    form_used: str  # the exact text whose tokenization was used, e.g. "negative" or " negative"
    continuation_ids: list[int]
    n_tokens: int


def resolve_label_tokenization(tokenizer, prompt_str: str, label: str) -> LabelTokenization:
    prefix_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    prefix_len = len(prefix_ids)
    for form in (label, " " + label):
        full_ids = tokenizer.encode(prompt_str + form, add_special_tokens=False)
        if full_ids[:prefix_len] == prefix_ids:
            return LabelTokenization(
                label=label, form_used=form,
                continuation_ids=full_ids[prefix_len:], n_tokens=len(full_ids) - prefix_len,
            )
    raise ValueError(
        f"Could not align tokenization for label={label!r} against this prompt: neither "
        f"{label!r} nor {(' ' + label)!r} preserved the prompt's own token boundary. "
        f"Prompt tail: {prompt_str[-80:]!r}"
    )


@dataclass
class ScoredExample:
    prefix_ids: list[int]
    labels: dict  # label -> LabelTokenization


def build_scored_example(tokenizer, prompt_str: str) -> ScoredExample:
    prefix_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    labels = {lab: resolve_label_tokenization(tokenizer, prompt_str, lab) for lab in LABELS}
    return ScoredExample(prefix_ids=prefix_ids, labels=labels)


@dataclass
class ScoringBatchResult:
    scores: list[dict]  # per example: {"negative": float, "positive": float}
    preds: list[str]
    ties: list[bool]
    wall_clock_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    input_token_counts: list[int]  # prefix length per example
    output_token_counts: dict  # label -> list[int] continuation lengths


@torch.inference_mode()
def score_batch(backend, scored_examples: list[ScoredExample]) -> ScoringBatchResult:
    """One forward pass covering BOTH candidate labels for every example in
    this batch. Sequences are left-padded (matching model_backend.py's
    generation convention) so that `logits_to_keep` reliably selects each
    row's own trailing real tokens regardless of padding length."""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    pad_id = backend.tokenizer.pad_token_id
    sequences: list[list[int]] = []
    seq_meta: list[tuple[int, str, int]] = []  # (example_idx, label, continuation_len)
    for i, ex in enumerate(scored_examples):
        for lab in LABELS:
            tok = ex.labels[lab]
            sequences.append(ex.prefix_ids + tok.continuation_ids)
            seq_meta.append((i, lab, tok.n_tokens))

    max_len = max(len(s) for s in sequences)
    max_cont = max(m[2] for m in seq_meta)
    logits_to_keep = max_cont + 1  # need the last `cont_len` predicting positions per row, plus 1 margin

    n_seq = len(sequences)
    input_ids = torch.full((n_seq, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n_seq, max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        L = len(seq)
        input_ids[row, max_len - L:] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, max_len - L:] = 1

    input_ids = input_ids.to(backend.model.device)
    attention_mask = attention_mask.to(backend.model.device)
    # Left-padded batches need EXPLICIT position_ids: unlike model.generate() (which
    # derives them from attention_mask internally via prepare_inputs_for_generation),
    # a raw forward() call defaults to a plain arange() over the padded sequence
    # dimension, which assigns each row's real tokens a RoPE offset equal to however
    # much left-padding precedes them in THIS batch. That makes a sequence's computed
    # log-likelihood depend on which other examples happen to share its batch --
    # verified empirically (same prompt scored in isolation vs. in an 8-example batch
    # gave different scores). Deriving position_ids from attention_mask (0 at the
    # first real token, ascending from there) makes every row's real tokens occupy the
    # same relative positions regardless of batch composition or padding amount.
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    out = backend.model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        use_cache=False, logits_to_keep=logits_to_keep,
    )
    # out.logits: [n_seq, K, V] where K = logits_to_keep (the LAST K columns of the
    # left-padded sequence, which equal each row's own last K real tokens).
    K = out.logits.shape[1]
    log_probs = F.log_softmax(out.logits.float(), dim=-1)

    scores = [dict() for _ in scored_examples]
    out_token_counts = {lab: [] for lab in LABELS}
    for row, (i, lab, cont_len) in enumerate(seq_meta):
        seq = sequences[row]
        total_logprob = 0.0
        for k in range(cont_len):
            from_end = cont_len - k + 1  # ranges cont_len+1 (first cont token) .. 2 (last cont token)
            col = K - from_end
            target_token = seq[len(seq) - cont_len + k]
            total_logprob += log_probs[row, col, target_token].item()
        scores[i][lab] = total_logprob
        out_token_counts[lab].append(cont_len)

    preds, ties = [], []
    for s in scores:
        if s["positive"] > s["negative"]:
            preds.append("positive")
        elif s["negative"] > s["positive"]:
            preds.append("negative")
        else:
            preds.append(TIE_BREAK_LABEL)
        ties.append(s["positive"] == s["negative"])

    wall = time.perf_counter() - t0
    return ScoringBatchResult(
        scores=scores, preds=preds, ties=ties, wall_clock_seconds=wall,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        input_token_counts=[len(ex.prefix_ids) for ex in scored_examples],
        output_token_counts=out_token_counts,
    )
