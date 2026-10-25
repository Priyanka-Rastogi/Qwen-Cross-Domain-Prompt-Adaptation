"""Source-only instruction proposal (condition B/D instruction pool).

Uses the SAME frozen Qwen checkpoint (no second LLM) to propose candidate task
descriptions from a small, FIXED sample of source-training (IMDb) examples. The
sample shown to the model is identical across experiment seeds; only the model's
stochastic generation (temperature/top_p sampling, seeded per experiment seed)
varies the wording produced, matching the spec's "seeds affect candidate
generation" rule while keeping the proposal sample itself fixed and documented.

Every candidate instruction is required to be domain-neutral: it must not mention
finance/economics/markets (the target domain) nor movies/films (the source
domain), so it reads as a generic binary-sentiment instruction. The fixed output
format directive ("Respond with exactly one word: positive or negative.") is
appended deterministically in code, not left to the model, so every candidate's
output-format phrasing is identical to the baseline's.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from evaluate import PhaseCost
from prompt_templates import CLASSIFICATION_REQUEST

BASELINE_INSTRUCTION = (
    "Classify the sentiment of the following text as positive or negative. "
    + CLASSIFICATION_REQUEST
)

PROPOSAL_SAMPLE_SIZE_PER_CLASS = 3  # fixed across seeds: 3 positive + 3 negative
MAX_CANDIDATES = 8
GENERATION_ATTEMPTS_PER_SEED = 16  # upper bound on sampling attempts before giving up
PROPOSAL_MAX_NEW_TOKENS = 60
PROPOSAL_TEMPERATURE = 0.9
PROPOSAL_TOP_P = 0.9

BANNED_SUBSTRINGS = [
    "financ", "stock", "invest", "econom", "bank", "market", "earnings", "revenue",
    "movie", "film", "cinema", "actor", "actress", "director", "review of the film",
]

PROPOSAL_EXAMPLE_TEXT_MAX_CHARS = 600  # keep the meta-prompt itself bounded


def _truncate_for_display(text: str) -> str:
    return text[:PROPOSAL_EXAMPLE_TEXT_MAX_CHARS]


def build_proposal_sample(source_train_df, seed_for_logging: int | None = None) -> list[dict]:
    """Fixed (seed-independent) sample: first 3 negative + first 3 positive rows
    of source_train.csv in file order (file order is itself a seed-42 random
    sample, so this is a deterministic sub-sample of it)."""
    neg = source_train_df[source_train_df["label"] == 0].head(PROPOSAL_SAMPLE_SIZE_PER_CLASS)
    pos = source_train_df[source_train_df["label"] == 1].head(PROPOSAL_SAMPLE_SIZE_PER_CLASS)
    sample = []
    for _, row in list(neg.iterrows()) + list(pos.iterrows()):
        sample.append({"id": row["id"], "text": row["text"], "label_name": row["label_name"]})
    return sample


def build_proposal_prompt(sample: list[dict]) -> str:
    lines = [
        "You are designing an instruction for a binary sentiment classifier that "
        "must work on text from ANY domain (it will be used on text you have not seen). "
        "Below are a few labeled examples for calibration only.",
        "",
    ]
    for ex in sample:
        lines.append(f"Text: {_truncate_for_display(ex['text'])}")
        lines.append(f"Sentiment: {ex['label_name']}")
        lines.append("")
    lines.append(
        "Write ONE short, general instruction sentence (no more than 25 words) that tells "
        "a model to classify a piece of text as expressing positive or negative sentiment. "
        "The instruction must be domain-neutral: do NOT mention movies, films, finance, "
        "stocks, or any other specific subject matter. Do NOT include an output-format "
        "directive (e.g. do not say how to respond); only describe the classification task "
        "itself. Output only the instruction sentence, nothing else."
    )
    return "\n".join(lines)


def _passes_domain_neutral_filter(candidate: str) -> str | None:
    lower = candidate.lower()
    for bad in BANNED_SUBSTRINGS:
        if bad in lower:
            return f"contains banned domain term: '{bad}'"
    if not candidate.strip():
        return "empty"
    if len(candidate.split()) > 40:
        return "too long (>40 words)"
    return None


def parse_candidate(raw_generation: str) -> str:
    text = raw_generation.strip()
    text = text.split("\n")[0].strip()
    text = text.strip('"\'')
    text = re.sub(r"^(Instruction:|Sentence:)\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_candidate(instruction_task_sentence: str) -> str:
    return re.sub(r"\s+", " ", instruction_task_sentence.strip().lower())


def finalize_instruction(task_sentence: str) -> str:
    sentence = task_sentence.strip()
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return f"{sentence} {CLASSIFICATION_REQUEST}"


@dataclass
class ProposalRecord:
    proposal_prompt: str
    sampled_example_ids: list[str]
    raw_generations: list[dict] = field(default_factory=list)  # {"attempt","raw_text","parsed","status"}
    accepted_candidates: list[str] = field(default_factory=list)  # final instruction strings
    cost: PhaseCost = field(default_factory=lambda: PhaseCost(phase="instruction_generation"))


def propose_instructions(backend, source_train_df, experiment_seed: int,
                          max_candidates: int = MAX_CANDIDATES,
                          generation_attempts: int = GENERATION_ATTEMPTS_PER_SEED) -> ProposalRecord:
    sample = build_proposal_sample(source_train_df)
    proposal_prompt = build_proposal_prompt(sample)

    rec = ProposalRecord(proposal_prompt=proposal_prompt, sampled_example_ids=[s["id"] for s in sample])

    accepted_task_sentences: list[str] = []
    accepted_norms: set[str] = set()

    baseline_task_sentence = "Classify the sentiment of the following text as positive or negative"
    accepted_task_sentences.append(baseline_task_sentence)
    accepted_norms.add(normalize_candidate(baseline_task_sentence))

    messages = [{"role": "user", "content": proposal_prompt}]
    prompt_str = backend.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    for attempt in range(generation_attempts):
        if len(accepted_task_sentences) >= max_candidates:
            break
        gen_seed = experiment_seed * 10_000 + attempt
        gen = backend.generate_batch(
            [prompt_str], max_new_tokens=PROPOSAL_MAX_NEW_TOKENS, do_sample=True,
            temperature=PROPOSAL_TEMPERATURE, top_p=PROPOSAL_TOP_P, seed=gen_seed,
        )
        rec.cost.wall_clock_seconds += gen.wall_clock_seconds
        rec.cost.model_invocations += 1
        rec.cost.input_tokens += sum(gen.input_token_counts)
        rec.cost.output_tokens += sum(gen.output_token_counts)
        rec.cost.peak_allocated_bytes = max(rec.cost.peak_allocated_bytes, gen.peak_allocated_bytes)
        rec.cost.peak_reserved_bytes = max(rec.cost.peak_reserved_bytes, gen.peak_reserved_bytes)

        raw_text = gen.raw_outputs[0]
        parsed = parse_candidate(raw_text)
        norm = normalize_candidate(parsed)
        reject_reason = _passes_domain_neutral_filter(parsed)
        if reject_reason is None and norm in accepted_norms:
            reject_reason = "duplicate of an already-accepted candidate"

        status = "accepted" if reject_reason is None else f"rejected: {reject_reason}"
        rec.raw_generations.append(
            {"attempt": attempt, "generation_seed": gen_seed, "raw_text": raw_text,
             "parsed": parsed, "status": status}
        )
        if reject_reason is None:
            accepted_task_sentences.append(parsed)
            accepted_norms.add(norm)

    rec.cost.examples_classified = len(rec.raw_generations)
    rec.accepted_candidates = [finalize_instruction(s) for s in accepted_task_sentences]
    return rec
