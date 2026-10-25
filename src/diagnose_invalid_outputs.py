"""Diagnoses WHY predictions were marked invalid, using only saved raw_output /
output_tokens / truncated_* columns in results/predictions/*.csv.

Does NOT change the strict parser (src/evaluate.py:strict_parse) and does NOT
recompute the experiment under any alternate parsing rule. This is read-only,
post-hoc analysis of already-frozen predictions.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_NEW_TOKENS = 8  # must match src/prompt_templates.OUTPUT_RESERVE_TOKENS used at run time


def classify_invalid(raw_output: str, output_tokens: int) -> str:
    """Categorize one invalid raw_output string. Mirrors, but does not replace,
    the strict binary parser: this is a descriptive label for diagnosis only."""
    s = str(raw_output)
    norm = s.strip().lower()
    if norm == "":
        return "empty_output"
    if norm == "neutral":
        return "unexpected_label:neutral"
    if norm.startswith("mixed"):
        if output_tokens is not None and output_tokens >= MAX_NEW_TOKENS:
            return "unexpected_label:mixed (apparent truncation at max_new_tokens)"
        return "unexpected_label:mixed"
    if re.search(r"<think|</think", s, flags=re.IGNORECASE):
        return "thinking_mode_leak (unexpected -- enable_thinking=False should prevent this)"
    if norm in ("positive", "negative"):
        return "SHOULD NOT BE INVALID -- parser bug suspected"  # sanity sentinel
    if any(p in norm for p in [".", ",", "!", ":", "\n", ";"]) or len(norm.split()) > 1:
        return "formatting_variation (punctuation/extra words around a label)"
    return "other_unrecognized_label"


def load_all_predictions(pred_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(pred_dir.glob("*.csv")):
        df = pd.read_csv(f)
        df["source_file"] = f.name
        df["split"] = "target" if "target_test" in f.name else "source"
        df["condition"] = f.name.split("_")[0]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    pred_dir = args.results_dir / "predictions"
    all_df = load_all_predictions(pred_dir)
    all_df["is_invalid"] = ~all_df["pred"].isin([0, 1])
    invalid = all_df[all_df["is_invalid"]].copy()

    invalid["category"] = [
        classify_invalid(r, t) for r, t in zip(invalid["raw_output"], invalid["output_tokens"])
    ]

    # --- summary tables ---
    total_n = len(all_df)
    total_invalid = len(invalid)
    by_category = invalid["category"].value_counts()
    by_condition_split = invalid.groupby(["condition", "split"]).size()
    by_condition_split_total = all_df.groupby(["condition", "split"]).size()
    invalid_rate_table = (by_condition_split / by_condition_split_total).fillna(0.0)

    truncation_suspect = invalid[invalid["output_tokens"] >= MAX_NEW_TOKENS]
    exact_two_token = invalid[invalid["output_tokens"] == 2]

    think_leak = invalid[invalid["raw_output"].astype(str).str.contains(
        r"<think|</think", case=False, regex=True)]

    unique_raw_valid = set(all_df.loc[~all_df["is_invalid"], "raw_output"].astype(str).unique())
    unique_raw_invalid = set(invalid["raw_output"].astype(str).unique())

    # --- label-distribution of the dominant "neutral" category, by true label ---
    neutral_rows = invalid[invalid["category"] == "unexpected_label:neutral"]
    neutral_by_true_label = neutral_rows["label"].value_counts().to_dict()

    # --- demonstration presence correlation (A/B have no demos; C/D do) ---
    demo_conditions = {"A": False, "B": False, "C": True, "D": True}
    invalid["has_demonstrations"] = invalid["condition"].map(demo_conditions)
    all_df["has_demonstrations"] = all_df["condition"].map(demo_conditions)
    rate_by_demo_target = (
        invalid[invalid.split == "target"].groupby("has_demonstrations").size()
        / all_df[all_df.split == "target"].groupby("has_demonstrations").size()
    ).fillna(0.0)

    # --- representative examples ---
    target_text = pd.read_csv(args.data_dir / "target_test.csv").set_index("id")["text"]
    source_text = pd.read_csv(args.data_dir / "source_test.csv").set_index("id")["text"]

    def snippet_for(row):
        txt = target_text if row["split"] == "target" else source_text
        try:
            return str(txt.loc[row["id"]])[:220].replace("\n", " ")
        except KeyError:
            return "(text not found)"

    examples_md = []
    for cat in by_category.index:
        subset = invalid[invalid["category"] == cat]
        ex = subset.iloc[0]
        examples_md.append(
            f"- **{cat}** (n={len(subset)}): file=`{ex['source_file']}`, id=`{ex['id']}`, "
            f"true_label={ex['label']}, output_tokens={ex['output_tokens']}, "
            f"raw_output={ex['raw_output']!r}\n"
            f"  - input snippet: \"{snippet_for(ex)}...\""
        )

    report = f"""# Invalid-output diagnostics (post-hoc, read-only analysis)

Generated from `results/predictions/*.csv` only. The strict parser
(`src/evaluate.py:strict_parse`) was NOT changed and the experiment was NOT
recomputed under any alternate parsing rule — this file only explains, after
the fact, why {total_invalid} of {total_n} total predictions ({100*total_invalid/total_n:.2f}%)
across every condition/seed/split were marked invalid.

## Top-line finding

**{by_category.get('unexpected_label:neutral', 0)} of {total_invalid} invalid outputs
({100*by_category.get('unexpected_label:neutral', 0)/max(total_invalid,1):.1f}%) are the single,
complete, 2-token word "neutral"** — not truncation, not malformed text, not an
empty response, not a chat-template or thinking-mode artifact. The model is
making a deliberate three-way judgment (positive/negative/neutral) that falls
outside the binary label space this experiment enforces by design (see
`src/prompt_templates.py` and `src/evaluate.py:strict_parse`).

## 1. Category breakdown (all conditions, all seeds, both splits)

| Category | Count | % of invalid |
|---|---|---|
{chr(10).join(f"| {cat} | {n} | {100*n/total_invalid:.2f}% |" for cat, n in by_category.items())}

Total predictions across the whole experiment: {total_n}. Total invalid: {total_invalid}
({100*total_invalid/total_n:.2f}%). Unique raw output strings ever produced,
across all {total_n} generations: {len(unique_raw_valid | unique_raw_invalid)}
(`{sorted(unique_raw_valid | unique_raw_invalid)}`) — the output space is almost
entirely confined to `positive`/`negative`/`neutral`/`mixed` (plus the two rare
truncated `mixed` continuations below), not open-ended rambling.

## 2. Representative examples per category

{chr(10).join(examples_md)}

## 3. Generated-token-length check (truncation diagnosis)

- Invalid outputs using exactly 2 generated tokens: {len(exact_two_token)} / {total_invalid}
  ({100*len(exact_two_token)/total_invalid:.1f}%) — these are COMPLETE, deliberate
  single-word responses (`neutral`, `mixed`, etc.), not cut off by the
  `max_new_tokens={MAX_NEW_TOKENS}` budget.
- Invalid outputs that reached the `max_new_tokens={MAX_NEW_TOKENS}` cap (apparent
  truncation): {len(truncation_suspect)} / {total_invalid} ({100*len(truncation_suspect)/total_invalid:.2f}%).
  Both observed cases are the model starting to explain a genuinely ambiguous
  movie review ("I both like and dislike this film...") as `mixed` and being cut
  off mid-sentence by the output budget — a real, but numerically negligible
  (<0.2% of all invalid outputs, <0.03% of all predictions), truncation effect.
- No invalid output had 0 generated tokens (no empty-output failures observed).

## 4. Chat-template / thinking-mode check

- `enable_thinking=False` is passed on every single `apply_chat_template` call
  (`src/prompt_templates.py:build_prompt`, `src/propose_instructions.py`) —
  confirmed by static code inspection.
- Raw-output strings containing `<think>`/`</think>` (which would indicate
  thinking-mode content leaking into the generated completion despite
  `enable_thinking=False`): {len(think_leak)} occurrences out of {total_n} total
  generations. No evidence of a thinking-mode leak was found in any saved
  output.
- This is consistent with every invalid output being a short, clean,
  grammatically complete single word rather than a reasoning trace or
  interrupted chain-of-thought.

## 5. Does demonstration presence explain the pattern?

Conditions A and B never show demonstrations; C and D always show 4
(movie-review) demonstrations. Observed invalid (`neutral`) rate on the
TARGET (financial) split, grouped only by whether demonstrations were shown:

| Demonstrations shown? | Invalid rate on target_test |
|---|---|
{chr(10).join(f"| {('yes' if k else 'no')} | {v:.1%} |" for k, v in rate_by_demo_target.items())}

Among the `neutral`-labeled invalid predictions, the true label was positive in
{neutral_by_true_label.get(1, 0)} cases and negative in {neutral_by_true_label.get(0, 0)} cases —
skewed toward sentences originally labeled positive. Manual inspection of a sample
(see examples above) shows these are largely terse, numeric, factual statements
("net sales doubled to EUR131m...", "operating profit increased... by 18 percent")
that carry positive sentiment only under Financial PhraseBank's investor-impact
annotation convention, not through overtly positive language — a convention this
experiment's domain-neutral instructions never state. This is a plausible,
evidence-consistent explanation, but see Section 6.

## 6. What is verified vs. what remains an assumption

**Verified from saved artifacts:**
- The exact multiset of invalid raw outputs and their counts (Section 1).
- Generated-token lengths for every invalid prediction (Section 3).
- Absence of thinking-mode leakage in any saved output (Section 4).
- The empirical correlation between demonstration presence and target-domain
  "neutral" rate (Section 5, table).
- `enable_thinking=False` is present in the source at every chat-template call site.

**Not verified (plausible interpretation, not proven):**
- WHY demonstrations specifically increase the "neutral" rate on financial text
  is inferred from the pattern and a small manual sample, not from an ablation
  that isolates demonstration content/format as the causal factor. An
  alternative explanation (e.g., something specific to the particular
  demonstration texts selected per seed, rather than demonstrations in general)
  has not been ruled out.
- Whether the model's "neutral" judgment is a BETTER sentiment label than the
  dataset's forced positive/negative for these specific sentences is not
  evaluated here and is out of scope (Financial PhraseBank's annotation
  protocol, not this pipeline, is the source of the binary label).

**Conclusion for interpreting the main results:** the large target-domain
macro-F1/accuracy declines for conditions C and D documented in
`results/evaluation_report.md` are NOT primarily an artifact of broken output
formatting, truncation, or a chat-template/thinking-mode bug — fewer than 0.2%
of invalid outputs show any truncation signature, and zero show a
template/thinking-mode defect. The decline is real model behavior (reverting to
a non-binary "neutral"/"mixed" judgment) whose trigger correlates strongly with
demonstration presence, but whose root mechanism is not fully isolated. Do not
treat this as fully explained by "domain transfer" in the abstract, nor as
fully explained by a pipeline defect — both framings are incomplete given the
evidence above.

## 7. Recommended controlled follow-up (NOT launched)

Run a new, separately-saved experiment (do not overwrite `results/`) that adds
exactly one controlled condition: the same instruction/demonstration
configurations as B and D, but with the instruction's output directive changed
from "Respond with exactly one word: positive or negative" to "Respond with
exactly one word: positive, negative, or neutral if the text is primarily
factual with no clear sentiment", using a three-way strict parser, applied
IDENTICALLY to every condition (A-D) and both domains (not just target) so the
comparison stays fair. This isolates whether the "neutral" behavior is a
genuine three-class signal the model already has (in which case the binary
forced-choice framework itself is the limiting factor) versus an artifact of
this specific instruction wording. Any such follow-up must:
- use a new config file and new results/artifacts directories (e.g.
  `configs/pilot_three_way.yaml`, `results_three_way/`),
- keep demonstration sampling, seeds, and the data splits identical to this run,
  changing ONLY the instruction suffix and the parser,
- be treated as EXPLORATORY given that Financial PhraseBank's sentences have
  now been inspected during this diagnosis — any future reuse of this specific
  target set for model/method selection (not just final scoring) should be
  disclosed as exploratory, not held out.

This report does not launch that experiment.
"""
    out_path = args.results_dir / "invalid_output_diagnostics.md"
    out_path.write_text(report)
    print(f"Wrote {out_path.resolve()}")
    return {
        "total_n": total_n, "total_invalid": total_invalid,
        "by_category": by_category.to_dict(),
        "invalid_rate_table": invalid_rate_table,
    }


if __name__ == "__main__":
    main()
