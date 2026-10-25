# Invalid-output diagnostics (post-hoc, read-only analysis)

Generated from `results/predictions/*.csv` only. The strict parser
(`src/evaluate.py:strict_parse`) was NOT changed and the experiment was NOT
recomputed under any alternate parsing rule — this file only explains, after
the fact, why 2164 of 13730 total predictions (15.76%)
across every condition/seed/split were marked invalid.

## Top-line finding

**2148 of 2164 invalid outputs
(99.3%) are the single,
complete, 2-token word "neutral"** — not truncation, not malformed text, not an
empty response, not a chat-template or thinking-mode artifact. The model is
making a deliberate three-way judgment (positive/negative/neutral) that falls
outside the binary label space this experiment enforces by design (see
`src/prompt_templates.py` and `src/evaluate.py:strict_parse`).

## 1. Category breakdown (all conditions, all seeds, both splits)

| Category | Count | % of invalid |
|---|---|---|
| unexpected_label:neutral | 2148 | 99.26% |
| unexpected_label:mixed | 12 | 0.55% |
| unexpected_label:mixed (apparent truncation at max_new_tokens) | 4 | 0.18% |

Total predictions across the whole experiment: 13730. Total invalid: 2164
(15.76%). Unique raw output strings ever produced,
across all 13730 generations: 9
(`['Mixed', 'Negative', 'Positive', 'mixed', 'mixed\n\n**Note:** The instruction asked', 'mixed\n\n**Note:** The text contains', 'negative', 'neutral', 'positive']`) — the output space is almost
entirely confined to `positive`/`negative`/`neutral`/`mixed` (plus the two rare
truncated `mixed` continuations below), not open-ended rambling.

## 2. Representative examples per category

- **unexpected_label:neutral** (n=2148): file=`B_seed0_target_test.csv`, id=`fpb_allagree_416`, true_label=1, output_tokens=2, raw_output='neutral'
  - input snippet: "Finnish flexible packaging manufacturer Suominen Corporation reports net sales of EUR 54.5 mn in the first quarter of 2008 , compared with EUR 54.3 mn a year earlier ...."
- **unexpected_label:mixed** (n=12): file=`A_source_test.csv`, id=`imdb_test_19358`, true_label=1, output_tokens=2, raw_output='Mixed'
  - input snippet: "Although this film looks like a Crime Thriller Noir, the plot is actually a bit simplistic and with very few surprise twists or turns at all - and those that do appear, are not exactly shockers.<br /><br />However, if yo..."
- **unexpected_label:mixed (apparent truncation at max_new_tokens)** (n=4): file=`B_seed0_source_test.csv`, id=`imdb_test_1225`, true_label=0, output_tokens=8, raw_output='mixed\n\n**Note:** The instruction asked'
  - input snippet: "Don't bother to check for logic. There is none. But on the other hand, there are MANY really great movies that totally lack logic, so why bother?<br /><br />I both like and dislike this film. I like it because the action..."

## 3. Generated-token-length check (truncation diagnosis)

- Invalid outputs using exactly 2 generated tokens: 2160 / 2164
  (99.8%) — these are COMPLETE, deliberate
  single-word responses (`neutral`, `mixed`, etc.), not cut off by the
  `max_new_tokens=8` budget.
- Invalid outputs that reached the `max_new_tokens=8` cap (apparent
  truncation): 4 / 2164 (0.18%).
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
  `enable_thinking=False`): 0 occurrences out of 13730 total
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
| no | 0.2% |
| yes | 40.4% |

Among the `neutral`-labeled invalid predictions, the true label was positive in
1728 cases and negative in 420 cases —
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
