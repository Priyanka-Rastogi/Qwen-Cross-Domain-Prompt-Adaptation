# Cross-domain prompt adaptation: IMDb -> Financial PhraseBank (Qwen3-8B, frozen weights)

Final report for the completed 3-seed pilot (`configs/pilot.yaml`, seeds=[0, 1, 2]).
Generated entirely from saved artifacts (`results/comparison.csv`,
`results/metrics/summary.json`, `results/evaluation_report.md`,
`results/invalid_output_diagnostics.md`) — no prompts were changed and no
result reported here was produced by re-running prompt search.

Every number in the table below traces back to `results/comparison.csv`, which `src/evaluate_experiment.py` built by independently recomputing accuracy/macro-F1/invalid-rate from the raw `results/predictions/*.csv` files (not by copying `summary.json`). That independent recomputation matched the pipeline's own stored metrics exactly for every completed condition/seed, with zero mismatches.

## 1. What was adapted (and what was not)

**Verified** (enforced in code, checked by `src/evaluate_experiment.py` against
the actual run's artifacts/logs):
- Qwen3-8B's weights were never updated. `src/model_backend.py` sets
  `requires_grad=False` on every parameter, asserts the trainable-parameter
  count equals exactly zero before any inference, keeps `model.eval()`, and
  wraps all generation in `torch.inference_mode()`. "Adaptation" means only:
  (a) the wording of a natural-language task instruction, and/or (b) which
  labeled IMDb examples are shown as in-context demonstrations — both chosen
  using IMDb (source) data only and inserted into the prompt at inference time.
- `enable_thinking=False` is passed on every chat-template call
  (`src/prompt_templates.py`, `src/propose_instructions.py`), and zero of the
  13,730 total generations produced in this run contain any `<think>` content
  (see `results/invalid_output_diagnostics.md` Section 4) — thinking mode was
  off in practice, not just in the call signature.
- Source/target isolation: `source_train.csv` (200 examples) fed instruction
  proposal and the demonstration pool; `source_validation.csv` (100 examples,
  disjoint) scored and selected the best instruction (B) / instruction+demo
  combination (D) via source-development macro-F1; `source_test.csv` (500) and
  `target_test.csv` (873) were touched only for final scoring, after each
  configuration was frozen and hashed into `artifacts/selected_prompts/*.json`.
  `src/evaluate_experiment.py` independently confirmed every dev-selection
  evaluation used exactly the 100 `source_validation` examples (never the 500-
  or 873-example test sets), and that every completed condition scored the
  identical source-test/target-test example-id sets.

**Assumption, not verified by this run:** that Qwen3-8B's pretraining data did
not include Financial PhraseBank or similar labeled financial-sentiment
corpora. "Unseen target domain" in this project means excluded from this
experiment's development process (instruction proposal, demonstration
selection, hyperparameter choice), not necessarily absent from pretraining —
that broader claim is unverifiable from this pipeline and is not made.

## 2. Results table (3 seeds, mean ± std where applicable)

| Condition   | Src Accuracy   | Src Macro-F1   | Tgt Accuracy   | Tgt Macro-F1   | Src Invalid Rate   | Tgt Invalid Rate   |   Search Runtime (s, total/3 seeds) |   Final Inference Runtime (s, src+tgt, total/3 seeds) | Seeds completed   |
|:------------|:---------------|:---------------|:---------------|:---------------|:-------------------|:-------------------|------------------------------------:|------------------------------------------------------:|:------------------|
| A           | 0.922 ± 0.000  | 0.923 ± 0.000  | 0.978 ± 0.000  | 0.976 ± 0.000  | 0.2%               | 0.0%               |                                   0 |                                                    26 | 1/1               |
| B           | 0.923 ± 0.001  | 0.924 ± 0.001  | 0.981 ± 0.002  | 0.980 ± 0.001  | 0.4%               | 0.2%               |                                  40 |                                                    53 | 3/3               |
| C           | 0.931 ± 0.014  | 0.933 ± 0.013  | 0.614 ± 0.057  | 0.761 ± 0.040  | 0.4%               | 35.1%              |                                   0 |                                                   505 | 3/3               |
| D           | 0.932 ± 0.004  | 0.941 ± 0.002  | 0.527 ± 0.028  | 0.702 ± 0.024  | 1.9%               | 45.7%              |                                3134 |                                                   498 | 3/3               |

"Seeds completed" confirms every row is the full 3/3 (or 1/1 for A, which has
no seed variation — deterministic, no sampling — and is therefore computed
once and shared). Search runtime is 0 for A and C because neither performs a
dev-set selection search (A is fixed; C's demonstrations are sampled once per
seed with no selection step). Full per-seed breakdowns, confusion matrices, and
per-class precision/recall/F1 are in `results/evaluation_report.md`.

## 3. D - A / D - B / D - C (percentage points, macro-F1)

| Comparison | Source | Target |
|---|---|---|
| D - A | +1.80 pp | -27.40 pp |
| D - B | +1.63 pp | -27.80 pp |
| D - C | +0.78 pp | -5.92 pp |

D's bounded instruction+demonstration search improved in-domain macro-F1 by a
small margin (+1.80 pp over the fixed baseline)
at a search cost of 3134s across 3 seeds (vs 40s for B's
much smaller instruction-only search — roughly 78x more
expensive). On the target domain, D's macro-F1 is -27.40 pp
relative to the zero-cost fixed baseline A — a clear regression, not an
improvement.

## 4. Did source-side optimization help cross-domain transfer? Was it worth the cost?

No. D (and C, which needs no search at all) both score substantially worse on
target than the fixed baseline A, while D is also the most expensive condition
to produce. **Before attributing this to "domain transfer failure," see the
invalid-output diagnosis below** — a large share of the apparent score drop is
driven by a specific, identifiable output pattern, not generic confusion or
degraded sentiment reasoning.

### Invalid-output diagnosis (full detail in `results/invalid_output_diagnostics.md`)

**2148 of 2164 invalid outputs
(99.3%) are the single,
complete, 2-token word "neutral"** — not truncation, not malformed text, not an
empty response, not a chat-template or thinking-mode artifact. The model is
making a deliberate three-way judgment (positive/negative/neutral) that falls
outside the binary label space this experiment enforces by design (see
`src/prompt_templates.py` and `src/evaluate.py:strict_parse`).

In short: fewer than 0.2% of all invalid outputs across the entire experiment
show any truncation signature, and zero show a chat-template or thinking-mode
defect. The dominant failure mode — the model answering "neutral" — is
concentrated almost entirely in the target domain and almost entirely in
conditions that show demonstrations (C, D): the target-domain invalid rate is
~0.2% when no demonstrations are shown (A, B) versus ~40% when they are (C, D).
This means the target-score decline for C and D is real model behavior
triggered by a specific, identifiable condition (demonstration presence on
terse/factual financial text), not an unexplained "domain transfer" effect in
the abstract, and not a parsing, truncation, or template bug. The causal
mechanism (why demonstrations specifically trigger this) is not fully isolated
— see the diagnosis file's Section 6 for what remains an open question.

## 5. Limitations

- 3 seeds is not enough to claim statistical significance; standard deviations
  above describe the observed spread across 3 seeds only.
- Condition D is a bounded random-search joint prompt baseline (<=80
  instruction x demonstration-set combinations per seed) — not MIPRO, EASE, or
  a gradient-based optimizer.
- Financial PhraseBank's `sentences_allagree` is evaluated as binary sentiment
  (neutral excluded from the gold label set) even though the model's own
  "neutral" judgments show it treats some of these sentences as better
  described by a third class the gold labels don't offer it.
- The target dataset has now been inspected (via this diagnosis) in a way that
  goes beyond pure held-out final scoring. Any further work that uses
  Financial PhraseBank to guide method or prompt choices (rather than only
  final scoring of an already-frozen configuration) must be labeled
  exploratory, not a fresh held-out evaluation.

## 6. Justified next experiment

See `results/invalid_output_diagnostics.md` Section 7 for the full proposal: a
controlled three-way-label variant (positive/negative/neutral), applied
identically to every condition and domain, to isolate whether "neutral" is a
genuine signal the forced-binary framework is suppressing, versus an artifact
of this specific instruction wording. This is NOT launched by this report.

## Reproducibility

```
python src/prepare_data.py
python -m pytest tests/ -q
PYTHONPATH=src python src/run_experiment.py --config configs/pilot.yaml
PYTHONPATH=src python src/evaluate_experiment.py --config configs/pilot.yaml
PYTHONPATH=src python src/diagnose_invalid_outputs.py
PYTHONPATH=src python src/generate_report.py
```
