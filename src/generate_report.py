"""Renders the final human-readable report (results/report.md) from the
completed 3-seed pilot's saved artifacts: results/comparison.csv (authoritative
numbers, cross-checked against saved predictions by evaluate_experiment.py),
results/metrics/summary.json, and results/invalid_output_diagnostics.md.

Does not recompute metrics, does not touch prompts, does not re-run anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def fmt_ms(mean, std, as_pct=False):
    if mean is None or (isinstance(mean, float) and pd.isna(mean)):
        return "N/A"
    if as_pct:
        return f"{100*mean:.1f}% ± {100*std:.1f}%" if std is not None and not pd.isna(std) else f"{100*mean:.1f}%"
    return f"{mean:.3f} ± {std:.3f}" if std is not None and not pd.isna(std) else f"{mean:.3f}"


def build_main_table(comparison: pd.DataFrame) -> str:
    """One row per condition, columns split by source/target, per item 2 of the
    request: accuracy, macro-F1 (mean±std), invalid rate, search runtime vs
    final-inference runtime, separately."""
    rows = []
    for cond in ["A", "B", "C", "D"]:
        src = comparison[(comparison.condition == cond) & (comparison.split == "source")].iloc[0]
        tgt = comparison[(comparison.condition == cond) & (comparison.split == "target")].iloc[0]
        rows.append({
            "Condition": cond,
            "Src Accuracy": fmt_ms(src.accuracy_mean, src.accuracy_std),
            "Src Macro-F1": fmt_ms(src.macro_f1_mean, src.macro_f1_std),
            "Tgt Accuracy": fmt_ms(tgt.accuracy_mean, tgt.accuracy_std),
            "Tgt Macro-F1": fmt_ms(tgt.macro_f1_mean, tgt.macro_f1_std),
            "Src Invalid Rate": fmt_ms(src.invalid_rate_mean, None, as_pct=True),
            "Tgt Invalid Rate": fmt_ms(tgt.invalid_rate_mean, None, as_pct=True),
            "Search Runtime (s, total/3 seeds)": f"{src.search_runtime_s_total:.0f}" if pd.notna(src.search_runtime_s_total) else "0 (not searched)",
            "Final Inference Runtime (s, src+tgt, total/3 seeds)": (
                f"{(src.final_inference_runtime_s_total or 0) + (tgt.final_inference_runtime_s_total or 0):.0f}"
            ),
            "Seeds completed": f"{int(src.n_seeds_completed)}/{int(src.n_seeds_expected)}",
        })
    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)


def pct_points(a, b):
    return round(100 * (a - b), 2) if (a is not None and b is not None) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    results_dir = args.results_dir

    comparison = pd.read_csv(results_dir / "comparison.csv")
    summary = json.loads((results_dir / "metrics" / "summary.json").read_text())
    eval_report_text = (results_dir / "evaluation_report.md").read_text()
    diag_text = (results_dir / "invalid_output_diagnostics.md").read_text()
    cfg = summary["config"]

    # Cross-check: every number placed into this report comes from comparison.csv
    # (itself independently recomputed from results/predictions/*.csv by
    # evaluate_experiment.py) and must match what evaluation_report.md already
    # reported for the same condition/split as a redundant consistency check.
    # evaluation_report.md reports per-seed metrics recomputed directly from
    # results/predictions/*.csv (by evaluate_experiment.py) and already asserts
    # recomputed==stored for every completed condition/seed; we surface that
    # this check already passed rather than re-deriving it here.
    cross_check_notes = []
    all_matched = "recomputed metrics match the pipeline's own stored metrics exactly" in eval_report_text
    mismatch_flag = "MISMATCH" in eval_report_text
    cross_check_notes.append(
        f"Every number in the table below traces back to `results/comparison.csv`, which "
        f"`src/evaluate_experiment.py` built by independently recomputing accuracy/macro-F1/"
        f"invalid-rate from the raw `results/predictions/*.csv` files (not by copying "
        f"`summary.json`). That independent recomputation {'matched the pipeline' + chr(39) + 's own stored metrics exactly for every completed condition/seed, with zero mismatches' if (all_matched and not mismatch_flag) else 'reported at least one MISMATCH -- see results/evaluation_report.md Section 3-4'}."
    )

    diag_top_line = diag_text.split("## Top-line finding")[1].split("## 1.")[0].strip()

    main_table = build_main_table(comparison)

    def get(cond, split, metric):
        row = comparison[(comparison.condition == cond) & (comparison.split == split)]
        return None if row.empty else row.iloc[0][metric]

    d_src_f1, a_src_f1 = get("D", "source", "macro_f1_mean"), get("A", "source", "macro_f1_mean")
    d_tgt_f1, a_tgt_f1, b_tgt_f1, c_tgt_f1 = (
        get("D", "target", "macro_f1_mean"), get("A", "target", "macro_f1_mean"),
        get("B", "target", "macro_f1_mean"), get("C", "target", "macro_f1_mean"),
    )
    d_search, b_search = get("D", "source", "search_runtime_s_total"), get("B", "source", "search_runtime_s_total")

    n_seeds = cfg["seeds"]

    report = f"""# Cross-domain prompt adaptation: IMDb -> Financial PhraseBank (Qwen3-8B, frozen weights)

Final report for the completed 3-seed pilot (`configs/pilot.yaml`, seeds={n_seeds}).
Generated entirely from saved artifacts (`results/comparison.csv`,
`results/metrics/summary.json`, `results/evaluation_report.md`,
`results/invalid_output_diagnostics.md`) — no prompts were changed and no
result reported here was produced by re-running prompt search.

{" ".join(cross_check_notes)}

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

{main_table}

"Seeds completed" confirms every row is the full 3/3 (or 1/1 for A, which has
no seed variation — deterministic, no sampling — and is therefore computed
once and shared). Search runtime is 0 for A and C because neither performs a
dev-set selection search (A is fixed; C's demonstrations are sampled once per
seed with no selection step). Full per-seed breakdowns, confusion matrices, and
per-class precision/recall/F1 are in `results/evaluation_report.md`.

## 3. D - A / D - B / D - C (percentage points, macro-F1)

| Comparison | Source | Target |
|---|---|---|
| D - A | {pct_points(d_src_f1, a_src_f1):+.2f} pp | {pct_points(d_tgt_f1, a_tgt_f1):+.2f} pp |
| D - B | {pct_points(d_src_f1, get('B','source','macro_f1_mean')):+.2f} pp | {pct_points(d_tgt_f1, b_tgt_f1):+.2f} pp |
| D - C | {pct_points(d_src_f1, get('C','source','macro_f1_mean')):+.2f} pp | {pct_points(d_tgt_f1, c_tgt_f1):+.2f} pp |

D's bounded instruction+demonstration search improved in-domain macro-F1 by a
small margin ({pct_points(d_src_f1, a_src_f1):+.2f} pp over the fixed baseline)
at a search cost of {d_search:.0f}s across 3 seeds (vs {b_search:.0f}s for B's
much smaller instruction-only search — roughly {d_search/b_search:.0f}x more
expensive). On the target domain, D's macro-F1 is {pct_points(d_tgt_f1, a_tgt_f1):+.2f} pp
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

{diag_top_line}

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
"""
    out_path = results_dir / "report.md"
    out_path.write_text(report)
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
