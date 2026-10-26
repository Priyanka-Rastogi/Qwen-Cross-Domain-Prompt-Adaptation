# Qwen3-8B cross-domain prompt adaptation (IMDb -> Financial PhraseBank)

Source-only prompt adaptation with a frozen Qwen3-8B: learn a natural-language
instruction and/or select labeled in-context demonstrations from IMDb movie
reviews, then evaluate, without any further adaptation, on Financial PhraseBank
financial sentences. No model weights are ever updated.

This is a controlled baseline for later research, not a claim that
joint instruction/demonstration search is novel (see Wan et al., EASE,
MIPRO/MIPROv2 for prior work this builds on).

## What this is / is not

- **Is**: in-context prompt adaptation (instruction wording + few-shot
  demonstration selection) under frozen weights, selected using only
  source-domain (IMDb) labels, then scored once on held-out source and target
  (financial) test sets.
- **Is not**: fine-tuning, LoRA/QLoRA, adapter or soft-prompt training,
  backpropagation of any kind, or training of a separate demonstration-selector
  model. See `src/model_backend.py` for the explicit frozen-parameter assertion.

## Hardware / environment this was built and run on

- GPU: 1x NVIDIA B200 MIG slice (`1g.45gb`, ~45GB VRAM; a second, unrelated
  process on the same host used part of that slice throughout this project —
  budgets below account for that).
- CUDA 12.8, driver 570.124.06.
- Python 3.12.3, project virtualenv at `.venv/` (`--system-site-packages`, reusing
  the host's NVIDIA-built PyTorch rather than reinstalling it).
- `pip install` in this environment is constrained by a global
  `PIP_CONSTRAINT` file; installing `datasets` required overriding it locally
  (`PIP_CONSTRAINT= pip install datasets==3.6.0`) — see `requirements.txt` for
  exact pinned versions actually installed.
- Do not assume these exact versions/paths on a different machine; re-run
  Section "Environment verification" below first.

## Model

`Qwen/Qwen3-8B` (Hugging Face Transformers), loaded once and reused for both
instruction proposal and classification. BF16, `device_map="cuda"`,
`model.eval()`, every parameter `requires_grad=False` (asserted in code),
`torch.inference_mode()` around all generation. Chat template applied with
`enable_thinking=False`. The resolved model revision is recorded in every
`results/metrics/*.json` cost block (`software_config.model_revision`) and
resolved at load time by `src/model_backend.py:_resolve_revision`. Dataset
revisions are recorded separately in `artifacts/manifests/data_manifest.json`.

## Data

See `data/DATASET_README.md` for full provenance, licensing, checksums, and
per-file row counts. Summary:

| File | Role | N | Balance |
|---|---|---|---|
| `data/source_train.csv` | demo pool + instruction proposal (IMDb train) | 200 | 100/100 |
| `data/source_validation.csv` | prompt selection (IMDb train, disjoint) | 100 | 50/50 |
| `data/source_test.csv` | final in-domain eval (IMDb test) | 500 | 250/250 |
| `data/target_test.csv` | final cross-domain eval (Financial PhraseBank, binary) | 873 | 570 pos / 303 neg |

IMDb's own `ClassLabel` metadata (`['neg','pos']`) was read explicitly and
confirmed to already match this project's `negative=0, positive=1` convention
before use. Financial PhraseBank's HF repo ships a legacy `datasets`
loading script that modern `datasets` versions cannot execute; `prepare_data.py`
instead downloads the original author archive and parses
`Sentences_AllAgree.txt` directly (latin-1, `text@label` lines) — see
`data/DATASET_README.md` for the exact archive/member checksums.

## Isolation protocol

- Source-training examples (`source_train.csv`) may guide instruction proposal
  and supply demonstrations.
- Source-development labels (`source_validation.csv`) may guide instruction/
  demonstration selection (macro-F1).
- Source-test and target-test labels are used only for final scoring, after
  every prompt configuration is frozen (`artifacts/selected_prompts/*.json`,
  written with a content hash before final evaluation runs).
- Target text is never shown during proposal or selection, and demonstration
  retrieval is never query-dependent (no retrieval against target text at all —
  demonstrations are drawn once per seed from the source pool and reused for
  every target example).
- "Unseen target domain" means excluded from this experiment's development
  process, not absent from Qwen3-8B's pretraining data (an unverifiable claim
  this report does not make).

## Experimental conditions

- **A**: fixed baseline instruction, no demonstrations. Deterministic (no
  sampling) — computed once and shared across all seeds via the evaluation
  cache, flagged `seed_invariant: true` in `results/metrics/A_final.json`.
- **B**: up to 8 source-proposed, domain-neutral instructions (baseline
  included), best selected by source-dev macro-F1. No demonstrations.
- **C**: baseline instruction + 4 demonstrations (2 pos / 2 neg) randomly
  sampled from the demo pool per seed, fixed ordering rule, not selected via
  dev performance.
- **D**: bounded random-search joint baseline — same instruction pool as B x
  up to 10 balanced demonstration sets (<= 80 combinations/seed, C's set
  included), best combination selected by source-dev macro-F1. This is a
  bounded random-search baseline, not MIPRO, EASE, or a gradient-based method.

Full details (truncation policy, strict parser, tie-break rule, caching, seeds):
see module docstrings in `src/prompt_templates.py`, `src/evaluate.py`,
`src/search_prompts.py`, `src/cache.py`.

## Project structure

```
configs/            smoke.yaml (tiny source-only check), pilot.yaml (full 3-seed run)
src/
  prepare_data.py       download + align + split + export data/*.csv
  text_utils.py          shared normalization/hashing
  model_backend.py      frozen Qwen3-8B load + batched generation
  prompt_templates.py   shared prompt builder + truncation policy
  propose_instructions.py  condition B/D instruction candidate generation
  search_prompts.py     demonstration sampling + selection tie-break
  evaluate.py            strict parser + cached batched classification
  metrics.py             accuracy/macro-F1/confusion with invalid handling
  cache.py               resumable JSONL evaluation cache
  run_experiment.py     orchestrates A-D across seeds, incremental + resumable
  generate_report.py    renders results/metrics/summary.json -> results/report.md
  score_labels.py        binary-label (positive/negative) conditional-likelihood
                          scoring: scoring rule, tokenization verification, tie-break
  run_binary_scoring.py  orchestrates binary-label scoring over the same frozen
                          A-D prompts/seeds as run_experiment.py
  verify_binary_scoring.py  independent audit of the scoring implementation
                             (live-model cross-checks, recomputed metrics, isolation)
  audit_comparison_table.py  from-scratch recomputation of both interfaces' metrics
                              directly from prediction CSVs (comparison-table audit)
tests/               pytest unit tests (no GPU required)
artifacts/
  manifests/           data_manifest.json (splits, hashes, provenance)
  instructions/        per-seed proposal prompts + raw generations + accepted pool
  selected_prompts/    frozen B/C/D configs (hash + timestamp) before final eval
  cache/                eval_cache.jsonl / smoke_cache.jsonl (gitignored)
results/
  predictions/          per-example predictions (one row per classified example)
  metrics/              per-condition/seed metrics + summary.json
  logs/                 run_log.jsonl, phase_costs.json
  report.md             human-readable final report (free generation)
  binary_label_scoring/  predictions/metrics/logs for the binary-scoring interface,
                          plus metric_verification.md (independent audit) and
                          near_tie_sensitivity.csv (batch-order-noise bound)
  comparison_audit/     from-scratch re-derivation of both interfaces' metrics,
                         with per-seed TP/FP/FN tables, from audit_comparison_table.py
data/
  *.csv, DATASET_README.md, raw_extract/ (Financial PhraseBank raw archive)
```

## CLI commands

```bash
# 0. Environment setup (one-time)
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
PIP_CONSTRAINT= pip install "transformers>=4.51.0" "datasets==3.6.0" accelerate \
    scikit-learn pandas pyyaml pytest huggingface_hub

# 1. Environment verification
python -c "import torch, transformers, datasets; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, datasets.__version__)"

# 2. Data preparation (downloads IMDb + Financial PhraseBank, writes data/*.csv)
python src/prepare_data.py

# 3. Unit tests
python -m pytest tests/ -q

# 4. Source-only smoke test (tiny, no target eval)
PYTHONPATH=src python src/run_experiment.py --config configs/smoke.yaml

# 5. Full bounded pilot (3 seeds, source + target eval)
PYTHONPATH=src python src/run_experiment.py --config configs/pilot.yaml

# 6. Resume an interrupted run
PYTHONPATH=src python src/run_experiment.py --config configs/pilot.yaml --resume

# 7. Regenerate the report from existing results
PYTHONPATH=src python src/generate_report.py --results-dir results
```

## Binary-label scoring: a second prediction interface on the same frozen prompts

The experiment above (free generation + strict parsing, `src/evaluate.py`)
sometimes lets the model emit something other than "positive"/"negative"
(mostly "neutral"-like text under conditions C/D on the financial target
domain); those get recorded as invalid and counted against accuracy
(`src/metrics.py`, `INVALID_PRED = -1`). At the request to see results using
only the two labels positive/negative, `src/score_labels.py` +
`src/run_binary_scoring.py` implement a **second, separate prediction
interface** on the exact same frozen model, instructions, demonstrations, and
seeds already selected above (hashes re-verified against
`artifacts/selected_prompts/*.json` before scoring) -- no prompt re-search, no
fine-tuning, no new data. Instead of greedy decoding, each example's label is
the argmax of the two candidate labels' teacher-forced log-likelihoods under
the identical chat-template prompt (scoring rule, tokenization-boundary
verification, and deterministic tie-break documented in
`src/score_labels.py`'s module docstring). Because the prediction is always
one of exactly two labels by construction, there is no invalid-output
category under this interface.

**Results** (accuracy / macro-F1; B/C/D are mean over 3 seeds, A is
seed-invariant; full per-seed tables in `results/binary_label_scoring/` and
`results/comparison_audit/`):

| Condition | Domain | Free generation | Binary-label scoring |
|---|---|---|---|
| A | source (movie) | 0.922 / 0.923 | 0.924 / 0.924 |
| A | target (finance) | 0.978 / 0.976 | 0.978 / 0.976 |
| B | source | 0.923 / 0.924 | 0.924 / 0.924 |
| B | target | 0.981 / 0.980 | 0.981 / 0.980 |
| C | source | 0.931 / 0.933 | 0.935 / 0.935 |
| C | target | 0.614 / 0.761 | 0.921 / 0.917 |
| D | source | 0.932 / 0.941 | 0.948 / 0.948 |
| D | target | 0.527 / 0.702 | 0.952 / 0.948 |

**How to read this.** On the source domain, the two interfaces agree closely
everywhere. The large C/D target-domain gap is **not** evidence that
likelihood scoring makes the model better at the underlying task -- free
generation's low C/D target accuracy was driven almost entirely by a large
invalid-output rate (35%/46%, see `results/invalid_output_diagnostics.md`),
and binary scoring mechanically removes that escape hatch. The jump is
substantially attributable to resolving previously-invalid cases, not to a
change in the model's underlying label preferences on cases it was already
answering validly (confirmed directly: `full_scale_agreement_check` in
`results/binary_label_scoring/logs/` shows ~99.8-100% agreement between the
two interfaces on every example where free generation *did* already produce a
valid single-token answer). Finance-set analysis beyond these aggregate
numbers is exploratory, not confirmatory -- it was not used to select any
setting.

**Verification.** `results/binary_label_scoring/metric_verification.md`
documents a full independent audit: a real bug (missing `position_ids` for
left-padded batched scoring, which made scores batch-order-dependent) was
found and fixed before any results were reported, with the entire first
(buggy) run deleted and re-run from scratch -- see that file for the before/
after evidence. The audit also found and bounded a narrower, inherent bf16
batched-inference limitation (not a logic bug): ~1.4% of predictions sit
close enough to their decision boundary that the predicted label can depend
on batch composition, bounded at <=2.18 percentage points of accuracy on any
single condition/domain. `results/comparison_audit/` separately re-derives
every free-generation and binary-scoring number directly from the raw
prediction CSVs (not by calling `src/metrics.py`) and confirms exact
agreement with the saved results -- see `comparison_audit_report.md` for the
full TP/FP/FN breakdown per seed.

## Negative results are valid

This project does not assume source-side prompt optimization (B, D) improves
cross-domain (target) performance, or that it improves in-domain performance at
all. `results/report.md` reports whatever the three seeds actually produced,
including the case where optimizing on IMDb helps in-domain scores but hurts
(or does nothing for) financial-text scores.
