# Binary-label scoring: independent metric/scoring verification

Read-only audit of `src/score_labels.py` and `src/run_binary_scoring.py` and of the already-completed run's saved outputs under `results/binary_label_scoring/`. Nothing in `results/binary_label_scoring/predictions` or `metrics` was changed or recomputed-and-overwritten by this audit; `source_test.csv`/`target_test.csv` labels were used only to SCORE already-produced predictions, never to influence any setting.

## 1. Completeness

All 10 expected configurations (A + {B,C,D} x seeds [0, 1, 2]) have both source_test and target_test predictions and a metrics JSON. Row counts match `data/source_test.csv` (500) and `data/target_test.csv` (873) exactly for every file.

## 2. Scoring code inspection

- **Answer strings:** the two candidate labels are the literal lowercase strings `"negative"` and `"positive"` (`src/score_labels.py:LABELS`), matching exactly what `src/evaluate.py:strict_parse` accepts as valid in the original free-generation pipeline.
- **Token IDs/lengths in the live chat-template context:** empirically checked across all 13730 saved prediction rows (every condition, every seed, both domains) via the `negative_form_used`/`positive_form_used`/`*_continuation_tokens` columns each row already records. Exactly 1 distinct (form, form, n_tokens, n_tokens) combination was observed: `[('negative', 'positive', 1, 1)]`. This confirms both labels tokenize as a single token with no leading space in this prompt's exact tail (`...Sentiment:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`), identically regardless of instruction wording or demonstration content -- expected, since the chat-template suffix after `add_generation_prompt=True` does not depend on either.
- **Shifted causal logits:** `score_batch` requests `logits_to_keep = max_cont_len + 1` so the model returns logits for the trailing `K` positions of the (left-padded) sequence; for a row with `cont_len` continuation tokens, continuation token `k` (0-indexed) is scored against the logits at `col = K - (cont_len - k + 1)`, i.e. the position immediately BEFORE that token -- standard shifted next-token scoring. This indexing was independently re-derived by hand and confirmed against a completely different code path in section 3 below.
- **Attention/padding masks and answer-token boundaries:** sequences are left-padded (`pad` on the left, matching `model_backend.py`'s generation convention) with a matching `attention_mask`. **A real bug was found and fixed here -- see the correction note below.**
- **Summed vs. length-normalized log-probabilities:** scores are SUMMED (`total_logprob += ...` over all continuation tokens), not divided by length. This makes no difference in practice because both candidate labels are always exactly 1 token in this prompt context (confirmed above) -- a length-normalization divisor of 1 is a no-op.
- **Special tokens:** `prompt_str` is the literal text produced by `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`, which already contains the chat special tokens (`<|im_start|>`, `<|im_end|>`, etc.) as literal text. It is then encoded with `add_special_tokens=False`, which means no EXTRA token (e.g. a redundant BOS) is added on top -- the template's own special tokens are included exactly once, via the tokenizer's normal vocabulary, which is the standard way to tokenize a pre-rendered chat template.
- **Only input text passed to the model:** confirmed by direct inspection of every `build_prompt(...)` call site in `src/run_binary_scoring.py` -- all pass `ex["text"]`, never `ex["label"]`. Call sites: ['        built = build_prompt(backend.tokenizer, instruction, demonstrations, ex["text"])']

### The correction referred to as "corrected scoring"

**Bug:** `score_batch` originally called `backend.model(input_ids=..., attention_mask=..., use_cache=False, logits_to_keep=...)` without passing `position_ids`. For a raw `forward()` call (as opposed to `model.generate()`, which derives correct position ids from `attention_mask` internally via `prepare_inputs_for_generation`), omitting `position_ids` defaults to a plain `arange()` over the padded sequence dimension. With LEFT padding, this assigns each row's real tokens a RoPE position offset equal to however much padding precedes them *in that specific batch* -- which, contrary to the naive expectation that RoPE's relative-position property would cancel out a constant per-row offset, was empirically found to change the computed log-likelihoods: the SAME prompt scored in isolation (batch of 1) gave a different score than when batched alongside other differently-sized examples. Concretely, example `fpb_allagree_340` scored `negative=-0.474, positive=-0.974` (favoring negative) in one batch composition but `negative=-0.826, positive=-0.576` (favoring positive, flipping the prediction) in another, for the identical prompt.

**Fix:** explicitly compute `position_ids = attention_mask.long().cumsum(-1) - 1` (masked to 0 where `attention_mask == 0`) and pass it into every forward call, so every row's real tokens occupy positions `0, 1, 2, ...` regardless of batch composition or padding amount.

**What it affected:** this bug was present during the FIRST full run of `src/run_binary_scoring.py`, which showed an entirely-free, full-scale self-consistency check (every example whose original free-generation output was already a valid single-token `positive`/`negative` must, by construction, also win the two-way likelihood comparison) passing at only ~99.4-99.9% instead of the expected ~100%, with 11 explicit violations found across ~9175 checkable cases. **Before producing any report from that run, the bug was found, the fix applied, and `artifacts/cache/binary_label_scoring_cache.jsonl` plus the entire `results/binary_label_scoring/` directory from that first run were deleted and the full A-D sweep (all 10 configurations, all 13730 examples) was re-run from scratch under the corrected code.** Every file currently present under `results/binary_label_scoring/` was produced by the corrected run; no output from the buggy run remains on disk. This audit verifies the corrected run.

## 3. Live model validation (not unit tests)

### 3a. Independent cross-check vs. transformers' own built-in `labels=` cross-entropy loss
Ran **24 fresh forward passes** on the live frozen model (not reading any cached/previously-computed score), sampled across conditions A, B, C, D and both domains, comparing `score_batch`'s optimized `logits_to_keep`+manual-gather computation against a COMPLETELY SEPARATE code path: a single unpadded sequence scored via `model(input_ids=..., labels=...).loss` (no `logits_to_keep`, no manual gather, no padding at all). Max absolute difference = `0.000000` (tolerance 0.01) -> **PASS**. This confirms the shifted-logits indexing math in `score_batch` is correct when there is no padding/batching involved.

### 3b/3c. A SECOND issue found during this audit: batch-order floating-point noise
While building the cross-check above, scoring the SAME prompt in different batch compositions (different padding amounts/positions) produced DIFFERENT raw scores -- e.g. one sampled example showed a difference of up to 0.67 log-prob units between an isolated (batch-of-1) score and a batched score. This is **not** the `position_ids` bug described above (which is fixed and re-verified in section 3a); it is ordinary bf16 floating-point non-associativity in GPU batched-matmul kernels, which is expected to vary slightly with batch shape for ANY batched bf16 neural-network inference, not specific to this codebase. For the vast majority of predictions this is harmless, because the winning label's score is overwhelmingly higher than the losing label's (median |margin| across all 13730 predictions is ~16 log-prob units) -- noise of ~0.1-0.7 units cannot flip a decision with that much headroom.

**However**, this DOES matter for near-tie examples. Pulling the **30 actual closest-margin examples** from the completed run's own saved predictions and rescoring each one under 4 different batch compositions (isolated, plus 3 batches of varying size/position): **16 of 30 (53%) had their PREDICTED LABEL flip** depending on which other examples shared their batch -- i.e. for these specific examples, "which label was predicted" is not a reproducible function of (model, prompt) under the current batched implementation; it also depends on an arbitrary, non-semantic factor (batch composition).

**Quantified, bounded impact on the headline numbers** (worst case: every example with |margin| < 1.0 -- the empirically observed noise magnitude -- flips away from its current prediction; in practice flips go in both directions, so the realistic impact is well below this bound, but this is the auditable ceiling):


| Condition | Domain | n | accuracy | n within noise band | worst-case accuracy | max pp swing |
|---|---|---|---|---|---|---|
| A | source_test | 500 | 0.9240 | 4 | 0.9200 | 0.40 pp |
| A | target_test | 873 | 0.9782 | 0 | 0.9782 | 0.00 pp |
| B_seed0 | source_test | 500 | 0.9240 | 3 | 0.9220 | 0.20 pp |
| B_seed0 | target_test | 873 | 0.9805 | 5 | 0.9771 | 0.34 pp |
| B_seed1 | source_test | 500 | 0.9240 | 6 | 0.9200 | 0.40 pp |
| B_seed1 | target_test | 873 | 0.9828 | 3 | 0.9817 | 0.11 pp |
| B_seed2 | source_test | 500 | 0.9240 | 3 | 0.9220 | 0.20 pp |
| B_seed2 | target_test | 873 | 0.9805 | 5 | 0.9771 | 0.34 pp |
| C_seed0 | source_test | 500 | 0.9160 | 4 | 0.9140 | 0.20 pp |
| C_seed0 | target_test | 873 | 0.9198 | 23 | 0.9061 | 1.37 pp |
| C_seed1 | source_test | 500 | 0.9420 | 7 | 0.9340 | 0.80 pp |
| C_seed1 | target_test | 873 | 0.9450 | 17 | 0.9324 | 1.26 pp |
| C_seed2 | source_test | 500 | 0.9480 | 8 | 0.9340 | 1.40 pp |
| C_seed2 | target_test | 873 | 0.8981 | 39 | 0.8763 | 2.18 pp |
| D_seed0 | source_test | 500 | 0.9500 | 7 | 0.9400 | 1.00 pp |
| D_seed0 | target_test | 873 | 0.9542 | 10 | 0.9462 | 0.80 pp |
| D_seed1 | source_test | 500 | 0.9520 | 6 | 0.9460 | 0.60 pp |
| D_seed1 | target_test | 873 | 0.9473 | 16 | 0.9347 | 1.26 pp |
| D_seed2 | source_test | 500 | 0.9420 | 5 | 0.9400 | 0.20 pp |
| D_seed2 | target_test | 873 | 0.9530 | 18 | 0.9381 | 1.49 pp |

Worst case across every condition/seed/domain: **2.18 percentage points** (C_seed2/target_test, which also has the most near-tie examples -- consistent with conditions C/D generally producing lower-confidence predictions on the target domain than A/B). This is a real, quantified limitation of the current (batched, bf16) implementation, not a logic error in the scoring rule itself (section 3a).

## 3d. Frozen-model check

- Trainable parameters: **0** / 8190735360 total.
- `model.training` flag: **False** (must be False).
- Training-related calls found in scoring source (`.backward(`, `requires_grad_(True`, `torch.optim`, `.step()`): **none**.

## 4/5/6. Per-run independently recomputed metrics

| Condition | Seed | Domain | n | correct | accuracy | macro-F1 | support(neg,pos) | matches saved JSON | ties | isolation issues |
|---|---|---|---|---|---|---|---|---|---|---|
| A | - | source/movie | 500 | 462 | 0.9240 | 0.9239 | (250,250) | True | 0 | none |
| A | - | target/finance | 873 | 854 | 0.9782 | 0.9763 | (303,570) | True | 0 | none |
| B | 0 | source/movie | 500 | 462 | 0.9240 | 0.9239 | (250,250) | True | 0 | none |
| B | 0 | target/finance | 873 | 856 | 0.9805 | 0.9788 | (303,570) | True | 2 | none |
| B | 1 | source/movie | 500 | 462 | 0.9240 | 0.9239 | (250,250) | True | 1 | none |
| B | 1 | target/finance | 873 | 858 | 0.9828 | 0.9813 | (303,570) | True | 0 | none |
| B | 2 | source/movie | 500 | 462 | 0.9240 | 0.9239 | (250,250) | True | 0 | none |
| B | 2 | target/finance | 873 | 856 | 0.9805 | 0.9788 | (303,570) | True | 2 | none |
| C | 0 | source/movie | 500 | 458 | 0.9160 | 0.9156 | (250,250) | True | 0 | none |
| C | 0 | target/finance | 873 | 803 | 0.9198 | 0.9153 | (303,570) | True | 3 | none |
| C | 1 | source/movie | 500 | 471 | 0.9420 | 0.9420 | (250,250) | True | 0 | none |
| C | 1 | target/finance | 873 | 825 | 0.9450 | 0.9408 | (303,570) | True | 2 | none |
| C | 2 | source/movie | 500 | 474 | 0.9480 | 0.9480 | (250,250) | True | 1 | none |
| C | 2 | target/finance | 873 | 784 | 0.8981 | 0.8934 | (303,570) | True | 4 | none |
| D | 0 | source/movie | 500 | 475 | 0.9500 | 0.9500 | (250,250) | True | 1 | none |
| D | 0 | target/finance | 873 | 833 | 0.9542 | 0.9504 | (303,570) | True | 1 | none |
| D | 1 | source/movie | 500 | 476 | 0.9520 | 0.9520 | (250,250) | True | 0 | none |
| D | 1 | target/finance | 873 | 827 | 0.9473 | 0.9435 | (303,570) | True | 3 | none |
| D | 2 | source/movie | 500 | 471 | 0.9420 | 0.9420 | (250,250) | True | 1 | none |
| D | 2 | target/finance | 873 | 832 | 0.9530 | 0.9485 | (303,570) | True | 2 | none |

## 5. Cross-condition isolation checks

- Identical source_test id set used by every condition/seed: **True**.
- Identical target_test id set used by every condition/seed: **True**.
- Demonstration ids (20 unique, across all B/C/D x seed manifests) overlapping source_test: `[]`; overlapping target_test: `[]` -> **clean**.
- No example is dropped from any denominator (row counts match the full source_test.csv/target_test.csv exactly, see Section 1) and there is no "neutral" category or remapping anywhere in this scoring interface -- every prediction is forced into {negative, positive} by construction (argmax of exactly two scores), so `invalid_count` is 0 by construction for every run (confirmed in Section 4/5/6's recomputation, not merely assumed).
- Original prompts/demonstrations reused: every B/C/D instruction and demonstration set was reloaded from `artifacts/selected_prompts/{condition}_seed{seed}.json` and its `instruction_hash`/`demonstrations_content_hash` independently recomputed and matched (recomputation would raise if mismatched -- see `demos_from_manifest` / `get_condition_config` in `src/run_binary_scoring.py`); this audit re-verified the same hashes again from scratch (Section 3's live-validation sample construction).

## 7. Aggregation methodology

Per-seed metrics are computed FIRST (one accuracy/macro-F1 per condition x seed x domain, each over the full, unpooled 500-example source_test or 873-example target_test set), and ONLY THEN averaged across the 3 stochastic seeds -- predictions from different seeds are never concatenated into a single larger evaluation set (that would pseudo-replicate a fixed 500/873-example test set as if it were 1500/2619 independent examples). Condition A has no seed dependence and is reported once (not averaged into B/C/D, not duplicated 3x).

**Note on standard deviation convention:** the ORIGINAL project convention (`src/metrics.py:aggregate_across_seeds`, used throughout `results/report.md` and `results/comparison.csv`) uses `numpy`'s default **population** standard deviation (`ddof=0`). This verification report instead computes the **sample** standard deviation (`ddof=1`) as explicitly requested, since it is the more standard choice for treating 3 seeds as a small sample. With n=3, sample std = population std x sqrt(3/2) ~= 1.22x larger. Both are reported below; this is a convention difference, not a correctness bug in either report.


| Condition | Domain | Accuracy mean | Accuracy sample std (ddof=1) | Accuracy population std (ddof=0, original convention) | Macro-F1 mean | Macro-F1 sample std (ddof=1) |
|---|---|---|---|---|---|---|
| A (no seed) | source | 0.9240 | - | - | 0.9239 | - |
| A (no seed) | target | 0.9782 | - | - | 0.9763 | - |
| B | source | 0.9240 | 0.0000 | 0.0000 | 0.9239 | 0.0000 |
| B | target | 0.9813 | 0.0013 | 0.0011 | 0.9796 | 0.0014 |
| C | source | 0.9353 | 0.0170 | 0.0139 | 0.9352 | 0.0172 |
| C | target | 0.9210 | 0.0235 | 0.0192 | 0.9165 | 0.0237 |
| D | source | 0.9480 | 0.0053 | 0.0043 | 0.9480 | 0.0053 |
| D | target | 0.9515 | 0.0037 | 0.0030 | 0.9475 | 0.0036 |

When comparing two conditions' accuracy/F1, differences are reported in **percentage points (pp)** -- e.g. 0.95 vs 0.92 is "+3.0 pp", NOT "+3.3%" (a relative change would be `(0.95-0.92)/0.92 = +3.3%`). This report does not compute relative percentage changes since the base-rate values here (accuracies near 0.9-0.98) make pp and relative-% diverge enough to matter; pp is the more directly interpretable quantity for accuracy/F1 comparisons and is what `results/report.md` already uses.

## Verdicts

- **Scoring correctness: FAILED (see scoring_full_bitexact_determinism)** -- split into two sub-claims because they have different answers:
  - Scoring RULE implementation (shifted-logit indexing, tokenization boundary detection, tie-break rule, no-label-leakage, frozen model): **VERIFIED** (Sections 2, 3a, 3d, and the per-run recomputation in Section 4/5/6 all independently agree with the pipeline's own numbers).
  - Full bit-exact determinism of the BATCHED execution: **FAILED** -- 16 of 30 tested near-tie examples have a batch-composition-dependent predicted label (Section 3b), bounded to a worst-case 2.18 pp impact on any single condition/domain's accuracy (Section 3c). This affects the SAME 189 examples (1.4% of all 13730 predictions, |margin|<1.0) identified from the saved data, not an unbounded or unknown fraction.
- **Metric correctness: VERIFIED** -- every saved accuracy/macro-F1/per-class/confusion-matrix number was independently recomputed from the raw prediction CSVs and matched the pipeline's own `summary.json`/`*_final.json` to 9 decimal places, for all 10 configurations and both domains.
- **Data isolation: VERIFIED** -- identical source/target test id sets across all conditions, zero overlap between any demonstration id and any test id, only `ex["text"]` (never `ex["label"]`) passed into prompt construction, and every B/C/D instruction/demonstration set's hash independently re-verified against its frozen manifest.


These verdicts are based on live model forward passes (Section 3), full independent recomputation of every saved metric from the raw prediction CSVs (Section 4/5/6, matched to 9 decimal places against the pipeline's own `summary`/`*_final.json` files), and explicit id-level/hash-level isolation checks -- not on `tests/test_pipeline.py` passing, which was not used as evidence for this audit.

**Recommendation (not applied by this audit, per instruction to document before correcting):** if fully reproducible per-example predictions are required (e.g. for a paper claim about a SPECIFIC example's label), re-score with `batch_size=1` (fully isolated, no padding) -- this eliminates the batch-order noise source entirely at the cost of losing the batching speedup. It would not be expected to change any condition's headline accuracy/macro-F1 by more than the bounds in Section 3c, and would most likely change it by much less (the 189 affected examples' flips are not systematically biased toward incorrect).
