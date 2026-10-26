# Comparison-table audit

Independent, from-scratch recomputation of every free-generation and binary-label-scoring metric from the raw per-row prediction CSVs, done without calling `src/metrics.py` or reusing any previously computed JSON, per the request to audit the comparison table before sharing it.

**Code path responsible for the existing saved free-generation numbers:** `src/metrics.py:compute_metrics`, called from `src/evaluate.py` during the original pilot run and saved to `results/metrics/{condition}_final.json` / aggregated into `results/comparison.csv` via `src/metrics.py:aggregate_across_seeds` (which uses **population** std, ddof=0).

**Result of independent recomputation: 20/20 free-generation (condition, seed, domain) metric sets matched the saved `results/metrics/*_final.json` files exactly (to floating-point precision, 0 mismatches found).** This means the originally reported finance numbers (C: accuracy 0.614, macro-F1 0.761; D: accuracy 0.527, macro-F1 0.702 -- both from `results/comparison.csv`, which averages 3 seeds) are confirmed correct under the documented invalid-handling rule (invalid predictions retained, never filtered, count toward FN for their true class and toward the incorrect/accuracy denominator, never count as FP for either real class). No discrepancy was found; no correction to the original comparison table is needed.
## Free-generation finance (and source) results: independent recomputation

Methodology: for every prediction CSV under `results/predictions/`, `pred == -1` (the project's documented invalid-output sentinel, see `src/metrics.py` docstring) is kept as its own value. No row is dropped. For each class c in {0,1}: TP = (true==c & pred==c), FP = (true!=c & pred==c), FN = (true==c & pred!=c) -- note FN explicitly includes rows where pred==-1, so every invalid prediction on a true-class-c example reduces that class's recall, exactly as the spec requires. Accuracy counts every row in the denominator; invalid predictions can never be 'correct'. This is a from-scratch reimplementation (does not call `src/metrics.py`), so agreement with the saved values is a genuine independent check.


### Per-seed table: condition C and D, finance (target) domain

| Condition | Seed | n | correct | TP(neg) | FP(neg) | FN(neg) | support(neg) | TP(pos) | FP(pos) | FN(pos) | support(pos) | invalid | accuracy | macro-F1 | matches saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C | 0 | 873 | 499 | 262 | 28 | 41 | 303 | 237 | 0 | 333 | 570 | 346 | 0.5716 | 0.7355 | True |
| C | 1 | 873 | 606 | 256 | 26 | 47 | 303 | 350 | 1 | 220 | 570 | 240 | 0.6942 | 0.8176 | True |
| C | 2 | 873 | 502 | 260 | 37 | 43 | 303 | 242 | 0 | 328 | 570 | 334 | 0.5750 | 0.7314 | True |
| D | 0 | 873 | 453 | 216 | 16 | 87 | 303 | 237 | 0 | 333 | 570 | 404 | 0.5189 | 0.6974 | True |
| D | 1 | 873 | 493 | 232 | 17 | 71 | 303 | 261 | 0 | 309 | 570 | 363 | 0.5647 | 0.7344 | True |
| D | 2 | 873 | 435 | 179 | 8 | 124 | 303 | 256 | 0 | 314 | 570 | 430 | 0.4983 | 0.6752 | True |

### Aggregated across seeds (C, D; both domains) -- mean, sample std (ddof=1), population std (ddof=0)

| Condition | Domain | Accuracy mean | Acc sample std | Acc population std | Macro-F1 mean | F1 sample std | F1 population std | Original comparison.csv accuracy | Original comparison.csv macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| A (no seed) | source/movie | 0.9220 | - | - | 0.9227 | - | - | 0.9220 | 0.9227 |
| A (no seed) | target/finance | 0.9782 | - | - | 0.9763 | - | - | 0.9782 | 0.9763 |
| B | source/movie | 0.9227 | 0.0012 | 0.0009 | 0.9244 | 0.0012 | 0.0010 | 0.9227 | 0.9244 |
| B | target/finance | 0.9805 | 0.0020 | 0.0016 | 0.9804 | 0.0008 | 0.0006 | 0.9805 | 0.9804 |
| C | source/movie | 0.9313 | 0.0168 | 0.0137 | 0.9329 | 0.0163 | 0.0133 | 0.9313 | 0.9329 |
| C | target/finance | 0.6136 | 0.0698 | 0.0570 | 0.7615 | 0.0487 | 0.0397 | 0.6136 | 0.7615 |
| D | source/movie | 0.9320 | 0.0053 | 0.0043 | 0.9408 | 0.0023 | 0.0018 | 0.9320 | 0.9408 |
| D | target/finance | 0.5273 | 0.0340 | 0.0278 | 0.7023 | 0.0299 | 0.0244 | 0.5273 | 0.7023 |
## Binary-label scoring: independent re-verification with three-seed standard deviations

Recomputed directly from `results/binary_label_scoring/predictions/*.csv` (independent of `results/binary_label_scoring/metrics/*_final.json` and of the earlier `verify_binary_scoring.py` audit pass -- this is a fresh, third computation). Invalid-by-construction count across all rows: 0 (binary scoring forces every prediction into {negative, positive}, so this must be 0).

| Condition | Domain | Accuracy mean | Acc sample std (ddof=1) | Acc population std (ddof=0) | Macro-F1 mean | F1 sample std (ddof=1) | F1 population std (ddof=0) |
|---|---|---|---|---|---|---|---|
| A (no seed) | source/movie | 0.9240 | - | - | 0.9239 | - | - |
| A (no seed) | target/finance | 0.9782 | - | - | 0.9763 | - | - |
| B | source/movie | 0.9240 | 0.0000 | 0.0000 | 0.9239 | 0.0000 | 0.0000 |
| B | target/finance | 0.9813 | 0.0013 | 0.0011 | 0.9796 | 0.0014 | 0.0012 |
| C | source/movie | 0.9353 | 0.0170 | 0.0139 | 0.9352 | 0.0172 | 0.0141 |
| C | target/finance | 0.9210 | 0.0235 | 0.0192 | 0.9165 | 0.0237 | 0.0194 |
| D | source/movie | 0.9480 | 0.0053 | 0.0043 | 0.9480 | 0.0053 | 0.0043 |
| D | target/finance | 0.9515 | 0.0037 | 0.0030 | 0.9475 | 0.0036 | 0.0029 |