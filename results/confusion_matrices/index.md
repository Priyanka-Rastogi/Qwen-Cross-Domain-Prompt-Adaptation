# Confusion matrices: index

Built directly from already-saved prediction CSVs under `results/predictions/` (free generation) and `results/binary_label_scoring/predictions/` (binary-label scoring). No inference was re-run.

## Condition mapping

| Code | Meaning |
|---|---|
| A | zero-shot, fixed baseline instruction, no demonstrations (deterministic -- reported once) |
| B | optimized instruction only, no demonstrations (source-dev selected, 3 seeds) |
| C | fixed baseline instruction + 4 random demonstrations (2 pos / 2 neg, 3 seeds) |
| D | selected instruction + selected demonstrations (bounded joint search, 3 seeds) |

## Label mapping

- Gold rows: `positive` (label==1), `negative` (label==0).
- Predicted columns: `positive` (pred==1), `negative` (pred==0), `neutral` (pred==-1 AND raw_output, stripped+lowercased, is exactly `"neutral"`), `other_invalid` (pred==-1 and raw_output is any other unparseable string, e.g. `"mixed"`-family outputs).
- `neutral` is **recovered from the saved `raw_output` text**, not assumed from `pred == -1` -- see `classify_pred_category()` in `src/generate_confusion_matrices.py`, which mirrors the strict-normalization rule already used for diagnosis in `src/diagnose_invalid_outputs.py:classify_invalid` (same `strip().lower()` exact-match rule as the production parser `src/evaluate.py:strict_parse`).
- Binary-label scoring never has a `neutral`/`other_invalid` prediction by construction (forced two-way argmax) -- those columns are present for schema consistency and are always 0 for that interface, which this script verifies rather than assumes.

## Verification

All 40 generated matrices: every example preserved (matrix total == prediction-CSV row count), and each gold-row sum matches the gold-label class count in `data/source_test.csv` (250/250) or `data/target_test.csv` (303 negative / 570 positive). **Overall result: ALL CHECKS PASSED**.

No missing prediction files -- every (interface, condition, seed, domain) combination expected to exist was found.

## Aggregation note

For B/C/D (3 seeds each), per-seed matrices are saved individually (`{condition}_seed{N}_{domain}.csv/.png`). A separate, explicitly labeled `{condition}_mean_of_3seeds_{domain}.csv/.png` is also saved: this is the **cell-wise arithmetic mean of the three seeds' matrices** (so cell values are generally non-integer), NOT the three seeds pooled into one 1500/2619-example confusion matrix. The per-seed matrices remain the primary evaluation artifact; the mean matrix is a convenience summary only. Condition A has no seed dependence and only one matrix per (interface, domain) is produced.

## Per-seed accuracy (sanity cross-check against saved metrics)

| Interface | Condition | Seed | Domain | n | correct | accuracy | CSV | PNG |
|---|---|---|---|---|---|---|---|---|
| free_generation | A | - | source_movie | 500 | 461 | 0.9220 | `results/confusion_matrices/free_generation/A_source_movie.csv` | `results/confusion_matrices/free_generation/A_source_movie.png` |
| free_generation | A | - | target_finance | 873 | 854 | 0.9782 | `results/confusion_matrices/free_generation/A_target_finance.csv` | `results/confusion_matrices/free_generation/A_target_finance.png` |
| free_generation | B | 0 | source_movie | 500 | 461 | 0.9220 | `results/confusion_matrices/free_generation/B_seed0_source_movie.csv` | `results/confusion_matrices/free_generation/B_seed0_source_movie.png` |
| free_generation | B | 1 | source_movie | 500 | 462 | 0.9240 | `results/confusion_matrices/free_generation/B_seed1_source_movie.csv` | `results/confusion_matrices/free_generation/B_seed1_source_movie.png` |
| free_generation | B | 2 | source_movie | 500 | 461 | 0.9220 | `results/confusion_matrices/free_generation/B_seed2_source_movie.csv` | `results/confusion_matrices/free_generation/B_seed2_source_movie.png` |
| free_generation | B | 0 | target_finance | 873 | 855 | 0.9794 | `results/confusion_matrices/free_generation/B_seed0_target_finance.csv` | `results/confusion_matrices/free_generation/B_seed0_target_finance.png` |
| free_generation | B | 1 | target_finance | 873 | 858 | 0.9828 | `results/confusion_matrices/free_generation/B_seed1_target_finance.csv` | `results/confusion_matrices/free_generation/B_seed1_target_finance.png` |
| free_generation | B | 2 | target_finance | 873 | 855 | 0.9794 | `results/confusion_matrices/free_generation/B_seed2_target_finance.csv` | `results/confusion_matrices/free_generation/B_seed2_target_finance.png` |
| free_generation | C | 0 | source_movie | 500 | 456 | 0.9120 | `results/confusion_matrices/free_generation/C_seed0_source_movie.csv` | `results/confusion_matrices/free_generation/C_seed0_source_movie.png` |
| free_generation | C | 1 | source_movie | 500 | 470 | 0.9400 | `results/confusion_matrices/free_generation/C_seed1_source_movie.csv` | `results/confusion_matrices/free_generation/C_seed1_source_movie.png` |
| free_generation | C | 2 | source_movie | 500 | 471 | 0.9420 | `results/confusion_matrices/free_generation/C_seed2_source_movie.csv` | `results/confusion_matrices/free_generation/C_seed2_source_movie.png` |
| free_generation | C | 0 | target_finance | 873 | 499 | 0.5716 | `results/confusion_matrices/free_generation/C_seed0_target_finance.csv` | `results/confusion_matrices/free_generation/C_seed0_target_finance.png` |
| free_generation | C | 1 | target_finance | 873 | 606 | 0.6942 | `results/confusion_matrices/free_generation/C_seed1_target_finance.csv` | `results/confusion_matrices/free_generation/C_seed1_target_finance.png` |
| free_generation | C | 2 | target_finance | 873 | 502 | 0.5750 | `results/confusion_matrices/free_generation/C_seed2_target_finance.csv` | `results/confusion_matrices/free_generation/C_seed2_target_finance.png` |
| free_generation | D | 0 | source_movie | 500 | 463 | 0.9260 | `results/confusion_matrices/free_generation/D_seed0_source_movie.csv` | `results/confusion_matrices/free_generation/D_seed0_source_movie.png` |
| free_generation | D | 1 | source_movie | 500 | 467 | 0.9340 | `results/confusion_matrices/free_generation/D_seed1_source_movie.csv` | `results/confusion_matrices/free_generation/D_seed1_source_movie.png` |
| free_generation | D | 2 | source_movie | 500 | 468 | 0.9360 | `results/confusion_matrices/free_generation/D_seed2_source_movie.csv` | `results/confusion_matrices/free_generation/D_seed2_source_movie.png` |
| free_generation | D | 0 | target_finance | 873 | 453 | 0.5189 | `results/confusion_matrices/free_generation/D_seed0_target_finance.csv` | `results/confusion_matrices/free_generation/D_seed0_target_finance.png` |
| free_generation | D | 1 | target_finance | 873 | 493 | 0.5647 | `results/confusion_matrices/free_generation/D_seed1_target_finance.csv` | `results/confusion_matrices/free_generation/D_seed1_target_finance.png` |
| free_generation | D | 2 | target_finance | 873 | 435 | 0.4983 | `results/confusion_matrices/free_generation/D_seed2_target_finance.csv` | `results/confusion_matrices/free_generation/D_seed2_target_finance.png` |
| binary_scoring | A | - | source_movie | 500 | 462 | 0.9240 | `results/confusion_matrices/binary_scoring/A_source_movie.csv` | `results/confusion_matrices/binary_scoring/A_source_movie.png` |
| binary_scoring | A | - | target_finance | 873 | 854 | 0.9782 | `results/confusion_matrices/binary_scoring/A_target_finance.csv` | `results/confusion_matrices/binary_scoring/A_target_finance.png` |
| binary_scoring | B | 0 | source_movie | 500 | 462 | 0.9240 | `results/confusion_matrices/binary_scoring/B_seed0_source_movie.csv` | `results/confusion_matrices/binary_scoring/B_seed0_source_movie.png` |
| binary_scoring | B | 1 | source_movie | 500 | 462 | 0.9240 | `results/confusion_matrices/binary_scoring/B_seed1_source_movie.csv` | `results/confusion_matrices/binary_scoring/B_seed1_source_movie.png` |
| binary_scoring | B | 2 | source_movie | 500 | 462 | 0.9240 | `results/confusion_matrices/binary_scoring/B_seed2_source_movie.csv` | `results/confusion_matrices/binary_scoring/B_seed2_source_movie.png` |
| binary_scoring | B | 0 | target_finance | 873 | 856 | 0.9805 | `results/confusion_matrices/binary_scoring/B_seed0_target_finance.csv` | `results/confusion_matrices/binary_scoring/B_seed0_target_finance.png` |
| binary_scoring | B | 1 | target_finance | 873 | 858 | 0.9828 | `results/confusion_matrices/binary_scoring/B_seed1_target_finance.csv` | `results/confusion_matrices/binary_scoring/B_seed1_target_finance.png` |
| binary_scoring | B | 2 | target_finance | 873 | 856 | 0.9805 | `results/confusion_matrices/binary_scoring/B_seed2_target_finance.csv` | `results/confusion_matrices/binary_scoring/B_seed2_target_finance.png` |
| binary_scoring | C | 0 | source_movie | 500 | 458 | 0.9160 | `results/confusion_matrices/binary_scoring/C_seed0_source_movie.csv` | `results/confusion_matrices/binary_scoring/C_seed0_source_movie.png` |
| binary_scoring | C | 1 | source_movie | 500 | 471 | 0.9420 | `results/confusion_matrices/binary_scoring/C_seed1_source_movie.csv` | `results/confusion_matrices/binary_scoring/C_seed1_source_movie.png` |
| binary_scoring | C | 2 | source_movie | 500 | 474 | 0.9480 | `results/confusion_matrices/binary_scoring/C_seed2_source_movie.csv` | `results/confusion_matrices/binary_scoring/C_seed2_source_movie.png` |
| binary_scoring | C | 0 | target_finance | 873 | 803 | 0.9198 | `results/confusion_matrices/binary_scoring/C_seed0_target_finance.csv` | `results/confusion_matrices/binary_scoring/C_seed0_target_finance.png` |
| binary_scoring | C | 1 | target_finance | 873 | 825 | 0.9450 | `results/confusion_matrices/binary_scoring/C_seed1_target_finance.csv` | `results/confusion_matrices/binary_scoring/C_seed1_target_finance.png` |
| binary_scoring | C | 2 | target_finance | 873 | 784 | 0.8981 | `results/confusion_matrices/binary_scoring/C_seed2_target_finance.csv` | `results/confusion_matrices/binary_scoring/C_seed2_target_finance.png` |
| binary_scoring | D | 0 | source_movie | 500 | 475 | 0.9500 | `results/confusion_matrices/binary_scoring/D_seed0_source_movie.csv` | `results/confusion_matrices/binary_scoring/D_seed0_source_movie.png` |
| binary_scoring | D | 1 | source_movie | 500 | 476 | 0.9520 | `results/confusion_matrices/binary_scoring/D_seed1_source_movie.csv` | `results/confusion_matrices/binary_scoring/D_seed1_source_movie.png` |
| binary_scoring | D | 2 | source_movie | 500 | 471 | 0.9420 | `results/confusion_matrices/binary_scoring/D_seed2_source_movie.csv` | `results/confusion_matrices/binary_scoring/D_seed2_source_movie.png` |
| binary_scoring | D | 0 | target_finance | 873 | 833 | 0.9542 | `results/confusion_matrices/binary_scoring/D_seed0_target_finance.csv` | `results/confusion_matrices/binary_scoring/D_seed0_target_finance.png` |
| binary_scoring | D | 1 | target_finance | 873 | 827 | 0.9473 | `results/confusion_matrices/binary_scoring/D_seed1_target_finance.csv` | `results/confusion_matrices/binary_scoring/D_seed1_target_finance.png` |
| binary_scoring | D | 2 | target_finance | 873 | 832 | 0.9530 | `results/confusion_matrices/binary_scoring/D_seed2_target_finance.csv` | `results/confusion_matrices/binary_scoring/D_seed2_target_finance.png` |