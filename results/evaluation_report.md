### Independent post-hoc evaluation report

Snapshot taken at: 2026-10-21T08:20:29.724272+00:00 (UTC). This report was generated strictly
from saved artifacts on disk — no prompts were changed, no prompt search was
re-run, and no test-set result influenced any prompt or hyperparameter choice.
If the underlying pilot run (`configs/pilot.yaml`, seeds=[0, 1, 2]) was still in
progress at snapshot time, conditions/seeds not yet finished are reported as
**incomplete**, never as zero or as failing scores.

## 1. Run completion status

- **A**: completed
- **B_seed0**: completed
- **B_seed1**: completed
- **B_seed2**: completed
- **C_seed0**: completed
- **C_seed1**: completed
- **C_seed2**: completed
- **D_seed0**: completed
- **D_seed1**: completed
- **D_seed2**: completed

## 2. Verification checks

- [PASS] frozen-model code path (requires_grad=False, assert trainable==0, model.eval(), torch.inference_mode, enable_thinking passed to chat template)
  - static source inspection of src/model_backend.py
- [PASS] model revision actually used by this run is recoverable from logs
  - found revision=b968826d9c46dd6066d109eabc6255188de91218 in /workspace/qwen_cross_domain_prompt_adaptation/results/logs/pilot_stdout.log
- [UNVERIFIABLE] trainable-parameter count == 0 was enforced at runtime (not just in source)
  - the assertion in QwenBackend.load() raises if violated and the run has produced downstream results without a crash at load time, which is consistent with it having passed, but no artifact persists the actual counted value -- THIS CANNOT BE FULLY VERIFIED FROM SAVED ARTIFACTS ALONE, only inferred from absence of a crash.
- [PASS] data split manifest exists and records seed=42
  - data_split_seed=42
- [PASS] manifest reports zero source/target overlap
  - source_target_overlap_count=0
- [PASS] manifest reports zero source_train/source_validation overlap
  - source_train_validation_overlap_count=0
- [PASS] independent recount of CSV ids matches manifest ids (not just trusting manifest counts)
  - source_train: n=200 (expected 200), id set matches manifest=True; source_validation: n=100 (expected 100), id set matches manifest=True; source_test: n=500 (expected 500), id set matches manifest=True; target_test: n=873, id set matches manifest=True
- [PASS] label<->label_name mapping (0=negative, 1=positive) holds in every CSV
  - source_train: consistent=True; source_validation: consistent=True; source_test: consistent=True; target_test: consistent=True
- [PASS] B_seed0: frozen instruction_hash matches independent recomputation
  - stored=9cfd07191ec6..., recomputed=9cfd07191ec6...
- [PASS] B_seed1: frozen instruction_hash matches independent recomputation
  - stored=00b4b7ecdac0..., recomputed=00b4b7ecdac0...
- [PASS] B_seed2: frozen instruction_hash matches independent recomputation
  - stored=9cfd07191ec6..., recomputed=9cfd07191ec6...
- [PASS] C_seed0: frozen instruction_hash matches independent recomputation
  - stored=528efe0cd51d..., recomputed=528efe0cd51d...
- [PASS] C_seed1: frozen instruction_hash matches independent recomputation
  - stored=528efe0cd51d..., recomputed=528efe0cd51d...
- [PASS] C_seed2: frozen instruction_hash matches independent recomputation
  - stored=528efe0cd51d..., recomputed=528efe0cd51d...
- [PASS] D_seed0: frozen instruction_hash matches independent recomputation
  - stored=4c54e4efd897..., recomputed=4c54e4efd897...
- [PASS] D_seed1: frozen instruction_hash matches independent recomputation
  - stored=528efe0cd51d..., recomputed=528efe0cd51d...
- [PASS] D_seed2: frozen instruction_hash matches independent recomputation
  - stored=9cfd07191ec6..., recomputed=9cfd07191ec6...
- [PASS] seed0: all accepted instruction candidates are domain-neutral (no finance/movie terms)
  - n_candidates=8, violations=[]
- [PASS] seed1: all accepted instruction candidates are domain-neutral (no finance/movie terms)
  - n_candidates=8, violations=[]
- [PASS] seed2: all accepted instruction candidates are domain-neutral (no finance/movie terms)
  - n_candidates=8, violations=[]
- [PASS] B_seed0: every dev-selection evaluation used exactly the 100 source_validation examples (never source_test[500] or target_test)
  - n_candidates_checked=8, mismatched_counts=[]
- [PASS] B_seed1: every dev-selection evaluation used exactly the 100 source_validation examples (never source_test[500] or target_test)
  - n_candidates_checked=8, mismatched_counts=[]
- [PASS] B_seed2: every dev-selection evaluation used exactly the 100 source_validation examples (never source_test[500] or target_test)
  - n_candidates_checked=8, mismatched_counts=[]
- [PASS] D_seed0: every dev-selection evaluation used exactly the 100 source_validation examples (never source_test[500] or target_test)
  - n_candidates_checked=80, mismatched_counts=[]
- [PASS] D_seed1: every dev-selection evaluation used exactly the 100 source_validation examples (never source_test[500] or target_test)
  - n_candidates_checked=80, mismatched_counts=[]
- [PASS] D_seed2: every dev-selection evaluation used exactly the 100 source_validation examples (never source_test[500] or target_test)
  - n_candidates_checked=80, mismatched_counts=[]

## 3-4. Recomputed metrics (from saved prediction CSVs, independent of the pipeline's own summary.json) + same-examples check

Same source-test example set used by every completed condition: **True**
Same target-test example set used by every completed condition: **True**
(per-condition detail: {"A.source_test": [true, 500, 500], "A.target_test": [true, 873, 873], "B_seed0.source_test": [true, 500, 500], "B_seed0.target_test": [true, 873, 873], "B_seed1.source_test": [true, 500, 500], "B_seed1.target_test": [true, 873, 873], "B_seed2.source_test": [true, 500, 500], "B_seed2.target_test": [true, 873, 873], "C_seed0.source_test": [true, 500, 500], "C_seed0.target_test": [true, 873, 873], "C_seed1.source_test": [true, 500, 500], "C_seed1.target_test": [true, 873, 873], "C_seed2.source_test": [true, 500, 500], "C_seed2.target_test": [true, 873, 873], "D_seed0.source_test": [true, 500, 500], "D_seed0.target_test": [true, 873, 873], "D_seed1.source_test": [true, 500, 500], "D_seed1.target_test": [true, 873, 873], "D_seed2.source_test": [true, 500, 500], "D_seed2.target_test": [true, 873, 873]})

### A
- source_test: n=500, accuracy=0.9220, macro_f1=0.9227, invalid_rate=0.0020
  - negative: precision=0.890, recall=0.968, f1=0.927
  - positive: precision=0.965, recall=0.876, f1=0.918
  - confusion: true_neg->[neg 242, pos 8, invalid 0]; true_pos->[neg 30, pos 219, invalid 1]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.9782, macro_f1=0.9763, invalid_rate=0.0000
  - negative: precision=0.944, recall=0.997, f1=0.970
  - positive: precision=0.998, recall=0.968, f1=0.983
  - confusion: true_neg->[neg 302, pos 1, invalid 0]; true_pos->[neg 18, pos 552, invalid 0]
  - recomputed metrics match the pipeline's own stored metrics exactly

### B_seed0
- source_test: n=500, accuracy=0.9220, macro_f1=0.9238, invalid_rate=0.0040
  - negative: precision=0.898, recall=0.956, f1=0.926
  - positive: precision=0.957, recall=0.888, f1=0.921
  - confusion: true_neg->[neg 239, pos 10, invalid 1]; true_pos->[neg 27, pos 222, invalid 1]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.9794, macro_f1=0.9799, invalid_rate=0.0034
  - negative: precision=0.953, recall=1.000, f1=0.976
  - positive: precision=1.000, recall=0.968, f1=0.984
  - confusion: true_neg->[neg 303, pos 0, invalid 0]; true_pos->[neg 15, pos 552, invalid 3]
  - recomputed metrics match the pipeline's own stored metrics exactly

### B_seed1
- source_test: n=500, accuracy=0.9240, macro_f1=0.9258, invalid_rate=0.0040
  - negative: precision=0.902, recall=0.956, f1=0.928
  - positive: precision=0.957, recall=0.892, f1=0.923
  - confusion: true_neg->[neg 239, pos 10, invalid 1]; true_pos->[neg 26, pos 223, invalid 1]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.9828, macro_f1=0.9813, invalid_rate=0.0000
  - negative: precision=0.953, recall=1.000, f1=0.976
  - positive: precision=1.000, recall=0.974, f1=0.987
  - confusion: true_neg->[neg 303, pos 0, invalid 0]; true_pos->[neg 15, pos 555, invalid 0]
  - recomputed metrics match the pipeline's own stored metrics exactly

### B_seed2
- source_test: n=500, accuracy=0.9220, macro_f1=0.9238, invalid_rate=0.0040
  - negative: precision=0.898, recall=0.956, f1=0.926
  - positive: precision=0.957, recall=0.888, f1=0.921
  - confusion: true_neg->[neg 239, pos 10, invalid 1]; true_pos->[neg 27, pos 222, invalid 1]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.9794, macro_f1=0.9799, invalid_rate=0.0034
  - negative: precision=0.953, recall=1.000, f1=0.976
  - positive: precision=1.000, recall=0.968, f1=0.984
  - confusion: true_neg->[neg 303, pos 0, invalid 0]; true_pos->[neg 15, pos 552, invalid 3]
  - recomputed metrics match the pipeline's own stored metrics exactly

### C_seed0
- source_test: n=500, accuracy=0.9120, macro_f1=0.9141, invalid_rate=0.0060
  - negative: precision=0.869, recall=0.984, f1=0.923
  - positive: precision=0.981, recall=0.840, f1=0.905
  - confusion: true_neg->[neg 246, pos 4, invalid 0]; true_pos->[neg 37, pos 210, invalid 3]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.5716, macro_f1=0.7355, invalid_rate=0.3963
  - negative: precision=0.903, recall=0.865, f1=0.884
  - positive: precision=1.000, recall=0.416, f1=0.587
  - confusion: true_neg->[neg 262, pos 0, invalid 41]; true_pos->[neg 28, pos 237, invalid 305]
  - recomputed metrics match the pipeline's own stored metrics exactly

### C_seed1
- source_test: n=500, accuracy=0.9400, macro_f1=0.9409, invalid_rate=0.0020
  - negative: precision=0.920, recall=0.968, f1=0.943
  - positive: precision=0.966, recall=0.912, f1=0.938
  - confusion: true_neg->[neg 242, pos 8, invalid 0]; true_pos->[neg 21, pos 228, invalid 1]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.6942, macro_f1=0.8176, invalid_rate=0.2749
  - negative: precision=0.908, recall=0.845, f1=0.875
  - positive: precision=0.997, recall=0.614, f1=0.760
  - confusion: true_neg->[neg 256, pos 1, invalid 46]; true_pos->[neg 26, pos 350, invalid 194]
  - recomputed metrics match the pipeline's own stored metrics exactly

### C_seed2
- source_test: n=500, accuracy=0.9420, macro_f1=0.9438, invalid_rate=0.0040
  - negative: precision=0.924, recall=0.972, f1=0.947
  - positive: precision=0.970, recall=0.912, f1=0.940
  - confusion: true_neg->[neg 243, pos 7, invalid 0]; true_pos->[neg 20, pos 228, invalid 2]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.5750, macro_f1=0.7314, invalid_rate=0.3826
  - negative: precision=0.875, recall=0.858, f1=0.867
  - positive: precision=1.000, recall=0.425, f1=0.596
  - confusion: true_neg->[neg 260, pos 0, invalid 43]; true_pos->[neg 37, pos 242, invalid 291]
  - recomputed metrics match the pipeline's own stored metrics exactly

### D_seed0
- source_test: n=500, accuracy=0.9260, macro_f1=0.9391, invalid_rate=0.0280
  - negative: precision=0.955, recall=0.924, f1=0.939
  - positive: precision=0.951, recall=0.928, f1=0.939
  - confusion: true_neg->[neg 231, pos 12, invalid 7]; true_pos->[neg 11, pos 232, invalid 7]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.5189, macro_f1=0.6974, invalid_rate=0.4628
  - negative: precision=0.931, recall=0.713, f1=0.807
  - positive: precision=1.000, recall=0.416, f1=0.587
  - confusion: true_neg->[neg 216, pos 0, invalid 87]; true_pos->[neg 16, pos 237, invalid 317]
  - recomputed metrics match the pipeline's own stored metrics exactly

### D_seed1
- source_test: n=500, accuracy=0.9340, macro_f1=0.9433, invalid_rate=0.0200
  - negative: precision=0.941, recall=0.956, f1=0.948
  - positive: precision=0.966, recall=0.912, f1=0.938
  - confusion: true_neg->[neg 239, pos 8, invalid 3]; true_pos->[neg 15, pos 228, invalid 7]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.5647, macro_f1=0.7344, invalid_rate=0.4158
  - negative: precision=0.932, recall=0.766, f1=0.841
  - positive: precision=1.000, recall=0.458, f1=0.628
  - confusion: true_neg->[neg 232, pos 0, invalid 71]; true_pos->[neg 17, pos 261, invalid 292]
  - recomputed metrics match the pipeline's own stored metrics exactly

### D_seed2
- source_test: n=500, accuracy=0.9360, macro_f1=0.9398, invalid_rate=0.0080
  - negative: precision=0.947, recall=0.936, f1=0.942
  - positive: precision=0.940, recall=0.936, f1=0.938
  - confusion: true_neg->[neg 234, pos 15, invalid 1]; true_pos->[neg 13, pos 234, invalid 3]
  - recomputed metrics match the pipeline's own stored metrics exactly
- target_test: n=873, accuracy=0.4983, macro_f1=0.6752, invalid_rate=0.4926
  - negative: precision=0.957, recall=0.591, f1=0.731
  - positive: precision=1.000, recall=0.449, f1=0.620
  - confusion: true_neg->[neg 179, pos 0, invalid 124]; true_pos->[neg 8, pos 256, invalid 306]
  - recomputed metrics match the pipeline's own stored metrics exactly


## 5. Comparison table

See `results/comparison.csv` (also rendered below). `status` shows how many of
the expected seeds are actually completed for that row; a row with 0 completed
seeds reports `None`/`null` for every score, never 0.

| condition   | split   |   n_seeds_completed |   n_seeds_expected |   accuracy_mean |   accuracy_std |   macro_f1_mean |   macro_f1_std |   invalid_rate_mean |   search_runtime_s_total |   final_inference_runtime_s_total |   input_tokens_total |   output_tokens_total | status   |
|:------------|:--------|--------------------:|-------------------:|----------------:|---------------:|----------------:|---------------:|--------------------:|-------------------------:|----------------------------------:|---------------------:|----------------------:|:---------|
| A           | source  |                   1 |                  1 |          0.9220 |         0.0000 |          0.9227 |         0.0000 |              0.0020 |                   0.0000 |                           15.4174 |               119888 |                  1000 | complete |
| A           | target  |                   1 |                  1 |          0.9782 |         0.0000 |          0.9763 |         0.0000 |              0.0000 |                   0.0000 |                           10.7317 |                67434 |                  1746 | complete |
| B           | source  |                   3 |                  3 |          0.9227 |         0.0009 |          0.9244 |         0.0010 |              0.0040 |                  40.1030 |                           30.9126 |               366664 |                  3024 | complete |
| B           | target  |                   3 |                  3 |          0.9805 |         0.0016 |          0.9804 |         0.0006 |              0.0023 |                  40.1030 |                           22.2579 |               214524 |                  5238 | complete |
| C           | source  |                   3 |                  3 |          0.9313 |         0.0137 |          0.9329 |         0.0133 |              0.0040 |                   0.0000 |                          208.5963 |              1762164 |                  3000 | complete |
| C           | target  |                   3 |                  3 |          0.6136 |         0.0570 |          0.7615 |         0.0397 |              0.3513 |                   0.0000 |                          296.7652 |              2651067 |                  5238 | complete |
| D           | source  |                   3 |                  3 |          0.9320 |         0.0043 |          0.9408 |         0.0018 |              0.0187 |                3134.1467 |                          204.6398 |              1743664 |                  3000 | complete |
| D           | target  |                   3 |                  3 |          0.5273 |         0.0278 |          0.7023 |         0.0244 |              0.4570 |                3134.1467 |                          293.1543 |              2618766 |                  5238 | complete |

## 6. D - A / D - B / D - C differences (percentage points)

[
  {
    "split": "source",
    "comparison": "D-A",
    "accuracy_diff_pp": 1.0,
    "macro_f1_diff_pp": 1.8
  },
  {
    "split": "source",
    "comparison": "D-B",
    "accuracy_diff_pp": 0.93,
    "macro_f1_diff_pp": 1.63
  },
  {
    "split": "source",
    "comparison": "D-C",
    "accuracy_diff_pp": 0.07,
    "macro_f1_diff_pp": 0.78
  },
  {
    "split": "target",
    "comparison": "D-A",
    "accuracy_diff_pp": -45.09,
    "macro_f1_diff_pp": -27.4
  },
  {
    "split": "target",
    "comparison": "D-B",
    "accuracy_diff_pp": -45.32,
    "macro_f1_diff_pp": -27.8
  },
  {
    "split": "target",
    "comparison": "D-C",
    "accuracy_diff_pp": -8.63,
    "macro_f1_diff_pp": -5.92
  }
]

## 7. Did source-side optimization help cross-domain transfer, and was it worth the compute?

Based on 3 of 3 completed seed(s) only — not enough to claim statistical significance; treat the following as a preliminary, not final, finding.

On the completed seed(s), source-side joint instruction+demonstration search (D) did not help (and on the completed seed(s), hurt) cross-domain transfer:
- D improved in-domain (source-test) macro-F1 over A by 1.8 pp.
- D changed cross-domain (target-test) macro-F1 relative to A by -27.4 pp (i.e. a regression).
- D's target invalid-output rate (45.7% mean) is far higher than A's (0.0% mean) — most of D's target-side degradation traces to the model failing to emit a parseable positive/negative label at all when D's selected demonstrations (drawn from movie reviews) are shown alongside financial sentences, not to confidently wrong sentiment judgments.
- D's dev-search cost (3134s on the completed seed(s)) is roughly 78x the cost of B's search (40s), for a source-side gain of only 1.8 pp macro-F1 and a target-side regression.

On this evidence, the ~78x extra search cost of D over B is NOT justified: it buys a small in-domain gain while making cross-domain transfer substantially worse than even the zero-cost fixed baseline (A). This is a legitimate negative result for D as implemented, not a pipeline failure — see Limitations.

## 8. Exploratory target-domain error analysis (post-evaluation only; not used to revise this experiment)

### A (target, n=873)
- False negatives (true positive -> predicted negative): 18
- False positives (true negative -> predicted positive): 1
- Invalid/unparseable outputs: 0
  - example false negative: "However , the proportion of the paid standing orders grew in 2009 ...." (raw_output='negative')
  - example false positive: "Finnish textiles and clothing group Marimekko Oyj posted a net profit of 7.99 mln euro $ 10.4 mln for 2006 , compared to 8.4 mln euro $ 10.9 mln for 2005 ...." (raw_output='positive')

### B_seed0 (target, n=873)
- False negatives (true positive -> predicted negative): 15
- False positives (true negative -> predicted positive): 0
- Invalid/unparseable outputs: 3
  - example false negative: "However , the proportion of the paid standing orders grew in 2009 ...." (raw_output='negative')

### B_seed1 (target, n=873)
- False negatives (true positive -> predicted negative): 15
- False positives (true negative -> predicted positive): 0
- Invalid/unparseable outputs: 0
  - example false negative: "However , the proportion of the paid standing orders grew in 2009 ...." (raw_output='negative')

### B_seed2 (target, n=873)
- False negatives (true positive -> predicted negative): 15
- False positives (true negative -> predicted positive): 0
- Invalid/unparseable outputs: 3
  - example false negative: "However , the proportion of the paid standing orders grew in 2009 ...." (raw_output='negative')

### C_seed0 (target, n=873)
- False negatives (true positive -> predicted negative): 28
- False positives (true negative -> predicted positive): 0
- Invalid/unparseable outputs: 346
  - example false negative: "EPS from continuing operations came in at 0.30 eur , up from 0.17 ...." (raw_output='negative')

### C_seed1 (target, n=873)
- False negatives (true positive -> predicted negative): 26
- False positives (true negative -> predicted positive): 1
- Invalid/unparseable outputs: 240
  - example false negative: "26 January 2011 - Finnish metal products company Componenta Oyj ( HEL : CTH1V ) said yesterday its net loss narrowed to EUR500 ,000 in the last quarter of 2010 ..." (raw_output='negative')
  - example false positive: "Profitability ( EBIT % ) was 13.6 % , compared to 14.3 % in Q2 2009 ...." (raw_output='positive')

### C_seed2 (target, n=873)
- False negatives (true positive -> predicted negative): 37
- False positives (true negative -> predicted positive): 0
- Invalid/unparseable outputs: 334
  - example false negative: "Diluted earnings per share ( EPS ) rose to EUR 0.29 from EUR 0.05 ...." (raw_output='negative')

### D_seed0 (target, n=873)
- False negatives (true positive -> predicted negative): 16
- False positives (true negative -> predicted positive): 0
- Invalid/unparseable outputs: 404
  - example false negative: "8 May 2009 - Finnish liquid handling products and diagnostic test systems maker Biohit Oyj ( HEL : BIOBV ) said today ( 8 May 2009 ) its net loss narrowed to EU..." (raw_output='negative')

### D_seed1 (target, n=873)
- False negatives (true positive -> predicted negative): 17
- False positives (true negative -> predicted positive): 0
- Invalid/unparseable outputs: 363
  - example false negative: "26 January 2011 - Finnish metal products company Componenta Oyj ( HEL : CTH1V ) said yesterday its net loss narrowed to EUR500 ,000 in the last quarter of 2010 ..." (raw_output='negative')

### D_seed2 (target, n=873)
- False negatives (true positive -> predicted negative): 8
- False positives (true negative -> predicted positive): 0
- Invalid/unparseable outputs: 430
  - example false negative: "8 May 2009 - Finnish liquid handling products and diagnostic test systems maker Biohit Oyj ( HEL : BIOBV ) said today ( 8 May 2009 ) its net loss narrowed to EU..." (raw_output='negative')


## 10. Limitations

- Three seeds (even once all complete) is not enough to claim statistical
  significance; standard deviations reported here describe the observed spread
  across 3 seeds only, nothing more.
- The zero-trainable-parameters guarantee is enforced by an in-process assertion
  in `src/model_backend.py` and is not independently re-verifiable from saved
  artifacts alone post-hoc (see check list above) — only its absence-of-crash is
  observable.
- Dev-phase (search) wall-clock/token totals are reconstructed by summing the
  per-candidate cost entries embedded in each condition's own final metrics
  JSON; `results/logs/phase_costs.json` (a separate aggregate file) is only
  written once `run_experiment.py` finishes entirely, so it will not exist yet
  for an in-progress run — this report does not depend on it.
- This script does not modify `results/metrics/summary.json`,
  `artifacts/selected_prompts/*.json`, or any prediction file; it only reads them.

## File paths

- This report: `/workspace/qwen_cross_domain_prompt_adaptation/results/evaluation_report.md`
- Comparison table: `/workspace/qwen_cross_domain_prompt_adaptation/results/comparison.csv`
- Source metrics read from: `/workspace/qwen_cross_domain_prompt_adaptation/results/metrics`
- Source predictions read from: `/workspace/qwen_cross_domain_prompt_adaptation/results/predictions`
- Data manifest: `/workspace/qwen_cross_domain_prompt_adaptation/artifacts/manifests/data_manifest.json`
- Selected-prompt manifests: `/workspace/qwen_cross_domain_prompt_adaptation/artifacts/selected_prompts`
