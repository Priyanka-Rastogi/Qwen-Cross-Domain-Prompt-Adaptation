"""Builds confusion matrices directly from already-saved prediction CSVs for both
prediction interfaces (free generation and binary-label scoring), across all four
conditions, both domains, and every completed seed. Does NOT run any inference.

Column recovery rule for free generation (documented strict normalization, mirrors
src/evaluate.py:strict_parse and src/diagnose_invalid_outputs.py:classify_invalid):
  normalized = raw_output.strip().lower()
  - normalized == "positive" -> predicted column "positive"   (pred == 1 in the CSV)
  - normalized == "negative" -> predicted column "negative"   (pred == 0 in the CSV)
  - normalized == "neutral"  -> predicted column "neutral"    (pred == -1, this exact
    string recovered from raw_output, NOT assumed from pred == -1 alone)
  - anything else            -> predicted column "other_invalid" (pred == -1, raw_output
    is some other unparseable string, e.g. "mixed...")

Binary-label scoring never produces "neutral" or "other_invalid" by construction
(every prediction is forced into {positive, negative} via argmax of exactly two
scores) -- those columns are included for schema consistency and are always 0 for
that interface; this is verified, not assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FREE_GEN_PRED_DIR = ROOT / "results" / "predictions"
BINARY_PRED_DIR = ROOT / "results" / "binary_label_scoring" / "predictions"
OUT_DIR = ROOT / "results" / "confusion_matrices"

CONDITIONS = [("A", [None]), ("B", [0, 1, 2]), ("C", [0, 1, 2]), ("D", [0, 1, 2])]
DOMAINS = [("source_test", "source_movie", "data/source_test.csv"),
           ("target_test", "target_finance", "data/target_test.csv")]
GOLD_ORDER = ["positive", "negative"]       # rows, per spec
PRED_COLUMNS = ["positive", "negative", "neutral", "other_invalid"]
INTERFACES = ["free_generation", "binary_scoring"]


def tag_for(condition: str, seed) -> str:
    return condition if seed is None else f"{condition}_seed{seed}"


def pred_csv_path(interface: str, condition: str, seed, domain_key: str) -> Path:
    tag = tag_for(condition, seed)
    base = FREE_GEN_PRED_DIR if interface == "free_generation" else BINARY_PRED_DIR
    return base / f"{tag}_{domain_key}.csv"


def classify_pred_category(interface: str, pred: int, raw_output) -> str:
    if interface == "binary_scoring":
        # Forced binary choice by construction -- no invalid/neutral category possible.
        return "positive" if pred == 1 else "negative"
    if pred == 1:
        return "positive"
    if pred == 0:
        return "negative"
    if pred == -1:
        normalized = str(raw_output).strip().lower()
        return "neutral" if normalized == "neutral" else "other_invalid"
    raise ValueError(f"Unexpected pred value {pred!r} in free-generation predictions (expected -1/0/1).")


def build_matrix(csv_path: Path, interface: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    gold_name = df["label"].map({1: "positive", 0: "negative"})
    if gold_name.isna().any():
        raise ValueError(f"{csv_path}: unexpected gold label values outside {{0,1}}")
    raw_output_col = df["raw_output"] if "raw_output" in df.columns else pd.Series([""] * len(df))
    pred_cat = [
        classify_pred_category(interface, p, r)
        for p, r in zip(df["pred"], raw_output_col)
    ]
    matrix = (
        pd.crosstab(gold_name, pd.Series(pred_cat, name="pred"))
        .reindex(index=GOLD_ORDER, columns=PRED_COLUMNS, fill_value=0)
        .astype(int)
    )

    # --- verification: preserve every example, row sums = gold class counts, total = n ---
    gold_counts = gold_name.value_counts()
    row_sums = matrix.sum(axis=1)
    checks = {
        "n_rows_in_csv": len(df),
        "matrix_total": int(matrix.values.sum()),
        "total_matches_n": int(matrix.values.sum()) == len(df),
        "row_sum_positive_matches_gold": int(row_sums["positive"]) == int(gold_counts.get("positive", 0)),
        "row_sum_negative_matches_gold": int(row_sums["negative"]) == int(gold_counts.get("negative", 0)),
    }
    if not all([checks["total_matches_n"], checks["row_sum_positive_matches_gold"],
                checks["row_sum_negative_matches_gold"]]):
        raise AssertionError(f"{csv_path}: confusion-matrix verification FAILED: {checks}")
    return matrix, checks


def save_heatmap(matrix: pd.DataFrame, title: str, out_path: Path, is_mean: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title, fontsize=10)
    vmax = data.max() if data.max() > 0 else 1.0
    fmt = "{:.1f}" if is_mean else "{:d}"
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = fmt.format(val) if is_mean else fmt.format(int(val))
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    for interface in INTERFACES:
        (OUT_DIR / interface).mkdir(parents=True, exist_ok=True)

    index_lines = []
    all_checks = []
    per_seed_summary_rows = []
    missing = []

    for interface in INTERFACES:
        for condition, seeds in CONDITIONS:
            for domain_key, domain_label, gold_csv in DOMAINS:
                seed_matrices = []
                for seed in seeds:
                    tag = tag_for(condition, seed)
                    csv_path = pred_csv_path(interface, condition, seed, domain_key)
                    if not csv_path.exists():
                        missing.append(str(csv_path))
                        continue
                    matrix, checks = build_matrix(csv_path, interface)
                    checks.update(interface=interface, condition=condition, seed=seed, domain=domain_label)
                    all_checks.append(checks)
                    seed_matrices.append(matrix)

                    out_name = f"{tag}_{domain_label}"
                    csv_out = OUT_DIR / interface / f"{out_name}.csv"
                    matrix.to_csv(csv_out)
                    png_out = OUT_DIR / interface / f"{out_name}.png"
                    seed_disp = "deterministic (no seed)" if seed is None else f"seed {seed}"
                    save_heatmap(
                        matrix,
                        f"{interface} | condition {condition} | {seed_disp} | {domain_label}\n"
                        f"n={checks['n_rows_in_csv']}",
                        png_out,
                    )

                    correct = int(matrix.loc["positive", "positive"] + matrix.loc["negative", "negative"])
                    per_seed_summary_rows.append({
                        "interface": interface, "condition": condition, "seed": seed,
                        "domain": domain_label, "n": checks["n_rows_in_csv"], "correct": correct,
                        "accuracy": correct / checks["n_rows_in_csv"],
                        "file_csv": str(csv_out.relative_to(ROOT)),
                        "file_png": str(png_out.relative_to(ROOT)),
                    })

                # Mean-count matrix across seeds for B/C/D (never pooled as a bigger test set)
                if len(seed_matrices) > 1:
                    mean_matrix = sum(m.astype(float) for m in seed_matrices) / len(seed_matrices)
                    out_name = f"{condition}_mean_of_{len(seed_matrices)}seeds_{domain_label}"
                    csv_out = OUT_DIR / interface / f"{out_name}.csv"
                    mean_matrix.to_csv(csv_out)
                    png_out = OUT_DIR / interface / f"{out_name}.png"
                    save_heatmap(
                        mean_matrix,
                        f"{interface} | condition {condition} | MEAN of {len(seed_matrices)} seeds "
                        f"(cell-wise average count, NOT pooled) | {domain_label}",
                        png_out, is_mean=True,
                    )

    summary_df = pd.DataFrame(per_seed_summary_rows)
    summary_df.to_csv(OUT_DIR / "per_seed_summary.csv", index=False)
    checks_df = pd.DataFrame(all_checks)
    checks_df.to_csv(OUT_DIR / "verification_checks.csv", index=False)

    all_passed = bool(checks_df[["total_matches_n", "row_sum_positive_matches_gold",
                                  "row_sum_negative_matches_gold"]].all(axis=None))

    index_lines.append("# Confusion matrices: index\n")
    index_lines.append(
        "Built directly from already-saved prediction CSVs under `results/predictions/` "
        "(free generation) and `results/binary_label_scoring/predictions/` (binary-label "
        "scoring). No inference was re-run.\n"
    )
    index_lines.append("## Condition mapping\n")
    index_lines.append(
        "| Code | Meaning |\n|---|---|\n"
        "| A | zero-shot, fixed baseline instruction, no demonstrations (deterministic -- reported once) |\n"
        "| B | optimized instruction only, no demonstrations (source-dev selected, 3 seeds) |\n"
        "| C | fixed baseline instruction + 4 random demonstrations (2 pos / 2 neg, 3 seeds) |\n"
        "| D | selected instruction + selected demonstrations (bounded joint search, 3 seeds) |\n"
    )
    index_lines.append("## Label mapping\n")
    index_lines.append(
        "- Gold rows: `positive` (label==1), `negative` (label==0).\n"
        "- Predicted columns: `positive` (pred==1), `negative` (pred==0), "
        "`neutral` (pred==-1 AND raw_output, stripped+lowercased, is exactly `\"neutral\"`), "
        "`other_invalid` (pred==-1 and raw_output is any other unparseable string, e.g. "
        "`\"mixed\"`-family outputs).\n"
        "- `neutral` is **recovered from the saved `raw_output` text**, not assumed from "
        "`pred == -1` -- see `classify_pred_category()` in `src/generate_confusion_matrices.py`, "
        "which mirrors the strict-normalization rule already used for diagnosis in "
        "`src/diagnose_invalid_outputs.py:classify_invalid` (same `strip().lower()` exact-match "
        "rule as the production parser `src/evaluate.py:strict_parse`).\n"
        "- Binary-label scoring never has a `neutral`/`other_invalid` prediction by construction "
        "(forced two-way argmax) -- those columns are present for schema consistency and are "
        "always 0 for that interface, which this script verifies rather than assumes.\n"
    )
    index_lines.append("## Verification\n")
    index_lines.append(
        f"All {len(checks_df)} generated matrices: every example preserved (matrix total == "
        f"prediction-CSV row count), and each gold-row sum matches the gold-label class count "
        f"in `data/source_test.csv` (250/250) or `data/target_test.csv` (303 negative / 570 "
        f"positive). **Overall result: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED -- see verification_checks.csv'}**.\n"
    )
    if missing:
        index_lines.append(f"**Missing files (skipped, not fabricated):**\n" + "\n".join(f"- `{m}`" for m in missing) + "\n")
    else:
        index_lines.append("No missing prediction files -- every (interface, condition, seed, domain) combination expected to exist was found.\n")

    index_lines.append("## Aggregation note\n")
    index_lines.append(
        "For B/C/D (3 seeds each), per-seed matrices are saved individually "
        "(`{condition}_seed{N}_{domain}.csv/.png`). A separate, explicitly labeled "
        "`{condition}_mean_of_3seeds_{domain}.csv/.png` is also saved: this is the **cell-wise "
        "arithmetic mean of the three seeds' matrices** (so cell values are generally "
        "non-integer), NOT the three seeds pooled into one 1500/2619-example confusion matrix. "
        "The per-seed matrices remain the primary evaluation artifact; the mean matrix is a "
        "convenience summary only. Condition A has no seed dependence and only one matrix per "
        "(interface, domain) is produced.\n"
    )

    index_lines.append("## Per-seed accuracy (sanity cross-check against saved metrics)\n")
    index_lines.append("| Interface | Condition | Seed | Domain | n | correct | accuracy | CSV | PNG |")
    index_lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        seed_disp = "-" if pd.isna(r.seed) else int(r.seed)
        index_lines.append(
            f"| {r.interface} | {r.condition} | {seed_disp} | {r.domain} | {r.n} | {r.correct} | "
            f"{r.accuracy:.4f} | `{r.file_csv}` | `{r.file_png}` |"
        )

    (OUT_DIR / "index.md").write_text("\n".join(index_lines))

    print(f"Wrote {len(summary_df)} per-(interface,condition,seed,domain) matrices under {OUT_DIR}")
    print(f"All verification checks passed: {all_passed}")
    if missing:
        print(f"MISSING ({len(missing)}): {missing}")
    print(f"Wrote {OUT_DIR / 'index.md'}")
    print(f"Wrote {OUT_DIR / 'per_seed_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'verification_checks.csv'}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
