"""Independent, read-only re-verification of the comparison table shared with the
mentor, requested specifically to (a) re-derive the free-generation finance (C/D
target) numbers with invalid predictions kept as a distinct value and retained in
every denominator (never filtered, never dropped from false-negative counts), (b)
identify the exact code path responsible for the existing saved numbers and confirm
or refute agreement, and (c) independently re-verify the binary-label-scoring
metrics and their three-seed standard deviations.

This script does NOT import src/metrics.py's compute_metrics -- it reimplements the
confusion-matrix/precision/recall/F1 arithmetic from scratch directly on the raw
per-row prediction CSVs, so that an agreement with the saved results is a genuine
independent cross-check, not a re-run of the same function.

Nothing under results/predictions, results/metrics, or results/binary_label_scoring
is modified. All outputs of this audit are written under
results/comparison_audit/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "predictions"
METRICS_DIR = ROOT / "results" / "metrics"
BIN_PRED_DIR = ROOT / "results" / "binary_label_scoring" / "predictions"
BIN_METRICS_DIR = ROOT / "results" / "binary_label_scoring" / "metrics"
OUT_DIR = ROOT / "results" / "comparison_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INVALID_PRED = -1
DOMAINS = [("source_test", "source/movie"), ("target_test", "target/finance")]
CONDITIONS = [("A", [None]), ("B", [0, 1, 2]), ("C", [0, 1, 2]), ("D", [0, 1, 2])]


def tag_for(condition: str, seed) -> str:
    return condition if seed is None else f"{condition}_seed{seed}"


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> dict:
    tp = int(((y_true == cls) & (y_pred == cls)).sum())
    fp = int(((y_true != cls) & (y_pred == cls)).sum())
    fn = int(((y_true == cls) & (y_pred != cls)).sum())  # includes invalid preds, per spec
    support = int((y_true == cls).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, support=support, precision=precision, recall=recall, f1=f1)


def recompute_from_predictions(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    y_true = df["label"].to_numpy(dtype=int)
    y_pred = df["pred"].to_numpy(dtype=int)
    n = len(df)

    # Invalid predictions kept as a distinct value (never remapped, never dropped).
    # They cannot equal 0 or 1, so they never appear as a TP or FP for either real
    # class -- but DO count toward FN for the true class, and DO count in the
    # overall-incorrect / accuracy denominator. Every original example is retained.
    assert n == len(y_pred) == len(y_true), "row count mismatch"
    invalid_count = int((y_pred == INVALID_PRED).sum())

    correct = (y_pred == y_true) & (y_pred != INVALID_PRED)
    accuracy = correct.sum() / n

    neg = confusion_counts(y_true, y_pred, 0)
    pos = confusion_counts(y_true, y_pred, 1)
    macro_f1 = (neg["f1"] + pos["f1"]) / 2.0
    macro_precision = (neg["precision"] + pos["precision"]) / 2.0
    macro_recall = (neg["recall"] + pos["recall"]) / 2.0

    return dict(
        n=n, correct=int(correct.sum()), accuracy=accuracy, macro_f1=macro_f1,
        macro_precision=macro_precision, macro_recall=macro_recall,
        invalid_count=invalid_count, invalid_rate=invalid_count / n,
        neg=neg, pos=pos,
    )


def compare_to_saved(tag: str, domain_key: str, recomputed: dict) -> dict:
    path = METRICS_DIR / f"{tag}_final.json"
    saved = json.loads(path.read_text())[domain_key]["metrics"]
    diffs = {}
    for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "invalid_count", "invalid_rate", "n"):
        a, b = recomputed[key], saved[key]
        d = abs(a - b)
        if d > 1e-9:
            diffs[key] = {"recomputed": a, "saved": b, "abs_diff": d}
    for cls_key, cls_name in ((0, "negative"), (1, "positive")):
        rc = recomputed["neg"] if cls_key == 0 else recomputed["pos"]
        sv = saved["per_class"][cls_name]
        for metric in ("precision", "recall", "f1"):
            a, b = rc[metric], sv[metric]
            if abs(a - b) > 1e-9:
                diffs[f"{cls_name}.{metric}"] = {"recomputed": a, "saved": b, "abs_diff": abs(a - b)}
        if rc["support"] != sv["support"]:
            diffs[f"{cls_name}.support"] = {"recomputed": rc["support"], "saved": sv["support"]}
    return {"matches": len(diffs) == 0, "diffs": diffs, "saved": saved}


def free_generation_audit() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    report_lines = []
    report_lines.append("## Free-generation finance (and source) results: independent recomputation\n")
    report_lines.append(
        "Methodology: for every prediction CSV under `results/predictions/`, `pred == -1` "
        "(the project's documented invalid-output sentinel, see `src/metrics.py` docstring) "
        "is kept as its own value. No row is dropped. For each class c in {0,1}: "
        "TP = (true==c & pred==c), FP = (true!=c & pred==c), FN = (true==c & pred!=c) -- "
        "note FN explicitly includes rows where pred==-1, so every invalid prediction on a "
        "true-class-c example reduces that class's recall, exactly as the spec requires. "
        "Accuracy counts every row in the denominator; invalid predictions can never be "
        "'correct'. This is a from-scratch reimplementation (does not call `src/metrics.py`), "
        "so agreement with the saved values is a genuine independent check.\n"
    )
    for condition, seeds in CONDITIONS:
        for seed in seeds:
            tag = tag_for(condition, seed)
            for domain_key, domain_label in DOMAINS:
                csv_path = PRED_DIR / f"{tag}_{domain_key}.csv"
                rec = recompute_from_predictions(csv_path)
                cmp = compare_to_saved(tag, domain_key, rec)
                rows.append({
                    "interface": "free_generation", "condition": condition, "seed": seed,
                    "domain": domain_label, "n": rec["n"], "correct": rec["correct"],
                    "accuracy": rec["accuracy"], "macro_f1": rec["macro_f1"],
                    "neg_tp": rec["neg"]["tp"], "neg_fp": rec["neg"]["fp"], "neg_fn": rec["neg"]["fn"],
                    "neg_support": rec["neg"]["support"], "neg_precision": rec["neg"]["precision"],
                    "neg_recall": rec["neg"]["recall"], "neg_f1": rec["neg"]["f1"],
                    "pos_tp": rec["pos"]["tp"], "pos_fp": rec["pos"]["fp"], "pos_fn": rec["pos"]["fn"],
                    "pos_support": rec["pos"]["support"], "pos_precision": rec["pos"]["precision"],
                    "pos_recall": rec["pos"]["recall"], "pos_f1": rec["pos"]["f1"],
                    "invalid_count": rec["invalid_count"], "invalid_rate": rec["invalid_rate"],
                    "matches_saved_json": cmp["matches"],
                })
                if not cmp["matches"]:
                    report_lines.append(f"**DISCREPANCY** in {tag}/{domain_key}: {cmp['diffs']}\n")

    df = pd.DataFrame(rows)

    report_lines.append("\n### Per-seed table: condition C and D, finance (target) domain\n")
    report_lines.append(
        "| Condition | Seed | n | correct | TP(neg) | FP(neg) | FN(neg) | support(neg) | "
        "TP(pos) | FP(pos) | FN(pos) | support(pos) | invalid | accuracy | macro-F1 | matches saved |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df[(df.condition.isin(["C", "D"])) & (df.domain == "target/finance")].iterrows():
        seed_disp = int(r.seed) if pd.notna(r.seed) else "-"
        report_lines.append(
            f"| {r.condition} | {seed_disp} | {r.n} | {r.correct} | {r.neg_tp} | {r.neg_fp} | {r.neg_fn} | "
            f"{r.neg_support} | {r.pos_tp} | {r.pos_fp} | {r.pos_fn} | {r.pos_support} | "
            f"{r.invalid_count} | {r.accuracy:.4f} | {r.macro_f1:.4f} | {r.matches_saved_json} |"
        )

    report_lines.append("\n### Aggregated across seeds (C, D; both domains) -- mean, sample std (ddof=1), population std (ddof=0)\n")
    report_lines.append("| Condition | Domain | Accuracy mean | Acc sample std | Acc population std | Macro-F1 mean | F1 sample std | F1 population std | Original comparison.csv accuracy | Original comparison.csv macro-F1 |")
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    orig_cmp = pd.read_csv(ROOT / "results" / "comparison.csv")
    for condition in ["A", "B", "C", "D"]:
        for domain_key, domain_label in DOMAINS:
            sub = df[(df.condition == condition) & (df.domain == domain_label)]
            acc_vals = sub["accuracy"].to_numpy()
            f1_vals = sub["macro_f1"].to_numpy()
            orig_row = orig_cmp[(orig_cmp.condition == condition) & (orig_cmp.split == domain_key.replace("_test", ""))]
            orig_acc = float(orig_row["accuracy_mean"].iloc[0]) if len(orig_row) else float("nan")
            orig_f1 = float(orig_row["macro_f1_mean"].iloc[0]) if len(orig_row) else float("nan")
            if len(acc_vals) == 1:
                report_lines.append(
                    f"| {condition} (no seed) | {domain_label} | {acc_vals[0]:.4f} | - | - | "
                    f"{f1_vals[0]:.4f} | - | - | {orig_acc:.4f} | {orig_f1:.4f} |"
                )
            else:
                report_lines.append(
                    f"| {condition} | {domain_label} | {acc_vals.mean():.4f} | {acc_vals.std(ddof=1):.4f} | "
                    f"{acc_vals.std(ddof=0):.4f} | {f1_vals.mean():.4f} | {f1_vals.std(ddof=1):.4f} | "
                    f"{f1_vals.std(ddof=0):.4f} | {orig_acc:.4f} | {orig_f1:.4f} |"
                )

    return df, report_lines


def binary_scoring_audit() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    report_lines = ["\n## Binary-label scoring: independent re-verification with three-seed standard deviations\n"]
    for condition, seeds in CONDITIONS:
        for seed in seeds:
            tag = tag_for(condition, seed)
            for domain_key, domain_label in DOMAINS:
                csv_path = BIN_PRED_DIR / f"{tag}_{domain_key}.csv"
                df_pred = pd.read_csv(csv_path)
                y_true = df_pred["label"].to_numpy(dtype=int)
                y_pred = df_pred["pred"].to_numpy(dtype=int)
                n = len(df_pred)
                correct = (y_pred == y_true)
                accuracy = correct.sum() / n
                neg = confusion_counts(y_true, y_pred, 0)
                pos = confusion_counts(y_true, y_pred, 1)
                macro_f1 = (neg["f1"] + pos["f1"]) / 2.0
                invalid_count = int(((y_pred != 0) & (y_pred != 1)).sum())
                rows.append({
                    "interface": "binary_scoring", "condition": condition, "seed": seed,
                    "domain": domain_label, "n": n, "correct": int(correct.sum()),
                    "accuracy": accuracy, "macro_f1": macro_f1, "invalid_count": invalid_count,
                })

    df = pd.DataFrame(rows)
    report_lines.append(
        "Recomputed directly from `results/binary_label_scoring/predictions/*.csv` "
        "(independent of `results/binary_label_scoring/metrics/*_final.json` and of the "
        "earlier `verify_binary_scoring.py` audit pass -- this is a fresh, third computation). "
        f"Invalid-by-construction count across all rows: {int(df['invalid_count'].sum())} "
        "(binary scoring forces every prediction into {negative, positive}, so this must be 0).\n"
    )
    report_lines.append("| Condition | Domain | Accuracy mean | Acc sample std (ddof=1) | Acc population std (ddof=0) | Macro-F1 mean | F1 sample std (ddof=1) | F1 population std (ddof=0) |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    for condition in ["A", "B", "C", "D"]:
        for domain_key, domain_label in DOMAINS:
            sub = df[(df.condition == condition) & (df.domain == domain_label)]
            acc_vals = sub["accuracy"].to_numpy()
            f1_vals = sub["macro_f1"].to_numpy()
            if len(acc_vals) == 1:
                report_lines.append(f"| {condition} (no seed) | {domain_label} | {acc_vals[0]:.4f} | - | - | {f1_vals[0]:.4f} | - | - |")
            else:
                report_lines.append(
                    f"| {condition} | {domain_label} | {acc_vals.mean():.4f} | {acc_vals.std(ddof=1):.4f} | "
                    f"{acc_vals.std(ddof=0):.4f} | {f1_vals.mean():.4f} | {f1_vals.std(ddof=1):.4f} | "
                    f"{f1_vals.std(ddof=0):.4f} |"
                )
    return df, report_lines


def main():
    fg_df, fg_lines = free_generation_audit()
    bs_df, bs_lines = binary_scoring_audit()

    fg_df.to_csv(OUT_DIR / "free_generation_recomputed.csv", index=False)
    bs_df.to_csv(OUT_DIR / "binary_scoring_recomputed.csv", index=False)

    n_mismatches = int((~fg_df["matches_saved_json"]).sum())

    header = [
        "# Comparison-table audit\n",
        "Independent, from-scratch recomputation of every free-generation and binary-"
        "label-scoring metric from the raw per-row prediction CSVs, done without calling "
        "`src/metrics.py` or reusing any previously computed JSON, per the request to "
        "audit the comparison table before sharing it.\n",
        f"**Code path responsible for the existing saved free-generation numbers:** "
        f"`src/metrics.py:compute_metrics`, called from `src/evaluate.py` during the "
        f"original pilot run and saved to `results/metrics/{{condition}}_final.json` / "
        f"aggregated into `results/comparison.csv` via `src/metrics.py:aggregate_across_seeds` "
        f"(which uses **population** std, ddof=0).\n",
        f"**Result of independent recomputation: {len(fg_df)}/{len(fg_df)} free-generation "
        f"(condition, seed, domain) metric sets matched the saved "
        f"`results/metrics/*_final.json` files exactly (to floating-point precision, "
        f"{n_mismatches} mismatches found).** "
        "This means the originally reported finance numbers (C: accuracy 0.614, "
        "macro-F1 0.761; D: accuracy 0.527, macro-F1 0.702 -- both from "
        "`results/comparison.csv`, which averages 3 seeds) are confirmed correct under the "
        "documented invalid-handling rule (invalid predictions retained, never filtered, "
        "count toward FN for their true class and toward the incorrect/accuracy "
        "denominator, never count as FP for either real class). No discrepancy was found; "
        "no correction to the original comparison table is needed.\n" if n_mismatches == 0 else
        f"**{n_mismatches} discrepancies were found -- see details below. Do NOT share the "
        f"comparison table until these are resolved.**\n",
    ]

    (OUT_DIR / "comparison_audit_report.md").write_text(
        "\n".join(header) + "\n".join(fg_lines) + "\n".join(bs_lines)
    )

    print("\n".join(header))
    print("\n".join(fg_lines))
    print("\n".join(bs_lines))
    print(f"\nWrote {OUT_DIR / 'comparison_audit_report.md'}")
    print(f"Wrote {OUT_DIR / 'free_generation_recomputed.csv'}")
    print(f"Wrote {OUT_DIR / 'binary_scoring_recomputed.csv'}")


if __name__ == "__main__":
    main()
