"""Metrics with explicit, documented handling of invalid (unparseable) predictions.

Invalid predictions are encoded as -1 (neither label 0 nor 1). They:
  - count as incorrect for accuracy (never dropped from the denominator),
  - are never counted as a false positive for either class (sklearn only scores
    labels=[0, 1], so a -1 prediction cannot match either class),
  - do reduce recall for their true class, because the true-class example was not
    correctly predicted,
  - are shown separately in the confusion table as their own column.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

INVALID_PRED = -1


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    n = len(y_true)

    correct = (y_pred == y_true) & (y_pred != INVALID_PRED)
    accuracy = correct.sum() / n if n else 0.0

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))

    invalid_count = int((y_pred == INVALID_PRED).sum())
    invalid_rate = invalid_count / n if n else 0.0

    confusion = {
        "negative": {
            "pred_negative": int(((y_true == 0) & (y_pred == 0)).sum()),
            "pred_positive": int(((y_true == 0) & (y_pred == 1)).sum()),
            "pred_invalid": int(((y_true == 0) & (y_pred == INVALID_PRED)).sum()),
        },
        "positive": {
            "pred_negative": int(((y_true == 1) & (y_pred == 0)).sum()),
            "pred_positive": int(((y_true == 1) & (y_pred == 1)).sum()),
            "pred_invalid": int(((y_true == 1) & (y_pred == INVALID_PRED)).sum()),
        },
    }

    return {
        "n": n,
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": {
            "negative": {"precision": float(precision[0]), "recall": float(recall[0]),
                         "f1": float(f1[0]), "support": int(support[0])},
            "positive": {"precision": float(precision[1]), "recall": float(recall[1]),
                         "f1": float(f1[1]), "support": int(support[1])},
        },
        "confusion_matrix": confusion,
        "invalid_count": invalid_count,
        "invalid_rate": float(invalid_rate),
    }


def aggregate_across_seeds(seed_metric_dicts: list[dict]) -> dict:
    """Mean/std of scalar metrics across seeds. Not a claim of independent datasets —
    seeds vary candidate generation/demonstration sampling only; the eval sets are fixed."""
    keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "invalid_rate"]
    out = {}
    for k in keys:
        vals = [m[k] for m in seed_metric_dicts]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    out["n_seeds"] = len(seed_metric_dicts)
    return out
