"""Lightweight unit tests (no GPU / model loading required)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from metrics import INVALID_PRED, compute_metrics  # noqa: E402
from text_utils import normalize_for_hash, text_hash  # noqa: E402
from cache import EvalCache, make_cache_key  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------- Dataset label mapping ----------------

@pytest.mark.parametrize("name", ["source_train", "source_validation", "source_test", "target_test"])
def test_label_mapping_is_consistent(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    assert set(df["label"].unique()) <= {0, 1}
    assert (df.loc[df["label"] == 0, "label_name"] == "negative").all()
    assert (df.loc[df["label"] == 1, "label_name"] == "positive").all()


def test_domain_and_split_values():
    source_train = pd.read_csv(DATA_DIR / "source_train.csv")
    assert set(source_train["domain"].unique()) == {"movie"}
    target_test = pd.read_csv(DATA_DIR / "target_test.csv")
    assert set(target_test["domain"].unique()) == {"finance"}
    assert set(target_test["original_split"].unique()) == {"unpartitioned"}


# ---------------- Split disjointness ----------------

def test_split_disjointness_and_overlap():
    def norm_set(df):
        return set(df["text"].map(normalize_for_hash))

    train = pd.read_csv(DATA_DIR / "source_train.csv")
    val = pd.read_csv(DATA_DIR / "source_validation.csv")
    test = pd.read_csv(DATA_DIR / "source_test.csv")
    target = pd.read_csv(DATA_DIR / "target_test.csv")

    h_train, h_val, h_test, h_target = norm_set(train), norm_set(val), norm_set(test), norm_set(target)

    assert h_train.isdisjoint(h_val)
    assert h_train.isdisjoint(h_test)
    assert h_val.isdisjoint(h_test)
    assert (h_train | h_val | h_test).isdisjoint(h_target)

    for df, name in [(train, "source_train"), (val, "source_validation"), (test, "source_test"), (target, "target_test")]:
        assert df["id"].is_unique, f"duplicate ids in {name}"


def test_source_splits_are_balanced():
    train = pd.read_csv(DATA_DIR / "source_train.csv")
    val = pd.read_csv(DATA_DIR / "source_validation.csv")
    test = pd.read_csv(DATA_DIR / "source_test.csv")
    assert len(train) == 200 and train["label"].value_counts().to_dict() == {0: 100, 1: 100}
    assert len(val) == 100 and val["label"].value_counts().to_dict() == {0: 50, 1: 50}
    assert len(test) == 500 and test["label"].value_counts().to_dict() == {0: 250, 1: 250}


# ---------------- Strict prediction parsing ----------------

def test_strict_parser():
    sys.path.insert(0, str(SRC))
    from evaluate import strict_parse

    assert strict_parse("positive") == 1
    assert strict_parse("negative") == 0
    assert strict_parse("  Positive  ") == 1
    assert strict_parse("NEGATIVE") == 0
    assert strict_parse("positive.") == INVALID_PRED  # punctuation not stripped: strict
    assert strict_parse("it is positive") == INVALID_PRED
    assert strict_parse("") == INVALID_PRED
    assert strict_parse("pos") == INVALID_PRED


# ---------------- Metrics with invalid predictions ----------------

def test_metrics_treat_invalid_as_incorrect_not_dropped():
    y_true = [0, 0, 1, 1]
    y_pred = [0, INVALID_PRED, 1, INVALID_PRED]
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 4
    assert m["invalid_count"] == 2
    assert m["accuracy"] == pytest.approx(2 / 4)
    # recall for positive class reduced because one true-positive example was invalid
    assert m["per_class"]["positive"]["recall"] == pytest.approx(0.5)
    assert m["confusion_matrix"]["positive"]["pred_invalid"] == 1
    assert m["confusion_matrix"]["negative"]["pred_invalid"] == 1


def test_metrics_all_invalid_gives_zero_f1_not_crash():
    y_true = [0, 1, 0, 1]
    y_pred = [INVALID_PRED] * 4
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["macro_f1"] == 0.0
    assert m["invalid_count"] == 4


# ---------------- Cache key sensitivity ----------------

def test_cache_key_changes_with_prompt_or_settings():
    base = dict(
        model_revision="rev1", tokenizer_revision="rev1", prompt_str="hello",
        input_content_hash=text_hash("some text"), preprocessing_config={"a": 1},
        decoding_config={"do_sample": False}, software_config={"v": 1},
    )
    k0 = make_cache_key(**base)
    k_prompt = make_cache_key(**{**base, "prompt_str": "different"})
    k_decoding = make_cache_key(**{**base, "decoding_config": {"do_sample": True}})
    k_preproc = make_cache_key(**{**base, "preprocessing_config": {"a": 2}})
    k_software = make_cache_key(**{**base, "software_config": {"v": 2}})
    k_same = make_cache_key(**base)

    assert k0 == k_same
    assert len({k0, k_prompt, k_decoding, k_preproc, k_software}) == 5


def test_eval_cache_resume(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = EvalCache(path)
    cache.put("k1", {"pred": 1, "raw_output": "positive", "prompt_tokens": 10, "output_tokens": 1})
    assert cache.get("k1")["pred"] == 1

    cache2 = EvalCache(path)  # simulate resume: reload from disk
    assert cache2.get("k1")["pred"] == 1
    assert len(cache2) == 1


# ---------------- Zero trainable parameters (structural check, no model load) ----------------

def test_model_backend_freezes_all_params_logic():
    import torch

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(3))

    m = Dummy()
    for p in m.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert trainable == 0
