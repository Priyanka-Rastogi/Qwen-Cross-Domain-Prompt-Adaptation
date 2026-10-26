"""Binary-label scoring experiment: re-evaluates the EXACT frozen prompts
(instruction + demonstrations) selected by the original pilot run
(`results/metrics/`, `artifacts/selected_prompts/`), replacing free-generation
decoding with conditional-likelihood scoring over the closed {negative,
positive} label set (see src/score_labels.py for the scoring rule).

What this script does NOT do (by requirement):
  - It does not fine-tune or otherwise update Qwen3-8B (still frozen, same
    QwenBackend.load() as the original run).
  - It does not re-run instruction proposal or demonstration search. Every
    instruction/demonstration set is read back from the original run's frozen
    manifests and independently hash-verified against the recorded
    instruction_hash / demonstrations_content_hash before use.
  - It does not touch results/predictions, results/metrics, results/report.md,
    or any other original free-generation artifact. All new outputs are
    written under results/binary_label_scoring/.
  - It never reads or maps the original "neutral"/invalid free-generation
    outputs onto positive/negative. Every prediction here comes from a fresh
    forward pass of the frozen model.

Usage:
  python src/run_binary_scoring.py --validate-only   # small source-only sanity check
  python src/run_binary_scoring.py                   # validation, then full A-D sweep
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from cache import EvalCache, make_cache_key
from evaluate import PhaseCost
from metrics import aggregate_across_seeds, compute_metrics
from model_backend import QwenBackend
from prompt_templates import build_prompt
from propose_instructions import BASELINE_INSTRUCTION
from run_experiment import json_default, save_json, save_predictions_csv
from score_labels import LABEL_TO_ID, SCORING_METHOD_CONFIG, build_scored_example, score_batch
from search_prompts import demo_row_for_prompt, order_demonstrations
from text_utils import text_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "binary_label_scoring"
ORIGINAL_RESULTS_DIR = REPO_ROOT / "results"  # read-only: original free-generation artifacts

SCORING_PREPROCESSING_CONFIG = {
    "per_example_max_tokens": 256,
    "truncation_rule": "head truncation (keep first N tokens) by tokenizer token count",
    "note": "identical prompt construction to the free-generation pipeline "
            "(src/prompt_templates.build_prompt); only the prediction interface differs",
}


def load_pilot_config() -> dict:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "pilot.yaml").read_text())
    cfg.setdefault("seeds", [0, 1, 2])
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("data_dir", "data")
    return cfg


def demos_from_manifest(source_train_df: pd.DataFrame, demo_ids: list[str], expected_hash: str) -> list[dict]:
    rows = []
    for did in demo_ids:
        match = source_train_df[source_train_df["id"] == did]
        if match.empty:
            raise ValueError(f"Demonstration id {did!r} from a frozen manifest not found in source_train.csv")
        row = match.iloc[0]
        rows.append({"id": row["id"], "text": row["text"], "label": int(row["label"]), "label_name": row["label_name"]})
    ordered = order_demonstrations(rows)
    recomputed_hash = text_hash("||".join(f"{d.get('label_name')}:{d.get('text')}" for d in ordered))
    if recomputed_hash != expected_hash:
        raise ValueError(
            f"Demonstration content hash mismatch while reconstructing {demo_ids}: "
            f"recomputed {recomputed_hash} != manifest {expected_hash}. Refusing to proceed "
            f"with an unverified demonstration set."
        )
    return [demo_row_for_prompt(d) for d in ordered]


def get_condition_config(condition: str, seed, source_train_df: pd.DataFrame) -> tuple[str, list[dict]]:
    """Returns (instruction, demonstration_prompt_rows), reusing the EXACT
    original frozen selection for this condition/seed. Raises if any
    hash-verification check fails (never silently proceeds on a mismatch)."""
    if condition == "A":
        return BASELINE_INSTRUCTION, []
    manifest_path = REPO_ROOT / "artifacts" / "selected_prompts" / f"{condition}_seed{seed}.json"
    manifest = json.loads(manifest_path.read_text())
    instruction = manifest["instruction"]
    if text_hash(instruction) != manifest["instruction_hash"]:
        raise ValueError(f"Instruction hash mismatch for {condition}_seed{seed} ({manifest_path})")
    if manifest["demonstration_ids"] is None:
        return instruction, []
    demos = demos_from_manifest(source_train_df, manifest["demonstration_ids"], manifest["demonstrations_content_hash"])
    return instruction, demos


def _build_result_row(ex: dict, value: dict) -> dict:
    return {
        "id": ex["id"], "label": ex["label"], "pred": LABEL_TO_ID[value["pred_label"]],
        "raw_label": value["pred_label"],
        "score_negative": value["score_negative"], "score_positive": value["score_positive"],
        "margin": value["score_positive"] - value["score_negative"],
        "tie": value["tie"],
        "negative_continuation_tokens": value["negative_continuation_tokens"],
        "positive_continuation_tokens": value["positive_continuation_tokens"],
        "negative_form_used": value["negative_form_used"],
        "positive_form_used": value["positive_form_used"],
        "prompt_tokens": value["prompt_tokens"],
        "truncated_demo": value["truncated_demo"], "truncated_query": value["truncated_query"],
        "from_cache": value.get("from_cache", False),
    }


def score_examples(backend, cache: EvalCache, examples: list[dict], instruction: str,
                    demonstrations: list[dict], batch_size: int, phase: str) -> tuple[list[dict], PhaseCost]:
    cost = PhaseCost(phase=phase)
    software_config = backend.software_config()
    results: list[dict | None] = [None] * len(examples)

    pending_idx, pending_scored, pending_keys, pending_built = [], [], [], []

    for i, ex in enumerate(examples):
        built = build_prompt(backend.tokenizer, instruction, demonstrations, ex["text"])
        key = make_cache_key(
            model_revision=backend.model_revision, tokenizer_revision=backend.model_revision,
            prompt_str=built.prompt_str, input_content_hash=text_hash(ex["text"]),
            preprocessing_config=SCORING_PREPROCESSING_CONFIG, decoding_config=SCORING_METHOD_CONFIG,
            software_config=software_config,
        )
        cached = cache.get(key)
        if cached is not None:
            cost.cache_hits += 1
            results[i] = _build_result_row(ex, {**cached, "from_cache": True})
        else:
            cost.cache_misses += 1
            pending_idx.append(i)
            pending_scored.append(build_scored_example(backend.tokenizer, built.prompt_str))
            pending_keys.append(key)
            pending_built.append(built)

    for start in range(0, len(pending_idx), batch_size):
        sl = slice(start, start + batch_size)
        idxs, batch_scored, keys, builts = pending_idx[sl], pending_scored[sl], pending_keys[sl], pending_built[sl]
        res = score_batch(backend, batch_scored)
        cost.model_invocations += 1
        cost.wall_clock_seconds += res.wall_clock_seconds
        cost.input_tokens += sum(res.input_token_counts) * 2  # prefix scored against both labels
        cost.output_tokens += sum(res.output_token_counts["negative"]) + sum(res.output_token_counts["positive"])
        cost.peak_allocated_bytes = max(cost.peak_allocated_bytes, res.peak_allocated_bytes)
        cost.peak_reserved_bytes = max(cost.peak_reserved_bytes, res.peak_reserved_bytes)

        for j, i in enumerate(idxs):
            ex, built, scored = examples[i], builts[j], batch_scored[j]
            value = {
                "pred_label": res.preds[j],
                "score_negative": res.scores[j]["negative"], "score_positive": res.scores[j]["positive"],
                "tie": res.ties[j],
                "negative_continuation_tokens": scored.labels["negative"].n_tokens,
                "positive_continuation_tokens": scored.labels["positive"].n_tokens,
                "negative_form_used": scored.labels["negative"].form_used,
                "positive_form_used": scored.labels["positive"].form_used,
                "prompt_tokens": len(scored.prefix_ids),
                "truncated_demo": any(r.truncated and r.role == "demo" for r in built.truncation_log),
                "truncated_query": any(r.truncated and r.role == "query" for r in built.truncation_log),
            }
            cache.put(keys[j], value)
            results[i] = _build_result_row(ex, {**value, "from_cache": False})

    cost.examples_classified = len(examples)
    assert all(r is not None for r in results)
    return results, cost


def validate_scoring(backend, cache: EvalCache, source_test_df: pd.DataFrame, n: int = 20) -> dict:
    """Pre-registered correctness check, run BEFORE the full sweep (req. 8).

    Key invariant exploited: for a SINGLE-TOKEN label, greedy free-generation's
    chosen first token is, by construction, the argmax over the ENTIRE
    vocabulary -- so if the original free-generation output for an example was
    already exactly "positive" or "negative" (i.e. valid), that label is
    certainly also the winner of the two-way {positive, negative} likelihood
    comparison this script performs. Any disagreement on such a case is not a
    matter of interpretation -- it indicates a tokenization-alignment or
    indexing bug in src/score_labels.py, and this function aborts before
    spending compute on the full run.
    """
    sample = source_test_df.head(n)[["id", "text", "label"]].to_dict("records")
    results, cost = score_examples(backend, cache, sample, BASELINE_INSTRUCTION, [],
                                    batch_size=4, phase="validation_sample")

    orig_path = ORIGINAL_RESULTS_DIR / "predictions" / "A_source_test.csv"
    orig_df = pd.read_csv(orig_path, encoding="utf-8")
    orig_map = {row["id"]: row for _, row in orig_df.iterrows()}

    report = {
        "n_samples": n, "cost": cost.as_dict(),
        "agreement_on_originally_valid": {"n_compared": 0, "n_agree": 0},
        "examples": [],
    }
    for r in results:
        orig = orig_map.get(r["id"])
        entry = {
            "id": r["id"], "true_label": r["label"], "scoring_pred": r["raw_label"],
            "score_negative": r["score_negative"], "score_positive": r["score_positive"],
            "margin": r["margin"],
            "negative_form_used": r["negative_form_used"], "positive_form_used": r["positive_form_used"],
            "negative_continuation_tokens": r["negative_continuation_tokens"],
            "positive_continuation_tokens": r["positive_continuation_tokens"],
        }
        if orig is not None:
            orig_norm = str(orig["raw_output"]).strip().lower()
            if orig_norm in ("positive", "negative"):
                entry["original_free_generation_pred"] = orig_norm
                entry["agrees_with_original_valid_generation"] = (orig_norm == r["raw_label"])
                report["agreement_on_originally_valid"]["n_compared"] += 1
                if entry["agrees_with_original_valid_generation"]:
                    report["agreement_on_originally_valid"]["n_agree"] += 1
        report["examples"].append(entry)

    save_json(OUT_DIR / "validation_sample.json", report)
    n_cmp = report["agreement_on_originally_valid"]["n_compared"]
    n_agree = report["agreement_on_originally_valid"]["n_agree"]
    print(f"[validation] {n} source-only samples scored. {n_cmp}/{n} had an originally-valid "
          f"free-generation label; scoring agreed on {n_agree}/{n_cmp}.", flush=True)
    if n_cmp > 0 and n_agree != n_cmp:
        raise RuntimeError(
            "Scoring disagrees with greedy free-generation on >=1 example where free-generation "
            "already produced a valid single-token label -- this is mathematically impossible if "
            "the scoring implementation is correct (see validate_scoring docstring). Aborting "
            "before the full run. Inspect results/binary_label_scoring/validation_sample.json."
        )
    print(f"[validation] PASSED. Saved {(OUT_DIR / 'validation_sample.json').resolve()}", flush=True)
    return report


def full_scale_agreement_check(all_condition_results: dict) -> dict:
    """Same invariant as validate_scoring, checked across the ENTIRE run
    (free, since it only needs the already-saved original predictions and the
    new scoring predictions -- no extra model calls)."""
    summary = {}
    for tag, paths in all_condition_results.items():
        for split, new_df in paths.items():
            orig_path = ORIGINAL_RESULTS_DIR / "predictions" / f"{tag}_{split}.csv"
            if not orig_path.exists():
                continue
            orig_df = pd.read_csv(orig_path, encoding="utf-8")
            orig_df["raw_output_norm"] = orig_df["raw_output"].astype(str).str.strip().str.lower()
            orig_valid = orig_df[orig_df["raw_output_norm"].isin(["positive", "negative"])]
            merged = orig_valid.merge(new_df[["id", "raw_label"]], on="id", how="left")
            n_compared = len(merged)
            n_agree = int((merged["raw_output_norm"] == merged["raw_label"]).sum())
            summary[f"{tag}_{split}"] = {"n_compared": n_compared, "n_agree": n_agree,
                                          "agreement_rate": (n_agree / n_compared) if n_compared else None}
    return summary


def run(validate_only: bool = False):
    cfg = load_pilot_config()
    data_dir = REPO_ROOT / cfg["data_dir"]
    source_train_df = pd.read_csv(data_dir / "source_train.csv", encoding="utf-8")
    source_test_df = pd.read_csv(data_dir / "source_test.csv", encoding="utf-8")
    target_test_df = pd.read_csv(data_dir / "target_test.csv", encoding="utf-8")
    source_test_examples = source_test_df[["id", "text", "label"]].to_dict("records")
    target_test_examples = target_test_df[["id", "text", "label"]].to_dict("records")

    for sub in ("predictions", "metrics", "logs"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    cache = EvalCache(REPO_ROOT / "artifacts" / "cache" / "binary_label_scoring_cache.jsonl")

    t0 = time.perf_counter()
    backend = QwenBackend.load()
    print(f"[model_loading] {time.perf_counter() - t0:.1f}s revision={backend.model_revision}", flush=True)

    validate_scoring(backend, cache, source_test_df, n=20)
    if validate_only:
        print("[validate-only] stopping after validation, as requested.")
        return

    batch_size = max(1, cfg.get("batch_size", 16) // 2)  # 2 sequences/example scored per forward call
    seeds = cfg["seeds"]

    all_costs = []
    results_summary = {}
    all_condition_dfs: dict[str, dict[str, pd.DataFrame]] = {}

    def run_condition(condition: str, seed, instruction: str, demos: list[dict]) -> dict:
        tag = "A" if condition == "A" else f"{condition}_seed{seed}"
        t_cond = time.perf_counter()
        src_results, src_cost = score_examples(backend, cache, source_test_examples, instruction, demos,
                                                batch_size, f"{tag}_source_test")
        tgt_results, tgt_cost = score_examples(backend, cache, target_test_examples, instruction, demos,
                                                batch_size, f"{tag}_target_test")
        src_metrics = compute_metrics([r["label"] for r in src_results], [r["pred"] for r in src_results])
        tgt_metrics = compute_metrics([r["label"] for r in tgt_results], [r["pred"] for r in tgt_results])

        src_df = pd.DataFrame(src_results)
        tgt_df = pd.DataFrame(tgt_results)
        save_predictions_csv(OUT_DIR / "predictions" / f"{tag}_source_test.csv", src_results)
        save_predictions_csv(OUT_DIR / "predictions" / f"{tag}_target_test.csv", tgt_results)
        all_condition_dfs[tag] = {"source_test": src_df, "target_test": tgt_df}

        all_costs.append({**src_cost.as_dict(), "phase": f"{tag}_source_test"})
        all_costs.append({**tgt_cost.as_dict(), "phase": f"{tag}_target_test"})
        result = {
            "condition": condition, "seed": seed, "instruction": instruction,
            "demonstration_ids": [d.get("id") for d in demos] if demos else None,
            "source_test": {"metrics": src_metrics, "cost": src_cost.as_dict()},
            "target_test": {"metrics": tgt_metrics, "cost": tgt_cost.as_dict()},
            "wall_clock_seconds_total": time.perf_counter() - t_cond,
        }
        save_json(OUT_DIR / "metrics" / f"{tag}_final.json", result)
        print(f"[{tag}] src_acc={src_metrics['accuracy']:.4f} macro_f1={src_metrics['macro_f1']:.4f}  "
              f"tgt_acc={tgt_metrics['accuracy']:.4f} macro_f1={tgt_metrics['macro_f1']:.4f}  "
              f"({result['wall_clock_seconds_total']:.1f}s)", flush=True)
        return result

    instruction_a, demos_a = get_condition_config("A", None, source_train_df)
    results_summary["A"] = run_condition("A", None, instruction_a, demos_a)

    for seed in seeds:
        for cond in ["B", "C", "D"]:
            instr, demos = get_condition_config(cond, seed, source_train_df)
            results_summary[f"{cond}_seed{seed}"] = run_condition(cond, seed, instr, demos)

    summary = {"A": results_summary["A"], "per_seed": results_summary}
    for cond in ["B", "C", "D"]:
        src_list = [results_summary[f"{cond}_seed{s}"]["source_test"]["metrics"] for s in seeds]
        tgt_list = [results_summary[f"{cond}_seed{s}"]["target_test"]["metrics"] for s in seeds]
        summary[cond] = {
            "source_test_aggregate": aggregate_across_seeds(src_list),
            "target_test_aggregate": aggregate_across_seeds(tgt_list),
        }
    save_json(OUT_DIR / "metrics" / "summary.json", summary)
    save_json(OUT_DIR / "logs" / "phase_costs.json", all_costs)

    agreement = full_scale_agreement_check(all_condition_dfs)
    save_json(OUT_DIR / "logs" / "full_scale_agreement_check.json", agreement)
    print("[agreement check, full scale]")
    for k, v in agreement.items():
        print(f"  {k}: {v['n_agree']}/{v['n_compared']} "
              f"({v['agreement_rate']*100:.2f}%)" if v["n_compared"] else f"  {k}: n/a (0 originally-valid cases)")

    print(f"\nWrote summary to {(OUT_DIR / 'metrics' / 'summary.json').resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run(validate_only=args.validate_only)


if __name__ == "__main__":
    main()
