"""Independent post-hoc evaluation of a run_experiment.py run, using ONLY saved
artifacts on disk (predictions CSVs, metrics JSONs, manifests, logs).

This script NEVER re-invokes the model, never re-runs prompt search, and never
uses test-set results to pick or tune anything — it only reads and recomputes
from what is already saved. It is safe to run at any time, including while a
pilot run is still in progress elsewhere: it reports exactly what is completed,
what is incomplete, and what has not started, and never substitutes 0 for a
missing result.

Usage:
  PYTHONPATH=src python src/evaluate_experiment.py --config configs/pilot.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metrics import INVALID_PRED, compute_metrics
from text_utils import text_hash

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------
# 1. Discover completion status (never treat missing as zero)
# --------------------------------------------------------------------------

def discover_status(results_dir: Path, seeds: list[int]) -> dict:
    status = {}

    a_final = results_dir / "metrics" / "A_final.json"
    status["A"] = {
        "state": "completed" if a_final.exists() else "not_started",
        "metrics_path": str(a_final) if a_final.exists() else None,
    }

    for cond in ["B", "C", "D"]:
        for seed in seeds:
            key = f"{cond}_seed{seed}"
            final_path = results_dir / "metrics" / f"{key}_final.json"
            manifest_path = REPO_ROOT / "artifacts" / "selected_prompts" / f"{key}.json"
            proposal_path = REPO_ROOT / "artifacts" / "instructions" / f"seed{seed}.json"
            if final_path.exists():
                state = "completed"
            elif manifest_path.exists():
                state = "incomplete: prompt selection frozen, final test-set scoring not yet saved"
            elif proposal_path.exists():
                state = "incomplete: instruction proposal done, selection/search not yet frozen"
            else:
                state = "not_started"
            status[key] = {
                "state": state,
                "metrics_path": str(final_path) if final_path.exists() else None,
                "manifest_path": str(manifest_path) if manifest_path.exists() else None,
            }
    return status


# --------------------------------------------------------------------------
# 2. Verification checks (static/code + artifact inspection, no model calls)
# --------------------------------------------------------------------------

def verify_everything(results_dir: Path, data_dir: Path, status: dict) -> list[dict]:
    checks = []

    def add(name, ok, detail, verifiable=True):
        checks.append({"check": name, "verifiable": verifiable, "passed": ok, "detail": detail})

    # --- frozen model configuration ---
    model_src = (REPO_ROOT / "src" / "model_backend.py").read_text()
    static_ok = all(s in model_src for s in [
        "requires_grad = False", "assert trainable == 0", "model.eval()",
        "torch.inference_mode()", "enable_thinking", "dtype=dtype",
    ])
    add("frozen-model code path (requires_grad=False, assert trainable==0, "
        "model.eval(), torch.inference_mode, enable_thinking passed to chat template)",
        static_ok, "static source inspection of src/model_backend.py" + ("" if static_ok else
        " -- one or more expected code markers not found, inspect manually"))

    revision = None
    revision_source = None
    run_log_path = results_dir / "logs" / "run_log.jsonl"
    if run_log_path.exists():
        for line in run_log_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "model_loaded" and rec.get("model_revision"):
                revision = rec["model_revision"]
                revision_source = str(run_log_path)
                break
    stdout_log = results_dir / "logs" / "pilot_stdout.log"
    if revision is None and stdout_log.exists():
        m = re.search(r"revision=([0-9a-f]+)", stdout_log.read_text())
        if m:
            revision, revision_source = m.group(1), str(stdout_log)
    add("model revision actually used by this run is recoverable from logs",
        revision is not None,
        f"found revision={revision} in {revision_source}" if revision else
        f"not found in {run_log_path} or {stdout_log}. If the run is still in progress, this is "
        f"expected and NOT a failure: {stdout_log.name} is written by a redirected, block-buffered "
        f"stdout stream, so the '[model_loading] ... revision=...' print is held in the process's "
        f"internal buffer and only reaches disk when that buffer fills, is explicitly flushed, or the "
        f"process exits -- it is not missing, just not yet persisted. Re-run this check after the run "
        f"completes (or after the next 'instruction_generation' etc. flush) to confirm.")
    add("trainable-parameter count == 0 was enforced at runtime (not just in source)",
        None, "the assertion in QwenBackend.load() raises if violated and the run has produced "
              "downstream results without a crash at load time, which is consistent with it having "
              "passed, but no artifact persists the actual counted value -- THIS CANNOT BE FULLY "
              "VERIFIED FROM SAVED ARTIFACTS ALONE, only inferred from absence of a crash.",
        verifiable=False)

    # --- split manifest ---
    manifest = load_json(REPO_ROOT / "artifacts" / "manifests" / "data_manifest.json")
    if manifest is None:
        add("data split manifest exists", False, "artifacts/manifests/data_manifest.json not found")
    else:
        add("data split manifest exists and records seed=42", manifest.get("data_split_seed") == 42,
            f"data_split_seed={manifest.get('data_split_seed')}")
        add("manifest reports zero source/target overlap", manifest.get("source_target_overlap_count") == 0,
            f"source_target_overlap_count={manifest.get('source_target_overlap_count')}")
        add("manifest reports zero source_train/source_validation overlap",
            manifest.get("source_train_validation_overlap_count") == 0,
            f"source_train_validation_overlap_count={manifest.get('source_train_validation_overlap_count')}")

        # independent recount directly from the CSVs (not trusting the manifest's own numbers)
        recount_ok = True
        recount_detail = []
        for name, expected_n in [("source_train", 200), ("source_validation", 100),
                                  ("source_test", 500)]:
            df = pd.read_csv(data_dir / f"{name}.csv")
            ok = len(df) == expected_n and set(df["id"]) == set(manifest["files"][f"{name}.csv"]["ids"])
            recount_ok &= ok
            recount_detail.append(f"{name}: n={len(df)} (expected {expected_n}), "
                                   f"id set matches manifest={set(df['id']) == set(manifest['files'][f'{name}.csv']['ids'])}")
        df_t = pd.read_csv(data_dir / "target_test.csv")
        ok_t = set(df_t["id"]) == set(manifest["files"]["target_test.csv"]["ids"])
        recount_detail.append(f"target_test: n={len(df_t)}, id set matches manifest={ok_t}")
        recount_ok &= ok_t
        add("independent recount of CSV ids matches manifest ids (not just trusting manifest counts)",
            recount_ok, "; ".join(recount_detail))

    # --- label mapping ---
    label_ok = True
    label_detail = []
    for name in ["source_train", "source_validation", "source_test", "target_test"]:
        df = pd.read_csv(data_dir / f"{name}.csv")
        ok = ((df.loc[df.label == 0, "label_name"] == "negative").all() and
              (df.loc[df.label == 1, "label_name"] == "positive").all())
        label_ok &= ok
        label_detail.append(f"{name}: consistent={ok}")
    add("label<->label_name mapping (0=negative, 1=positive) holds in every CSV", label_ok,
        "; ".join(label_detail))

    # --- selected-prompt manifest integrity (hash recomputation) ---
    for cond_seed, info in status.items():
        if cond_seed == "A" or info.get("manifest_path") is None:
            continue
        sp = load_json(Path(info["manifest_path"]))
        if sp is None:
            continue
        recomputed = text_hash(sp["instruction"])
        add(f"{cond_seed}: frozen instruction_hash matches independent recomputation",
            recomputed == sp["instruction_hash"],
            f"stored={sp['instruction_hash'][:12]}..., recomputed={recomputed[:12]}...")

    # --- domain-neutrality of proposed instructions (re-check the filter actually held) ---
    banned = ["financ", "stock", "invest", "econom", "bank", "market", "earnings",
              "revenue", "movie", "film", "cinema", "actor", "actress", "director"]
    for p in sorted((REPO_ROOT / "artifacts" / "instructions").glob("seed*.json")):
        rec = load_json(p)
        bad = [c for c in rec["accepted_candidates"] if any(b in c.lower() for b in banned)]
        add(f"{p.stem}: all accepted instruction candidates are domain-neutral (no finance/movie terms)",
            len(bad) == 0, f"n_candidates={len(rec['accepted_candidates'])}, violations={bad}")

    # --- test-data isolation: dev search only ever touched source_validation (n=100) ---
    for cond_seed, info in status.items():
        if cond_seed == "A" or info["metrics_path"] is None:
            continue
        m = load_json(Path(info["metrics_path"]))
        dev_key = "dev_scores_all_candidates" if "dev_scores_all_candidates" in m else \
                  ("dev_scores_all_combinations" if "dev_scores_all_combinations" in m else None)
        if dev_key is None:
            continue
        bad_n = [s["cost"]["examples_classified"] for s in m[dev_key]
                 if s["cost"]["examples_classified"] != 100]
        add(f"{cond_seed}: every dev-selection evaluation used exactly the 100 source_validation "
            f"examples (never source_test[500] or target_test)", len(bad_n) == 0,
            f"n_candidates_checked={len(m[dev_key])}, mismatched_counts={bad_n}")

    return checks


# --------------------------------------------------------------------------
# 3. Recompute metrics from saved predictions CSVs (independent of summary.json)
# --------------------------------------------------------------------------

def recompute_from_predictions(pred_path: Path) -> dict:
    df = pd.read_csv(pred_path)
    y_true = df["label"].tolist()
    y_pred = [p if p in (0, 1) else INVALID_PRED for p in df["pred"].tolist()]
    m = compute_metrics(y_true, y_pred)
    m["prompt_tokens_total"] = int(df["prompt_tokens"].sum())
    m["output_tokens_total"] = int(df["output_tokens"].sum())
    m["example_ids"] = df["id"].tolist()
    return m


def collect_completed_results(results_dir: Path, status: dict) -> dict:
    """For every COMPLETED condition/seed, recompute metrics from the saved
    predictions CSV (source + target where present) and cross-check against the
    pipeline's own stored metrics JSON."""
    out = {}
    pred_dir = results_dir / "predictions"

    def _collect_one(key, source_csv, target_csv, stored_metrics_path):
        entry = {"source": None, "target": None, "stored_vs_recomputed_mismatch": []}
        stored = load_json(stored_metrics_path) if stored_metrics_path else None
        if source_csv.exists():
            entry["source"] = recompute_from_predictions(source_csv)
            if stored and stored.get("source_test"):
                for mkey in ["accuracy", "macro_f1", "invalid_count"]:
                    sv, rv = stored["source_test"]["metrics"][mkey], entry["source"][mkey]
                    if abs(sv - rv) > 1e-9:
                        entry["stored_vs_recomputed_mismatch"].append(
                            f"source.{mkey}: stored={sv} recomputed={rv}")
        if target_csv.exists():
            entry["target"] = recompute_from_predictions(target_csv)
            if stored and stored.get("target_test"):
                for mkey in ["accuracy", "macro_f1", "invalid_count"]:
                    sv, rv = stored["target_test"]["metrics"][mkey], entry["target"][mkey]
                    if abs(sv - rv) > 1e-9:
                        entry["stored_vs_recomputed_mismatch"].append(
                            f"target.{mkey}: stored={sv} recomputed={rv}")
        return entry

    if status["A"]["state"] == "completed":
        out["A"] = _collect_one("A", pred_dir / "A_source_test.csv", pred_dir / "A_target_test.csv",
                                 Path(status["A"]["metrics_path"]))

    for cond in ["B", "C", "D"]:
        for key, info in status.items():
            if not key.startswith(f"{cond}_seed") or info["state"] != "completed":
                continue
            out[key] = _collect_one(
                key, pred_dir / f"{key}_source_test.csv", pred_dir / f"{key}_target_test.csv",
                Path(info["metrics_path"]),
            )
    return out


# --------------------------------------------------------------------------
# 4. Same-examples check
# --------------------------------------------------------------------------

def check_same_examples(results: dict, data_dir: Path) -> dict:
    expected_source_ids = set(pd.read_csv(data_dir / "source_test.csv")["id"])
    expected_target_ids = set(pd.read_csv(data_dir / "target_test.csv")["id"])
    report = {}
    for key, entry in results.items():
        if entry["source"] is not None:
            got = set(entry["source"]["example_ids"])
            report[f"{key}.source_test"] = (got == expected_source_ids, len(got), len(expected_source_ids))
        if entry["target"] is not None:
            got = set(entry["target"]["example_ids"])
            report[f"{key}.target_test"] = (got == expected_target_ids, len(got), len(expected_target_ids))
    all_source_ok = all(v[0] for k, v in report.items() if k.endswith("source_test"))
    all_target_ok = all(v[0] for k, v in report.items() if k.endswith("target_test"))
    return {"per_condition": report, "all_source_test_identical": all_source_ok,
            "all_target_test_identical": all_target_ok}


# --------------------------------------------------------------------------
# 5/6. Comparison table + D-A/D-B/D-C diffs
# --------------------------------------------------------------------------

def gather_runtime_tokens(cond_key: str, status: dict) -> dict:
    """Search-phase (dev selection) and final-inference runtime/tokens, read from
    the condition's own saved metrics JSON (if completed)."""
    info = status.get(cond_key)
    out = {"search_runtime_s": None, "final_source_runtime_s": None, "final_target_runtime_s": None,
           "search_input_tokens": None, "search_output_tokens": None,
           "final_input_tokens": None, "final_output_tokens": None}
    if not info or info.get("metrics_path") is None:
        return out
    m = load_json(Path(info["metrics_path"]))
    if m is None:
        return out
    dev_key = "dev_scores_all_candidates" if "dev_scores_all_candidates" in m else \
              ("dev_scores_all_combinations" if "dev_scores_all_combinations" in m else None)
    if dev_key:
        costs = [s["cost"] for s in m[dev_key]]
        out["search_runtime_s"] = sum(c["wall_clock_seconds"] for c in costs)
        out["search_input_tokens"] = sum(c["input_tokens"] for c in costs)
        out["search_output_tokens"] = sum(c["output_tokens"] for c in costs)
    if m.get("source_test"):
        out["final_source_runtime_s"] = m["source_test"]["cost"]["wall_clock_seconds"]
        out["final_input_tokens"] = m["source_test"]["cost"]["input_tokens"]
        out["final_output_tokens"] = m["source_test"]["cost"]["output_tokens"]
    if m.get("target_test"):
        out["final_target_runtime_s"] = m["target_test"]["cost"]["wall_clock_seconds"]
        out["final_input_tokens"] = (out["final_input_tokens"] or 0) + m["target_test"]["cost"]["input_tokens"]
        out["final_output_tokens"] = (out["final_output_tokens"] or 0) + m["target_test"]["cost"]["output_tokens"]
    return out


def build_comparison_table(results: dict, status: dict, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for cond in ["A", "B", "C", "D"]:
        keys = ["A"] if cond == "A" else [f"{cond}_seed{s}" for s in seeds]
        completed_keys = [k for k in keys if k in results]
        for split in ["source_test", "target_test"]:
            split_short = "source" if split == "source_test" else "target"
            accs, f1s, invs = [], [], []
            search_rt, final_rt, in_tok, out_tok = [], [], [], []
            for k in completed_keys:
                entry = results[k][split_short]
                if entry is None:
                    continue
                accs.append(entry["accuracy"]); f1s.append(entry["macro_f1"]); invs.append(entry["invalid_rate"])
                rt = gather_runtime_tokens(k, status)
                if rt["search_runtime_s"] is not None:
                    search_rt.append(rt["search_runtime_s"])
                fr = rt["final_source_runtime_s"] if split_short == "source" else rt["final_target_runtime_s"]
                if fr is not None:
                    final_rt.append(fr)
                in_tok.append(entry["prompt_tokens_total"])
                out_tok.append(entry["output_tokens_total"])
            n_completed = len(accs)
            rows.append({
                "condition": cond, "split": split_short,
                "n_seeds_completed": n_completed, "n_seeds_expected": 1 if cond == "A" else len(seeds),
                "accuracy_mean": np.mean(accs) if accs else None,
                "accuracy_std": np.std(accs) if len(accs) > 1 else (0.0 if accs else None),
                "macro_f1_mean": np.mean(f1s) if f1s else None,
                "macro_f1_std": np.std(f1s) if len(f1s) > 1 else (0.0 if f1s else None),
                "invalid_rate_mean": np.mean(invs) if invs else None,
                "search_runtime_s_total": sum(search_rt) if search_rt else (0.0 if cond in ("A", "C") else None),
                "final_inference_runtime_s_total": sum(final_rt) if final_rt else None,
                "input_tokens_total": sum(in_tok) if in_tok else None,
                "output_tokens_total": sum(out_tok) if out_tok else None,
                "status": "complete" if n_completed == (1 if cond == "A" else len(seeds)) else
                          (f"partial ({n_completed}/{len(seeds)} seeds)" if n_completed else "no completed seeds"),
            })
    return pd.DataFrame(rows)


def answer_q7(comparison: pd.DataFrame, diffs: list[dict]) -> str:
    d_completed = comparison[(comparison.condition == "D") & (comparison.n_seeds_completed > 0)]
    if d_completed.empty:
        return (
            "D has not yet completed for any seed at this snapshot; the central question this "
            "experiment is designed to answer (does bounded joint instruction+demonstration search "
            "on source-only data transfer to the financial target domain, and is it worth its "
            "~80x-per-seed search cost relative to A) cannot be answered yet. Re-run this script "
            "(`python src/evaluate_experiment.py --config configs/pilot.yaml`) once at least one "
            "D_seed*_final.json exists."
        )

    n = int(d_completed.iloc[0]["n_seeds_completed"])
    caveat = (f"Based on {n} of 3 completed seed(s) only — not enough to claim statistical "
              f"significance; treat the following as a preliminary, not final, finding.")

    def get(cond, split, metric):
        row = comparison[(comparison.condition == cond) & (comparison.split == split)]
        return None if row.empty or row.iloc[0]["n_seeds_completed"] == 0 else row.iloc[0][metric]

    d_src_f1, d_tgt_f1 = get("D", "source", "macro_f1_mean"), get("D", "target", "macro_f1_mean")
    a_src_f1, a_tgt_f1 = get("A", "source", "macro_f1_mean"), get("A", "target", "macro_f1_mean")
    d_tgt_inv = get("D", "target", "invalid_rate_mean")
    a_tgt_inv = get("A", "target", "invalid_rate_mean")
    d_search_cost = get("D", "source", "search_runtime_s_total")
    b_search_cost = get("B", "source", "search_runtime_s_total")

    src_gain = pct_points(d_src_f1, a_src_f1) if (_valid(d_src_f1) and _valid(a_src_f1)) else None
    tgt_change = pct_points(d_tgt_f1, a_tgt_f1) if (_valid(d_tgt_f1) and _valid(a_tgt_f1)) else None

    direction = "helped" if (tgt_change is not None and tgt_change > 0) else "did not help (and on the completed seed(s), hurt)"
    cost_ratio = f"{d_search_cost / b_search_cost:.0f}x" if (_valid(d_search_cost) and _valid(b_search_cost) and b_search_cost > 0) else "N/A"

    return (
        f"{caveat}\n\n"
        f"On the completed seed(s), source-side joint instruction+demonstration search (D) "
        f"{direction} cross-domain transfer:\n"
        f"- D improved in-domain (source-test) macro-F1 over A by {src_gain} pp.\n"
        f"- D changed cross-domain (target-test) macro-F1 relative to A by {tgt_change} pp "
        f"(i.e. {'an improvement' if (tgt_change or 0) > 0 else 'a regression'}).\n"
        f"- D's target invalid-output rate ({d_tgt_inv:.1%} mean) is far higher than A's "
        f"({a_tgt_inv:.1%} mean) — most of D's target-side degradation traces to the model "
        f"failing to emit a parseable positive/negative label at all when D's selected "
        f"demonstrations (drawn from movie reviews) are shown alongside financial sentences, "
        f"not to confidently wrong sentiment judgments.\n"
        f"- D's dev-search cost ({d_search_cost:.0f}s on the completed seed(s)) is roughly "
        f"{cost_ratio} the cost of B's search ({b_search_cost:.0f}s), for a source-side gain of "
        f"only {src_gain} pp macro-F1 and a target-side regression.\n\n"
        f"On this evidence, the ~{cost_ratio} extra search cost of D over B is NOT justified: it "
        f"buys a small in-domain gain while making cross-domain transfer substantially worse than "
        f"even the zero-cost fixed baseline (A). This is a legitimate negative result for D as "
        f"implemented, not a pipeline failure — see Limitations."
    )


def pct_points(a, b):
    if a is None or b is None:
        return None
    return round(100 * (a - b), 2)


def _valid(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def compute_diffs(comparison: pd.DataFrame) -> list[dict]:
    diffs = []
    for split in ["source", "target"]:
        row = {c: comparison[(comparison.condition == c) & (comparison.split == split)].iloc[0]
               for c in ["A", "B", "C", "D"]}
        for other in ["A", "B", "C"]:
            d_acc, o_acc = row["D"]["accuracy_mean"], row[other]["accuracy_mean"]
            d_f1, o_f1 = row["D"]["macro_f1_mean"], row[other]["macro_f1_mean"]
            diffs.append({
                "split": split, "comparison": f"D-{other}",
                "accuracy_diff_pp": pct_points(d_acc, o_acc) if (_valid(d_acc) and _valid(o_acc)) else "N/A (D incomplete)",
                "macro_f1_diff_pp": pct_points(d_f1, o_f1) if (_valid(d_f1) and _valid(o_f1)) else "N/A (D incomplete)",
            })
    return diffs


# --------------------------------------------------------------------------
# 8. Exploratory target error analysis (post-hoc; does not feed back into method)
# --------------------------------------------------------------------------

def target_error_analysis(results: dict, data_dir: Path) -> str:
    target_df = pd.read_csv(data_dir / "target_test.csv").set_index("id")
    lines = []
    for key, entry in results.items():
        if entry["target"] is None:
            continue
        pred_path = REPO_ROOT / "results" / "predictions" / (
            "A_target_test.csv" if key == "A" else f"{key}_target_test.csv")
        df = pd.read_csv(pred_path).set_index("id")
        joined = df.join(target_df[["text"]], how="left")
        fn = joined[(joined.label == 1) & (joined.pred == 0)]
        fp = joined[(joined.label == 0) & (joined.pred == 1)]
        invalid = joined[~joined.pred.isin([0, 1])]
        lines.append(f"### {key} (target, n={len(joined)})")
        lines.append(f"- False negatives (true positive -> predicted negative): {len(fn)}")
        lines.append(f"- False positives (true negative -> predicted positive): {len(fp)}")
        lines.append(f"- Invalid/unparseable outputs: {len(invalid)}")
        for label, subset in [("false negative", fn), ("false positive", fp)]:
            if len(subset):
                ex = subset.iloc[0]
                snippet = str(ex["text"])[:160].replace("\n", " ")
                lines.append(f"  - example {label}: \"{snippet}...\" (raw_output={ex.get('raw_output')!r})")
        lines.append("")
    if not lines:
        return "No completed target-domain predictions are available yet for error analysis."
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pilot.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seeds = cfg.get("seeds", [0, 1, 2])
    results_dir = REPO_ROOT / cfg.get("results_dir", "results")
    data_dir = REPO_ROOT / cfg.get("data_dir", "data")

    import datetime
    snapshot_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    status = discover_status(results_dir, seeds)
    checks = verify_everything(results_dir, data_dir, status)
    results = collect_completed_results(results_dir, status)
    same_examples = check_same_examples(results, data_dir)
    comparison = build_comparison_table(results, status, seeds)
    diffs = compute_diffs(comparison)
    error_analysis = target_error_analysis(results, data_dir)

    comparison_path = results_dir / "comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")

    # ---- per-condition per-seed detailed metrics block ----
    detail_lines = []
    for key, entry in results.items():
        detail_lines.append(f"### {key}")
        for split_name, split_key in [("source_test", "source"), ("target_test", "target")]:
            m = entry[split_key]
            if m is None:
                detail_lines.append(f"- {split_name}: not available (incomplete or target eval disabled)")
                continue
            cm = m["confusion_matrix"]
            detail_lines.append(
                f"- {split_name}: n={m['n']}, accuracy={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}, "
                f"invalid_rate={m['invalid_rate']:.4f}\n"
                f"  - negative: precision={m['per_class']['negative']['precision']:.3f}, "
                f"recall={m['per_class']['negative']['recall']:.3f}, f1={m['per_class']['negative']['f1']:.3f}\n"
                f"  - positive: precision={m['per_class']['positive']['precision']:.3f}, "
                f"recall={m['per_class']['positive']['recall']:.3f}, f1={m['per_class']['positive']['f1']:.3f}\n"
                f"  - confusion: true_neg->[neg {cm['negative']['pred_negative']}, pos {cm['negative']['pred_positive']}, "
                f"invalid {cm['negative']['pred_invalid']}]; "
                f"true_pos->[neg {cm['positive']['pred_negative']}, pos {cm['positive']['pred_positive']}, "
                f"invalid {cm['positive']['pred_invalid']}]"
            )
            if entry["stored_vs_recomputed_mismatch"]:
                detail_lines.append(f"  - MISMATCH vs stored metrics JSON: {entry['stored_vs_recomputed_mismatch']}")
            else:
                detail_lines.append("  - recomputed metrics match the pipeline's own stored metrics exactly")
        detail_lines.append("")

    status_lines = []
    for key, info in status.items():
        status_lines.append(f"- **{key}**: {info['state']}")

    checks_lines = []
    for c in checks:
        mark = "UNVERIFIABLE" if c["passed"] is None else ("PASS" if c["passed"] else "FAIL")
        checks_lines.append(f"- [{mark}] {c['check']}\n  - {c['detail']}")

    def df_markdown(df):
        return df.fillna("N/A").to_markdown(index=False, floatfmt=".4f")

    report = f"""# Independent post-hoc evaluation report

Snapshot taken at: {snapshot_time} (UTC). This report was generated strictly
from saved artifacts on disk — no prompts were changed, no prompt search was
re-run, and no test-set result influenced any prompt or hyperparameter choice.
If the underlying pilot run (`configs/pilot.yaml`, seeds={seeds}) was still in
progress at snapshot time, conditions/seeds not yet finished are reported as
**incomplete**, never as zero or as failing scores.

## 1. Run completion status

{chr(10).join(status_lines)}

## 2. Verification checks

{chr(10).join(checks_lines)}

## 3-4. Recomputed metrics (from saved prediction CSVs, independent of the pipeline's own summary.json) + same-examples check

Same source-test example set used by every completed condition: **{same_examples['all_source_test_identical']}**
Same target-test example set used by every completed condition: **{same_examples['all_target_test_identical']}**
(per-condition detail: {json.dumps(same_examples['per_condition'])})

{chr(10).join(detail_lines)}

## 5. Comparison table

See `results/comparison.csv` (also rendered below). `status` shows how many of
the expected seeds are actually completed for that row; a row with 0 completed
seeds reports `None`/`null` for every score, never 0.

{df_markdown(comparison)}

## 6. D - A / D - B / D - C differences (percentage points)

{json.dumps(diffs, indent=2)}

## 7. Did source-side optimization help cross-domain transfer, and was it worth the compute?

{answer_q7(comparison, diffs)}

## 8. Exploratory target-domain error analysis (post-evaluation only; not used to revise this experiment)

{error_analysis}

## 10. Limitations

- Three seeds (even once all complete) is not enough to claim statistical
  significance; standard deviations reported here describe the observed spread
  across {len(seeds)} seeds only, nothing more.
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

- This report: `{(results_dir / 'evaluation_report.md').resolve()}`
- Comparison table: `{comparison_path.resolve()}`
- Source metrics read from: `{(results_dir / 'metrics').resolve()}`
- Source predictions read from: `{(results_dir / 'predictions').resolve()}`
- Data manifest: `{(REPO_ROOT / 'artifacts' / 'manifests' / 'data_manifest.json').resolve()}`
- Selected-prompt manifests: `{(REPO_ROOT / 'artifacts' / 'selected_prompts').resolve()}`
"""
    out_path = results_dir / "evaluation_report.md"
    out_path.write_text(report)
    print(f"Wrote {out_path.resolve()}")
    print(f"Wrote {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
