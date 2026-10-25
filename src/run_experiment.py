"""Orchestrates conditions A-D across seeds, with incremental saves + resume.

Usage:
  python src/run_experiment.py --config configs/smoke.yaml
  python src/run_experiment.py --config configs/pilot.yaml
  python src/run_experiment.py --config configs/pilot.yaml --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from cache import EvalCache
from evaluate import PhaseCost, evaluate_condition
from model_backend import QwenBackend
from metrics import aggregate_across_seeds
from propose_instructions import BASELINE_INSTRUCTION, propose_instructions
from search_prompts import (
    MAX_COMBINATIONS_FOR_D, SelectionCandidate, condition_c_demo_set, condition_d_demo_sets,
    demo_row_for_prompt, select_best,
)
from text_utils import text_hash

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("max_instruction_candidates", 8)
    cfg.setdefault("instruction_generation_attempts", 16)
    cfg.setdefault("max_demo_sets_d", 10)
    cfg.setdefault("run_target_eval", True)
    cfg.setdefault("seeds", [0, 1, 2])
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("results_dir", "results")
    cfg.setdefault("cache_path", "artifacts/cache/eval_cache.jsonl")
    return cfg


def df_to_examples(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if limit is not None:
        df = df.head(limit)
    return df[["id", "text", "label"]].to_dict("records")


def balanced_head(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Take the first `limit` rows while preserving class balance (limit must be
    even); needed because within each CSV, rows are grouped by label."""
    per_class = limit // 2
    parts = [df[df["label"] == lab].head(per_class) for lab in (0, 1)]
    return pd.concat(parts, ignore_index=True)


def load_data(data_dir: Path, cfg: dict) -> dict:
    out = {}
    for name in ["source_train", "source_validation", "source_test", "target_test"]:
        df = pd.read_csv(data_dir / f"{name}.csv", encoding="utf-8")
        limit = cfg.get(f"{name}_limit")
        out[name] = balanced_head(df, limit) if limit else df
    return out


def json_default(o):
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default))


def save_predictions_csv(path: Path, predictions: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(path, index=False, encoding="utf-8")


def freeze_manifest(path: Path, *, condition: str, seed, instruction: str, demonstrations: list[dict],
                     selection_score: dict | None):
    demo_ids = [d["id"] for d in demonstrations] if demonstrations and "id" in (demonstrations[0] or {}) else None
    manifest = {
        "condition": condition,
        "seed": seed,
        "instruction": instruction,
        "instruction_hash": text_hash(instruction),
        "demonstration_ids": demo_ids,
        "demonstrations_content_hash": text_hash(
            "||".join(f"{d.get('label_name')}:{d.get('text')}" for d in demonstrations)
        ) if demonstrations else None,
        "selection_score": selection_score,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(path, manifest)
    return manifest


def run(cfg: dict, resume: bool = False):
    try:
        sys.stdout.reconfigure(line_buffering=True)  # so progress is visible in a redirected log, not just at exit
    except Exception:
        pass
    data_dir = REPO_ROOT / cfg["data_dir"]
    results_dir = REPO_ROOT / cfg["results_dir"]
    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "artifacts" / "instructions").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "artifacts" / "selected_prompts").mkdir(parents=True, exist_ok=True)

    bs = cfg["batch_size"]
    run_target = cfg["run_target_eval"]
    # Namespace by config name so different configs (e.g. smoke.yaml vs pilot.yaml)
    # sharing the same seed values never read/overwrite each other's frozen
    # instruction pools or selected prompts (results/ is already namespaced via
    # results_dir, but artifacts/instructions and artifacts/selected_prompts are
    # shared across configs unless tagged here).
    run_tag = cfg.get("name", "run")

    data = load_data(data_dir, cfg)
    source_train_df = data["source_train"]
    source_validation_examples = df_to_examples(data["source_validation"])
    source_test_examples = df_to_examples(data["source_test"])
    target_test_examples = df_to_examples(data["target_test"]) if run_target else []

    cache = EvalCache(REPO_ROOT / cfg["cache_path"])

    t_load_start = time.perf_counter()
    backend = QwenBackend.load()
    load_cost = PhaseCost(
        phase="model_loading", wall_clock_seconds=time.perf_counter() - t_load_start,
        model_invocations=0, examples_classified=0,
    )
    print(f"[model_loading] {load_cost.wall_clock_seconds:.1f}s, revision={backend.model_revision}", flush=True)

    all_phase_costs = [load_cost.as_dict()]
    run_log_path = results_dir / "logs" / "run_log.jsonl"

    def log(event: dict):
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **event}, default=json_default) + "\n")

    log({"event": "model_loaded", "model_revision": backend.model_revision,
         "software_config": backend.software_config(), **load_cost.as_dict()})

    # ---------------- Condition A (seed-invariant) ----------------
    a_metrics_path = results_dir / "metrics" / "A_final.json"
    if resume and a_metrics_path.exists():
        print("[A] resume: metrics already exist, skipping recompute")
        a_result = json.loads(a_metrics_path.read_text())
    else:
        a_source = evaluate_condition(backend, cache, source_test_examples, BASELINE_INSTRUCTION, [],
                                       phase="A_final_source_test", batch_size=bs)
        a_target = (evaluate_condition(backend, cache, target_test_examples, BASELINE_INSTRUCTION, [],
                                        phase="A_final_target_test", batch_size=bs) if run_target else None)
        a_result = {
            "condition": "A", "seed_invariant": True, "instruction": BASELINE_INSTRUCTION,
            "source_test": {"metrics": a_source["metrics"], "cost": a_source["cost"]},
            "target_test": ({"metrics": a_target["metrics"], "cost": a_target["cost"]} if a_target else None),
        }
        save_json(a_metrics_path, a_result)
        save_predictions_csv(results_dir / "predictions" / "A_source_test.csv", a_source["predictions"])
        if a_target:
            save_predictions_csv(results_dir / "predictions" / "A_target_test.csv", a_target["predictions"])
        all_phase_costs.append({**a_source["cost"], "phase": "A_final_source_test"})
        if a_target:
            all_phase_costs.append({**a_target["cost"], "phase": "A_final_target_test"})
        log({"event": "condition_A_done", "accuracy_source": a_result["source_test"]["metrics"]["accuracy"]})
    print(f"[A] source_test macro_f1={a_result['source_test']['metrics']['macro_f1']:.4f} "
          f"accuracy={a_result['source_test']['metrics']['accuracy']:.4f}")

    per_seed_summary = {"A": a_result}

    for seed in cfg["seeds"]:
        print(f"\n===== SEED {seed} =====")
        seed_tag = f"seed{seed}"

        # ---------------- Instruction proposal (shared by B & D) ----------------
        instr_path = REPO_ROOT / "artifacts" / "instructions" / f"{run_tag}_{seed_tag}.json"
        if resume and instr_path.exists():
            print(f"[{seed_tag}] resume: loading cached instruction proposal")
            proposal_saved = json.loads(instr_path.read_text())
            accepted_candidates = proposal_saved["accepted_candidates"]
        else:
            proposal_record = propose_instructions(
                backend, source_train_df, seed,
                max_candidates=cfg["max_instruction_candidates"],
                generation_attempts=cfg["instruction_generation_attempts"],
            )
            accepted_candidates = proposal_record.accepted_candidates
            save_json(instr_path, {
                "seed": seed,
                "proposal_prompt": proposal_record.proposal_prompt,
                "sampled_example_ids": proposal_record.sampled_example_ids,
                "raw_generations": proposal_record.raw_generations,
                "accepted_candidates": accepted_candidates,
                "cost": proposal_record.cost.as_dict(),
            })
            all_phase_costs.append({**proposal_record.cost.as_dict(), "phase": f"instruction_generation_{seed_tag}"})
            log({"event": "instruction_proposal_done", "seed": seed, "n_candidates": len(accepted_candidates)})
        print(f"[{seed_tag}] instruction pool size = {len(accepted_candidates)}")

        # ---------------- Condition B: instruction-only selection ----------------
        b_manifest_path = REPO_ROOT / "artifacts" / "selected_prompts" / f"{run_tag}_B_{seed_tag}.json"
        b_metrics_path = results_dir / "metrics" / f"B_{seed_tag}_final.json"
        if resume and b_metrics_path.exists():
            print(f"[B/{seed_tag}] resume: skipping")
            b_result = json.loads(b_metrics_path.read_text())
        else:
            b_candidates = [
                SelectionCandidate(candidate_id=f"B_instr{idx}", instruction=instr, demonstrations=[])
                for idx, instr in enumerate(accepted_candidates)
            ]
            b_best, b_scored = select_best(
                b_candidates, backend, cache, source_validation_examples,
                phase=f"B_dev_{seed_tag}", batch_size=bs,
            )
            for s in b_scored:
                all_phase_costs.append({**s["cost"], "phase": f"B_dev_{seed_tag}", "candidate_id": s["candidate_id"]})
            freeze_manifest(b_manifest_path, condition="B", seed=seed, instruction=b_best.instruction,
                             demonstrations=[], selection_score=b_scored[0])

            b_source = evaluate_condition(backend, cache, source_test_examples, b_best.instruction, [],
                                           phase=f"B_final_source_test_{seed_tag}", batch_size=bs)
            b_target = (evaluate_condition(backend, cache, target_test_examples, b_best.instruction, [],
                                            phase=f"B_final_target_test_{seed_tag}", batch_size=bs)
                        if run_target else None)
            b_result = {
                "condition": "B", "seed": seed, "selected_instruction": b_best.instruction,
                "dev_scores_all_candidates": b_scored,
                "source_test": {"metrics": b_source["metrics"], "cost": b_source["cost"]},
                "target_test": ({"metrics": b_target["metrics"], "cost": b_target["cost"]} if b_target else None),
            }
            save_json(b_metrics_path, b_result)
            save_predictions_csv(results_dir / "predictions" / f"B_{seed_tag}_source_test.csv", b_source["predictions"])
            if b_target:
                save_predictions_csv(results_dir / "predictions" / f"B_{seed_tag}_target_test.csv", b_target["predictions"])
            all_phase_costs.append({**b_source["cost"], "phase": f"B_final_source_test_{seed_tag}"})
            if b_target:
                all_phase_costs.append({**b_target["cost"], "phase": f"B_final_target_test_{seed_tag}"})
            log({"event": "condition_B_done", "seed": seed, "accuracy_source": b_result["source_test"]["metrics"]["accuracy"]})
        print(f"[B/{seed_tag}] source_test macro_f1={b_result['source_test']['metrics']['macro_f1']:.4f}")

        # ---------------- Condition C: fixed instruction + random demos ----------------
        c_metrics_path = results_dir / "metrics" / f"C_{seed_tag}_final.json"
        c_demo_rows = condition_c_demo_set(source_train_df, seed)
        c_demo_prompt_rows = [demo_row_for_prompt(d) for d in c_demo_rows]
        if resume and c_metrics_path.exists():
            print(f"[C/{seed_tag}] resume: skipping")
            c_result = json.loads(c_metrics_path.read_text())
        else:
            freeze_manifest(REPO_ROOT / "artifacts" / "selected_prompts" / f"{run_tag}_C_{seed_tag}.json",
                             condition="C", seed=seed, instruction=BASELINE_INSTRUCTION,
                             demonstrations=c_demo_rows, selection_score=None)
            c_source = evaluate_condition(backend, cache, source_test_examples, BASELINE_INSTRUCTION,
                                           c_demo_prompt_rows, phase=f"C_final_source_test_{seed_tag}", batch_size=bs)
            c_target = (evaluate_condition(backend, cache, target_test_examples, BASELINE_INSTRUCTION,
                                            c_demo_prompt_rows, phase=f"C_final_target_test_{seed_tag}", batch_size=bs)
                        if run_target else None)
            c_result = {
                "condition": "C", "seed": seed, "demonstration_ids": [d["id"] for d in c_demo_rows],
                "source_test": {"metrics": c_source["metrics"], "cost": c_source["cost"]},
                "target_test": ({"metrics": c_target["metrics"], "cost": c_target["cost"]} if c_target else None),
            }
            save_json(c_metrics_path, c_result)
            save_predictions_csv(results_dir / "predictions" / f"C_{seed_tag}_source_test.csv", c_source["predictions"])
            if c_target:
                save_predictions_csv(results_dir / "predictions" / f"C_{seed_tag}_target_test.csv", c_target["predictions"])
            all_phase_costs.append({**c_source["cost"], "phase": f"C_final_source_test_{seed_tag}"})
            if c_target:
                all_phase_costs.append({**c_target["cost"], "phase": f"C_final_target_test_{seed_tag}"})
            log({"event": "condition_C_done", "seed": seed, "accuracy_source": c_result["source_test"]["metrics"]["accuracy"]})
        print(f"[C/{seed_tag}] source_test macro_f1={c_result['source_test']['metrics']['macro_f1']:.4f}")

        # ---------------- Condition D: instruction + demo combination search ----------------
        d_metrics_path = results_dir / "metrics" / f"D_{seed_tag}_final.json"
        if resume and d_metrics_path.exists():
            print(f"[D/{seed_tag}] resume: skipping")
            d_result = json.loads(d_metrics_path.read_text())
        else:
            d_demo_sets = condition_d_demo_sets(source_train_df, seed, c_demo_rows,
                                                 max_demo_sets=cfg["max_demo_sets_d"])
            combos = []
            for i, instr in enumerate(accepted_candidates):
                for j, demo_set in enumerate(d_demo_sets):
                    combos.append(SelectionCandidate(
                        candidate_id=f"D_instr{i}_demo{j}", instruction=instr,
                        demonstrations=[demo_row_for_prompt(d) for d in demo_set],
                    ))
            if len(combos) > MAX_COMBINATIONS_FOR_D:
                combos = combos[:MAX_COMBINATIONS_FOR_D]
            print(f"[D/{seed_tag}] evaluating {len(combos)} instruction x demo-set combinations on dev")

            d_best, d_scored = select_best(
                combos, backend, cache, source_validation_examples,
                phase=f"D_dev_{seed_tag}", batch_size=bs,
            )
            for s in d_scored:
                all_phase_costs.append({**s["cost"], "phase": f"D_dev_{seed_tag}", "candidate_id": s["candidate_id"]})
            freeze_manifest(REPO_ROOT / "artifacts" / "selected_prompts" / f"{run_tag}_D_{seed_tag}.json",
                             condition="D", seed=seed, instruction=d_best.instruction,
                             demonstrations=d_best.demonstrations, selection_score=d_scored[0])

            d_source = evaluate_condition(backend, cache, source_test_examples, d_best.instruction,
                                           d_best.demonstrations, phase=f"D_final_source_test_{seed_tag}", batch_size=bs)
            d_target = (evaluate_condition(backend, cache, target_test_examples, d_best.instruction,
                                            d_best.demonstrations, phase=f"D_final_target_test_{seed_tag}", batch_size=bs)
                        if run_target else None)
            d_result = {
                "condition": "D", "seed": seed, "selected_instruction": d_best.instruction,
                "selected_candidate_id": d_best.candidate_id, "n_combinations_evaluated": len(combos),
                "dev_scores_all_combinations": d_scored,
                "source_test": {"metrics": d_source["metrics"], "cost": d_source["cost"]},
                "target_test": ({"metrics": d_target["metrics"], "cost": d_target["cost"]} if d_target else None),
            }
            save_json(d_metrics_path, d_result)
            save_predictions_csv(results_dir / "predictions" / f"D_{seed_tag}_source_test.csv", d_source["predictions"])
            if d_target:
                save_predictions_csv(results_dir / "predictions" / f"D_{seed_tag}_target_test.csv", d_target["predictions"])
            all_phase_costs.append({**d_source["cost"], "phase": f"D_final_source_test_{seed_tag}"})
            if d_target:
                all_phase_costs.append({**d_target["cost"], "phase": f"D_final_target_test_{seed_tag}"})
            log({"event": "condition_D_done", "seed": seed, "accuracy_source": d_result["source_test"]["metrics"]["accuracy"]})
        print(f"[D/{seed_tag}] source_test macro_f1={d_result['source_test']['metrics']['macro_f1']:.4f}")

        per_seed_summary[f"B_{seed_tag}"] = b_result
        per_seed_summary[f"C_{seed_tag}"] = c_result
        per_seed_summary[f"D_{seed_tag}"] = d_result

    # ---------------- Aggregate across seeds ----------------
    summary = {"config": cfg, "A": a_result, "per_seed": {}}
    for cond in ["B", "C", "D"]:
        seed_metrics_source = []
        seed_metrics_target = []
        for seed in cfg["seeds"]:
            r = per_seed_summary[f"{cond}_seed{seed}"]
            seed_metrics_source.append(r["source_test"]["metrics"])
            if r["target_test"] is not None:
                seed_metrics_target.append(r["target_test"]["metrics"])
        summary[cond] = {
            "source_test_aggregate": aggregate_across_seeds(seed_metrics_source),
            "target_test_aggregate": aggregate_across_seeds(seed_metrics_target) if seed_metrics_target else None,
        }

    save_json(results_dir / "metrics" / "summary.json", summary)
    save_json(results_dir / "logs" / "phase_costs.json", all_phase_costs)
    print(f"\nWrote summary to {(results_dir / 'metrics' / 'summary.json').resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
