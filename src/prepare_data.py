"""Download, align, split, and export the source (IMDb) and target (Financial
PhraseBank) datasets used by the cross-domain prompt-adaptation pilot.

Produces four CSVs under data/:
  source_train.csv       200 balanced IMDb train examples (demo pool + instruction proposal)
  source_validation.csv  100 balanced IMDb train examples, disjoint from source_train
  source_test.csv        500 balanced IMDb official-test examples (final in-domain eval)
  target_test.csv        all eligible pos/neg Financial PhraseBank (sentences_allagree) examples

Also writes data/DATASET_README.md and artifacts/manifests/data_manifest.json.

Label convention (shared across domains): negative = 0, positive = 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import hf_hub_download

from text_utils import normalize_for_hash, text_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_extract"
MANIFEST_DIR = REPO_ROOT / "artifacts" / "manifests"

IMDB_REPO = "stanfordnlp/imdb"
FPB_REPO = "takala/financial_phrasebank"
FPB_ZIP_PATH_IN_REPO = "data/FinancialPhraseBank-v1.0.zip"
FPB_MEMBER = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"

DATA_SPLIT_SEED = 42

N_SOURCE_TRAIN = 200       # 100 neg + 100 pos
N_SOURCE_VALIDATION = 100  # 50 neg + 50 pos
N_SOURCE_TEST = 500        # 250 neg + 250 pos

CSV_COLUMNS = ["id", "text", "label", "label_name", "domain", "original_split"]
LABEL_NAMES = {0: "negative", 1: "positive"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DatasetProvenance:
    name: str
    hf_repo_id: str
    resolved_revision: str
    config: str | None = None
    extra_files: dict | None = None
    license: str | None = None
    download_timestamp_utc: str | None = None


# --------------------------------------------------------------------------
# IMDb loading
# --------------------------------------------------------------------------

def load_imdb():
    from huggingface_hub import HfApi
    from datasets import load_dataset

    api = HfApi()
    info = api.dataset_info(IMDB_REPO)
    revision = info.sha

    ds = load_dataset(IMDB_REPO, revision=revision)
    assert ds["train"].features["label"].names == ["neg", "pos"], (
        "IMDb label metadata changed from expected ['neg', 'pos'] ordering; "
        "refusing to assume label alignment."
    )
    provenance = DatasetProvenance(
        name="IMDb (source domain, movie reviews)",
        hf_repo_id=IMDB_REPO,
        resolved_revision=revision,
        config="plain_text",
        download_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    return ds["train"], ds["test"], provenance


# --------------------------------------------------------------------------
# Financial PhraseBank loading (legacy script dataset -> parse raw archive)
# --------------------------------------------------------------------------

def load_financial_phrasebank():
    zip_path = Path(
        hf_hub_download(repo_id=FPB_REPO, repo_type="dataset", filename=FPB_ZIP_PATH_IN_REPO)
    )
    zip_sha256 = sha256_file(zip_path)

    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(FPB_REPO)
    repo_revision = info.sha

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        member_bytes = zf.read(FPB_MEMBER)
        license_bytes = zf.read("FinancialPhraseBank-v1.0/License.txt")
        # Persist the extracted sentences file + license for provenance/audit.
        (RAW_DIR / "Sentences_AllAgree.txt").write_bytes(member_bytes)
        (RAW_DIR / "License.txt").write_bytes(license_bytes)

    sentences_sha256 = hashlib.sha256(member_bytes).hexdigest()
    text = member_bytes.decode("latin-1")
    lines = [l for l in text.split("\r\n") if l.strip()]

    rows = []
    for line in lines:
        # Format: "<sentence text>@<label>" where label in {positive, negative, neutral}
        sentence, _, label_str = line.rpartition("@")
        label_str = label_str.strip().lower()
        rows.append({"text": sentence, "raw_label": label_str})

    license_text = license_bytes.decode("utf-8", errors="replace")

    provenance = DatasetProvenance(
        name="Financial PhraseBank (target domain, financial news sentences), sentences_allagree",
        hf_repo_id=FPB_REPO,
        resolved_revision=repo_revision,
        config="sentences_allagree (parsed directly from original author archive; "
        "the HF repo's legacy loading script is not compatible with modern `datasets`)",
        extra_files={
            "zip_path_in_repo": FPB_ZIP_PATH_IN_REPO,
            "zip_sha256": zip_sha256,
            "member": FPB_MEMBER,
            "member_sha256": sentences_sha256,
            "member_encoding": "latin-1 (ISO-8859-1), CRLF line terminators",
        },
        license="CC BY-NC-SA 3.0 Unported (see data/raw_extract/License.txt)",
        download_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    return rows, provenance, license_text


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------

def dedup_by_hash(records: list[dict]) -> tuple[list[dict], int]:
    """Drop records whose normalized text hash repeats; keep first occurrence."""
    seen = set()
    kept = []
    removed = 0
    for r in records:
        h = r["_hash"]
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        kept.append(r)
    return kept, removed


def stratified_take(records: list[dict], n_per_class: int, rng: random.Random, exclude_hashes: set[str]):
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for r in records:
        if r["_hash"] in exclude_hashes:
            continue
        by_label[r["label"]].append(r)
    for lab in (0, 1):
        rng.shuffle(by_label[lab])
    taken = by_label[0][:n_per_class] + by_label[1][:n_per_class]
    if len(by_label[0]) < n_per_class or len(by_label[1]) < n_per_class:
        raise RuntimeError(f"Not enough examples per class to sample {n_per_class} each.")
    return taken


def to_rows(records: Iterable[dict], domain: str, original_split: str, id_prefix: str) -> list[dict]:
    out = []
    for r in records:
        out.append(
            {
                "id": f"{id_prefix}_{r['orig_idx']}",
                "text": r["text"],
                "label": r["label"],
                "label_name": LABEL_NAMES[r["label"]],
                "domain": domain,
                "original_split": original_split,
            }
        )
    return out


def validate_rows(rows: list[dict], name: str):
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    assert df["text"].notna().all() and (df["text"].str.len() > 0).all(), f"{name}: missing/empty text"
    assert set(df["label"].unique()) <= {0, 1}, f"{name}: label values outside {{0,1}}"
    bad_map = df[(df["label"] == 0) & (df["label_name"] != "negative")]
    bad_map2 = df[(df["label"] == 1) & (df["label_name"] != "positive")]
    assert bad_map.empty and bad_map2.empty, f"{name}: label/label_name mismatch"
    assert df["id"].is_unique, f"{name}: duplicate ids"
    norm_hashes = df["text"].map(text_hash)
    assert norm_hashes.is_unique, f"{name}: duplicate (normalized) texts within file"
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DATA_SPLIT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading IMDb (source domain) ...")
    imdb_train, imdb_test, imdb_provenance = load_imdb()

    def materialize(ds, split_name):
        recs = []
        for i in range(len(ds)):
            ex = ds[i]
            recs.append(
                {
                    "orig_idx": i,
                    "text": ex["text"],
                    "label": int(ex["label"]),
                    "_hash": text_hash(ex["text"]),
                }
            )
        return recs

    imdb_train_recs = materialize(imdb_train, "train")
    imdb_test_recs = materialize(imdb_test, "test")

    imdb_train_recs, train_dupes_removed = dedup_by_hash(imdb_train_recs)
    imdb_test_recs, test_dupes_removed = dedup_by_hash(imdb_test_recs)
    print(f"  IMDb train: {len(imdb_train_recs)} unique texts ({train_dupes_removed} exact-duplicate removed)")
    print(f"  IMDb test:  {len(imdb_test_recs)} unique texts ({test_dupes_removed} exact-duplicate removed)")

    rng = random.Random(args.seed)

    source_train_recs = stratified_take(imdb_train_recs, N_SOURCE_TRAIN // 2, rng, exclude_hashes=set())
    used_hashes = {r["_hash"] for r in source_train_recs}
    source_validation_recs = stratified_take(
        imdb_train_recs, N_SOURCE_VALIDATION // 2, rng, exclude_hashes=used_hashes
    )
    used_hashes |= {r["_hash"] for r in source_validation_recs}

    source_test_recs = stratified_take(imdb_test_recs, N_SOURCE_TEST // 2, rng, exclude_hashes=set())

    print("[2/5] Loading Financial PhraseBank (target domain) ...")
    fpb_raw_rows, fpb_provenance, fpb_license_text = load_financial_phrasebank()
    fpb_total = len(fpb_raw_rows)
    label_counts_raw: dict[str, int] = {}
    for r in fpb_raw_rows:
        label_counts_raw[r["raw_label"]] = label_counts_raw.get(r["raw_label"], 0) + 1

    fpb_label_map = {"negative": 0, "positive": 1}
    excluded_neutral = label_counts_raw.get("neutral", 0)
    unexpected_labels = {k: v for k, v in label_counts_raw.items() if k not in ("negative", "positive", "neutral")}
    if unexpected_labels:
        raise RuntimeError(f"Unexpected Financial PhraseBank labels encountered: {unexpected_labels}")

    fpb_binary_recs = []
    for idx, r in enumerate(fpb_raw_rows):
        if r["raw_label"] not in fpb_label_map:
            continue
        fpb_binary_recs.append(
            {
                "orig_idx": idx,
                "text": r["text"],
                "label": fpb_label_map[r["raw_label"]],
                "_hash": text_hash(r["text"]),
            }
        )
    fpb_binary_recs, fpb_dupes_removed = dedup_by_hash(fpb_binary_recs)
    print(f"  FPB allagree total={fpb_total}, label_counts={label_counts_raw}, "
          f"excluded_neutral={excluded_neutral}, binary_kept={len(fpb_binary_recs)} "
          f"(exact-duplicate removed={fpb_dupes_removed})")

    print("[3/5] Checking source/target overlap ...")
    source_hashes = {r["_hash"] for r in source_train_recs + source_validation_recs + source_test_recs}
    target_hashes = {r["_hash"] for r in fpb_binary_recs}
    overlap = source_hashes & target_hashes
    if overlap:
        raise RuntimeError(f"Found {len(overlap)} source/target text overlaps; refusing to proceed silently.")
    print(f"  source/target exact-text overlap: {len(overlap)}")

    demo_val_overlap = {r["_hash"] for r in source_train_recs} & {r["_hash"] for r in source_validation_recs}
    assert not demo_val_overlap, "source_train/source_validation overlap detected"

    print("[4/5] Writing CSVs ...")
    source_train_rows = to_rows(source_train_recs, "movie", "train", "imdb_train")
    source_validation_rows = to_rows(source_validation_recs, "movie", "train", "imdb_train")
    source_test_rows = to_rows(source_test_recs, "movie", "test", "imdb_test")
    target_test_rows = to_rows(fpb_binary_recs, "finance", "unpartitioned", "fpb_allagree")

    df_source_train = validate_rows(source_train_rows, "source_train")
    df_source_validation = validate_rows(source_validation_rows, "source_validation")
    df_source_test = validate_rows(source_test_rows, "source_test")
    df_target_test = validate_rows(target_test_rows, "target_test")

    df_source_train.to_csv(out_dir / "source_train.csv", index=False, encoding="utf-8")
    df_source_validation.to_csv(out_dir / "source_validation.csv", index=False, encoding="utf-8")
    df_source_test.to_csv(out_dir / "source_test.csv", index=False, encoding="utf-8")
    df_target_test.to_csv(out_dir / "target_test.csv", index=False, encoding="utf-8")

    for name, df in [
        ("source_train", df_source_train),
        ("source_validation", df_source_validation),
        ("source_test", df_source_test),
        ("target_test", df_target_test),
    ]:
        print(f"  {name}.csv: n={len(df)}, class_balance={df['label_name'].value_counts().to_dict()}")

    print("[5/5] Writing manifest and README ...")
    manifest = {
        "data_split_seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_mapping": {"negative": 0, "positive": 1},
        "imdb_provenance": asdict(imdb_provenance),
        "financial_phrasebank_provenance": asdict(fpb_provenance),
        "imdb_train_dedup": {"unique_texts": len(imdb_train_recs),
                              "exact_duplicates_removed": train_dupes_removed},
        "imdb_test_dedup": {"exact_duplicates_removed": test_dupes_removed},
        "fpb_filtering": {
            "total_sentences_allagree": fpb_total,
            "raw_label_counts": label_counts_raw,
            "excluded_neutral_count": excluded_neutral,
            "final_binary_count": len(fpb_binary_recs),
            "exact_duplicates_removed": fpb_dupes_removed,
        },
        "source_target_overlap_count": len(overlap),
        "source_train_validation_overlap_count": len(demo_val_overlap),
        "files": {
            "source_train.csv": {
                "n": len(df_source_train),
                "ids": df_source_train["id"].tolist(),
                "normalization_rule": "lowercase; collapse whitespace runs to single space; strip",
            },
            "source_validation.csv": {"n": len(df_source_validation), "ids": df_source_validation["id"].tolist()},
            "source_test.csv": {"n": len(df_source_test), "ids": df_source_test["id"].tolist()},
            "target_test.csv": {"n": len(df_target_test), "ids": df_target_test["id"].tolist()},
        },
    }
    manifest_path = MANIFEST_DIR / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {manifest_path}")

    write_dataset_readme(
        out_dir, imdb_provenance, fpb_provenance, df_source_train, df_source_validation,
        df_source_test, df_target_test, label_counts_raw, excluded_neutral, args.seed,
        train_dupes_removed, test_dupes_removed, fpb_dupes_removed,
    )

    print("\nAbsolute paths of generated CSVs:")
    for name in ["source_train.csv", "source_validation.csv", "source_test.csv", "target_test.csv"]:
        print(f"  {(out_dir / name).resolve()}")


def write_dataset_readme(out_dir, imdb_prov, fpb_prov, df_train, df_val, df_test, df_target,
                          fpb_label_counts_raw, excluded_neutral, seed,
                          train_dupes, test_dupes, fpb_dupes):
    readme = f"""# Dataset documentation

Generated by `src/prepare_data.py`. Do not hand-edit the CSVs; regenerate instead.

## Sources

### Source domain: IMDb movie reviews
- URL: https://huggingface.co/datasets/stanfordnlp/imdb
- Resolved HF revision (commit sha): `{imdb_prov.resolved_revision}`
- Config: `{imdb_prov.config}`
- Citation: Maas, A. L., et al. (2011). "Learning Word Vectors for Sentiment Analysis."
  Proceedings of the 49th Annual Meeting of the ACL: HLT, pp. 142-150.
- License: see original ACL IMDb dataset terms (academic research use).
- Downloaded: {imdb_prov.download_timestamp_utc}

### Target domain: Financial PhraseBank (sentences_allagree)
- URL: https://huggingface.co/datasets/takala/financial_phrasebank
- Resolved HF repo revision (commit sha): `{fpb_prov.resolved_revision}`
- The HF repo ships a legacy Hugging Face `datasets` loading script
  (`financial_phrasebank.py`) that modern `datasets` versions (script-based loading
  was removed) cannot execute. We instead downloaded the original author archive
  referenced by that repo directly (`{fpb_prov.extra_files['zip_path_in_repo']}`,
  sha256 `{fpb_prov.extra_files['zip_sha256']}`) and parsed
  `{fpb_prov.extra_files['member']}` (sha256 `{fpb_prov.extra_files['member_sha256']}`,
  encoding: {fpb_prov.extra_files['member_encoding']}) directly.
- The raw extracted file and its license are preserved under `data/raw_extract/` for
  provenance/audit.
- Citation: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
  "Good debt or bad debt: Detecting semantic orientations in economic texts."
  Journal of the Association for Information Science and Technology, 65(4), 782-796.
- License: {fpb_prov.license}
- Downloaded: {fpb_prov.download_timestamp_utc}

## Label mapping (shared across domains)
- `negative` -> 0
- `positive` -> 1
- IMDb's own label metadata (`ClassLabel(names=['neg','pos'])`) was read explicitly
  and confirmed to already align with this mapping (0=neg, 1=pos); it was not assumed.

## Financial PhraseBank filtering
This project uses **binary financial sentiment evaluation**, not the full three-class
Financial PhraseBank benchmark.

- Total `sentences_allagree` sentences: {sum(fpb_label_counts_raw.values())}
- Raw label counts: {fpb_label_counts_raw}
- Excluded (neutral): {excluded_neutral}
- Final binary (positive + negative) examples retained: {len(df_target)}

## Splits

Data-split seed: `{seed}` (fixed; independent of the experiment seeds {{0,1,2}} used
for instruction/demonstration search). Deduplication (exact-text, normalized) was
applied within each raw split before sampling: IMDb train removed {train_dupes}
duplicate(s), IMDb test removed {test_dupes} duplicate(s), Financial PhraseBank
binary subset removed {fpb_dupes} duplicate(s). Normalization rule: lowercase,
collapse whitespace runs to a single space, strip leading/trailing whitespace.
Source/target exact-text overlap was checked and found to be zero.

| File | Role | N | Class balance | Original split |
|---|---|---|---|---|
| `source_train.csv` | Demonstration-candidate pool + instruction proposal | {len(df_train)} | {df_train['label_name'].value_counts().to_dict()} | IMDb `train` |
| `source_validation.csv` | Source-development set (prompt selection) | {len(df_val)} | {df_val['label_name'].value_counts().to_dict()} | IMDb `train` (disjoint from source_train) |
| `source_test.csv` | Final in-domain evaluation | {len(df_test)} | {df_test['label_name'].value_counts().to_dict()} | IMDb `test` |
| `target_test.csv` | Final cross-domain evaluation | {len(df_target)} | {df_target['label_name'].value_counts().to_dict()} | Financial PhraseBank: `unpartitioned` (the dataset defines no official train/validation/test split) |

## CSV schema
`id,text,label,label_name,domain,original_split`
- `id`: stable unique id, `{{source}}_{{orig_idx}}` where `orig_idx` is the example's
  index in its original HF split (e.g. `imdb_train_4821`, `fpb_allagree_17`).
- `text`: full original text, unmodified and untruncated. Model-input truncation is
  applied only at inference time, never in these files.
- `label` / `label_name`: 0/negative or 1/positive.
- `domain`: `movie` or `finance`.
- `original_split`: the example's split in its *original* dataset (`train`, `test`,
  or `unpartitioned` for Financial PhraseBank, which has no official partition).

## Isolation note
Target labels and target text are reserved for final scoring only. Source-training
examples may guide instruction proposal and demonstration selection; source-development
labels may guide prompt/instruction selection; target data must not influence any of
those choices (see `artifacts/manifests/data_manifest.json` for the frozen membership
lists and `README.md` for the full isolation protocol).

See `artifacts/manifests/data_manifest.json` for exact row-level id lists, checksums,
and provenance metadata.
"""
    (out_dir / "DATASET_README.md").write_text(readme)
    print(f"  wrote {(out_dir / 'DATASET_README.md').resolve()}")


if __name__ == "__main__":
    main()
