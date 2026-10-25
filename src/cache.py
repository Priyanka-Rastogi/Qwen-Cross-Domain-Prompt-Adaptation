"""Append-only, resumable evaluation cache.

Cache key includes everything that could change the model's output or that we
want cache invalidation to be sensitive to: model/tokenizer revision, the exact
serialized prompt string, the input example's content hash, the preprocessing
configuration, decoding settings, and relevant software versions. Any change to
any of these yields a different key, so stale results are never silently reused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock


def make_cache_key(*, model_revision: str, tokenizer_revision: str, prompt_str: str,
                    input_content_hash: str, preprocessing_config: dict,
                    decoding_config: dict, software_config: dict) -> str:
    payload = {
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "prompt_str": prompt_str,
        "input_content_hash": input_content_hash,
        "preprocessing_config": preprocessing_config,
        "decoding_config": decoding_config,
        "software_config": software_config,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class EvalCache:
    """JSONL-backed cache. Safe for a single process; writes are flushed
    immediately so a killed run can resume from the last completed line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._store: dict[str, dict] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated last line from a prior kill
                self._store[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self._store.get(key)

    def put(self, key: str, value: dict):
        rec = {"key": key, **value}
        with self._lock:
            self._store[key] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    def __len__(self):
        return len(self._store)

    def hits_for_keys(self, keys: list[str]) -> int:
        return sum(1 for k in keys if k in self._store)
