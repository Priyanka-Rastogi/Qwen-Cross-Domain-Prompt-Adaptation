"""Shared text-normalization and hashing helpers (duplicate detection, cache keys)."""
import hashlib
import re


def normalize_for_hash(text: str) -> str:
    """lowercase; collapse whitespace runs to a single space; strip."""
    return re.sub(r"\s+", " ", text.strip().lower())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()
