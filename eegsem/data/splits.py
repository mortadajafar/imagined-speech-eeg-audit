"""Deterministic, machine-independent sentence-disjoint splits.
A sentence's split is a pure function of its text, so every subject / phase / machine agrees,
and no test sentence ever appears in any training set (in-subject, cross-subject, or cross-phase).
"""

import hashlib

SPLIT_BUCKETS = {"train": tuple(range(0, 8)), "val": (8,), "test": (9,)}  # 80 / 10 / 10


def norm_text(t: str) -> str:
    return " ".join(str(t).strip().split())


def bucket(text: str) -> int:
    h = hashlib.sha1(norm_text(text).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 10


def split_of(text: str) -> str:
    b = bucket(text)
    for name, bs in SPLIT_BUCKETS.items():
        if b in bs:
            return name
    raise RuntimeError


def split_mask(texts, name):
    import numpy as np

    return np.array([split_of(t) == name for t in texts], dtype=bool)
