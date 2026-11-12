"""Lexical baselines (Section 2.7): exact SHA-256, MinHash over word 5-shingles, TF-IDF cosine."""
from __future__ import annotations

import hashlib
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_P = (1 << 61) - 1  # Mersenne prime for universal hashing


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def exact_match(a: str, b: str) -> float:
    da = hashlib.sha256(normalise(a).encode()).digest()
    return float(da == hashlib.sha256(normalise(b).encode()).digest())


class MinHash:
    """MinHash signatures over word k-shingles (Broder 1997)."""

    def __init__(self, num_perm: int = 128, k: int = 5, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _P, size=num_perm, dtype=np.int64).astype(object)
        self.b = rng.integers(0, _P, size=num_perm, dtype=np.int64).astype(object)
        self.k = k

    def signature(self, text: str) -> np.ndarray:
        words = normalise(text).split()
        shingles = {" ".join(words[i:i + self.k]) for i in range(max(1, len(words) - self.k + 1))}
        xs = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=7).digest(), "big") for s in shingles], dtype=object)
        return np.array([int(((a * xs + b) % _P).min()) for a, b in zip(self.a, self.b)], dtype=object)

    def similarity(self, a: str, b: str) -> float:
        return float(np.mean(self.signature(a) == self.signature(b)))


def tfidf_cosine(fit_corpus: list[str], left: list[str], right: list[str]) -> np.ndarray:
    """Cosine similarity of paired texts. The vectoriser is fitted on fit_corpus (training split only)."""
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2).fit(fit_corpus)
    L, R = vec.transform(left), vec.transform(right)
    return np.asarray(L.multiply(R).sum(axis=1)).ravel()  # rows are L2-normalised by default
