"""Sign random projection fingerprints (Section 2.4).

An embedding e in R^d is reduced to b bits: c_i = 1 if r_i . e >= 0 else 0.
For two embeddings at angle theta, each bit differs with probability theta/pi
(Charikar 2002), so E[Hamming] = b * theta / pi.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


class SignRandomProjection:
    def __init__(self, matrix: np.ndarray):
        self.R = np.asarray(matrix, dtype=np.float32)
        self.bits, self.dim = self.R.shape
        if self.bits % 8:
            raise ValueError("number of bits must be a multiple of 8")

    @classmethod
    def generate(cls, dim: int, bits: int, seed: int) -> "SignRandomProjection":
        rng = np.random.default_rng(seed)
        return cls(rng.standard_normal((bits, dim)).astype(np.float32))

    # The consortium distributes the matrix file itself and checks its digest:
    # random streams are not guaranteed identical across NumPy versions.
    def save(self, path: str | Path) -> str:
        np.save(path, self.R)
        return self.digest()

    @classmethod
    def load(cls, path: str | Path, expected_digest: str | None = None) -> "SignRandomProjection":
        srp = cls(np.load(path))
        if expected_digest and srp.digest() != expected_digest:
            raise ValueError("projection matrix digest mismatch")
        return srp

    def digest(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.R).tobytes()).hexdigest()

    def encode(self, emb: np.ndarray) -> np.ndarray:
        """Embeddings (n, d) or (d,) -> bit array (n, bits) of uint8."""
        emb = np.atleast_2d(np.asarray(emb, dtype=np.float32))
        return (emb @ self.R.T >= 0).astype(np.uint8)


def to_hex(bits: np.ndarray) -> list[str]:
    """Bit array (n, b) -> list of lowercase hex strings, as stored on the ledger."""
    packed = np.packbits(np.atleast_2d(bits), axis=1)
    return [row.tobytes().hex() for row in packed]


def from_hex(codes: list[str]) -> np.ndarray:
    return np.unpackbits(np.array([np.frombuffer(bytes.fromhex(c), dtype=np.uint8) for c in codes]), axis=1)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Hamming distance between two bit arrays of equal shape."""
    return np.count_nonzero(np.atleast_2d(a) != np.atleast_2d(b), axis=1)


def estimated_cosine(h: np.ndarray | int, bits: int) -> np.ndarray | float:
    return np.cos(math.pi * np.asarray(h) / bits)


def author_pseudonym(consortium_key: bytes, orcid: str) -> str:
    """a = HMAC-SHA256(k, ORCID) (Section 2.3)."""
    import hmac
    return hmac.new(consortium_key, orcid.strip().encode(), hashlib.sha256).hexdigest()
