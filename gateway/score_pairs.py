"""Score every benchmark pair with every method and write the CSV that evaluate.py reads.

Usage:
    python score_pairs.py pairs.jsonl texts.jsonl embeddings.npz scores.csv --bits 128 256 512 --matrix-dir matrices/

pairs.jsonl : {pair_id, source_id, left_id, right_id, split, label, level, kind}
texts.jsonl : {id, title, abstract, body}
embeddings.npz : output of embed.py for every id in texts.jsonl
Scores are oriented so that higher = more similar (Hamming distances are negated).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import MinHash, exact_match, tfidf_cosine
from fingerprint import SignRandomProjection, hamming


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("texts"); ap.add_argument("embeddings"); ap.add_argument("out")
    ap.add_argument("--bits", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--matrix-dir", default="matrices")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    pairs = pd.DataFrame([json.loads(l) for l in open(a.pairs, encoding="utf-8")])
    texts = {t["id"]: t for t in map(json.loads, open(a.texts, encoding="utf-8"))}
    emb = np.load(a.embeddings, allow_pickle=False)
    row = {i: k for k, i in enumerate(emb["ids"].tolist())}
    li, ri = pairs.left_id.map(row).to_numpy(), pairs.right_id.map(row).to_numpy()
    H, B = emb["header"], emb["body"]

    full = lambda i: " ".join([texts[i]["title"], texts[i]["abstract"], texts[i].get("body") or ""])
    head = lambda i: texts[i]["title"] + " " + texts[i]["abstract"]

    pairs["exact"] = [exact_match(full(l), full(r)) for l, r in zip(pairs.left_id, pairs.right_id)]
    mh = MinHash()
    sig = {i: mh.signature(full(i)) for i in set(pairs.left_id) | set(pairs.right_id)}
    pairs["minhash"] = [float(np.mean(sig[l] == sig[r])) for l, r in zip(pairs.left_id, pairs.right_id)]
    train_ids = set(pairs[pairs.split == "train"].left_id) | set(pairs[pairs.split == "train"].right_id)
    if not train_ids:  # no training split (e.g. the pilot): fit IDF on original papers only, never on duplicates
        train_ids = set(pairs.left_id) | set(pairs[pairs.label == 0].right_id)
    pairs["tfidf"] = tfidf_cosine([head(i) for i in sorted(train_ids)], [head(i) for i in pairs.left_id], [head(i) for i in pairs.right_id])
    pairs["emb_cos_h"] = np.sum(H[li] * H[ri], axis=1)   # raw embeddings: upper bound, not private
    pairs["emb_cos_b"] = np.sum(B[li] * B[ri], axis=1)

    Path(a.matrix_dir).mkdir(exist_ok=True)
    for b in a.bits:
        path = Path(a.matrix_dir) / f"srp_{H.shape[1]}x{b}.npy"
        srp = SignRandomProjection.load(path) if path.exists() else SignRandomProjection.generate(H.shape[1], b, a.seed)
        if not path.exists():
            print(f"wrote {path} sha256={srp.save(path)}")
        cH, cB = srp.encode(H), srp.encode(B)
        pairs[f"srp{b}_h"] = -hamming(cH[li], cH[ri])
        pairs[f"srp{b}_b"] = -hamming(cB[li], cB[ri])
    pairs.to_csv(a.out, index=False)
    print(f"scored {len(pairs)} pairs -> {a.out}")


if __name__ == "__main__":
    main()
