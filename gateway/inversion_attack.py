"""Privacy evaluation: how many words of a text can an attacker recover from its representation?

Attack model (Section 2.1): an honest-but-curious consortium member who knows the encoder and
the projection matrix. The attacker trains a multi-label classifier on public (text, representation)
pairs to predict which vocabulary words occur in the text, then applies it to held-out
representations. This is the word-recovery attack of Song and Raghunathan (2020), trained directly
on the representation being attacked (raw embeddings, or SubmitChain codes), i.e. adapted to the defence.

Reported: micro token precision / recall / F1 on held-out texts (the attacker outputs the k most
probable words per text, k = mean number of vocabulary words per text), and the same for a prior-only
baseline that ignores the representation (always predicts the most frequent words). Leakage is the
gap between the attack and that baseline.

Usage:
    python inversion_attack.py texts.jsonl embeddings.npz out.json --view header --bits 128 256 512
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from fingerprint import SignRandomProjection


def token_prf(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = float(np.sum(pred & truth)); p = tp / max(pred.sum(), 1); r = tp / max(truth.sum(), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(pred, np.argsort(-scores, axis=1)[:, :k], True, axis=1)
    return pred


def run_attack(X_train, Y_train, X_test, Y_test, seed: int = 0, hidden: int = 512) -> dict:
    """Two attackers on standardised inputs: a linear probe (ridge) and an MLP. The stronger one is reported,
    so that a badly trained attacker cannot make a representation look safer than it is.
    Both output the k most probable words per text (same word budget as the prior baseline)."""
    sc = StandardScaler().fit(X_train)
    Xtr, Xte = sc.transform(X_train), sc.transform(X_test)
    k = int(round(Y_train.sum(axis=1).mean()))
    truth = Y_test.astype(bool)
    ridge = RidgeCV(alphas=(1.0, 10.0, 100.0, 1000.0)).fit(Xtr, Y_train.astype(np.float32))
    res = {"linear": token_prf(_topk(ridge.predict(Xte), k), truth)}
    mlp = MLPClassifier(hidden_layer_sizes=(hidden,), early_stopping=True, max_iter=300, random_state=seed).fit(Xtr, Y_train)
    res["mlp"] = token_prf(_topk(mlp.predict_proba(Xte), k), truth)
    best = max(res, key=lambda n: res[n]["f1"])
    return {**res[best], "attacker": best, "all": res}


def prior_baseline(Y_train, Y_test) -> dict:
    """Predict, for every text, the k most frequent training words (k = mean words per text)."""
    k = int(round(Y_train.sum(axis=1).mean()))
    top = np.argsort(-Y_train.sum(axis=0))[:k]
    pred = np.zeros_like(Y_test, dtype=bool); pred[:, top] = True
    return token_prf(pred, Y_test.astype(bool))


def evaluate_leakage(texts: list[str], emb: np.ndarray, bits: list[int], matrix_seed: int = 2026,
                     vocab: int = 2000, test_frac: float = 0.2, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts)); n_test = int(len(texts) * test_frac)
    te, tr = order[:n_test], order[n_test:]
    vec = CountVectorizer(binary=True, stop_words="english", max_features=vocab).fit([texts[i] for i in tr])
    Y = np.asarray(vec.transform(texts).todense()).astype(np.uint8)
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "vocab": int(Y.shape[1]),
           "prior_baseline": prior_baseline(Y[tr], Y[te]),
           "raw_embedding": run_attack(emb[tr], Y[tr], emb[te], Y[te], seed)}
    for b in bits:
        codes = SignRandomProjection.generate(emb.shape[1], b, matrix_seed).encode(emb).astype(np.float32) * 2 - 1
        out[f"srp_{b}"] = run_attack(codes[tr], Y[tr], codes[te], Y[te], seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("texts"); ap.add_argument("embeddings"); ap.add_argument("out")
    ap.add_argument("--view", choices=["header", "body"], default="header")
    ap.add_argument("--bits", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--vocab", type=int, default=2000)
    a = ap.parse_args()
    emb = np.load(a.embeddings, allow_pickle=False)
    by_id = {t["id"]: t for t in map(json.loads, open(a.texts, encoding="utf-8"))}
    field = (lambda t: t["title"] + " " + t["abstract"]) if a.view == "header" else (lambda t: t.get("body") or t["abstract"])
    texts = [field(by_id[i]) for i in emb["ids"].tolist()]
    res = evaluate_leakage(texts, emb[a.view], a.bits, vocab=a.vocab)
    json.dump(res, open(a.out, "w"), indent=2)
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:<16} token-F1={v['f1']:.3f}  P={v['precision']:.3f}  R={v['recall']:.3f}")


if __name__ == "__main__":
    main()
