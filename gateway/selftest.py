"""End-to-end check of the pipeline on SYNTHETIC data. It verifies that the code runs and that the
sign-projection estimator behaves as the theory predicts. Its numbers are NOT experimental results
and must never be reported in the paper.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

from baselines import MinHash, exact_match
from fingerprint import SignRandomProjection, author_pseudonym, from_hex, hamming, to_hex
from inversion_attack import evaluate_leakage

rng = np.random.default_rng(0)
d = 768

# 1. E[H] = b*theta/pi
for b in (128, 256, 512):
    srp = SignRandomProjection.generate(d, b, 1)
    for cos in (0.95, 0.8, 0.5):
        x = rng.standard_normal((4000, d)); x /= np.linalg.norm(x, axis=1, keepdims=True)
        n = rng.standard_normal((4000, d)); n -= (n * x).sum(1, keepdims=True) * x; n /= np.linalg.norm(n, axis=1, keepdims=True)
        y = cos * x + math.sqrt(1 - cos ** 2) * n
        h = hamming(srp.encode(x), srp.encode(y)).mean()
        expect = b * math.acos(cos) / math.pi
        assert abs(h - expect) < 0.03 * b, (b, cos, h, expect)
print("ok  sign projection matches E[H] = b*theta/pi")

srp = SignRandomProjection.generate(d, 256, 1)
bits = srp.encode(rng.standard_normal((5, d)))
assert (from_hex(to_hex(bits)) == bits).all() and len(to_hex(bits)[0]) == 64
with tempfile.TemporaryDirectory() as tmp:
    dig = srp.save(os.path.join(tmp, "m.npy")); SignRandomProjection.load(os.path.join(tmp, "m.npy"), dig)
print("ok  hex round trip and matrix digest")
assert len(author_pseudonym(b"k", "0000-0002-1825-0097")) == 64
assert exact_match("A  b, c!", "a b c") == 1.0 and MinHash().similarity("one two three four five six seven", "one two three four five six seven") == 1.0
print("ok  pseudonym and lexical baselines")

# 2. full scoring + evaluation on synthetic pairs
with tempfile.TemporaryDirectory() as tmp:
    vocab = [f"w{i}" for i in range(3000)]
    n_src = 300; texts, ids, H, B, pairs = [], [], [], [], []
    topic = rng.standard_normal((n_src, d))
    def unit(v): return v / np.linalg.norm(v)
    def doc(words): return " ".join(words)
    for s in range(n_src):
        w = list(rng.choice(vocab, 120)); sid = f"s{s}"
        ids.append(sid); H.append(unit(topic[s])); B.append(unit(topic[s] + 0.2 * rng.standard_normal(d)))
        texts.append({"id": sid, "title": doc(w[:8]), "abstract": doc(w[8:60]), "body": doc(w)})
        split = "train" if s < 210 else "val" if s < 240 else "test"
        for level, noise, keep in (("light", 0.25, 0.7), ("medium", 0.5, 0.4), ("heavy", 0.9, 0.1)):
            rid = f"{sid}::{level}"; w2 = [x if rng.random() < keep else rng.choice(vocab) for x in w]
            ids.append(rid); H.append(unit(topic[s] + noise * rng.standard_normal(d) * np.linalg.norm(topic[s]) / math.sqrt(d) * 6))
            B.append(unit(topic[s] + noise * rng.standard_normal(d) * np.linalg.norm(topic[s]) / math.sqrt(d) * 6))
            texts.append({"id": rid, "title": doc(w2[:8]), "abstract": doc(w2[8:60]), "body": doc(w2)})
            pairs.append({"pair_id": len(pairs), "source_id": sid, "left_id": sid, "right_id": rid, "split": split, "label": 1, "level": level, "kind": "duplicate"})
        lo, hi = (0, 210) if split == "train" else (210, 240) if split == "val" else (240, 300)
        for kind in ("topical", "version"):
            o = int(rng.integers(lo, hi))
            if o != s:
                pairs.append({"pair_id": len(pairs), "source_id": sid, "left_id": sid, "right_id": f"s{o}", "split": split, "label": 0, "level": "none", "kind": kind})
    p = lambda n: os.path.join(tmp, n)
    open(p("texts.jsonl"), "w").write("\n".join(map(json.dumps, texts)))
    open(p("pairs.jsonl"), "w").write("\n".join(map(json.dumps, pairs)))
    np.savez(p("emb.npz"), ids=np.array(ids), header=np.stack(H).astype(np.float32), body=np.stack(B).astype(np.float32))
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    run("score_pairs.py", p("pairs.jsonl"), p("texts.jsonl"), p("emb.npz"), p("scores.csv"), "--bits", "128", "256", "--matrix-dir", p("mat"))
    run("evaluate.py", p("scores.csv"), p("res.json"), "--n-boot", "100", "--target-fpr", "0.05",
        "--method", "Exact hash", "exact", "--method", "MinHash", "minhash", "--method", "TF-IDF", "tfidf",
        "--method", "Raw embedding", "emb_cos_h", "emb_cos_b", "--method", "SubmitChain-256", "srp256_h", "srp256_b")
    res = json.load(open(p("res.json")))["results"]
    assert all(set(r["by_level"]) == {"light", "medium", "heavy"} for r in res)
    run("plots.py", "detection", p("res.json"), p("fig5"))
    # 3. inversion attack on representations that do encode the words (so recovery must beat the prior)
    words = [list(rng.choice(vocab[:400], 30)) for _ in range(1500)]
    proj = rng.standard_normal((400, 256)); idx = {w: i for i, w in enumerate(vocab[:400])}
    emb = np.stack([unit(sum(proj[idx[w]] for w in ws)) for ws in words]).astype(np.float32)
    leak = evaluate_leakage([" ".join(ws) for ws in words], emb, [128, 256], vocab=400)
    json.dump(leak, open(p("leak.json"), "w")); run("plots.py", "privacy", p("leak.json"), p("fig7"))
    assert leak["raw_embedding"]["f1"] > leak["prior_baseline"]["f1"]
    print("ok  scoring, evaluation, attack and plotting run end to end (synthetic data; not results)")

# 4. triage logic with a scripted stand-in for the LLM (checks parsing, validation, retry, ablation maths)
from triage import FIELDS, compare, evaluate, summarise  # noqa: E402
replies = iter(["Sure! ```json\n{" + ", ".join(f'"{f}": "x"' for f in FIELDS) + "}\n```"])
s = summarise({"title": "t", "abstract": "a", "body": "b"}, lambda p: next(replies))
assert set(s) == set(FIELDS)
seq = iter(["not json", '{"rating": "same_work", "overlapping_fields": ["method", "bogus"], "rationale": "r"}'])
r = compare(s, s, lambda p: next(seq), shared_author=True)
assert r["rating"] == "same_work" and r["overlapping_fields"] == ["method"] and r["shared_author"]
script = {("a", "b"): "same_work", ("c", "d"): "related", ("e", "f"): "distinct"}
pairs = [{"left_id": "a", "right_id": "b", "label": 1}, {"left_id": "c", "right_id": "d", "label": 0}, {"left_id": "e", "right_id": "f", "label": 0}]
sums = {k: {f: k for f in FIELDS} for k in "abcdef"}
ev = evaluate(pairs, sums, lambda p: '{"rating": "%s"}' % script[(json.loads(p.split("SUMMARY A\n")[1].split("\n\nSUMMARY B")[0])["method"], json.loads(p.split("SUMMARY B\n")[1])["method"])])
assert abs(ev["precision_before"] - 1 / 3) < 1e-9 and ev["precision_after"] == 1.0 and ev["duplicates_kept"] == 1.0
print("ok  triage parsing, validation, retry and ablation arithmetic")
