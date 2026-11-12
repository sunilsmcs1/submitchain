"""Build the CPU-only pilot benchmark: texts.jsonl, pairs.jsonl and embeddings.npz in the formats
that submitchain/gateway/score_pairs.py, evaluate.py and inversion_attack.py read.

Encoder: latent semantic analysis (LSA) -- TF-IDF over word 1-2-grams and character 3-5-grams, reduced by
truncated SVD to 256 dimensions and L2-normalised. It is fitted only on corpus papers that are NOT
benchmark sources, so no source or duplicate text influences the encoder. It stands in for SPECTER,
which cannot be run in this environment; the body view is set equal to the header view because only
titles and abstract excerpts are available.
"""
from __future__ import annotations

import collections
import json
import random

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from rewrite import rewrite

rng = random.Random(0)
C = json.load(open("data/corpus.json"))
S = json.load(open("data/sources.json"))
idx = {c["id"]: c for c in C}
heavy = {h["src"]: h for h in map(json.loads, open("data/heavy.jsonl"))}
sources = S["sources"]

# split by source paper, stratified by topic: 40% validation (threshold selection), 60% test
split = {}
by_topic = collections.defaultdict(list)
for s in sources:
    by_topic[idx[s]["file"]].append(s)
for t, ss in by_topic.items():
    hv = [s for s in ss if s in heavy]; nh = [s for s in ss if s not in heavy]
    for group in (hv, nh):
        g = group[:]; rng.shuffle(g); k = round(0.4 * len(g))
        for i, s in enumerate(g):
            split[s] = "val" if i < k else "test"

texts = [{"id": c["id"], "title": c["title"], "abstract": c["abstract"], "body": ""} for c in C]
pairs = []
for n, s in enumerate(sources):
    c = idx[s]
    for lv in ("light", "medium"):
        t, a = rewrite(c["title"], c["abstract"], lv, seed=1000 * n + (lv == "medium"))
        did = f"{s}_{lv}"; texts.append({"id": did, "title": t, "abstract": a, "body": ""})
        pairs.append(dict(source_id=s, left_id=s, right_id=did, split=split[s], label=1, level=lv, kind="duplicate"))
    if s in heavy:
        did = f"{s}_heavy"; texts.append({"id": did, "title": heavy[s]["title"], "abstract": heavy[s]["abstract"], "body": ""})
        pairs.append(dict(source_id=s, left_id=s, right_id=did, split=split[s], label=1, level="heavy", kind="duplicate"))

# negatives: distinct corpus papers from the same topic -- the 5 most similar by word TF-IDF ("near")
# and 5 drawn at random ("random"). Benchmark sources are excluded from the negative pool.
src_set = set(sources)
pool = [c for c in C if c["id"] not in src_set]
head = lambda c: c["title"] + " " + c["abstract"]
tf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit([head(c) for c in C])
P = tf.transform([head(c) for c in pool])
for s in sources:
    c = idx[s]
    same = [i for i, p in enumerate(pool) if p["file"] == c["file"]]
    sims = cosine_similarity(tf.transform([head(c)]), P[same]).ravel()
    near = [same[i] for i in np.argsort(-sims)[:5]]
    rest = [i for i in same if i not in near]
    for kind, chosen in (("topical-near", near), ("topical-random", rng.sample(rest, 5))):
        for i in chosen:
            pairs.append(dict(source_id=s, left_id=s, right_id=pool[i]["id"], split=split[s], label=0, level="none", kind=kind))
for i, p in enumerate(pairs):
    p["pair_id"] = i

# LSA encoder fitted on the negative pool only
fit = [head(p) for p in pool]
wv = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english").fit(fit)
cv = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(fit)
X = lambda docs: hstack([wv.transform(docs), cv.transform(docs)]).tocsr()
svd = TruncatedSVD(n_components=256, random_state=0).fit(X(fit))
all_text = [t["title"] + " " + t["abstract"] for t in texts]
E = normalize(svd.transform(X(all_text))).astype(np.float32)
np.savez("data/embeddings.npz", ids=np.array([t["id"] for t in texts]), header=E, body=E)

with open("data/texts.jsonl", "w") as f:
    for t in texts: f.write(json.dumps(t) + "\n")
with open("data/pairs.jsonl", "w") as f:
    for p in pairs: f.write(json.dumps(p) + "\n")

cnt = collections.Counter((p["split"], p["level"] if p["label"] else p["kind"]) for p in pairs)
srcs = collections.Counter(split[s] for s in sources)
summary = {"n_corpus": len(C), "n_sources": len(sources), "n_pool": len(pool), "sources_by_split": srcs,
           "pairs_by_split_and_type": {f"{a}/{b}": n for (a, b), n in sorted(cnt.items())},
           "svd_explained_variance": float(svd.explained_variance_ratio_.sum())}
json.dump(summary, open("data/benchmark_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
