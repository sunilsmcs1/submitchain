"""Build the paraphrased-duplicate benchmark (Section 2.7).

Step 1 (this script, `paraphrase`): rewrite each source paper at three levels with one or more LLMs
        served through an OpenAI-compatible chat endpoint (for example a local Ollama server).
Step 2 (`embed.py`): embed sources and rewrites.
Step 3 (this script, `pairs`): assemble duplicate pairs, topical negatives and version negatives,
        and split 70/10/20 by source paper.

Source file sources.jsonl: {id, title, abstract, body, category, authors: [..]} from openly licensed arXiv papers.

    python build_benchmark.py paraphrase sources.jsonl texts.jsonl --models llama3.1 qwen2.5 --base-url http://localhost:11434/v1
    python embed.py texts.jsonl embeddings.npz
    python build_benchmark.py pairs sources.jsonl texts.jsonl embeddings.npz pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np

LEVELS = {
    "light": "Rewrite the text sentence by sentence. Change the wording of every sentence but keep the sentence order, "
             "all technical terms, all numbers and the meaning.",
    "medium": "Rewrite the text. Change the wording, and also reorder, split and merge sentences within each paragraph. "
              "Keep all claims, numbers and technical content.",
    "heavy": "Rewrite the text as if a different author were describing the same work. Use a different structure and "
             "different phrasing throughout, while preserving every claim, method detail and result.",
}
SYSTEM = "You rewrite scientific text. Output only the rewritten text, with no preface or commentary."


def chat(base_url: str, model: str, system: str, user: str, api_key: str = "none", temperature: float = 0.7) -> str:
    body = json.dumps({"model": model, "temperature": temperature,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def cmd_paraphrase(a) -> None:
    done = set()
    try:
        done = {json.loads(l)["id"] for l in open(a.texts, encoding="utf-8")}
    except FileNotFoundError:
        pass
    with open(a.texts, "a", encoding="utf-8") as out:
        for line in open(a.sources, encoding="utf-8"):
            s = json.loads(line)
            if s["id"] not in done:
                out.write(json.dumps({**s, "source_id": s["id"], "level": "none", "model": None}) + "\n")
            for model in a.models:
                for level, instruction in LEVELS.items():
                    rid = f"{s['id']}::{model}::{level}"
                    if rid in done:
                        continue  # resumable
                    rw = {k: chat(a.base_url, model, SYSTEM, f"{instruction}\n\nTEXT:\n{s[k]}", a.api_key)
                          for k in ("abstract", "body") if s.get(k)}
                    title = s["title"] if level != "heavy" else chat(
                        a.base_url, model, SYSTEM, "Write a new title for this paper that shares few words with the "
                        f"original title.\n\nORIGINAL TITLE: {s['title']}\n\nABSTRACT:\n{s['abstract']}", a.api_key)
                    out.write(json.dumps({"id": rid, "source_id": s["id"], "level": level, "model": model, "title": title,
                                          "abstract": rw.get("abstract", ""), "body": rw.get("body", ""),
                                          "category": s.get("category"), "authors": s.get("authors", [])}) + "\n")
                    out.flush()


def cmd_pairs(a) -> None:
    rng = np.random.default_rng(a.seed)
    sources = [json.loads(l) for l in open(a.sources, encoding="utf-8")]
    texts = [json.loads(l) for l in open(a.texts, encoding="utf-8")]
    emb = np.load(a.embeddings, allow_pickle=False)
    row = {i: k for k, i in enumerate(emb["ids"].tolist())}
    ids = [s["id"] for s in sources]
    H = emb["header"][[row[i] for i in ids]]
    perm = rng.permutation(len(ids))
    n_tr, n_va = int(0.7 * len(ids)), int(0.1 * len(ids))
    split = {ids[j]: ("train" if r < n_tr else "val" if r < n_tr + n_va else "test") for r, j in enumerate(perm)}
    sim = H @ H.T
    np.fill_diagonal(sim, -1)
    pairs, n = [], 0

    def add(src, left, right, label, level, kind):
        nonlocal n
        pairs.append({"pair_id": n, "source_id": src, "left_id": left, "right_id": right, "split": split[src],
                      "label": label, "level": level, "kind": kind}); n += 1

    for t in texts:
        if t["level"] != "none":
            add(t["source_id"], t["source_id"], t["id"], 1, t["level"], "duplicate")
    for k, s in enumerate(sources):
        same_split = [j for j in range(len(ids)) if split[ids[j]] == split[s["id"]] and j != k]
        # topical negatives: most similar distinct papers in the same category and split
        cand = [j for j in same_split if sources[j].get("category") == s.get("category")]
        for j in sorted(cand, key=lambda j: -sim[k, j])[:a.topical]:
            add(s["id"], s["id"], ids[j], 0, "none", "topical")
        # version negatives: distinct papers sharing an author, in the same category and split
        mine = set(s.get("authors", []))
        ver = [j for j in cand if mine & set(sources[j].get("authors", []))]
        for j in sorted(ver, key=lambda j: -sim[k, j])[:a.version]:
            add(s["id"], s["id"], ids[j], 0, "none", "version")
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    kinds = {}
    for p in pairs:
        kinds[(p["split"], p["kind"], p["level"])] = kinds.get((p["split"], p["kind"], p["level"]), 0) + 1
    for k in sorted(kinds):
        print(k, kinds[k])


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paraphrase"); p.add_argument("sources"); p.add_argument("texts")
    p.add_argument("--models", nargs="+", required=True); p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--api-key", default="none"); p.set_defaults(fn=cmd_paraphrase)
    q = sub.add_parser("pairs"); q.add_argument("sources"); q.add_argument("texts"); q.add_argument("embeddings"); q.add_argument("out")
    q.add_argument("--topical", type=int, default=3); q.add_argument("--version", type=int, default=2)
    q.add_argument("--seed", type=int, default=0); q.set_defaults(fn=cmd_pairs)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
