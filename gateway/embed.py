"""Compute header-view and body-view embeddings with a scientific-document encoder.

Usage:
    python embed.py papers.jsonl embeddings.npz --model allenai/specter

papers.jsonl: one JSON object per line with keys  id, title, abstract, body.
Output .npz: ids, header (n, d), body (n, d); all vectors L2-normalised.
Requires torch and transformers. A GPU is strongly recommended.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def load_encoder(model_name: str, device: str | None = None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tok, model, device


def _cls(tok, model, device, texts: list[str], max_len: int = 512) -> np.ndarray:
    import torch

    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        return model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def header_embedding(tok, model, device, title: str, abstract: str) -> np.ndarray:
    return _unit(_cls(tok, model, device, [title + tok.sep_token + abstract])[0])


def body_embedding(tok, model, device, body: str, words_per_segment: int = 300, batch: int = 8) -> np.ndarray:
    """Split the body into consecutive segments, embed each, mean-pool, normalise."""
    words = body.split()
    segs = [" ".join(words[i:i + words_per_segment]) for i in range(0, len(words), words_per_segment)] or [""]
    vecs = [_cls(tok, model, device, segs[i:i + batch]) for i in range(0, len(segs), batch)]
    return _unit(np.concatenate(vecs).mean(axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("papers")
    ap.add_argument("out")
    ap.add_argument("--model", default="allenai/specter")
    args = ap.parse_args()
    tok, model, device = load_encoder(args.model)
    ids, H, B = [], [], []
    with open(args.papers, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            ids.append(p["id"])
            H.append(header_embedding(tok, model, device, p["title"], p["abstract"]))
            B.append(body_embedding(tok, model, device, p.get("body") or p["abstract"]))
    np.savez_compressed(args.out, ids=np.array(ids), header=np.stack(H), body=np.stack(B), model=args.model)
    print(f"embedded {len(ids)} papers with {args.model} -> {args.out}")


if __name__ == "__main__":
    main()
