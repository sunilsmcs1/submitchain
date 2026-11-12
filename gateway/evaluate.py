"""Detection metrics with bootstrap confidence intervals (Section 2.7).

Input CSV: one row per pair with columns
    pair_id, source_id, split (train|val|test), label (1 duplicate, 0 negative),
    level (light|medium|heavy|none), kind (duplicate|topical|version),
    and one or more score columns, where HIGHER means MORE similar.
For SubmitChain pass the two views as --or-rule score_h score_b (negated Hamming distances).

Thresholds are chosen on the validation split only and applied unchanged to the test split.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def threshold_for_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold t such that FPR of (score >= t) on negatives is <= target_fpr."""
    neg = np.sort(scores[labels == 0])[::-1]
    if len(neg) == 0:
        raise ValueError("no negatives in validation split")
    k = int(np.floor(target_fpr * len(neg)))  # number of negatives allowed above the threshold
    return float(np.nextafter(neg[k], np.inf)) if k < len(neg) else float(neg[-1])


def prf(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    neg = int(np.sum(y == 0))
    return {"precision": p, "recall": r, "f1": f, "fpr": fp / neg if neg else 0.0}


def bootstrap_ci(df: pd.DataFrame, pred_col: str, n_boot: int, seed: int) -> dict:
    """95% CI by resampling source papers (the unit of independence), not pairs."""
    rng = np.random.default_rng(seed)
    groups = {k: g.index.to_numpy() for k, g in df.groupby("source_id")}
    keys = np.array(list(groups))
    stats = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        m = prf(df.loc[idx, pred_col].to_numpy(bool), df.loc[idx, "label"].to_numpy())
        for s in stats:
            stats[s].append(m[s])
    return {s: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for s, v in stats.items()}


def evaluate_method(df: pd.DataFrame, name: str, cols: list[str], target_fpr: float, n_boot: int, seed: int) -> dict:
    val, test = df[df.split == "val"], df[df.split == "test"].copy()
    # OR rule over several views: split the false-positive budget equally among views.
    taus = {c: threshold_for_fpr(val[c].to_numpy(float), val.label.to_numpy(), target_fpr / len(cols)) for c in cols}
    test["_pred"] = np.logical_or.reduce([test[c].to_numpy(float) >= taus[c] for c in cols])
    y = test.label.to_numpy()
    out = {"method": name, "thresholds": taus, "n_test_pairs": int(len(test)), "overall": prf(test._pred.to_numpy(bool), y)}
    out["overall"]["ci95"] = bootstrap_ci(test, "_pred", n_boot, seed)
    combined = np.max(np.stack([(test[c] - val[c].mean()) / (val[c].std() + 1e-12) for c in cols]), axis=0)
    out["overall"]["roc_auc"] = float(roc_auc_score(y, combined)) if len(set(y)) > 1 else None
    out["by_level"] = {}
    negs = test[test.label == 0]
    for level in ["light", "medium", "heavy"]:
        sub = pd.concat([test[(test.label == 1) & (test.level == level)], negs])
        if (sub.label == 1).any():
            m = prf(sub._pred.to_numpy(bool), sub.label.to_numpy())
            m["ci95"] = bootstrap_ci(sub, "_pred", n_boot, seed)
            out["by_level"][level] = m
    out["fpr_by_kind"] = {k: float(g._pred.mean()) for k, g in negs.groupby("kind")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_csv")
    ap.add_argument("out_json")
    ap.add_argument("--method", action="append", nargs="+", metavar=("NAME", "SCORE_COL"), required=True,
                    help="method name followed by one score column, or several columns combined with the OR rule")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.pairs_csv)
    res = [evaluate_method(df, m[0], m[1:], a.target_fpr, a.n_boot, a.seed) for m in a.method]
    json.dump({"target_fpr": a.target_fpr, "results": res}, open(a.out_json, "w"), indent=2)
    for r in res:
        o = r["overall"]
        print(f"{r['method']:<22} P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} FPR={o['fpr']:.4f}")


if __name__ == "__main__":
    main()
