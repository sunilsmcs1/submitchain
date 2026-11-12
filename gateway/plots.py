"""Result figures (Figs. 5-7) from the JSON written by evaluate.py and inversion_attack.py,
and from a CSV of Caliper measurements. Vector PDF plus 600-dpi PNG; colour-blind-safe palette.

    python plots.py detection results.json fig5_detection
    python plots.py system caliper.csv fig6_system        # columns: orgs, mode, active, latency_ms, tps
    python plots.py privacy leakage.json fig7_privacy
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#666666"]  # Okabe-Ito
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linewidth": 0.5, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


def save(fig, stem: str) -> None:
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=600); print("wrote", stem + ".pdf/.png")


def detection(path: str, stem: str) -> None:
    res = json.load(open(path))["results"]; levels = ["light", "medium", "heavy"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2)); w = 0.8 / len(res)
    for k, r in enumerate(res):
        f1 = [r["by_level"][l]["f1"] for l in levels]
        ci = np.array([r["by_level"][l]["ci95"]["f1"] for l in levels]).T
        x = np.arange(len(levels)) + k * w - 0.4 + w / 2
        ax.bar(x, f1, w, label=r["method"], color=PALETTE[k % len(PALETTE)],
               yerr=[np.array(f1) - ci[0], ci[1] - np.array(f1)], error_kw={"linewidth": 0.8, "capsize": 2})
    ax.set_xticks(range(len(levels)), [l.capitalize() + " rewriting" for l in levels])
    ax.set_ylabel("F1 on test split"); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=min(3, len(res)), loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, stem)


def system(path: str, stem: str) -> None:
    df = pd.read_csv(path); fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for (ax, col, lab) in [(axes[0], "latency_ms", "Mean latency (ms)"), (axes[1], "tps", "Throughput (tx/s)")]:
        for k, ((orgs, mode), g) in enumerate(df.groupby(["orgs", "mode"])):
            g = g.sort_values("active")
            ax.plot(g.active, g[col], marker="o", ms=3, lw=1.2, color=PALETTE[k % len(PALETTE)],
                    linestyle="-" if mode == "scan" else "--", label=f"{orgs} orgs, {mode}")
        ax.set_xscale("log"); ax.set_xlabel("Active submissions"); ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, stem)


def privacy(path: str, stem: str) -> None:
    res = json.load(open(path)); keys = [k for k, v in res.items() if isinstance(v, dict)]
    names = {"prior_baseline": "No access\n(prior only)", "raw_embedding": "Raw\nembedding"}
    labels = [names.get(k, k.replace("srp_", "Code\n") + (" bits" if k.startswith("srp_") else "")) for k in keys]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    colours = ["#666666" if k == "prior_baseline" else "#D55E00" if k == "raw_embedding" else "#0072B2" for k in keys]
    ax.bar(labels, [res[k]["f1"] for k in keys], color=colours, width=0.6)
    ax.set_ylabel("Token-level F1 of recovered words"); ax.set_ylim(0, 1.0)
    save(fig, stem)


if __name__ == "__main__":
    {"detection": detection, "system": system, "privacy": privacy}[sys.argv[1]](sys.argv[2], sys.argv[3])
