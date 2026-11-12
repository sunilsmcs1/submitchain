"""Fig. 7 (detection) and Fig. 8 (leakage vs detection) for the CPU-only pilot. Same style as gateway/plots.py."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/submitchain/gateway")
from plots import PALETTE, plt, save  # noqa: E402

R = {r["method"]: r for r in json.load(open("data/results.json"))["results"]}
df = pd.read_csv("data/scores.csv")
levels = ["light", "medium", "heavy"]
methods = ["MinHash 5-shingle", "TF-IDF", "Raw LSA embedding", "SubmitChain-128", "SubmitChain-256", "SubmitChain-512"]
labels = ["MinHash", "TF-IDF", "Raw LSA (not private)", "SubmitChain 128 b", "SubmitChain 256 b", "SubmitChain 512 b"]
cols = [PALETTE[6], PALETTE[1], PALETTE[3], PALETTE[4], PALETTE[0], PALETTE[2]]

# ---- Fig 7: (a) detection rate by level with 95% CI; (b) Hamming distance distributions at b = 256
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.35, 1]})
w = 0.8 / len(methods)
for k, (m, lab, c) in enumerate(zip(methods, labels, cols)):
    rec = np.array([R[m]["by_level"][l]["recall"] for l in levels])
    ci = np.array([R[m]["by_level"][l]["ci95"]["recall"] for l in levels]).T
    x = np.arange(3) + k * w - 0.4 + w / 2
    a.bar(x, rec, w, color=c, label=lab, yerr=[rec - ci[0], ci[1] - rec], error_kw={"linewidth": 0.7, "capsize": 1.5})
a.set_xticks(range(3), ["Light", "Medium", "Heavy (LLM)"])
a.set_ylabel("Duplicates detected (test)"); a.set_ylim(0, 1.05)
a.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
a.set_title("(a)", loc="left", fontsize=9)

te = df[df.split == "test"]
tau = int(np.floor(-R["SubmitChain-256"]["thresholds"]["srp256_h"]))
bins = np.arange(0, 140, 4)
groups = [("Light", (te.label == 1) & (te.level == "light"), PALETTE[4]),
          ("Medium", (te.label == 1) & (te.level == "medium"), PALETTE[0]),
          ("Heavy", (te.label == 1) & (te.level == "heavy"), PALETTE[2]),
          ("Near negatives", te.kind == "topical-near", PALETTE[5]),
          ("Random negatives", te.kind == "topical-random", PALETTE[6])]
for name, mask, c in groups:
    h = -te.loc[mask, "srp256_h"].to_numpy()
    b.hist(h, bins=bins, density=True, histtype="step", linewidth=1.3, color=c, label=f"{name} (n={len(h)})")
b.axvline(tau, color="black", linestyle="--", linewidth=0.9)
b.text(tau + 2, 0.054, f"τ = {tau}", fontsize=7.5, ha="left")
b.set_xlabel("Hamming distance, 256-bit codes"); b.set_ylabel("Density")
b.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
b.set_title("(b)", loc="left", fontsize=9)
fig.tight_layout(); save(fig, "/home/claude/submitchain/figures/fig7_detection")

# ---- Fig 8: (a) word recovery, mean ± SD over 5 splits; (b) leakage vs detection trade-off
L = json.load(open("data/leakage_5seeds.json"))
keys = ["srp_64", "srp_128", "srp_256", "srp_512", "raw_embedding"]
names = ["64 b", "128 b", "256 b", "512 b", "Raw LSA"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
mu = [L[k]["f1_mean"] for k in keys]; sd = [L[k]["f1_sd"] for k in keys]
a.bar(range(5), mu, 0.6, yerr=sd, color=[PALETTE[0]] * 4 + [PALETTE[3]], error_kw={"capsize": 2, "linewidth": 0.8})
a.axhline(L["prior_baseline"]["f1_mean"], color="black", linestyle="--", linewidth=0.9)
a.text(-0.4, L["prior_baseline"]["f1_mean"] + 0.015, "prior-only baseline", fontsize=7)
a.set_xticks(range(5), names); a.set_ylabel("Token F1 of recovered words"); a.set_ylim(0, 0.75)
a.set_title("(a)", loc="left", fontsize=9)
det_keys = ["SubmitChain-64", "SubmitChain-128", "SubmitChain-256", "SubmitChain-512", "Raw LSA embedding"]
for k, dk, n in zip(keys, det_keys, names):
    x = L[k]["f1_mean"] - L["prior_baseline"]["f1_mean"]; y = R[dk]["by_level"]["heavy"]["recall"]
    b.scatter(x, y, s=28, color=PALETTE[3] if k == "raw_embedding" else PALETTE[0], zorder=3)
    b.annotate(n, (x, y), xytext=(5, -10) if n == "512 b" else ((-30, 4) if n == "256 b" else (4, 3)), textcoords="offset points", fontsize=7.5)
b.set_xlabel("Leakage (token F1 above baseline)"); b.set_ylabel("Heavy duplicates detected")
b.set_xlim(0, 0.48); b.set_ylim(0.2, 1.0)
b.set_title("(b)", loc="left", fontsize=9)
fig.tight_layout(); save(fig, "/home/claude/submitchain/figures/fig8_leakage")
