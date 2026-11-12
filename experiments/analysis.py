"""Analyses reported in Section 3.1-3.2 that need no benchmark data.

1. Estimator check: empirical mean Hamming distance between sign-projection codes of unit vectors at
   known angles, against the theoretical value b*theta/pi (Charikar 2002).
2. Operating characteristic: P(flag | cosine) = P[Binomial(b, theta/pi) <= tau], for illustrative
   cosine levels. Shows how code length sharpens the separation; actual levels come from the benchmark.
3. Index recall: probability that a pair at Hamming distance d shares at least one index key, for disjoint
   bands and for the deployed bit-sampling tables, with measured points; and expected spurious candidates.
Writes analysis.json and figures fig5_estimator.(pdf|png), fig6_index.(pdf|png).
"""
import json
import math
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

import sys
sys.path.insert(0, "gateway")
from fingerprint import SignRandomProjection, hamming  # noqa: E402

PAL = {"128": "#E69F00", "256": "#0072B2", "512": "#009E73", "scan": "#D55E00", "index": "#0072B2"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linewidth": 0.5, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "legend.frameon": False})
out = {}
rng = np.random.default_rng(2026)
d = 768                      # SPECTER embedding dimension
BITS = [128, 256, 512]

# ---- 1. estimator check ----------------------------------------------------------------------
cosines = np.array([0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.30, 0.0])
est = {}
for b in BITS:
    srp = SignRandomProjection.generate(d, b, 7)
    rows = []
    for c in cosines:
        x = rng.standard_normal((5000, d)); x /= np.linalg.norm(x, axis=1, keepdims=True)
        n = rng.standard_normal((5000, d)); n -= (n * x).sum(1, keepdims=True) * x; n /= np.linalg.norm(n, axis=1, keepdims=True)
        y = c * x + math.sqrt(1 - c * c) * n
        h = hamming(srp.encode(x), srp.encode(y))
        th = math.acos(c) / math.pi
        rows.append({"cos": float(c), "mean_H": float(h.mean()), "sd_H": float(h.std()), "theory_mean": b * th,
                     "theory_sd": math.sqrt(b * th * (1 - th))})
    est[str(b)] = rows
out["estimator"] = est
worst = max(abs(r["mean_H"] - r["theory_mean"]) / b for b in BITS for r in est[str(b)])
out["estimator_max_abs_error_fraction_of_bits"] = worst

# ---- 2. operating characteristic -------------------------------------------------------------
c_dup, c_neg = 0.90, 0.80    # illustrative only
oc = {}
for b in BITS:
    p_dup, p_neg = math.acos(c_dup) / math.pi, math.acos(c_neg) / math.pi
    tau = int(math.floor(b * (p_dup + p_neg) / 2))
    oc[str(b)] = {"tau": tau, "P_flag_at_0.90": float(binom.cdf(tau, b, p_dup)), "P_flag_at_0.80": float(binom.cdf(tau, b, p_neg)),
                  "curve": [[float(c), float(binom.cdf(tau, b, math.acos(c) / math.pi))] for c in np.linspace(0.6, 1.0, 161)]}
out["operating_characteristic"] = {"illustrative_cosines": [c_dup, c_neg], "by_bits": oc}

# ---- 3. index recall ---------------------------------------------------------------------
def p_bands(n_bits, w, dist):
    """Disjoint w-bit bands: P(at least one band has no flipped bit | dist flips), inclusion-exclusion."""
    m = n_bits // w
    miss = sum((-1) ** j * comb(m, j) * comb(n_bits - j * w, dist) for j in range(m + 1) if n_bits - j * w >= dist) / comb(n_bits, dist)
    return 1 - miss

def p_sampled(n_bits, tables, k, dist):
    """L tables of k randomly sampled positions: 1 - (1 - C(b-k,d)/C(b,d))^L."""
    q = comb(n_bits - k, dist) / comb(n_bits, dist) if n_bits - k >= dist else 0.0
    return 1 - (1 - q) ** tables

configs = {"16 disjoint 16-bit bands": ("bands", 16, 16), "L=64, k=16 (deployed)": ("sampled", 64, 16),
           "L=128, k=20": ("sampled", 128, 20)}
recall = {}
for name, (kind, L_, k) in configs.items():
    f = (lambda d_: p_bands(256, k, d_)) if kind == "bands" else (lambda d_, L_=L_, k=k: p_sampled(256, L_, k, d_))
    recall[name] = {"curve": [[dd, f(dd)] for dd in range(0, 129)], "at": {str(dd): f(dd) for dd in (24, 32, 40, 44, 48, 56)},
                    "spurious_per_view_per_active": L_ * 2.0 ** -k, "keys_per_view": L_}
out["index_recall_256bits"] = recall
out["index_recall_empirical"] = json.load(open("experiments/index_recall.json"))

# ---- chaincode benchmark (measured) -------------------------------------------------------
bench = json.load(open("experiments/chaincode_bench.json"))
out["chaincode_bench"] = bench
json.dump(out, open("experiments/analysis.json", "w"), indent=1)

# ---- Figure 5 ------------------------------------------------------------------------------
fig, (a, bx) = plt.subplots(1, 2, figsize=(7.0, 2.9))
xs = np.linspace(0, 1, 200)
for b in BITS:
    rows = est[str(b)]
    a.plot(xs, b * xs, color=PAL[str(b)], lw=1, alpha=0.7)
    th = [math.acos(r["cos"]) / math.pi for r in rows]
    a.errorbar(th, [r["mean_H"] / 1 for r in rows], yerr=[2 * r["sd_H"] for r in rows], fmt="o", ms=3, color=PAL[str(b)],
               elinewidth=0.8, capsize=1.5, label=f"b = {b}")
a.set_xlabel(r"Angle between embeddings, $\theta/\pi$"); a.set_ylabel("Hamming distance between codes")
a.set_xlim(0, 0.52); a.set_ylim(0, 270)
a.text(-0.16, 1.02, "(a)", transform=a.transAxes, fontweight="bold", va="bottom")
a.legend(loc="upper left", fontsize=7.5, title="points: mean ± 2 SD\nlines: b·θ/π", title_fontsize=7, alignment="left")
for b in BITS:
    cur = np.array(oc[str(b)]["curve"])
    bx.plot(cur[:, 0], cur[:, 1], color=PAL[str(b)], lw=1.4, label=f"b = {b} (τ = {oc[str(b)]['tau']})")
for c in (c_neg, c_dup):
    bx.axvline(c, color="#888", lw=0.8, ls="--")
bx.text(c_neg, 1.03, "0.80", ha="center", fontsize=7, color="#555"); bx.text(c_dup, 1.03, "0.90", ha="center", fontsize=7, color="#555")
bx.set_xlabel("Cosine similarity of the two embeddings"); bx.set_ylabel("Probability the pair is flagged")
bx.set_xlim(0.7, 1.0); bx.set_ylim(-0.02, 1.08)
bx.text(-0.16, 1.02, "(b)", transform=bx.transAxes, fontweight="bold", va="bottom")
bx.legend(loc="lower right", fontsize=7.5)
fig.tight_layout()
fig.savefig("figures/fig5_estimator.pdf"); fig.savefig("figures/fig5_estimator.png", dpi=600)

# ---- Figure 6 ------------------------------------------------------------------------------
fig, (a, bx) = plt.subplots(1, 2, figsize=(7.0, 2.9))
for mode in ("scan", "index"):
    rows = [r for r in bench["rows"] if r["mode"] == mode]
    n = [r["active"] for r in rows]; med = [r["median_ms"] for r in rows]; p95 = [r["p95_ms"] for r in rows]
    a.errorbar(n, med, yerr=[np.zeros(len(n)), np.array(p95) - np.array(med)], marker="o", ms=3.5, lw=1.4, color=PAL[mode],
               elinewidth=0.8, capsize=2, label=f"{mode}")
    for x, y, r in zip(n, med, rows):
        a.annotate(f"{r['median_reads']}", (x, y), textcoords="offset points", xytext=(-9, 4) if mode == "scan" else (0, -11),
                   ha="center", fontsize=6.3, color=PAL[mode])
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("Active submissions on the ledger"); a.set_ylabel("Contract execution time (ms)")
a.text(-0.16, 1.02, "(a)", transform=a.transAxes, fontweight="bold", va="bottom")
a.legend(loc="upper left", fontsize=7.5); a.set_ylim(0.1, 2000)
a.text(0.98, 0.04, "points: median; whiskers: 95th percentile\nlabels: state keys read per registration", transform=a.transAxes,
       ha="right", fontsize=6.3, color="#555")
cols = {"16 disjoint 16-bit bands": "#999999", "L=64, k=16 (deployed)": "#0072B2", "L=128, k=20": "#009E73"}
for name, r in recall.items():
    cur = np.array(r["curve"]); bx.plot(cur[:, 0], cur[:, 1], color=cols[name], lw=1.4, ls="--" if "bands" in name else "-", label=name)
emp = out["index_recall_empirical"]["recall"]
for key, name in (("L64_k16", "L=64, k=16 (deployed)"), ("L128_k20", "L=128, k=20")):
    e = np.array(emp[key]); bx.plot(e[:, 0], e[:, 1], "o", ms=3, color=cols[name], mfc="white", mew=1)
bx.axvline(44, color="#888", lw=0.8, ls=":"); bx.text(45, 1.03, "τ = 44", fontsize=7, color="#555")
bx.set_xlabel("Hamming distance between 256-bit codes"); bx.set_ylabel("P(pair retrieved by the index)")
bx.set_xlim(0, 100); bx.set_ylim(-0.02, 1.09)
bx.text(-0.16, 1.02, "(b)", transform=bx.transAxes, fontweight="bold", va="bottom")
bx.legend(loc="lower left", fontsize=7, title="lines: analytic, circles: measured", title_fontsize=6.5)
fig.tight_layout()
fig.savefig("figures/fig6_index.pdf"); fig.savefig("figures/fig6_index.png", dpi=600)

print("estimator max |error| as fraction of b:", round(worst, 4))
for b in BITS:
    o = oc[str(b)]; print(b, "tau", o["tau"], "P(flag|0.90)=%.4f P(flag|0.80)=%.2e" % (o["P_flag_at_0.90"], o["P_flag_at_0.80"]))
for name, r in recall.items():
    print(name, {k: round(v, 4) for k, v in r["at"].items()}, "spurious/view/active", r["spurious_per_view_per_active"])
