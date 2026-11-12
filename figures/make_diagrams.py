"""Figures 1-4 (design diagrams). Vector PDF + 600-dpi PNG. Run: python make_diagrams.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, LINE = "#1b2a33", "#55656f", "#7b8a93"
BLUE, BLUE_F = "#0072B2", "#E3F0F8"
GREY_F, ORANGE, ORANGE_F, GREEN, GREEN_F, RED, RED_F = "#F1F4F6", "#B36A00", "#FBF0DC", "#00795A", "#E0F3EC", "#B4372F", "#F9E5E3"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.axis("off"); return fig, ax


def box(ax, x, y, w, h, title, lines=(), fc="white", ec=LINE, lw=1.0, tsize=9, lsize=7.5, tcol=INK, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw, zorder=2))
    n = len(lines); step = 2.9; top = y + h / 2 + (n * step) / 2
    tx = x + w / 2 if align == "center" else x + 1.6
    ax.text(tx, top, title, ha=align, va="center", fontsize=tsize, fontweight="bold", color=tcol, zorder=3)
    for i, l in enumerate(lines):
        ax.text(tx, top - (i + 1) * step - 0.3, l, ha=align, va="center", fontsize=lsize, color=MUTED, zorder=3)


def arrow(ax, p, q, label=None, lpos=None, color=LINE, style="-|>", ls="-", rad=0.0, lha="left"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=1.0, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=0, shrinkB=0))
    if label:
        lx, ly = lpos or ((p[0] + q[0]) / 2 + 1, (p[1] + q[1]) / 2)
        ax.text(lx, ly, label, fontsize=7, color=MUTED, ha=lha, va="center", zorder=3)


def save(fig, name):
    fig.savefig(name + ".pdf"); fig.savefig(name + ".png", dpi=600); plt.close(fig); print("wrote", name)


def fig1():
    fig, ax = canvas(7.0, 5.2)
    for x, name in zip([2, 25, 48], ["Journal A", "Journal B", "Journal C"]):
        ax.add_patch(FancyBboxPatch((x, 33.5), 20, 17, boxstyle="round,pad=0,rounding_size=1.2", fc=GREY_F, ec=LINE, lw=1.0, zorder=1))
        ax.text(x + 10, 48.4, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
        box(ax, x + 1.5, 39.8, 17, 6.6, "Journal gateway", ["manuscript and embeddings"], lsize=6.2, tsize=7.4)
        box(ax, x + 1.5, 34.6, 17, 4.2, "Peer node", fc="white", tsize=7.4)
        arrow(ax, (x + 10, 34.6), (x + 10, 30.4))
    ax.text(13, 32, "ID, pseudonyms,", fontsize=6.2, color=MUTED, ha="left", va="center")
    ax.text(36, 32, "fingerprints,", fontsize=6.2, color=MUTED, ha="left", va="center")
    ax.text(59, 32, "status", fontsize=6.2, color=MUTED, ha="left", va="center")
    ax.add_patch(FancyBboxPatch((2, 13.6), 66, 16.8, boxstyle="round,pad=0,rounding_size=1.2", fc=BLUE_F, ec=BLUE, lw=1.3, zorder=1))
    ax.text(4, 28.2, "Consortium ledger (Hyperledger Fabric channel, all member journals)", fontsize=8, fontweight="bold", color=INK, va="center")
    box(ax, 4, 15.2, 19.5, 10.6, "Submission registry", ["one record per submission,", "no manuscript text"], lsize=6.4, tsize=7.4)
    box(ax, 25.2, 15.2, 20.5, 10.6, "Lifecycle smart contract", ["records status changes,", "matches active submissions only"], lsize=6.2, tsize=7.2)
    box(ax, 47.4, 15.2, 18.8, 10.6, "Ordering service", ["nodes spread across", "member journals"], lsize=6.4, tsize=7.4)
    arrow(ax, (23.5, 20.5), (25.2, 20.5))
    box(ax, 2, 1.5, 20, 8.6, "Duplicate alert", ["to the two journals", "concerned"], fc=RED_F, ec=RED, lsize=6.4, tsize=7.6)
    box(ax, 25, 1.5, 20, 8.6, "LLM-assisted triage", ["needs both editors' consent;", "editors decide"], fc=ORANGE_F, ec=ORANGE, lsize=6.4, tsize=7.6)
    box(ax, 48, 1.5, 20, 8.6, "Submission explorer", ["author, member editor", "and public views"], fc=GREEN_F, ec=GREEN, lsize=6.4, tsize=7.6)
    arrow(ax, (12, 13.6), (12, 10.1)); arrow(ax, (22, 5.8), (25, 5.8)); arrow(ax, (58, 13.6), (58, 10.1), style="<|-|>")
    save(fig, "fig1_architecture")


def fig2():
    fig, ax = canvas(7.0, 3.4)
    ax.add_patch(plt.Circle((3, 17), 1.0, color=INK, zorder=3))
    box(ax, 19, 12.5, 19, 9, "UNDER_REVIEW", ["active: can trigger alerts"], fc=BLUE_F, ec=BLUE, lw=1.3, tsize=8.5, lsize=6.6)
    arrow(ax, (4, 17), (19, 17)); ax.text(11.5, 18.6, "RegisterSubmission", fontsize=6.6, color=MUTED, ha="center")
    for y, name, note, fc, ec in [(24.5, "ACCEPTED", "final", GREEN_F, GREEN), (13, "REJECTED", "final; may be submitted elsewhere", RED_F, RED),
                                  (1.5, "WITHDRAWN", "final; may be submitted elsewhere", ORANGE_F, ORANGE)]:
        box(ax, 46, y, 23, 8, name, [note], fc=fc, ec=ec, lsize=6.2, tsize=8.5)
    arrow(ax, (38, 19.5), (46, 28.5), rad=-0.15); arrow(ax, (38, 17), (46, 17)); arrow(ax, (38, 14.5), (46, 5.5), rad=0.15)
    ax.text(42, 20.6, "UpdateStatus", fontsize=6.2, color=MUTED, ha="center")
    arrow(ax, (24, 12.5), (33, 12.5), rad=1.2, ls="--")
    ax.text(28.5, 4.3, "revision rounds: no state change", fontsize=6.6, color=MUTED, ha="center", va="center")
    ax.text(28.5, 26.5, "only the registering journal\nmay change the state", fontsize=6.6, color=MUTED, ha="center", va="center", style="italic")
    save(fig, "fig2_lifecycle")


def fig3():
    fig, ax = canvas(7.0, 3.4)
    ax.add_patch(FancyBboxPatch((1, 1), 52, 32, boxstyle="round,pad=0,rounding_size=1.2", fc=GREY_F, ec=LINE, lw=1.0, zorder=0))
    ax.text(2.5, 31, "Inside the journal gateway", fontsize=8.5, fontweight="bold", color=INK, va="center")
    ax.add_patch(FancyBboxPatch((55, 1), 14, 32, boxstyle="round,pad=0,rounding_size=1.2", fc=BLUE_F, ec=BLUE, lw=1.3, zorder=0))
    ax.text(62, 31, "On the ledger", fontsize=8.5, fontweight="bold", color=INK, va="center", ha="center")
    for y, v, t in [(18.5, "Header view", "title + abstract"), (4.5, "Body view", "text segments")]:
        box(ax, 2.5, y, 11, 9, v, [t], lsize=6.2, tsize=7.2)
        box(ax, 15.5, y, 11, 9, "Encoder", ["SPECTER"], lsize=6.2, tsize=7.2)
        box(ax, 28.5, y, 11, 9, "Embedding", ["unit vector,", "d values"], lsize=6.2, tsize=7.2)
        box(ax, 41.5, y, 10.5, 9, "Sign random", ["projection:", "keep signs only"], lsize=6.2, tsize=7.2)
        box(ax, 56.5, y, 11, 9, "b-bit code", ["0110 1001 ..."], fc="white", ec=BLUE, lsize=6.4, tsize=7.2)
        for p, q in [(13.5, 15.5), (26.5, 28.5), (39.5, 41.5), (52, 56.5)]:
            arrow(ax, (p, y + 4.5), (q, y + 4.5))
    ax.text(21, 15.6, "segment vectors are mean-pooled", fontsize=6.2, color=MUTED, ha="center", style="italic")
    ax.text(62, 15.6, "Hamming distance\nto active codes", fontsize=6.2, color=MUTED, ha="center", va="center")
    save(fig, "fig3_fingerprint_pipeline")


def fig4():
    fig, ax = canvas(7.0, 3.6)
    box(ax, 19, 29.5, 32, 5.5, "Query by submission identifier", fc=GREY_F, tsize=8.5)
    cols = [(1, "Public", "no sign-in", GREY_F, LINE, ["Identifier is registered: yes / no", "Registration time"], ["Journal", "Status and history", "Authors", "Fingerprints, alerts"]),
            (24, "Member editor", "Fabric identity, owning journal", BLUE_F, BLUE, ["Status and history", "Whether an active match exists", "Matching organisation, distances"], ["Other journals' submissions", "Any manuscript content"]),
            (47, "Author", "ORCID sign-in; pseudonym in record", GREEN_F, GREEN, ["Receiving journal", "Every status change with time,", "  block and transaction ID"], ["Fingerprints", "Alerts and other submissions"])]
    for x, name, how, fc, ec, sees, hidden in cols:
        ax.add_patch(FancyBboxPatch((x, 1), 22, 24.5, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=1.2, zorder=1))
        ax.text(x + 11, 23.2, name, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(x + 11, 20.6, how, ha="center", va="center", fontsize=6.0, color=MUTED, style="italic")
        ax.text(x + 1.5, 17.8, "Returned", fontsize=7, fontweight="bold", color=INK, va="center")
        for i, s in enumerate(sees):
            ax.text(x + 1.5, 15.3 - i * 2.3, ("   " if s.startswith("  ") else "• ") + s.strip(), fontsize=6.1, color=INK, va="center")
        ax.text(x + 1.5, 7.6, "Not returned", fontsize=7, fontweight="bold", color=MUTED, va="center")
        for i, s in enumerate(hidden[:2]):
            ax.text(x + 1.5, 5.3 - i * 2.2, "• " + s, fontsize=6.1, color=MUTED, va="center")
        arrow(ax, (35, 29.5) if x == 24 else (28 if x == 1 else 42, 29.5), (x + 11, 25.5))
    save(fig, "fig4_explorer_views")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
