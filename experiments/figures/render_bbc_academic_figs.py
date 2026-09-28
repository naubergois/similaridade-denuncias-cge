#!/usr/bin/env python3
"""Academic infographic figures for the BBC SOTA lab article.

Numbers from experiments/bbc_final.json, bbc_llm_ensemble.json,
and Keraghel & Nadif Table 2 (published, not re-run). 300 dpi.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1] / "paper" / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl"))
PAPER.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.7,
        "legend.fontsize": 8,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "pdf.fonttype": 42,
    }
)

BLUE = "#2060cc"
NAVY = "#1b365d"
TEAL = "#1f6f63"
ORANGE = "#cc7020"
RED = "#b03a2e"
GREEN = "#208040"
GRAY = "#555555"
LIGHT = "#e8eef6"
SAND = "#f4efe4"
MINT = "#dceee8"
PEACH = "#f6e6d4"
PUB_NMI, PUB_ARI, PUB_ACC = 0.9512, 0.9666, 0.9761

FLOORS = [
    ("1", "Majority of 11", 0.9234, 0.9451, 0.9771, 51),
    ("2", "Sonnet pair-80", 0.9358, 0.9554, 0.9816, 41),
    ("3", "Opus flips+close-3", 0.9381, 0.9575, 0.9825, 39),
    ("4", "Odd-one-out", 0.9430, 0.9607, 0.9838, 36),
    ("5", "Revolt 4/7", 0.9466, 0.9628, 0.9847, 34),
    ("7", "Foreign NN", 0.9482, 0.9640, 0.9852, 33),
    ("12", "Rim exemplars", 0.9487, 0.9639, 0.9852, 33),
    ("14", "Event kind", 0.9496, 0.9663, 0.9861, 31),
    ("15", "Closest-pair keep", 0.9520, 0.9673, 0.9865, 30),
]

PANEL = [
    ("mpnet", 0.9067),
    ("bge", 0.8946),
    ("concat", 0.9172),
    ("mix", 0.9182),
    ("mpnet+tfidf", 0.9087),
    ("chunk", 0.9002),
    ("title", 0.9092),
    ("factorial", 0.9164),
    ("consensus", 0.9167),
    ("GMM", 0.9165),
    ("spectral", 0.6689),
    ("majority", 0.9234),
]

GCC_BBC = [
    ("GCC(kNN, co)", 0.8876, 0.7626, 0.7597),
    ("GCC(NER, co)", 0.9543, 0.8618, 0.8927),
    ("GCC(kNN, LLM)", 0.9582, 0.8741, 0.9012),
    (r"GCC$^*$", 0.9761, 0.9512, 0.9666),
]


def _save(fig, name: str) -> Path:
    dest_exp = HERE / name
    dest_paper = PAPER / name
    fig.savefig(dest_exp, facecolor="white")
    fig.savefig(dest_paper, facecolor="white")
    plt.close(fig)
    print(dest_paper)
    return dest_paper


def _box(ax, x, y, w, h, text, fc, ec=NAVY, fs=8, weight="regular"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        fc=fc,
        ec=ec,
        lw=0.9,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=NAVY,
        fontweight=weight,
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            color=NAVY,
            lw=1.0,
        )
    )


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Gated audit of an ensemble partition (laboratory protocol)", pad=8)

    ax.add_patch(Rectangle((0.25, 0.4), 7.3, 9.2, fc=LIGHT, ec="#b8c4d6", lw=0.6))
    ax.add_patch(Rectangle((7.8, 0.4), 13.95, 9.2, fc=SAND, ec="#d4c6aa", lw=0.6))
    ax.text(3.9, 9.2, "Block A  —  unsupervised geometry", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.text(14.7, 9.2, "Block B  —  gated language-model audit", ha="center", fontsize=9, fontweight="bold", color=NAVY)

    views = [
        (0.5, 7.55, "MPNet / BGE"),
        (0.5, 6.35, "concat / mix"),
        (0.5, 5.15, "chunk / title"),
        (0.5, 3.95, "factorial / GMM"),
        (0.5, 2.75, "consensus"),
        (0.5, 1.55, "spectral kNN"),
    ]
    for x, y, t in views:
        _box(ax, x, y, 2.6, 0.95, t, "#ffffff", fs=7.5)
    _box(ax, 3.5, 3.9, 3.5, 2.3, "Hungarian align\nto mix view\n\nmajority vote\nNMI 0.9234", MINT, fs=7.5, weight="bold")
    _arrow(ax, 3.15, 2.05, 3.5, 4.4)
    _arrow(ax, 3.15, 6.8, 3.5, 5.5)

    gates = [
        (8.15, 7.4, "2  pair on thin 80\n0.9358", LIGHT),
        (11.55, 7.4, "3  Opus close-3\n0.9381", LIGHT),
        (14.95, 7.4, "4  odd-one-out\n0.9430", LIGHT),
        (8.15, 5.15, "5  revolt 4/7\n0.9466", LIGHT),
        (11.55, 5.15, "7  foreign NN\n0.9482", LIGHT),
        (14.95, 5.15, "12  rim exemplars\n0.9487", LIGHT),
        (8.15, 2.7, "14  event kind\n0.9496", PEACH),
        (11.55, 2.7, "15  keep closest pair\n0.9520", MINT),
        (14.95, 2.7, "30 leftover\nerrors", "#f4d6d2"),
    ]
    for x, y, t, fc in gates:
        _box(ax, x, y, 3.15, 1.55, t, fc, fs=7.2)
    _arrow(ax, 7.05, 5.05, 8.15, 6.0)
    for a, b in [(11.3, 11.55), (14.7, 14.95)]:
        _arrow(ax, a, 8.15, b, 8.15)
    _arrow(ax, 9.7, 7.4, 9.7, 6.7)
    _arrow(ax, 13.1, 7.4, 13.1, 6.7)
    _arrow(ax, 16.5, 7.4, 16.5, 6.7)
    _arrow(ax, 11.3, 5.9, 11.55, 5.9)
    _arrow(ax, 14.7, 5.9, 14.95, 5.9)
    _arrow(ax, 9.7, 5.15, 9.7, 4.25)
    _arrow(ax, 13.1, 5.15, 13.1, 4.25)
    _arrow(ax, 11.3, 3.45, 11.55, 3.45)
    _arrow(ax, 14.7, 3.45, 14.95, 3.45)

    ax.text(
        11.0,
        0.75,
        "Gold never picks the set.  Desk names are not the answer vocabulary.  Source: bbc_final.json.",
        ha="center",
        fontsize=7.2,
        color=GRAY,
    )
    _save(fig, "fig_pipeline.png")


def fig_metrics_compare() -> None:
    methods = ["Majority\n(Floor 1)", "Floor 15\n(kept)", r"GCC$^*$" + "\n(published)"]
    nmi = [0.9234, 0.9520, 0.9512]
    ari = [0.9451, 0.9673, 0.9666]
    acc = [0.9771, 0.9865, 0.9761]
    colors = [BLUE, TEAL, ORANGE]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8), sharey=False)
    for ax, vals, title, ymin in zip(
        axes,
        [nmi, ari, acc],
        ["NMI", "ARI", "Hungarian ACC"],
        [0.90, 0.93, 0.96],
    ):
        bars = ax.bar([0, 1, 2], vals, color=colors, width=0.62, edgecolor="white")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(methods, fontsize=7.5)
        ax.set_ylim(ymin, 1.0)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.axhline(vals[2], color=ORANGE, ls=":", lw=0.8, alpha=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        ax.text(0.02, 0.04, f"axis from {ymin:.2f}", transform=ax.transAxes, fontsize=6.5, color=GRAY)
    fig.suptitle("BBC News: kept laboratory stack against published GCC*", fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, "Floor 1/15: bbc_final.json (measured). GCC*: Keraghel & Nadif, Table 2.", ha="center", fontsize=7.5, color=GRAY)
    _save(fig, "fig_metrics_compare.png")


def fig_staircase() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    xs = list(range(len(FLOORS)))
    ys = [r[2] for r in FLOORS]
    ax.plot(xs, ys, color=NAVY, lw=1.8, zorder=2)
    for i, (fid, _name, nmi, *_rest) in enumerate(FLOORS):
        c = TEAL if nmi >= PUB_NMI else BLUE
        ax.scatter(i, nmi, s=42, color=c, zorder=3, edgecolors="white", linewidths=0.4)
        ax.annotate(f"{nmi:.4f}", (i, nmi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=c)
    ax.axhline(PUB_NMI, color=ORANGE, ls="--", lw=1.1, label=f"GCC* published NMI {PUB_NMI:.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r[0]}\n{r[1]}" for r in FLOORS], fontsize=6.6)
    ax.set_ylabel("NMI (sklearn, arithmetic)")
    ax.set_ylim(0.918, 0.956)
    ax.set_title("Kept floors: NMI on BBC News (gold does not pick documents)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    fig.text(0.5, -0.02, "Source: experiments/bbc_final.json. Zoomed y-axis to show the last 0.03.", ha="center", fontsize=7.5, color=GRAY)
    path = _save(fig, "fig_nmi_staircase.png")
    shutil.copy(path, PAPER / "fig_bbc_sota_staircase.png")
    shutil.copy(path, HERE / "bbc_sota_staircase.png")


def fig_errors() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    xs = list(range(len(FLOORS)))
    ys = [r[5] for r in FLOORS]
    bars = ax.bar(xs, ys, color=[TEAL if e <= 30 else BLUE for e in ys], width=0.7, edgecolor="white")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"F{r[0]}" for r in FLOORS])
    ax.set_ylabel("Disagreements with gold desks")
    ax.set_ylim(0, 58)
    ax.set_title("Remaining errors on the kept path")
    for b, e in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, e + 0.8, str(e), ha="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    fig.text(0.5, -0.03, "N = 2225. Source: bbc_final.json. Bars start at zero.", ha="center", fontsize=7.5, color=GRAY)
    _save(fig, "fig_errors.png")


def fig_gcc_published() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    names = [r[0] for r in GCC_BBC]
    acc = [r[1] for r in GCC_BBC]
    nmi = [r[2] for r in GCC_BBC]
    ari = [r[3] for r in GCC_BBC]
    x = range(len(names))
    w = 0.25
    ax.bar([i - w for i in x], acc, w, label="ACC", color=BLUE)
    ax.bar(list(x), nmi, w, label="NMI", color=TEAL)
    ax.bar([i + w for i in x], ari, w, label="ARI", color=ORANGE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Published score (unit interval)")
    ax.set_title("BBC News column of Keraghel and Nadif, Table 2")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    fig.text(0.5, -0.04, "Not re-run here. Percentages in the source table divided by 100.", ha="center", fontsize=7.5, color=GRAY)
    _save(fig, "fig_gcc_published.png")


def fig_panel() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    items = list(PANEL)
    names = [p[0] for p in items]
    vals = [p[1] for p in items]
    colors = []
    for n in names:
        if n == "majority":
            colors.append(TEAL)
        elif n == "spectral":
            colors.append(GRAY)
        else:
            colors.append(BLUE)
    ys = list(range(len(names)))
    ax.barh(ys, vals, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_xlim(0.60, 0.95)
    ax.axvline(PUB_NMI, color=ORANGE, ls="--", lw=1.0, label=f"GCC* {PUB_NMI:.4f}")
    ax.set_xlabel("NMI")
    ax.set_title("Single-view panel versus majority (BBC News)")
    for y, v in zip(ys, vals):
        ax.text(v + 0.003, y, f"{v:.4f}", va="center", fontsize=7)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.5, -0.03, "Source: bbc_llm_ensemble.json. Spectral kNN is a weak voter; majority still wins.", ha="center", fontsize=7.5, color=GRAY)
    _save(fig, "fig_panel.png")


def fig_protocol() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Fair protocol versus probes that use gold", pad=6)

    _box(ax, 0.4, 5.6, 6.0, 3.6, "FAIR  (headline)\n\nGate from geometry only\nPair / 5-way exemplars\nNo desk-name answers\n\nFloor 15   NMI 0.9520", MINT, fs=8, weight="bold")
    _box(ax, 7.0, 5.6, 6.0, 3.6, "PROBE  (not comparable)\n\nGold picks the 21\nborder errors\n\nSonnet pair   NMI 0.9468", PEACH, fs=8)
    _box(ax, 13.6, 5.6, 6.0, 3.6, "ORACLE  (ceiling)\n\nGold flips the 21\nthinnest-margin errors\n\nn=80   NMI 0.9527", LIGHT, fs=8)
    _box(ax, 0.4, 0.6, 19.2, 4.2, "", "#fafafa", ec="#cccccc")
    ax.text(10.2, 4.2, "Why the gold-picked probe is not next to GCC*", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.text(
        10.2,
        2.35,
        "If gold chooses which articles the model reads, the score answers a different question.\n"
        "The fair Floor 15 sits above that probe (0.9520 > 0.9468).\n"
        "Oracle n=120 reaches 0.9609 and is a ceiling, not a method.\n"
        "Sources: bbc_final.json, bbc_claude_border_errors.json, bbc_nmi_lift.json.",
        ha="center",
        va="center",
        fontsize=8,
        color=GRAY,
    )
    _save(fig, "fig_protocol.png")


def fig_gates() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("What the language model is allowed to see", pad=6)
    steps = [
        (0.3, "Document\nin mix space", LIGHT),
        (3.7, "Thin margin\nor neighbour\nrevolt?", SAND),
        (7.1, "Sonnet pair\n0 / 1", LIGHT),
        (10.5, "Opus only\non flips /\nrevolt", PEACH),
        (13.9, "Keep if flip\nstays on\nclosest pair", MINT),
    ]
    for x, t, fc in steps:
        _box(ax, x, 2.6, 3.05, 3.4, t, fc, fs=8.5, weight="bold")
    for x in (3.35, 6.75, 10.15, 13.55):
        _arrow(ax, x, 4.3, x + 0.35, 4.3)
    ax.text(5.2, 1.6, "no  →  keep majority label", ha="center", fontsize=8, color=GRAY)
    ax.text(15.4, 1.6, "off-pair flip discarded", ha="center", fontsize=8, color=RED)
    ax.text(9.0, 0.55, "Core of each cluster is never sent to the model. Source: run_bbc_llm_arch*.py.", ha="center", fontsize=7.5, color=GRAY)
    _save(fig, "fig_gates.png")


def fig_leak() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("The leak that held NMI down: brand versus event", pad=6)
    _box(ax, 0.4, 2.2, 4.6, 3.6, "Embedding cosine\n\nVodafone, handset,\nsite, gadget words\n→ digital pile", LIGHT, fs=8.5)
    _box(ax, 5.7, 2.2, 4.6, 3.6, "BBC desk gold\n\nBoard appointment,\nregulator, results\n→ business pile", SAND, fs=8.5)
    _box(ax, 11.0, 2.2, 4.6, 3.6, "Floor 14–15 move\n\nMatch event kind,\nkeep only closest\ncentroid pair", MINT, fs=8.5)
    _arrow(ax, 5.05, 4.0, 5.65, 4.0)
    _arrow(ax, 10.35, 4.0, 10.95, 4.0)
    ax.text(8.0, 0.7, "Not a measured confusion cell. Qualitative residual after Floor 15 (30 errors). bbc_llm_arch9.json.", ha="center", fontsize=7.5, color=GRAY)
    _save(fig, "fig_leak.png")


def main() -> None:
    fig_pipeline()
    fig_metrics_compare()
    fig_staircase()
    fig_errors()
    fig_gcc_published()
    fig_panel()
    fig_protocol()
    fig_gates()
    fig_leak()
    print("done")


if __name__ == "__main__":
    main()
