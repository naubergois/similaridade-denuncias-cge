#!/usr/bin/env python3
"""Academic figures for the Similaridade practice paper. 300 dpi, colorblind-safe.

No generated-art scenes. Boxes, arrows, and bars only.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUT, ".mpl"))

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.linewidth": 0.7,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

NAVY = "#1b365d"
BLUE = "#2c5aa0"
TEAL = "#1f6f63"
ORANGE = "#c46b1a"
RED = "#a33b32"
GRAY = "#4a4a4a"
INK = "#1a1a1a"
PALE = "#f3f5f8"
PALE_T = "#e8f2ef"
PALE_O = "#f8efe4"
PALE_R = "#f6eaea"
LINE = "#8a8a8a"


def box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8.4, weight="normal"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.05,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=weight,
        wrap=True,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=LINE,
        )
    )


def fig_abstract():
    fig, ax = plt.subplots(figsize=(11.4, 4.55))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.55)
    ax.axis("off")

    ax.text(0.25, 4.22, "Graphical abstract", fontsize=8, color=GRAY)
    ax.text(
        0.25,
        3.86,
        "Grouping duplicate citizen complaints in a state ombudsman (CGE-CE)",
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        0.25,
        3.50,
        "Same event, different wording.  n = 4,389 real complaints.  Grouping rate ρ = |G|/n  is not precision.",
        fontsize=9,
        color=INK,
    )

    cards = (
        (0.25, "62.4%", "TF-IDF + DBSCAN", "production  ·  lexical", PALE, BLUE),
        (4.05, "82.5%", "embeddings + HDBSCAN", "inferred  ·  3,621 / 4,389", PALE_T, TEAL),
        (7.85, "84.9%", "+ leftover LLM", "ablation  ·  filter unknown", PALE_O, ORANGE),
    )
    for x, rate, name, note, face, edge in cards:
        box(ax, x, 1.55, 3.30, 1.70, "", face, edge, 8)
        ax.text(x + 1.65, 2.82, rate, ha="center", fontsize=20, fontweight="bold", color=edge)
        ax.text(x + 1.65, 2.28, name, ha="center", fontsize=9.2, color=INK)
        ax.text(x + 1.65, 1.88, note, ha="center", fontsize=8, color=GRAY)

    ax.add_patch(Rectangle((0.25, 0.22), 10.90, 1.10, facecolor=PALE, edgecolor=NAVY, linewidth=0.9))
    ax.text(
        5.70,
        0.92,
        "Accepted cluster:  |C| ≥ 2  and  mean pairwise cosine  s̄(C) ≥ 0.75.",
        ha="center",
        fontsize=9,
        color=INK,
    )
    ax.text(
        5.70,
        0.52,
        "No officer gold.  Leftover path 768 → 27 → 104 is not an auditable operator.  An officer still closes the case.",
        ha="center",
        fontsize=8.4,
        color=GRAY,
    )
    fig.savefig(os.path.join(OUT, "fig_abstract.png"), facecolor="white")
    plt.close()


def fig_jobs():
    fig, ax = plt.subplots(figsize=(11.2, 4.15))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.15)
    ax.axis("off")
    ax.text(0.20, 3.82, "Three jobs on citizen text", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(
        0.20,
        3.48,
        "Only the right-hand job is Similaridade.  A higher grouping rate can still mix two events.",
        fontsize=8.6,
        color=GRAY,
    )

    cols = (
        (0.20, "Route", "Assign a unit or service", "Das; Chen; Silva (Fala.BR)\nRakhimzhanov et al.", "Not this paper", PALE, BLUE),
        (3.86, "Theme", "Discover latent topics", "BERTopic; Tang et al.\nEsperança et al.", "Not this paper", PALE_T, TEAL),
        (7.52, "Case", "Same real-world event", "Fellegi–Sunter; Broder\nRabbi et al. (consumer)", "Similaridade", PALE_O, ORANGE),
    )
    for x, title, job, cites, flag, face, edge in cols:
        box(ax, x, 0.28, 3.46, 3.00, "", face, edge)
        ax.text(x + 1.73, 2.90, title, ha="center", fontsize=13, fontweight="bold", color=edge)
        ax.text(x + 1.73, 2.42, job, ha="center", fontsize=9.2, color=INK)
        ax.text(x + 1.73, 1.70, cites, ha="center", fontsize=8.0, color=GRAY)
        ax.text(x + 1.73, 0.72, flag, ha="center", fontsize=9.0, fontweight="bold", color=edge)
    fig.savefig(os.path.join(OUT, "fig_jobs.png"), facecolor="white")
    plt.close()


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.3, 5.35))
    ax.set_xlim(0, 11.3)
    ax.set_ylim(0, 5.35)
    ax.axis("off")

    box(ax, 0.18, 2.20, 1.55, 0.95, "Complaint\nbatch\nn = 4,389", PALE, NAVY, 8.2)

    ax.text(4.85, 5.08, "Production  (live, August 2026)", fontsize=10, fontweight="bold", color=BLUE)
    box(ax, 2.05, 3.85, 1.70, 0.88, "TF-IDF\n1–3 grams", PALE, BLUE)
    box(ax, 4.00, 3.85, 1.70, 0.88, "DBSCAN\nε, m unknown", PALE, BLUE)
    box(ax, 5.95, 3.85, 1.85, 0.88, "Accept if\ns̄(C) ≥ 0.75", PALE, BLUE)
    box(ax, 8.05, 3.85, 2.95, 0.88, "ρ = 62.4%\n≈ 2,739 grouped", PALE, BLUE, 8.2)

    ax.text(4.85, 3.12, "Homolog  (same batch, no leftover model)", fontsize=10, fontweight="bold", color=TEAL)
    box(ax, 2.05, 1.90, 1.70, 0.88, "Sentence\nembeddings", PALE_T, TEAL)
    box(ax, 4.00, 1.90, 1.70, 0.88, "HDBSCAN\ncut unknown", PALE_T, TEAL)
    box(ax, 5.95, 1.90, 1.85, 0.88, "Implied G\nwithout R", PALE_T, TEAL)
    box(ax, 8.05, 1.90, 2.95, 0.88, "ρ = 82.5%\n3,621 grouped", PALE_T, TEAL, 8.2)

    ax.text(4.85, 1.18, "Ablation  (leftover LLM; not a deployment claim)", fontsize=10, fontweight="bold", color=ORANGE)
    box(ax, 2.05, 0.18, 1.70, 0.72, "L = 768\nleftovers", PALE_O, ORANGE, 8)
    box(ax, 4.00, 0.18, 1.70, 0.72, "K = 27\nrule unknown", PALE_O, ORANGE, 8)
    box(ax, 5.95, 0.18, 1.85, 0.72, "R = 104\nin 21 groups", PALE_O, ORANGE, 8)
    box(ax, 8.05, 0.18, 2.95, 0.72, "ρ = 84.9%\n3,725 grouped", PALE_O, ORANGE, 8.2)

    arrow(ax, 1.73, 2.67, 2.05, 4.29)
    arrow(ax, 1.73, 2.67, 2.05, 2.34)
    for y in (4.29, 2.34):
        arrow(ax, 3.75, y, 4.00, y)
        arrow(ax, 5.70, y, 5.95, y)
        arrow(ax, 7.80, y, 8.05, y)
    arrow(ax, 3.75, 0.54, 4.00, 0.54)
    arrow(ax, 5.70, 0.54, 5.95, 0.54)
    arrow(ax, 7.80, 0.54, 8.05, 0.54)

    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), facecolor="white")
    plt.close()


def fig_grouping():
    fig, ax = plt.subplots(figsize=(6.8, 4.15))
    labels = [
        "TF-IDF + DBSCAN\n(production)",
        "Embeddings + HDBSCAN\n(inferred)",
        "HDBSCAN + leftover LLM\n(reported ablation)",
    ]
    vals = [62.4, 82.5, 84.9]
    colors = [BLUE, TEAL, ORANGE]
    bars = ax.bar(labels, vals, color=colors, width=0.58, edgecolor="white")
    ax.set_ylabel(r"Grouping rate  $\rho$  (%)")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    notes = ["reported", "3,621 / 4,389", "includes 104 recovered"]
    for b, v, n in zip(bars, vals, notes):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v + 7.4, n, ha="center", fontsize=7.4, color=GRAY)
    ax.set_title("Same batch, n = 4,389.  ρ is not precision.", fontsize=9, color=GRAY, pad=8)
    fig.savefig(os.path.join(OUT, "fig2_grouping_rate.png"), facecolor="white")
    plt.close()


def fig_funnel():
    fig, ax = plt.subplots(figsize=(6.9, 3.85))
    stages = [
        ("Leftovers after HDBSCAN  (L)", 768, ORANGE),
        ("Candidates sent to LLM  (K)", 27, GOLD := "#b07a20"),
        ("Recovered complaints  (R)", 104, TEAL),
        ("Leftover groups reported", 21, BLUE),
    ]
    y = list(range(len(stages)))[::-1]
    ax.barh(y, [s[1] for s in stages], color=[s[2] for s in stages], height=0.58, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stages])
    ax.set_xlabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for yi, w in zip(y, [s[1] for s in stages]):
        ax.text(w + 14, yi, str(w), va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 980)
    ax.set_title("Map L → K is unknown, so R is not auditable.", fontsize=9, color=GRAY, pad=6)
    fig.savefig(os.path.join(OUT, "fig3_outlier_funnel.png"), facecolor="white")
    plt.close()


def fig_bands():
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 3.15)
    ax.axis("off")
    ax.text(0.15, 2.82, "Provisional review queues  (cuts unknown)", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(
        0.15,
        2.48,
        "Fellegi–Sunter pair decisions, applied to accepted clusters.  Not a measured safety threshold.",
        fontsize=8.4,
        color=GRAY,
    )
    cols = (
        (0.15, "Green", "Suggested link", "≈ bulk linking", "≈ 87% of G", PALE_T, TEAL),
        (2.50, "Yellow", "Possible link", "officer confirms", "share unknown", PALE_O, ORANGE),
        (4.85, "Red", "Read with care", "treat as non-link", "share unknown", PALE_R, RED),
    )
    for x, name, fs, job, note, face, edge in cols:
        box(ax, x, 0.22, 2.20, 2.05, "", face, edge)
        ax.text(x + 1.10, 1.85, name, ha="center", fontsize=12, fontweight="bold", color=edge)
        ax.text(x + 1.10, 1.42, fs, ha="center", fontsize=8.6, color=INK)
        ax.text(x + 1.10, 1.00, job, ha="center", fontsize=8.6, color=INK)
        ax.text(x + 1.10, 0.55, note, ha="center", fontsize=8.0, color=GRAY)
    fig.savefig(os.path.join(OUT, "fig_bands.png"), facecolor="white")
    plt.close()


def fig_operator():
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    labels = ["Grouped\ncomplaints  |G|", "Arithmetic remainder\nif green is trusted"]
    vals = [3725, 484]
    bars = ax.bar(labels, vals, color=[BLUE, TEAL], width=0.5, edgecolor="white")
    ax.set_ylabel("Complaints")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 70, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 4300)
    ax.set_title("0.13 × 3,725 ≈ 484.  Not officer minutes.", fontsize=9, color=GRAY, pad=6)
    fig.savefig(os.path.join(OUT, "fig4_operator_load.png"), facecolor="white")
    plt.close()


def fig_day():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), sharey=False)
    ax = axes[0]
    ax.bar(["Cluster 31", "Leftover"], [48, 1], color=[BLUE, GRAY], width=0.55, edgecolor="white")
    ax.set_title("08:00  ·  49 complaints")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 85)
    ax.text(0, 52, "48  ·  85.68%", ha="center", fontsize=8)
    ax.text(1, 4.5, "1", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.bar(["Cluster 32", "Cluster 33", "Leftover"], [73, 3, 2], color=[TEAL, ORANGE, GRAY], width=0.6, edgecolor="white")
    ax.set_title("12:00  ·  78 complaints")
    ax.set_ylim(0, 85)
    ax.text(0, 76, "73  ·  100%", ha="center", fontsize=8)
    ax.text(1, 6.8, "3  ·  85.76%", ha="center", fontsize=8)
    ax.text(2, 5.2, "2", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Production day, 28 August 2026  (TF-IDF + DBSCAN)", fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_production_day.png"), facecolor="white")
    plt.close()


if __name__ == "__main__":
    fig_abstract()
    fig_jobs()
    fig_pipeline()
    fig_grouping()
    fig_funnel()
    fig_bands()
    fig_operator()
    fig_day()
    print("wrote academic figures in", OUT)
