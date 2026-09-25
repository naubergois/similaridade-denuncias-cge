#!/usr/bin/env python3
"""Publication figures for the Similaridade practice paper. 300 dpi, colorblind-safe."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
        "savefig.pad_inches": 0.12,
    }
)

BLUE = "#2060cc"
GREEN = "#208040"
ORANGE = "#cc7020"
RED = "#cc3030"
GOLD = "#b08020"
GRAY = "#555555"
LIGHT = "#f4f6f8"


def box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8.5):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.1,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#1a1a1a")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=GRAY,
        )
    )


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    box(ax, 0.25, 2.15, 1.7, 0.95, "Complaint\ntext", "#e8eef7", BLUE, 9)

    ax.text(3.55, 4.85, "Production (live, Aug 2026)", ha="center", fontsize=10, fontweight="bold", color=BLUE)
    box(ax, 2.3, 3.55, 1.55, 0.85, "TF-IDF\nn-grams", "#e8eef7", BLUE)
    box(ax, 4.15, 3.55, 1.55, 0.85, "DBSCAN\nfixed radius", "#e8eef7", BLUE)
    box(ax, 6.0, 3.55, 1.7, 0.85, "Keep if mean\ncosine ≥ 0.75", "#e8eef7", BLUE)
    box(ax, 8.05, 3.55, 2.8, 0.85, "Groups + outliers\n62.4% grouped (n=4,389)", "#e8eef7", BLUE, 8)

    ax.text(3.55, 2.05, "Homolog (same 4,389 complaints)", ha="center", fontsize=10, fontweight="bold", color=GREEN)
    box(ax, 2.3, 0.75, 1.55, 0.85, "Sentence\nembeddings", "#e8f3ea", GREEN)
    box(ax, 4.15, 0.75, 1.55, 0.85, "HDBSCAN\nno fixed radius", "#e8f3ea", GREEN)
    box(ax, 6.0, 0.75, 1.7, 0.85, "DeepSeek R1\non leftovers", "#fff4e8", ORANGE)
    box(ax, 8.05, 0.75, 2.8, 0.85, "Groups + recovered leftovers\n84.9% grouped", "#e8f3ea", GREEN, 8)

    arrow(ax, 1.95, 2.62, 2.3, 3.95)
    arrow(ax, 1.95, 2.62, 2.3, 1.17)
    arrow(ax, 3.85, 3.97, 4.15, 3.97)
    arrow(ax, 5.7, 3.97, 6.0, 3.97)
    arrow(ax, 7.7, 3.97, 8.05, 3.97)
    arrow(ax, 3.85, 1.17, 4.15, 1.17)
    arrow(ax, 5.7, 1.17, 6.0, 1.17)
    arrow(ax, 7.7, 1.17, 8.05, 1.17)

    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), facecolor="white")
    plt.close()


def fig_grouping():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    labels = [
        "TF-IDF +\nDBSCAN\n(production)",
        "Embeddings +\nHDBSCAN\n(inferred)",
        "HDBSCAN +\nleftover LLM\n(reported)",
    ]
    vals = [62.4, 82.5, 84.9]
    colors = [BLUE, GREEN, ORANGE]
    bars = ax.bar(labels, vals, color=colors, width=0.58, edgecolor="white")
    ax.set_ylabel("Share of complaints grouped (%)")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    notes = ["", "3,621 of 4,389", "includes 104 recovered"]
    for b, v, n in zip(bars, vals, notes):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        if n:
            ax.text(b.get_x() + b.get_width() / 2, v + 7.4, n, ha="center", fontsize=7.2, color=GRAY)
    ax.set_title("Same 4,389 complaints. Grouping rate, not precision.", fontsize=8.5, color=GRAY, pad=8)
    fig.savefig(os.path.join(OUT, "fig2_grouping_rate.png"), facecolor="white")
    plt.close()


def fig_funnel():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    stages = [
        ("Ungrouped leftovers", 768, ORANGE),
        ("Candidates sent to LLM", 27, GOLD),
        ("Complaints recovered", 104, GREEN),
        ("Themes found", 21, BLUE),
    ]
    y = list(range(len(stages)))[::-1]
    widths = [s[1] for s in stages]
    colors = [s[2] for s in stages]
    ax.barh(y, widths, color=colors, height=0.62, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stages])
    ax.set_xlabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for yi, w in zip(y, widths):
        ax.text(w + 12, yi, str(w), va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 920)
    fig.savefig(os.path.join(OUT, "fig3_outlier_funnel.png"), facecolor="white")
    plt.close()


def fig_operator():
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels = ["All grouped\ncomplaints", "Cases left for\nthe officer (est.)"]
    vals = [3725, 484]
    colors = [BLUE, GREEN]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white")
    ax.set_ylabel("Complaints to inspect")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 4300)
    ax.text(
        0.5,
        4000,
        "Green band ≈ 87% of grouped items (homolog rule)",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    fig.savefig(os.path.join(OUT, "fig4_operator_load.png"), facecolor="white")
    plt.close()


def fig_day():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=False)
    # 08:00
    ax = axes[0]
    ax.bar(["Cluster 31", "Outlier"], [48, 1], color=[BLUE, GRAY], width=0.55, edgecolor="white")
    ax.set_title("08:00  ·  49 complaints")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 85)
    ax.text(0, 51, "48  ·  85.68%", ha="center", fontsize=8)
    ax.text(1, 4, "1", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.bar(
        ["Cluster 32", "Cluster 33", "Outlier"],
        [73, 3, 2],
        color=[GREEN, GOLD, GRAY],
        width=0.6,
        edgecolor="white",
    )
    ax.set_title("12:00  ·  78 complaints")
    ax.set_ylim(0, 85)
    ax.text(0, 76, "73  ·  100%", ha="center", fontsize=8)
    ax.text(1, 6.5, "3  ·  85.76%", ha="center", fontsize=8)
    ax.text(2, 5, "2", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Production day, 28 August 2026 (TF-IDF + DBSCAN)", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_production_day.png"), facecolor="white")
    plt.close()


if __name__ == "__main__":
    fig_pipeline()
    fig_grouping()
    fig_funnel()
    fig_operator()
    fig_day()
    print("wrote figures in", OUT)
