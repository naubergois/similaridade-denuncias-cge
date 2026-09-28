#!/usr/bin/env python3
"""NMI staircase to the published BBC line. Measured numbers only. 300 dpi."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUT, ".mpl"))

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)

NAVY = "#1b365d"
TEAL = "#1f6f63"
ORANGE = "#c46b1a"
GRAY = "#5a5a5a"
PUB = 0.9512

# Kept path only. Source: experiments/bbc_final.json
STEPS = [
    ("1 voto de 11", 0.9234),
    ("2 par Sonnet", 0.9358),
    ("3 Opus borda", 0.9381),
    ("4 odd-one-out", 0.9430),
    ("5 revolta 4/7", 0.9466),
    ("7 vizinho alheio", 0.9482),
    ("12 borda dos grupos", 0.9487),
    ("14 tipo de evento", 0.9496),
    ("15 só o par mais perto", 0.9520),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    xs = list(range(len(STEPS)))
    ys = [s[1] for s in STEPS]
    ax.plot(xs, ys, color=NAVY, lw=2.0, marker="o", ms=6, zorder=3)
    for i, (name, y) in enumerate(STEPS):
        color = TEAL if y >= PUB else NAVY
        ax.plot(i, y, "o", color=color, ms=7, zorder=4)
        ax.annotate(
            f"{y:.4f}",
            (i, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=color,
        )
    ax.axhline(PUB, color=ORANGE, ls="--", lw=1.2, label="GCC* publicado 0,9512")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in STEPS], rotation=28, ha="right")
    ax.set_ylabel("NMI (sklearn, média aritmética)")
    ax.set_ylim(0.918, 0.956)
    ax.set_title("BBC News: pisos justos até passar o publicado")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d8d8", lw=0.6)
    path = os.path.join(OUT, "bbc_sota_staircase.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
