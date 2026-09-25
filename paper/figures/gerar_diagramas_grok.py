#!/usr/bin/env python3
"""Grok diagrams + typeset academic infographics.

Grok draws the scene with no letters. This file sets the title, rates,
and captions in Times so the paper summary stays readable.

    python3 paper/figures/gerar_diagramas_grok.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gerar_figuras_grok import ESTILO, ARRANJO, gerar_grok, gravar_png, load_env  # noqa: E402

NAVY = (22, 35, 63)
TEAL = (31, 111, 99)
BRASS = (201, 162, 39)
CREAM = (243, 237, 225)
INK = (26, 26, 26)
MUTED = (90, 90, 90)
RED = (168, 53, 43)
WHITE = (255, 255, 255)

DIAGS = (
    {
        "id": "grok_diag_hero",
        "arquivo": "grok_diag_hero.png",
        "tipo": "cena",
        "cena": (
            "A wide cream field. In the lower two-thirds, an oak ombudsman "
            "desk: loose cream envelopes on the left, gold threads gathering "
            "them into a neat stack of navy folders in the centre, a wooden "
            "stamp waiting on the right. The upper third is empty cream, "
            "reserved for a later title. No writing, no numerals."
        ),
    },
    {
        "id": "grok_diag_jobs",
        "arquivo": "grok_diag_jobs.png",
        "tipo": "comparacao",
        "cena": (
            "Three equal vertical panels on cream, separated by two thin navy "
            "rules. Left panel: one cream envelope and a thin navy arrow "
            "pointing to a single wooden inbox tray — routing to a unit. "
            "Centre panel: a mixed tall pile of many cream sheets — a theme "
            "heap. Right panel: one open navy folder holding two clipped "
            "cream sheets of different length — the same event. No writing."
        ),
    },
    {
        "id": "grok_diag_rates",
        "arquivo": "grok_diag_rates.png",
        "tipo": "comparacao",
        "cena": (
            "Three equal columns on cream, thin navy rules between them. "
            "Each column is a stack of closed navy folders on an oak plinth. "
            "Left stack shortest. Middle stack clearly taller. Right stack "
            "only a little taller than the middle, with a small brick-red "
            "tab on the top folder. Generous empty cream above each stack. "
            "No writing, no numerals, no percent signs."
        ),
    },
    {
        "id": "grok_diag_flow",
        "arquivo": "grok_diag_flow.png",
        "tipo": "fluxo",
        "cena": (
            "Five geometric stations in one left-to-right row on cream, "
            "connected by thin navy arrows: a cream envelope; a stack of "
            "index cards; a teal ring of dots; a leftover envelope with a "
            "brick-red tab; a closed navy folder with a wooden stamp beside "
            "it. Wide empty cream above the row for later labels. No writing."
        ),
    },
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        candidates = (
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/Library/Fonts/Times New Roman Bold.ttf",
            "/System/Library/Fonts/Times.ttc",
        )
    else:
        candidates = (
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/Library/Fonts/Times New Roman.ttf",
            "/System/Library/Fonts/Times.ttc",
        )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def compose_summary(hero: Path, dest: Path) -> None:
    W, H = 2400, 1680
    canvas = Image.new("RGB", (W, H), CREAM)
    draw = ImageDraw.Draw(canvas)
    title = font(44, bold=True)
    sub = font(22)
    body = font(26)
    small = font(20)
    num = font(52, bold=True)
    label = font(22)

    draw.text((80, 36), "GRAPHICAL ABSTRACT", font=small, fill=TEAL)
    draw.text((80, 78), "Grouping Duplicate Citizen Complaints in a State Ombudsman", font=title, fill=NAVY)
    draw.text(
        (80, 140),
        "CGE-CE  ·  Ouvidoria of Ceará, Brazil  ·  practice account, not submitted",
        font=sub,
        fill=MUTED,
    )
    problem = (
        "Many wordings can be one event. If each arrival opens its own file, "
        "the officer reads the pile twice and the citizen waits."
    )
    y = 184
    for line in wrap(draw, problem, body, W - 160):
        draw.text((80, y), line, font=body, fill=INK)
        y += 34

    with Image.open(hero) as raw:
        scene = ImageOps.fit(raw.convert("RGB"), (2240, 780), Image.Resampling.LANCZOS)
    canvas.paste(scene, (80, 270))
    draw.rectangle((80, 270, 2320, 1050), outline=NAVY, width=1)

    draw.text((80, 1074), "Same historical batch of 4,389 real complaints. Grouping rate, not precision.", font=small, fill=MUTED)

    cards = (
        ("62.4%", "TF-IDF + DBSCAN", "production, lexical"),
        ("82.5%", "embeddings + HDBSCAN", "inferred, before any LLM"),
        ("84.9%", "+ leftover LLM", "undocumented filter"),
    )
    y = 1118
    gap = 24
    cw = (2240 - 2 * gap) // 3
    for i, (rate, name, note) in enumerate(cards):
        x = 80 + i * (cw + gap)
        color = (NAVY, TEAL, RED)[i]
        draw.rounded_rectangle((x, y, x + cw, y + 150), radius=10, fill=WHITE, outline=color, width=2)
        draw.text((x + 24, y + 18), rate, font=num, fill=color)
        draw.text((x + 24, y + 84), name, font=label, fill=INK)
        draw.text((x + 24, y + 114), note, font=small, fill=MUTED)

    y = 1290
    limit = (
        "None of these shares is a precision. The Ouvidoria has not labeled “same case.” "
        "The leftover model saw 27 of 768 ungrouped texts; how those 27 were chosen is unknown. "
        "84.9% is an ablation, not a deployment claim."
    )
    for line in wrap(draw, limit, small, 2240):
        draw.text((80, y), line, font=small, fill=MUTED)
        y += 28
    y += 10
    claim = (
        "Lexical clustering already catches copy-paste waves. A semantic stack grouped more "
        "of the same historical set. Daily production is still being earned. Similaridade groups; "
        "an officer still closes the case."
    )
    for line in wrap(draw, claim, body, 2240):
        draw.text((80, y), line, font=body, fill=NAVY)
        y += 34

    draw.line((80, H - 64, 2320, H - 64), fill=BRASS, width=1)
    draw.text(
        (80, H - 46),
        "Draft v0.4  ·  Scene: Grok Imagine  ·  Type set from team counts  ·  ACM DGOV practice paper",
        font=small,
        fill=MUTED,
    )
    canvas.save(dest, "PNG", optimize=True)


def compose_jobs(src: Path, dest: Path) -> None:
    W, H = 2160, 1180
    canvas = Image.new("RGB", (W, H), CREAM)
    draw = ImageDraw.Draw(canvas)
    with Image.open(src) as raw:
        scene = ImageOps.fit(raw.convert("RGB"), (2040, 820), Image.Resampling.LANCZOS)
    canvas.paste(scene, (60, 150))
    title = font(36, bold=True)
    cap = font(24, bold=True)
    small = font(20)
    draw.text((60, 36), "Three jobs that are easy to mix up", font=title, fill=NAVY)
    draw.text(
        (60, 88),
        "Only the right-hand job is Similaridade: same event, not same theme or same department.",
        font=small,
        fill=MUTED,
    )
    thirds = (180, 860, 1540)
    labels = (
        ("Route", "send the text to a unit"),
        ("Theme", "what citizens talk about"),
        ("Case", "two tellings of one event"),
    )
    for x, (name, note) in zip(thirds, labels):
        draw.text((x, 990), name, font=cap, fill=NAVY)
        draw.text((x, 1028), note, font=small, fill=MUTED)
    draw.text((60, 1136), "Diagram: Grok Imagine. Labels set in type.", font=small, fill=MUTED)
    canvas.save(dest, "PNG", optimize=True)


def compose_rates(src: Path, dest: Path) -> None:
    W, H = 2160, 1180
    canvas = Image.new("RGB", (W, H), CREAM)
    draw = ImageDraw.Draw(canvas)
    with Image.open(src) as raw:
        scene = ImageOps.fit(raw.convert("RGB"), (2040, 780), Image.Resampling.LANCZOS)
    canvas.paste(scene, (60, 150))
    title = font(36, bold=True)
    num = font(40, bold=True)
    small = font(20)
    draw.text((60, 32), "Three treatments on one batch of 4,389 complaints", font=title, fill=NAVY)
    draw.text(
        (60, 88),
        "Stack height is schematic, not to scale. Printed rates are the team counts. Not precision.",
        font=small,
        fill=MUTED,
    )
    cols = (
        (160, "62.4%", "TF-IDF + DBSCAN", "production"),
        (860, "82.5%", "embeddings + HDBSCAN", "inferred, no LLM"),
        (1560, "84.9%", "+ leftover LLM", "filter unknown"),
    )
    colors = (NAVY, TEAL, RED)
    for (x, rate, name, note), color in zip(cols, colors):
        draw.text((x, 960), rate, font=num, fill=color)
        draw.text((x, 1014), name, font=small, fill=INK)
        draw.text((x, 1046), note, font=small, fill=MUTED)
    draw.text((60, 1136), "Diagram: Grok Imagine. Numbers set in type from the team note.", font=small, fill=MUTED)
    canvas.save(dest, "PNG", optimize=True)


def compose_flow(src: Path, dest: Path) -> None:
    W, H = 2160, 980
    canvas = Image.new("RGB", (W, H), CREAM)
    draw = ImageDraw.Draw(canvas)
    with Image.open(src) as raw:
        scene = ImageOps.fit(raw.convert("RGB"), (2160, 760), Image.Resampling.LANCZOS)
    canvas.paste(scene, (0, 130))
    title = font(34, bold=True)
    small = font(18)
    draw.text((50, 28), "Homolog path  ·  same incoming text", font=title, fill=NAVY)
    labels = (
        (70, "Complaint"),
        (480, "Embed"),
        (900, "HDBSCAN"),
        (1320, "Leftover"),
        (1740, "Officer"),
    )
    for x, name in labels:
        draw.text((x, 100), name, font=small, fill=NAVY)
    draw.text(
        (50, 920),
        "The leftover station is an ablation. Host and candidate rule are unknown. The stamp is still human.",
        font=small,
        fill=MUTED,
    )
    canvas.save(dest, "PNG", optimize=True)


def main() -> int:
    load_env()
    import time

    for spec in DIAGS:
        dest = HERE / spec["arquivo"]
        print(f"GROK {spec['id']}", flush=True)
        prompt = " ".join(f"{ESTILO} {ARRANJO[spec['tipo']]} Scene: {spec['cena']}".split())
        raw = gerar_grok(prompt)
        gravar_png(raw, dest)
        print(f"OK {dest.name}", flush=True)
        time.sleep(1)

    compose_summary(HERE / "grok_diag_hero.png", HERE / "grok_paper_summary.png")
    compose_jobs(HERE / "grok_diag_jobs.png", HERE / "grok_infographic_jobs.png")
    compose_rates(HERE / "grok_diag_rates.png", HERE / "grok_infographic_rates.png")
    compose_flow(HERE / "grok_diag_flow.png", HERE / "grok_infographic_flow.png")
    print("composed summary + 3 infographics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
