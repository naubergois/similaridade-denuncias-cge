#!/usr/bin/env python3
"""Grok scene + typeset academic plates.

Grok draws a silent Nature-style illustration. This file writes every
rate, count, and caveat in Times from the team note. The paper includes
the composed fig_*.png files, not the raw scenes.

    python3 paper/figures/gerar_diagramas_grok.py
    python3 paper/figures/gerar_diagramas_grok.py --compose-only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gerar_figuras_grok import gerar_grok, gravar_png, load_env  # noqa: E402

# Academic plate — white paper, not magazine cream.
WHITE = (255, 255, 255)
PAPER = (250, 251, 252)
NAVY = (27, 54, 93)
TEAL = (31, 111, 99)
ORANGE = (196, 107, 26)
RED = (163, 59, 50)
INK = (26, 26, 26)
MUTED = (88, 92, 98)
LINE = (210, 214, 220)
PALE_B = (236, 242, 250)
PALE_T = (232, 242, 239)
PALE_O = (248, 239, 228)
PALE_R = (246, 234, 234)

ESTILO = (
    "Scientific journal illustration for an ACM digital-government article, "
    "Nature graphical-abstract register. Flat vector, white background, "
    "thin navy linework, muted teal and ochre, soft even daylight, generous "
    "empty margins reserved for later typesetting. Simple geometric paper "
    "sheets, folders, nodes, and arrows. Looks like a printed figure, not a "
    "magazine spread and not a photo. NO text, NO letters, NO numerals, "
    "NO percent signs, NO logos, NO watermarks, NO faces, NO photoreal wood, "
    "NO vintage stationery, NO cream parchment, NO neon, NO robots, "
    "NO glowing brains, NO sci-fi interface. Aspect ratio 16:9."
)

DIAGS = (
    {
        "id": "scene_abstract",
        "arquivo": "grok_scene_abstract.png",
        "cena": (
            "Wide white field. Left: a loose scatter of many plain white "
            "complaint sheets of slightly different sizes. Thin navy threads "
            "gather them toward the centre into three neat navy folders of "
            "increasing height: short, taller, only a little taller than the "
            "middle. Right: a simple wooden stamp waiting unused. Empty white "
            "band across the top third and the bottom third. No writing."
        ),
    },
    {
        "id": "scene_jobs",
        "arquivo": "grok_scene_jobs.png",
        "cena": (
            "Three equal white panels divided by two thin navy rules. "
            "Left: one white sheet and a thin arrow into a single inbox tray. "
            "Centre: a mixed tall pile of many unmarked sheets. "
            "Right: one open navy folder holding two clipped sheets of "
            "different length. Empty white band across the top fifth and "
            "the bottom fifth. No writing."
        ),
    },
    {
        "id": "scene_pipeline",
        "arquivo": "grok_scene_pipeline.png",
        "cena": (
            "A quiet left-to-right stream on white: loose sheets become a "
            "small constellation of dots, then a few closed navy folders. "
            "One leftover sheet with a thin ochre tab sits aside. Wide empty "
            "white above and below the stream. No boxes with labels. No writing."
        ),
    },
    {
        "id": "scene_rates",
        "arquivo": "grok_scene_rates.png",
        "cena": (
            "Exactly three equal columns on white, thin navy rules between "
            "them. Each column is a stack of closed navy folders on a low "
            "plinth. Left stack shortest. Middle stack clearly taller. "
            "Right stack only a little taller than the middle, with one "
            "small ochre tab on the top folder. Empty white across the top "
            "half. No writing, no numerals."
        ),
    },
    {
        "id": "scene_funnel",
        "arquivo": "grok_scene_funnel.png",
        "cena": (
            "A wide geometric funnel on white. Many small white sheets enter "
            "at the top. Most fall aside onto a quiet side tray. A short "
            "row of a few sheets continues down. A small navy stack leaves "
            "the narrow outlet. Empty white on both sides for later labels. "
            "No writing, no numerals."
        ),
    },
    {
        "id": "scene_bands",
        "arquivo": "grok_scene_bands.png",
        "cena": (
            "Three document trays in a row on white. Left tray teal, a closed "
            "folder ready. Centre tray ochre, a folder half open. Right tray "
            "brick-red tab, a folder still being read. Empty white above the "
            "trays. No writing, no numerals."
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


def scene_fit(path: Path, size: tuple[int, int], fade: float = 0.88) -> Image.Image:
    with Image.open(path) as raw:
        fitted = ImageOps.fit(raw.convert("RGB"), size, Image.Resampling.LANCZOS)
    fitted = ImageEnhance.Color(fitted).enhance(0.72)
    fitted = ImageEnhance.Contrast(fitted).enhance(0.96)
    white = Image.new("RGB", size, PAPER)
    return Image.blend(white, fitted, fade)


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline, width=2, radius=12) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_block(draw, x, y, text, fnt, fill, max_w) -> int:
    for line in wrap(draw, text, fnt, max_w):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(fnt.size * 1.35)
    return y


def compose_abstract(src: Path, dest: Path) -> None:
    W, H = 2400, 1200
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)
    title = font(40, bold=True)
    body = font(22)
    small = font(20)
    num = font(56, bold=True)
    label = font(22, bold=True)
    note = font(18)

    draw.text((72, 40), "Grouping duplicate citizen complaints in a state ombudsman (CGE-CE)", font=title, fill=NAVY)
    text_block(
        draw,
        72,
        100,
        "Same event, different wording. One historical batch, n = 4,389 real complaints. "
        "Grouping rate ρ = |G|/n is not precision.",
        body,
        MUTED,
        W - 144,
    )

    scene = scene_fit(src, (2256, 520), fade=0.90)
    canvas.paste(scene, (72, 170))
    draw.rectangle((72, 170, 2328, 690), outline=LINE, width=1)

    cards = (
        ("62.4%", "TF-IDF + DBSCAN", "production · lexical", "≈ 2,739 / 4,389", NAVY, PALE_B),
        ("82.5%", "embeddings + HDBSCAN", "inferred · no leftover model", "3,621 / 4,389", TEAL, PALE_T),
        ("84.9%", "+ leftover LLM", "ablation · filter unknown", "3,725 / 4,389", ORANGE, PALE_O),
    )
    gap = 24
    cw = (2256 - 2 * gap) // 3
    y = 720
    for i, (rate, name, role, count, color, pale) in enumerate(cards):
        x = 72 + i * (cw + gap)
        rounded(draw, (x, y, x + cw, y + 210), pale, color, width=2, radius=10)
        draw.text((x + 28, y + 18), rate, font=num, fill=color)
        draw.text((x + 28, y + 92), name, font=label, fill=INK)
        draw.text((x + 28, y + 128), role, font=note, fill=MUTED)
        draw.text((x + 28, y + 160), count, font=small, fill=INK)

    y = 960
    rounded(draw, (72, y, 2328, y + 200), PAPER, LINE, width=1, radius=8)
    y = text_block(
        draw,
        96,
        y + 20,
        "Accepted cluster: |C| ≥ 2 and mean pairwise cosine s̄(C) ≥ 0.75. "
        "The middle rate is inferred (3,725 − 104). The right-hand 84.9% includes an undocumented leftover filter.",
        body,
        INK,
        2200,
    )
    text_block(
        draw,
        96,
        y + 8,
        "No officer gold. Leftover path 768 → 27 → 104 in 21 groups is not an auditable operator. "
        "An officer still closes the case.",
        small,
        MUTED,
        2200,
    )
    canvas.save(dest, "PNG", optimize=True)


def compose_jobs(src: Path, dest: Path) -> None:
    W, H = 2400, 1320
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)
    title = font(38, bold=True)
    cap = font(28, bold=True)
    body = font(22)
    small = font(19)

    draw.text((72, 36), "Three jobs on citizen text", font=title, fill=NAVY)
    text_block(
        draw,
        72,
        92,
        "Only the right-hand job is Similaridade. A higher grouping rate can still mix two events.",
        body,
        MUTED,
        W - 144,
    )

    scene = scene_fit(src, (2256, 620), fade=0.90)
    canvas.paste(scene, (72, 150))
    draw.rectangle((72, 150, 2328, 770), outline=LINE, width=1)

    cols = (
        ("Route", "Assign a unit or service", "Das; Chen; Silva (Fala.BR); Rakhimzhanov et al.", "Not this paper", NAVY, PALE_B),
        ("Theme", "Discover latent topics", "BERTopic; Tang et al.; Esperança et al.", "Not this paper", TEAL, PALE_T),
        ("Case", "Same real-world event", "Fellegi–Sunter; Broder; Rabbi et al. (consumer)", "Similaridade", ORANGE, PALE_O),
    )
    gap = 24
    cw = (2256 - 2 * gap) // 3
    y = 800
    for i, (name, job, cite, mark, color, pale) in enumerate(cols):
        x = 72 + i * (cw + gap)
        rounded(draw, (x, y, x + cw, y + 430), pale, color, width=2, radius=10)
        draw.text((x + 28, y + 22), name, font=cap, fill=color)
        draw.text((x + 28, y + 72), job, font=body, fill=INK)
        cy = text_block(draw, x + 28, y + 118, cite, small, MUTED, cw - 56)
        draw.text((x + 28, max(cy + 16, y + 360)), mark, font=font(22, bold=True), fill=color)
    canvas.save(dest, "PNG", optimize=True)


def _arrow(draw, x1, y, x2, color=MUTED) -> None:
    draw.line((x1, y, x2 - 8, y), fill=color, width=2)
    draw.polygon([(x2, y), (x2 - 12, y - 6), (x2 - 12, y + 6)], fill=color)


def compose_pipeline(src: Path, dest: Path) -> None:
    W, H = 2400, 1580
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)
    title = font(36, bold=True)
    body = font(21)
    small = font(18)
    box_f = font(18)
    rate_f = font(28, bold=True)
    row_f = font(20, bold=True)

    draw.text((72, 32), "Three treatments on the same incoming batch", font=title, fill=NAVY)
    text_block(
        draw,
        72,
        86,
        "n = 4,389. Production is live. Homolog HDBSCAN is inferred by removing 104 recoveries. "
        "The leftover language-model path is an ablation: the map L → K is unknown.",
        body,
        MUTED,
        W - 144,
    )

    scene = scene_fit(src, (2256, 320), fade=0.82)
    canvas.paste(scene, (72, 160))
    draw.rectangle((72, 160, 2328, 480), outline=LINE, width=1)

    rows = (
        (
            "Production (live, August 2026)",
            NAVY,
            PALE_B,
            (("TF-IDF", "1–3 grams"), ("DBSCAN", "ε, m unknown"), ("Accept if", "s̄(C) ≥ 0.75"), ("ρ = 62.4%", "≈ 2,739 grouped")),
        ),
        (
            "Homolog (same batch, no leftover model)",
            TEAL,
            PALE_T,
            (("Sentence", "embeddings"), ("HDBSCAN", "cut unknown"), ("Implied G", "without R"), ("ρ = 82.5%", "3,621 grouped")),
        ),
        (
            "Ablation (leftover LLM; not a deployment claim)",
            ORANGE,
            PALE_O,
            (("L = 768", "leftovers"), ("K = 27", "rule unknown"), ("R = 104", "in 21 groups"), ("ρ = 84.9%", "3,725 grouped")),
        ),
    )
    y = 510
    for heading, color, pale, boxes in rows:
        rounded(draw, (72, y, 2328, y + 320), pale, color, width=2, radius=10)
        draw.text((96, y + 18), heading, font=row_f, fill=color)
        bw, bh, gap = 430, 170, 28
        bx0, by = 110, y + 70
        for i, (a, b) in enumerate(boxes):
            x = bx0 + i * (bw + gap)
            rounded(draw, (x, by, x + bw, by + bh), WHITE, color, width=2, radius=8)
            draw.text((x + 24, by + 36), a, font=rate_f, fill=color)
            draw.text((x + 24, by + 96), b, font=small, fill=INK)
            if i < 3:
                _arrow(draw, x + bw + 2, by + bh // 2, x + bw + gap - 2, color)
        y += 340
    canvas.save(dest, "PNG", optimize=True)


def compose_rates(src: Path, dest: Path) -> None:
    W, H = 1600, 1360
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)
    title = font(32, bold=True)
    body = font(20)
    num = font(48, bold=True)
    small = font(18)

    draw.text((56, 28), "Grouping rate ρ on one batch", font=title, fill=NAVY)
    text_block(
        draw,
        56,
        78,
        "n = 4,389. Stack height is schematic. Printed rates are the team counts. ρ is not precision.",
        body,
        MUTED,
        W - 112,
    )

    scene = scene_fit(src, (1488, 620), fade=0.90)
    canvas.paste(scene, (56, 150))
    draw.rectangle((56, 150, 1544, 770), outline=LINE, width=1)

    cols = (
        ("62.4%", "TF-IDF + DBSCAN", "production, reported", "≈ 2,739 grouped", NAVY, PALE_B),
        ("82.5%", "embeddings + HDBSCAN", "inferred (3,725 − 104)", "3,621 grouped", TEAL, PALE_T),
        ("84.9%", "HDBSCAN + leftover LLM", "reported ablation", "includes 104 recovered", ORANGE, PALE_O),
    )
    gap = 20
    cw = (1488 - 2 * gap) // 3
    y = 800
    for i, (rate, name, note, count, color, pale) in enumerate(cols):
        x = 56 + i * (cw + gap)
        rounded(draw, (x, y, x + cw, y + 280), pale, color, width=2, radius=10)
        draw.text((x + 20, y + 20), rate, font=num, fill=color)
        cy = text_block(draw, x + 20, y + 92, name, font(20, bold=True), INK, cw - 40)
        cy = text_block(draw, x + 20, cy + 8, note, small, MUTED, cw - 40)
        text_block(draw, x + 20, cy + 8, count, small, INK, cw - 40)

    y = 1110
    rounded(draw, (56, y, 1544, y + 220), PAPER, LINE, width=1, radius=8)
    text_block(
        draw,
        76,
        y + 24,
        "The cleaner comparison is 62.4% versus 82.5% (20.1 points on one historical batch). "
        "A more permissive merge would also raise the rate. The leftover model adds 2.4 points "
        "only if the 104 recoveries are accepted at face value. None of these shares is a precision: "
        "the Ouvidoria has not labeled “same case.”",
        body,
        INK,
        1440,
    )
    canvas.save(dest, "PNG", optimize=True)


def compose_funnel(src: Path, dest: Path) -> None:
    W, H = 1600, 1480
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)
    title = font(30, bold=True)
    body = font(20)
    num = font(40, bold=True)
    small = font(18)

    draw.text((56, 28), "Homolog leftover path", font=title, fill=NAVY)
    text_block(
        draw,
        56,
        76,
        "Map L → K is unknown, so recovered set R is not auditable.",
        body,
        MUTED,
        W - 112,
    )

    scene = scene_fit(src, (1488, 520), fade=0.88)
    canvas.paste(scene, (56, 130))
    draw.rectangle((56, 130, 1544, 650), outline=LINE, width=1)

    steps = (
        ("768", "Leftovers after HDBSCAN (L)", "ungrouped on the homolog pass", TEAL, PALE_T),
        ("27", "Candidates sent to LLM (K)", "selection rule unknown", ORANGE, PALE_O),
        ("104", "Recovered complaints (R)", "counted inside 3,725", NAVY, PALE_B),
        ("21", "Leftover groups reported", "called themes in the note", RED, PALE_R),
    )
    y = 678
    for rate, name, note, color, pale in steps:
        rounded(draw, (56, y, 1544, y + 150), pale, color, width=2, radius=8)
        draw.text((80, y + 36), rate, font=num, fill=color)
        draw.text((280, y + 32), name, font=font(22, bold=True), fill=INK)
        draw.text((280, y + 78), note, font=small, fill=MUTED)
        y += 166
    canvas.save(dest, "PNG", optimize=True)


def compose_bands(src: Path, dest: Path) -> None:
    W, H = 1600, 1040
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)
    title = font(30, bold=True)
    body = font(20)
    cap = font(26, bold=True)
    small = font(18)

    draw.text((56, 28), "Provisional review queues", font=title, fill=NAVY)
    text_block(
        draw,
        56,
        76,
        "Fellegi–Sunter pair decisions applied to accepted clusters. Cuts that send a cluster to a queue are unknown. Not a measured safety threshold.",
        body,
        MUTED,
        W - 112,
    )

    scene = scene_fit(src, (1488, 420), fade=0.90)
    canvas.paste(scene, (56, 170))
    draw.rectangle((56, 170, 1544, 590), outline=LINE, width=1)

    cols = (
        ("Green", "Suggested link", "≈ bulk linking", "≈ 87% of |G|", TEAL, PALE_T),
        ("Yellow", "Possible link", "officer confirms", "share unknown", ORANGE, PALE_O),
        ("Red", "Read with care", "treat as non-link", "share unknown", RED, PALE_R),
    )
    gap = 20
    cw = (1488 - 2 * gap) // 3
    y = 618
    for i, (name, job, how, share, color, pale) in enumerate(cols):
        x = 56 + i * (cw + gap)
        rounded(draw, (x, y, x + cw, y + 280), pale, color, width=2, radius=10)
        draw.text((x + 22, y + 20), name, font=cap, fill=color)
        draw.text((x + 22, y + 72), job, font=font(20, bold=True), fill=INK)
        draw.text((x + 22, y + 118), how, font=small, fill=MUTED)
        draw.text((x + 22, y + 200), share, font=body, fill=INK)

    text_block(
        draw,
        56,
        930,
        "If green were trusted, 0.13 × 3,725 ≈ 484 remaining complaints. That 484 is an arithmetic remainder, not officer minutes.",
        small,
        MUTED,
        W - 112,
    )
    canvas.save(dest, "PNG", optimize=True)


def generate_scenes() -> None:
    load_env()
    for spec in DIAGS:
        dest = HERE / spec["arquivo"]
        print(f"GROK {spec['id']}", flush=True)
        prompt = " ".join(f"{ESTILO} Scene: {spec['cena']}".split())
        raw = gerar_grok(prompt)
        gravar_png(raw, dest)
        print(f"OK {dest.name} {dest.stat().st_size} bytes", flush=True)
        time.sleep(1)


def compose_all() -> None:
    missing = [s["arquivo"] for s in DIAGS if not (HERE / s["arquivo"]).exists()]
    if missing:
        raise SystemExit(f"Missing Grok scenes: {', '.join(missing)}")
    compose_abstract(HERE / "grok_scene_abstract.png", HERE / "fig_abstract.png")
    compose_jobs(HERE / "grok_scene_jobs.png", HERE / "fig_jobs.png")
    compose_pipeline(HERE / "grok_scene_pipeline.png", HERE / "fig1_pipeline.png")
    compose_rates(HERE / "grok_scene_rates.png", HERE / "fig2_grouping_rate.png")
    compose_funnel(HERE / "grok_scene_funnel.png", HERE / "fig3_outlier_funnel.png")
    compose_bands(HERE / "grok_scene_bands.png", HERE / "fig_bands.png")
    print("composed fig_abstract, fig_jobs, fig1_pipeline, fig2, fig3, fig_bands")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    if not args.compose_only:
        generate_scenes()
    compose_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
