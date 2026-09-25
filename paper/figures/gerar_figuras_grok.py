#!/usr/bin/env python3
"""Conceptual paper figures via Grok Imagine (xAI).

Numeric charts stay in render_figures.py. This file draws the editorial
scenes: abstract, two pipelines, case vs theme, leftover funnel, bands.

    python3 paper/figures/gerar_figuras_grok.py
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
XAI_URL = "https://api.x.ai/v1/images/generations"
MODELO = os.environ.get("GROK_IMAGINE_MODEL", "grok-imagine-image-quality")

ESTILO = (
    "16:9 scholarly editorial illustration for an ACM digital-government paper. "
    "Flat vector style with soft paper grain, generous empty space, calm daylight. "
    "Palette: cream #F3EDE1 background, deep navy #16233F, teal #1F6F63, "
    "brass gold #C9A227, brick red #A8352B only for leftovers or failure. "
    "The composition must READ as a diagram: objects explain a relationship. "
    "NO text, NO letters, NO numbers, NO logos, NO brand names, NO readable "
    "screens, NO faces of real people, NO neon, NO cyberpunk, NO hologram, "
    "NO glowing brain, NO robot, NO sci-fi interface. Aspect ratio 16:9."
)

ARRANJO = {
    "fluxo": (
        "Composition: a single left-to-right horizontal flow of stations "
        "connected by thin navy arrows, evenly spaced."
    ),
    "dois_fluxos": (
        "Composition: two stacked left-to-right flows, a thin navy rule "
        "between them. Top flow is simpler and shorter. Bottom flow has "
        "one extra station."
    ),
    "comparacao": (
        "Composition: two halves side by side, a thin vertical navy rule. "
        "Left is the intended case. Right is the nearby but different case."
    ),
    "funil": (
        "Composition: a wide cream intake at the top narrowing to one "
        "narrow outlet at the bottom."
    ),
    "mapa": (
        "Composition: a top-down flat-lay of an oak desk, objects arranged "
        "as a small map with thin gold connecting lines."
    ),
    "cena": (
        "Composition: one clear institutional scene, subject centred, "
        "wide margin around it."
    ),
}

FIGS = (
    {
        "id": "grok_abstract",
        "tipo": "cena",
        "arquivo": "grok_abstract.png",
        "cena": (
            "A wide oak ombudsman desk. On the left, a loose pile of many "
            "cream complaint envelopes of slightly different sizes. Thin gold "
            "lines gather them toward the centre, where three navy folders "
            "sit closed and neat. On the far right, one unused wooden stamp "
            "waits on a blotter. Empty cream space above. No writing."
        ),
    },
    {
        "id": "grok_pipeline",
        "tipo": "dois_fluxos",
        "arquivo": "grok_pipeline.png",
        "cena": (
            "Top row, four stations: a single cream envelope, a stack of "
            "index cards, a small navy ring of dots, a closed folder. "
            "Bottom row, five stations: the same envelope, a brass compass, "
            "a taller teal ring of dots, a small leftover envelope with a "
            "brick-red tab, then a closed folder. Same cream field. No writing."
        ),
    },
    {
        "id": "grok_case_vs_theme",
        "tipo": "comparacao",
        "arquivo": "grok_case_vs_theme.png",
        "cena": (
            "Left half: two cream letters of different length that both "
            "sit on the same navy folder, a thin gold clip joining them — "
            "the same water outage told twice. Right half: a wide mixed "
            "pile of letters about many subjects, loosely stacked, no clip, "
            "a brick-red tab on the pile. Oak desk, cream field. No writing."
        ),
    },
    {
        "id": "grok_funnel",
        "tipo": "funil",
        "arquivo": "grok_funnel.png",
        "cena": (
            "Many small cream envelopes enter a wide cream funnel at the "
            "top. Midway, most envelopes fall aside onto a quiet side tray. "
            "A short row of a few envelopes continues down. At the bottom "
            "outlet, a small navy stack of recovered folders leaves. "
            "Brick-red only on the side tray. No writing."
        ),
    },
    {
        "id": "grok_bands",
        "tipo": "mapa",
        "arquivo": "grok_bands.png",
        "cena": (
            "Three wooden document trays on an oak desk in a gentle arc. "
            "Left tray teal-green with a closed navy folder ready to stamp. "
            "Centre tray brass-gold with a folder half open, stamp waiting. "
            "Right tray with a brick-red tab and a folder still being read. "
            "Empty gold square brackets near the green tray. No writing."
        ),
    },
)


def load_env() -> None:
    for path in (Path.home() / ".zshrc", Path.home() / ".zprofile", Path.home() / ".env"):
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            key = key.strip()
            if key in os.environ:
                continue
            os.environ[key] = value.strip().strip("\"'")


def prompt_de(spec: dict) -> str:
    arranjo = ARRANJO[spec["tipo"]]
    return " ".join(f"{ESTILO} {arranjo} Scene: {spec['cena']}".split())


def gerar_grok(prompt: str) -> bytes:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("XAI_API_KEY ausente")
    corpo = {
        "model": MODELO,
        "prompt": prompt,
        "n": 1,
        "aspect_ratio": "16:9",
        "resolution": "2k",
        "response_format": "b64_json",
    }
    req = urllib.request.Request(
        XAI_URL,
        data=json.dumps(corpo).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:400]
        if "resolution" in detail or "aspect_ratio" in detail:
            corpo.pop("resolution", None)
            corpo.pop("aspect_ratio", None)
            req = urllib.request.Request(
                XAI_URL,
                data=json.dumps(corpo).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        else:
            raise SystemExit(f"Grok HTTP {error.code}: {detail}") from None
    items = data.get("data") or []
    if not items:
        raise SystemExit("Grok não devolveu imagem")
    item = items[0]
    b64 = item.get("b64_json")
    if b64:
        return base64.b64decode(b64)
    url = item.get("url")
    if not url:
        raise SystemExit("Grok concluiu sem URL nem base64")
    req_img = urllib.request.Request(
        url,
        headers={"User-Agent": "similaridade-denuncias-cge/0.3", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req_img, timeout=120) as img:
        return img.read()


def gravar_png(raw: bytes, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(raw)) as source:
        image = ImageOps.fit(source.convert("RGB"), (1920, 1080), Image.Resampling.LANCZOS)
    image.save(destino, "PNG", optimize=True)


def main() -> int:
    load_env()
    for spec in FIGS:
        destino = HERE / spec["arquivo"]
        print(f"GROK {spec['id']} -> {destino.name}", flush=True)
        raw = gerar_grok(prompt_de(spec))
        gravar_png(raw, destino)
        print(f"OK {destino.stat().st_size} bytes", flush=True)
        time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
