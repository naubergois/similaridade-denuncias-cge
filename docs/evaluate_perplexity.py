#!/usr/bin/env python3
"""Academic review of the Similaridade draft via Perplexity Sonar (OpenRouter).

Direct PERPLEXITY_API_KEY is preferred. OPENROUTER_API_KEY is the fallback
used on this machine.

    python3 docs/evaluate_perplexity.py
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "article.tex"
OUT = ROOT / "docs" / "perplexity-review.md"

SYSTEM = """You are a strict associate editor for ACM Digital Government: Research and Practice (DGOV), practice-paper track.
Score only from evidence in the manuscript plus current public literature. Do not invent citations that you cannot name with title, year, and venue.
If a claim is operational and honest, say so. If it is a vendor slide, say so.
Write in English. Be concrete."""

RUBRIC = """Review the manuscript below.

Return Markdown with exactly these headings:

## Verdict
One line: Do not submit / Poster or short practice note / Revise for DGOV practice paper / Ready for DGOV practice paper.

## Score
A table with rows (0–5 each) and a mean /5:
- Fit to DGOV practice track
- Honesty of claims
- Method reproducibility
- Evidence quality
- State of the art coverage
- Operational usefulness
- Ethics and data handling

## What holds
3–6 bullets of claims that a reviewer should accept.

## What blocks submission
Numbered blockers. For each: why it matters to DGOV, and the smallest fix that would unblock it.

## Missing literature
Papers the authors should read or cite, with title, year, venue, and why. Prefer 2022–2026 public-sector complaint / ombudsman / near-duplicate work. Mark any item you are unsure about.

## Numbers to pin
Which figures or percentages a reviewer will ask to see in a log or a label set.

## Suggested next version
Five concrete edits for v0.4. No generic advice.

Manuscript follows.
"""


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


def tex_to_text(src: str) -> str:
    text = src
    text = re.sub(r"\\begin\{CCSXML\}.*?\\end\{CCSXML\}", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", r"\nABSTRACT\n\1\n", text, flags=re.S)
    text = re.sub(r"\\(title|subtitle|keywords)\{([^}]*)\}", r"\n\2\n", text)
    text = re.sub(r"\\(section|subsection|paragraph)\*?\{([^}]*)\}", r"\n\n## \2\n", text)
    text = re.sub(r"\\caption\{([^}]*)\}", r" [Figure: \1] ", text)
    text = re.sub(r"\\cite\{([^}]+)\}", r" [\1] ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^}]*)\}", r" \2 ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("~", " ").replace("``", '"').replace("''", '"')
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chat(messages: list[dict]) -> tuple[str, list[str], str, str]:
    pplx = os.environ.get("PERPLEXITY_API_KEY", "").strip()
    if pplx:
        url = "https://api.perplexity.ai/chat/completions"
        key = pplx
        model = "sonar-pro"
        provider = "perplexity"
        extra = {"search_mode": "academic", "web_search_options": {"search_context_size": "high"}}
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise SystemExit("Need PERPLEXITY_API_KEY or OPENROUTER_API_KEY")
        model = "perplexity/sonar-pro"
        provider = "openrouter"
        extra = {}
    body = {"model": model, "messages": messages, "temperature": 0.2, **extra}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    citations = data.get("citations") or message.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    return content, [str(c) for c in citations], provider, model


def main() -> int:
    load_env()
    manuscript = tex_to_text(TEX.read_text(encoding="utf-8"))
    user = RUBRIC + "\n\n---\n\n" + manuscript[:24000]
    content, citations, provider, model = chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    header = (
        "# Perplexity review — Similaridade v0.3\n\n"
        f"- Date: 23 September 2026\n"
        f"- Provider: `{provider}` / `{model}`\n"
        f"- Source: `paper/article.tex`\n\n"
        "Search-backed editorial read. Not a human associate editor.\n\n"
    )
    cites = ""
    if citations:
        cites = "\n\n## Citations returned by Sonar\n\n" + "\n".join(f"- {c}" for c in citations) + "\n"
    OUT.write_text(header + content.strip() + cites, encoding="utf-8")
    print(f"Wrote {OUT} ({len(content)} chars, {len(citations)} citations)")
    print(content[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
