#!/usr/bin/env python3
"""Lift BBC NMI without using gold to pick documents.

Majority of 11 clusterings, then Sonnet 4.6 picks among five existing
groups (exemplars only, no desk names) on a thin-margin border, reading
a long slice of the article. A second ring adds articles whose vote and
nearest centroid disagree, or that trip a lexical clash (game/prize,
roaming, chancellor/debt) — still no gold.

    .venv/bin/python experiments/run_bbc_nmi_lift.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bbc_beat_published import PUB, score  # noqa: E402
from run_bbc_claude_border_errors import (  # noqa: E402
    K,
    SONNET,
    build_majority,
    chat,
    fill,
)
from run_bbc_llm_border import (  # noqa: E402
    DESKS,
    cksum,
    clip,
    exemplars,
    load_cache,
    load_titles_and_ledes,
    margins,
    oracle as gold_oracle,
)
from run_bbc_llm_ensemble import gold_map  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

OUT = HERE / "bbc_nmi_lift.json"
CLASH = re.compile(
    r"\b(gamer|gamers|tournament|roaming|vodafone|t-mobile|"
    r"chancellor|g8|plasma|call centre|call center|blogger)\b",
    re.I,
)


def prompt_five(title: str, body: str, ex: dict[int, list[str]]) -> str:
    blocks = []
    for c in range(K):
        lines = "\n".join(ex[c]) if ex.get(c) else "- (empty)"
        blocks.append(f"Group {c}:\n{lines}")
    return (
        "You are grouping news articles into five existing clusters. "
        "Do not invent a new group and do not name topics. "
        f"Reply with a single digit 0-{K - 1}.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nNew article:\nTitle: {title}\nText: {clip(body, 900)}\n\n"
        "Which group does it belong to?"
    )


def parse_five(raw: str) -> int | None:
    m = re.search(r"[0-4]", (raw or "").strip())
    return int(m.group()) if m else None


def apply_five(pred: np.ndarray, indices, keys, cache) -> np.ndarray:
    out = pred.copy()
    for i, key in zip(indices, keys):
        choice = parse_five((cache.get(key) or {}).get("text", ""))
        if choice is not None:
            out[int(i)] = choice
    return out


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    print("fitting majority …", flush=True)
    Z, majority = build_majority(texts)
    rows = [score("majority of 11", majority, gold)]
    gmap = gold_map(majority, gold)
    aligned = np.array([gmap[int(p)] for p in majority])
    gap, first, second = margins(Z, majority)
    cents = normalize(
        np.stack([Z[majority == c].mean(axis=0) if (majority == c).any() else np.zeros(Z.shape[1]) for c in range(K)])
    )
    sims = Z @ cents.T
    nearest = sims.argmax(axis=1)
    order = np.argsort(gap)
    border80 = [int(i) for i in order[:80]]
    border120 = [int(i) for i in order[:120]]
    clash = [i for i, t in enumerate(texts) if CLASH.search(t) and i not in set(border120)]
    clash = clash[:40]
    geo = [i for i in range(len(texts)) if int(nearest[i]) != int(majority[i]) and i not in set(border120)]
    extra = sorted(set(clash) | set(geo[:40]))
    print(f"border80={len(border80)} border120={len(border120)} extra={len(extra)} clash={len(clash)} geo={len(geo)}", flush=True)

    ex = exemplars(Z, majority, titles, ledes)
    cache = load_cache()
    jobs: list[tuple[str, str, str]] = []

    def pack(indices: list[int]) -> list[str]:
        keys = []
        for i in indices:
            pp = prompt_five(titles[i], texts[i], ex)
            pk = cksum("nmi_five", SONNET, titles[i], pp)
            keys.append(pk)
            jobs.append((pk, pp, SONNET))
        return keys

    k120 = pack(border120)
    k80 = [k120[border120.index(i)] for i in border80]
    kextra = pack(extra)
    cache = fill(jobs, cache)

    pred80 = apply_five(majority, border80, k80, cache)
    pred120 = apply_five(majority, border120, k120, cache)
    pred_all = apply_five(pred120, extra, kextra, cache)
    rows.append(score("sonnet 5-way border n=80 long-text", pred80, gold))
    rows.append(score("sonnet 5-way border n=120 long-text", pred120, gold))
    rows.append(score("sonnet 5-way n=120 + clash/geo ring", pred_all, gold))
    rows.append(score("oracle majority-border n=80", gold_oracle(majority, gold, order, 80), gold))
    rows.append(score("oracle majority-border n=120", gold_oracle(majority, gold, order, 120), gold))

    def n_err(pred) -> int:
        return int((np.array([gmap[int(p)] for p in pred]) != gold).sum())

    a80 = np.array([gmap[int(p)] for p in pred80])
    a120 = np.array([gmap[int(p)] for p in pred120])
    detail = {
        "model": SONNET,
        "majority_errors": int((aligned != gold).sum()),
        "five80_errors": n_err(pred80),
        "five120_errors": n_err(pred120),
        "five120_plus_ring_errors": n_err(pred_all),
        "five80_fixed": int(((aligned != gold) & (a80 == gold)).sum()),
        "five80_broke": int(((aligned == gold) & (a80 != gold)).sum()),
        "five120_fixed": int(((aligned != gold) & (a120 == gold)).sum()),
        "five120_broke": int(((aligned == gold) & (a120 != gold)).sum()),
        "extra_n": len(extra),
        "note": "five-way exemplars, no desk names, gold not used to pick docs",
    }
    print(json.dumps(detail, indent=2), flush=True)
    OUT.write_text(
        json.dumps(
            {
                "disclaimer": "Laboratory only. CGE production stays TF-IDF + DBSCAN at cosine 0.75.",
                "published": PUB,
                "n": len(texts),
                "detail": detail,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
