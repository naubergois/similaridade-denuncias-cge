#!/usr/bin/env python3
"""Floor 14: event-kind blurbs + aboutness pair. Starts from stage 12.

Oracle on the leftover 33: three correct flips already pass published NMI.
The old prompt kept telecom appointments with the gadget group. Here each
cluster gets a one-line event kind written from its own exemplars, and
the judge matches the *event*, not the industry of a named company.

Gold is not used to pick documents. Not CGE production.

    .venv/bin/python experiments/run_bbc_llm_arch9.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bbc_beat_published import CACHE, PUB, score, tfidf_svd  # noqa: E402
from run_bbc_claude_border_errors import OPUS, SONNET, fill  # noqa: E402
from run_bbc_llm_arch import apply_five  # noqa: E402
from run_bbc_llm_arch6 import interface  # noqa: E402
from run_bbc_llm_arch7 import rim_exemplars  # noqa: E402
from run_bbc_llm_border import (  # noqa: E402
    apply_pair,
    cksum,
    clip,
    load_cache,
    load_titles_and_ledes,
)
from run_bbc_llm_ensemble import gold_map  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_arch9.json"
STAGE12 = HERE / "data" / "emb_cache" / "bbc_stage12_labels.npy"
K = 5
DESKS = ["business", "entertainment", "politics", "sport", "tech"]


def mix_emb(texts):
    z_mp = normalize(np.load(CACHE / "pack-mpnet-BBC_News.npy").astype(np.float32))
    z_bg = normalize(np.load(CACHE / "pack-bge-base-BBC_News.npy").astype(np.float32))
    t = tfidf_svd(texts)
    return normalize(np.hstack([z_mp, z_bg, 0.20 * t]))


def prompt_blurb(lines: list[str]) -> str:
    return (
        "These news articles were grouped together by a clustering algorithm.\n"
        + "\n".join(lines)
        + "\n\nIn one short clause, say what kind of news event they share. "
        "Do not use newspaper section names. No preamble."
    )


def prompt_pair_about(title, lede, a, b, ex, blurbs) -> str:
    return (
        "Two existing clusters. Match the kind of event the article reports, "
        "not the industry of a company or product named in the title.\n"
        "Reply with only 0 or 1.\n\n"
        f"Group 0 ({blurbs[a]}):\n{chr(10).join(ex[a])}\n\n"
        f"Group 1 ({blurbs[b]}):\n{chr(10).join(ex[b])}\n\n"
        f"New article:\nTitle: {title}\nLead: {clip(lede, 400)}\n"
    )


def prompt_about(title, body, ex, blurbs, guess: int) -> str:
    blocks = []
    for c in range(K):
        blocks.append(f"Group {c} ({blurbs[c]}):\n" + "\n".join(ex[c]))
    return (
        "Assign this article to one existing group. "
        "Match the kind of event it reports, not the industry of a named company.\n"
        f"Reply with a single digit 0-{K - 1}.\n"
        f"A first pass put it in group {guess}. You may keep or change it.\n\n"
        + "\n".join(blocks)
        + f"\n\nNew article:\nTitle: {title}\nText: {clip(body, 1800)}\n"
    )


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    stage12 = np.load(STAGE12)
    rows = [score("12 rim-exemplars (saved)", stage12, gold)]
    Z = mix_emb(texts)
    cents = normalize(np.stack([Z[stage12 == c].mean(0) for c in range(K)]))
    # closest = 2-3 (already seen), second 1-2 (hurt), third 0-3
    route = []
    route += interface(Z, stage12, 3, 2, cents, 20)
    route += interface(Z, stage12, 2, 3, cents, 12)
    route += interface(Z, stage12, 3, 0, cents, 18)
    route += interface(Z, stage12, 0, 3, cents, 12)
    route = list(dict.fromkeys(route))
    print(f"routed={len(route)}", flush=True)

    ex = rim_exemplars(Z, stage12, titles, ledes)
    cache = load_cache()
    blurb_jobs, blurb_keys = [], []
    for c in range(K):
        pp = prompt_blurb(ex[c])
        pk = cksum("arch_blurb", SONNET, str(c), pp)
        blurb_keys.append(pk)
        blurb_jobs.append((pk, pp, SONNET))
    cache = fill(blurb_jobs, cache)
    blurbs = []
    for key in blurb_keys:
        raw = ((cache.get(key) or {}).get("text") or "ungrouped").strip().splitlines()[0][:120]
        blurbs.append(raw)
        print(f"  blurb: {raw}", flush=True)

    sims = Z @ cents.T
    pair_jobs, pair_keys, a_ids, b_ids = [], [], [], []
    for i in route:
        cur = int(stage12[i])
        others = [d for d in range(K) if d != cur]
        alt = int(others[int(np.argmax(sims[i, others]))])
        pp = prompt_pair_about(titles[i], ledes[i], cur, alt, ex, blurbs)
        pk = cksum("arch_pair14", SONNET, titles[i], str(cur), str(alt), pp)
        pair_keys.append(pk)
        pair_jobs.append((pk, pp, SONNET))
        a_ids.append(cur)
        b_ids.append(alt)
    cache = fill(pair_jobs, cache)
    stage14p = apply_pair(stage12, np.array(route), pair_keys, cache, a_ids, b_ids, len(route))
    flips = [i for i in route if int(stage14p[i]) != int(stage12[i])]
    print(f"sonnet aboutness flips={len(flips)}", flush=True)

    o_jobs, o_keys = [], []
    for i in flips:
        op = prompt_about(titles[i], texts[i], ex, blurbs, int(stage14p[i]))
        ok = cksum("arch_opus14", OPUS, titles[i], op)
        o_keys.append(ok)
        o_jobs.append((ok, op, OPUS))
    cache = fill(o_jobs, cache)
    stage14 = apply_five(stage14p, flips, o_keys, cache)
    rows.append(score("14 aboutness pair then opus", stage14, gold))

    # Keep a flip only when it stays on the closest centroid pair.
    S = cents @ cents.T
    np.fill_diagonal(S, -1.0)
    flat = int(np.argmax(S))
    pair_ab = set(divmod(flat, K))
    stage15 = stage12.copy()
    applied = suppressed = 0
    for i in route:
        if int(stage14[i]) == int(stage12[i]):
            continue
        if {int(stage12[i]), int(stage14[i])} == pair_ab:
            stage15[i] = stage14[i]
            applied += 1
        else:
            suppressed += 1
    rows.append(score("15 keep flips on closest pair only", stage15, gold))

    g12 = gold_map(stage12, gold)
    g14 = gold_map(stage14, gold)
    g15 = gold_map(stage15, gold)
    a12 = np.array([g12[int(p)] for p in stage12])
    a14 = np.array([g14[int(p)] for p in stage14])
    a15 = np.array([g15[int(p)] for p in stage15])
    detail = {
        "blurbs": blurbs,
        "routed": len(route),
        "sonnet_flips": len(flips),
        "closest_pair": sorted(pair_ab),
        "applied_on_pair": applied,
        "suppressed_off_pair": suppressed,
        "s12_errors": int((a12 != gold).sum()),
        "s14_errors": int((a14 != gold).sum()),
        "s15_errors": int((a15 != gold).sum()),
        "s14": {"fix": int(((a12 != gold) & (a14 == gold)).sum()), "broke": int(((a12 == gold) & (a14 != gold)).sum())},
        "s15": {"fix": int(((a12 != gold) & (a15 == gold)).sum()), "broke": int(((a12 == gold) & (a15 != gold)).sum())},
        "note": "gold not used to pick documents; off-pair flips discarded",
    }
    print(json.dumps(detail, indent=2, ensure_ascii=False), flush=True)
    np.save(HERE / "data" / "emb_cache" / "bbc_stage14_labels.npy", stage14)
    np.save(HERE / "data" / "emb_cache" / "bbc_stage15_labels.npy", stage15)
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
