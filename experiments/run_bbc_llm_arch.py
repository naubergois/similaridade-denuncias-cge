#!/usr/bin/env python3
"""Three-stage LLM architecture on BBC News. Not the CGE product.

  1. Majority of 11 clusterings (no LLM).
  2. Sonnet pair on the 80 thinnest margins; 5-way only if the pick
     leaves the two nearest groups.
  3. Router sends a short list to Opus: every pair flip, plus docs
     whose third centroid is almost as close as the second.
     Opus reads a long slice and picks among five exemplar groups.

Gold is not used to choose documents.

    .venv/bin/python experiments/run_bbc_llm_arch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bbc_beat_published import PUB, score  # noqa: E402
from run_bbc_claude_border_errors import (  # noqa: E402
    K,
    OPUS,
    SONNET,
    build_majority,
    fill,
)
from run_bbc_llm_border import (  # noqa: E402
    apply_pair,
    cksum,
    clip,
    exemplars,
    load_cache,
    load_titles_and_ledes,
    margins,
    parse_pair,
    prompt_pair,
)
from run_bbc_llm_ensemble import gold_map  # noqa: E402
from run_bbc_nmi_lift import parse_five, prompt_five  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_arch.json"


def prompt_opus(title: str, body: str, ex: dict[int, list[str]], sonnet_guess: int | None) -> str:
    blocks = []
    for c in range(K):
        lines = "\n".join(ex[c]) if ex.get(c) else "- (empty)"
        blocks.append(f"Group {c}:\n{lines}")
    prior = f"A first model guessed group {sonnet_guess}. You may keep or change it.\n" if sonnet_guess is not None else ""
    return (
        "You are the second reader in a clustering pipeline. "
        "Assign this news article to one existing group. "
        "Do not invent a group and do not name topics. "
        f"Reply with a single digit 0-{K - 1}.\n\n"
        + prior
        + "\n".join(blocks)
        + f"\n\nNew article:\nTitle: {title}\nText: {clip(body, 1200)}\n"
    )


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
    print("stage 1 majority …", flush=True)
    Z, majority = build_majority(texts)
    rows = [score("1 majority of 11", majority, gold)]
    gmap = gold_map(majority, gold)
    aligned = np.array([gmap[int(p)] for p in majority])

    gap, first, second = margins(Z, majority)
    cents = normalize(
        np.stack([Z[majority == c].mean(0) if (majority == c).any() else np.zeros(Z.shape[1]) for c in range(K)])
    )
    sims = Z @ cents.T
    order3 = np.argsort(-sims, axis=1)
    third = order3[:, 2]
    gap23 = sims[np.arange(len(Z)), second] - sims[np.arange(len(Z)), third]
    order = np.argsort(gap)
    border = [int(i) for i in order[:80]]
    ex = exemplars(Z, majority, titles, ledes)
    cache = load_cache()

    pair_keys, a_ids, b_ids = [], [], []
    for i in border:
        a, b = int(first[i]), int(second[i])
        pp = prompt_pair(titles[i], ledes[i], a, b, ex)
        pk = cksum("claude_pair", SONNET, titles[i], str(a), str(b), pp)
        pair_keys.append(pk)
        a_ids.append(a)
        b_ids.append(b)
    stage2 = apply_pair(majority, np.array(border), pair_keys, cache, a_ids, b_ids, 80)
    five_keys = []
    for i in border:
        fp = prompt_five(titles[i], texts[i], ex)
        fk = cksum("nmi_five", SONNET, titles[i], fp)
        five_keys.append(fk)
        if fk not in cache:
            cache = fill([(fk, fp, SONNET)], cache)
        ch = parse_five((cache.get(fk) or {}).get("text", ""))
        if ch is not None and ch not in (int(first[i]), int(second[i])):
            stage2[i] = ch
    rows.append(score("2 sonnet pair80 + 5-way outside top-2", stage2, gold))

    flips = [i for i in border if int(stage2[i]) != int(majority[i])]
    close3 = [i for i in border if float(gap23[i]) < 0.02]
    route = sorted(set(flips) | set(close3))
    print(f"stage 3 route flips={len(flips)} close3={len(close3)} unique={len(route)}", flush=True)

    opus_jobs = []
    opus_keys = []
    for i in route:
        guess = int(stage2[i])
        op = prompt_opus(titles[i], texts[i], ex, guess)
        ok = cksum("arch_opus", OPUS, titles[i], op)
        opus_keys.append(ok)
        opus_jobs.append((ok, op, OPUS))
    cache = fill(opus_jobs, cache)
    stage3 = apply_five(stage2, route, opus_keys, cache)
    rows.append(score("3 opus audit on flips+close3", stage3, gold))

    # stage 3b: if Opus equals majority on a flip, that is a revert — already in stage3
    a2 = np.array([gmap[int(p)] for p in stage2])
    a3 = np.array([gmap[int(p)] for p in stage3])
    detail = {
        "architecture": ["majority", "sonnet pair+exception", "opus audit routed"],
        "route_n": len(route),
        "flips": len(flips),
        "close3": len(close3),
        "majority_errors": int((aligned != gold).sum()),
        "stage2_errors": int((a2 != gold).sum()),
        "stage3_errors": int((a3 != gold).sum()),
        "opus_fixed": int(((a2 != gold) & (a3 == gold)).sum()),
        "opus_broke": int(((a2 == gold) & (a3 != gold)).sum()),
        "opus_reverted_flips": int(sum(int(stage3[i]) == int(majority[i]) and int(stage2[i]) != int(majority[i]) for i in route)),
        "note": "gold not used to pick documents",
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
