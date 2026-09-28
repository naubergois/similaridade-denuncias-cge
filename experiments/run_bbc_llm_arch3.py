#!/usr/bin/env python3
"""Floors 6–7 on the BBC stack. Replay 1–5 from cache. Not CGE production.

  6. Neighbor revolt again after labels moved, then a softer 3-of-7 revolt.
  7. Foreign neighbor: the closest doc in another cluster beats the
     second-closest doc in the assigned cluster.

Gold is not used to pick documents.

    .venv/bin/python experiments/run_bbc_llm_arch3.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bbc_beat_published import PUB, score  # noqa: E402
from run_bbc_claude_border_errors import OPUS, SONNET, build_majority, fill  # noqa: E402
from run_bbc_llm_arch import apply_five, prompt_opus  # noqa: E402
from run_bbc_llm_arch2 import parse_yesno, prompt_fit  # noqa: E402
from run_bbc_llm_border import (  # noqa: E402
    apply_pair,
    cksum,
    exemplars,
    load_cache,
    load_titles_and_ledes,
    margins,
    prompt_pair,
)
from run_bbc_llm_ensemble import gold_map  # noqa: E402
from run_bbc_nmi_lift import parse_five, prompt_five  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_arch3.json"
K = 5


def revolt_ids(Z, labels, k=7, need=4, skip=None):
    skip = set(skip or [])
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z)
    _, idx = nn.kneighbors(Z)
    out = []
    for i in range(len(labels)):
        if i in skip:
            continue
        labs = [int(labels[j]) for j in idx[i, 1 : k + 1]]
        top, ntop = Counter(labs).most_common(1)[0]
        if top != int(labels[i]) and ntop >= need:
            out.append(i)
    return out


def foreign_ids(Z, labels, skip=None, margin=0.015):
    skip = set(skip or [])
    S = Z @ Z.T
    np.fill_diagonal(S, -2.0)
    labs = np.asarray(labels)
    out = []
    for i in range(len(labs)):
        if i in skip:
            continue
        same = np.where(labs == labs[i])[0]
        other = np.where(labs != labs[i])[0]
        if len(same) < 2 or len(other) == 0:
            continue
        same_s = np.partition(S[i, same], -2)[-2]
        other_s = float(S[i, other].max())
        if other_s > same_s + margin:
            out.append(i)
    return out


def opus_on(pred, indices, texts, titles, ex, cache, tag: str):
    jobs, keys = [], []
    for i in indices:
        op = prompt_opus(titles[i], texts[i], ex, int(pred[i]))
        ok = cksum(tag, OPUS, titles[i], op)
        keys.append(ok)
        jobs.append((ok, op, OPUS))
    cache = fill(jobs, cache)
    return apply_five(pred, indices, keys, cache), cache


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    print("replay 1–5 …", flush=True)
    Z, majority = build_majority(texts)
    gap, first, second = margins(Z, majority)
    cents = normalize(
        np.stack([Z[majority == c].mean(0) if (majority == c).any() else np.zeros(Z.shape[1]) for c in range(K)])
    )
    sims = Z @ cents.T
    order3 = np.argsort(-sims, axis=1)
    gap23 = sims[np.arange(len(Z)), order3[:, 1]] - sims[np.arange(len(Z)), order3[:, 2]]
    order = np.argsort(gap)
    border80 = [int(i) for i in order[:80]]
    border160 = [int(i) for i in order[:160]]
    ex = exemplars(Z, majority, titles, ledes)
    cache = load_cache()
    rows = [score("1 majority", majority, gold)]

    pair_keys, a_ids, b_ids = [], [], []
    for i in border80:
        a, b = int(first[i]), int(second[i])
        pp = prompt_pair(titles[i], ledes[i], a, b, ex)
        pair_keys.append(cksum("claude_pair", SONNET, titles[i], str(a), str(b), pp))
        a_ids.append(a)
        b_ids.append(b)
    stage2 = apply_pair(majority, np.array(border80), pair_keys, cache, a_ids, b_ids, 80)
    for i in border80:
        fp = prompt_five(titles[i], texts[i], ex)
        ch = parse_five((cache.get(cksum("nmi_five", SONNET, titles[i], fp)) or {}).get("text", ""))
        if ch is not None and ch not in (int(first[i]), int(second[i])):
            stage2[i] = ch
    rows.append(score("2 sonnet pair", stage2, gold))

    flips = [i for i in border80 if int(stage2[i]) != int(majority[i])]
    close3 = [i for i in border80 if float(gap23[i]) < 0.02]
    route3 = set(flips) | set(close3)
    keys3 = [cksum("arch_opus", OPUS, titles[i], prompt_opus(titles[i], texts[i], ex, int(stage2[i]))) for i in sorted(route3)]
    stage3 = apply_five(stage2, sorted(route3), keys3, cache)
    rows.append(score("3 opus flips+close3", stage3, gold))

    ask4 = [i for i in border160 if i not in route3]
    nos = []
    for i in ask4:
        pk = cksum("arch_fit", SONNET, titles[i], str(int(stage3[i])), prompt_fit(titles[i], texts[i], ex[int(stage3[i])]))
        if parse_yesno((cache.get(pk) or {}).get("text", "")) is False:
            nos.append(i)
    keys4 = [cksum("arch_opus4", OPUS, titles[i], prompt_opus(titles[i], texts[i], ex, int(stage3[i]))) for i in nos]
    stage4 = apply_five(stage3, nos, keys4, cache)
    rows.append(score("4 odd-one-out", stage4, gold))

    revolt = revolt_ids(Z, stage4, need=4, skip=route3 | set(nos))
    keys5 = [cksum("arch_opus5", OPUS, titles[i], prompt_opus(titles[i], texts[i], ex, int(stage4[i]))) for i in revolt]
    stage5 = apply_five(stage4, revolt, keys5, cache)
    rows.append(score("5 neighbor 4/7", stage5, gold))
    seen = set(route3) | set(nos) | set(revolt)

    # 6a: revolt again at 4/7 after labels moved
    r6 = revolt_ids(Z, stage5, need=4, skip=seen)
    print(f"stage 6a new 4/7 revolt n={len(r6)}", flush=True)
    stage6a, cache = opus_on(stage5, r6, texts, titles, ex, cache, "arch_opus6a")
    rows.append(score("6a iterate 4/7 revolt", stage6a, gold))
    seen |= set(r6)

    # 6b: softer 3/7
    r6b = revolt_ids(Z, stage6a, need=3, skip=seen)
    print(f"stage 6b 3/7 revolt n={len(r6b)}", flush=True)
    stage6, cache = opus_on(stage6a, r6b, texts, titles, ex, cache, "arch_opus6b")
    rows.append(score("6b softer 3/7 revolt", stage6, gold))
    seen |= set(r6b)

    # 7: foreign neighbor closer than second home neighbor
    foreign = foreign_ids(Z, stage6, skip=seen, margin=0.015)
    print(f"stage 7 foreign-neighbor n={len(foreign)}", flush=True)
    stage7, cache = opus_on(stage6, foreign, texts, titles, ex, cache, "arch_opus7")
    rows.append(score("7 foreign neighbor then opus", stage7, gold))

    gmap = gold_map(majority, gold)

    def nerr(pred):
        return int((np.array([gmap[int(p)] for p in pred]) != gold).sum())

    def fb(a, b):
        aa = np.array([gmap[int(p)] for p in a])
        bb = np.array([gmap[int(p)] for p in b])
        return int(((aa != gold) & (bb == gold)).sum()), int(((aa == gold) & (bb != gold)).sum())

    f6a, b6a = fb(stage5, stage6a)
    f6, b6 = fb(stage6a, stage6)
    f7, b7 = fb(stage6, stage7)
    detail = {
        "s5_errors": nerr(stage5),
        "s6a_errors": nerr(stage6a),
        "s6b_errors": nerr(stage6),
        "s7_errors": nerr(stage7),
        "n": {"r6a": len(r6), "r6b": len(r6b), "foreign": len(foreign)},
        "s6a": {"fix": f6a, "broke": b6a},
        "s6b": {"fix": f6, "broke": b6},
        "s7": {"fix": f7, "broke": b7},
        "note": "gold not used to pick documents",
    }
    print(json.dumps(detail, indent=2), flush=True)
    np.save(HERE / "data" / "emb_cache" / "bbc_stage7_labels.npy", stage7)
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
