#!/usr/bin/env python3
"""Floors 8–9 on the BBC stack. Replay 1–7 from cache. Not CGE production.

After labels moved, the old majority margin is stale.

  8. Rebuild centroids from the current partition. Sonnet pair on the
     60 thinnest new margins that Opus has not seen; 5-way only if the
     pick leaves the two nearest groups; Opus on those flips.
  9. Foreign-neighbor again (tighter 0.008) on whatever is still new.

Gold is not used to pick documents.

    .venv/bin/python experiments/run_bbc_llm_arch4.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bbc_beat_published import PUB, score  # noqa: E402
from run_bbc_claude_border_errors import OPUS, SONNET, build_majority, fill  # noqa: E402
from run_bbc_llm_arch import apply_five, prompt_opus  # noqa: E402
from run_bbc_llm_arch2 import parse_yesno, prompt_fit  # noqa: E402
from run_bbc_llm_arch3 import foreign_ids, opus_on, revolt_ids  # noqa: E402
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

OUT = HERE / "bbc_llm_arch4.json"
K = 5
DESKS = ["business", "entertainment", "politics", "sport", "tech"]


def remargin(Z, labels):
    cents = normalize(
        np.stack([Z[labels == c].mean(0) if (labels == c).any() else np.zeros(Z.shape[1]) for c in range(K)])
    )
    sims = Z @ cents.T
    order = np.argsort(-sims, axis=1)
    first = order[:, 0]
    second = order[:, 1]
    gap = sims[np.arange(len(Z)), first] - sims[np.arange(len(Z)), second]
    return gap, first, second, cents


def replay_to_7(texts, titles, ledes, gold):
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

    r6 = revolt_ids(Z, stage5, need=4, skip=seen)
    stage6a, cache = opus_on(stage5, r6, texts, titles, ex, cache, "arch_opus6a")
    seen |= set(r6)
    r6b = revolt_ids(Z, stage6a, need=3, skip=seen)
    stage6, cache = opus_on(stage6a, r6b, texts, titles, ex, cache, "arch_opus6b")
    seen |= set(r6b)
    foreign = foreign_ids(Z, stage6, skip=seen, margin=0.015)
    stage7, cache = opus_on(stage6, foreign, texts, titles, ex, cache, "arch_opus7")
    seen |= set(foreign)
    rows.append(score("7 foreign neighbor", stage7, gold))
    return Z, majority, ex, cache, rows, stage7, seen


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    print("replay 1–7 …", flush=True)
    Z, majority, ex, cache, rows, stage7, seen = replay_to_7(texts, titles, ledes, gold)
    gmap = gold_map(majority, gold)

    def nerr(pred):
        return int((np.array([gmap[int(p)] for p in pred]) != gold).sum())

    def fb(a, b):
        aa = np.array([gmap[int(p)] for p in a])
        bb = np.array([gmap[int(p)] for p in b])
        return int(((aa != gold) & (bb == gold)).sum()), int(((aa == gold) & (bb != gold)).sum())

    # 8: remargin on current labels; exemplars follow the moved partition
    ex8 = exemplars(Z, stage7, titles, ledes)
    gap8, first8, second8, _ = remargin(Z, stage7)
    thin = [int(i) for i in np.argsort(gap8) if int(i) not in seen][:60]
    print(f"stage 8 remargin thin60 unseen={len(thin)}", flush=True)
    pair_jobs, pair_keys, a_ids, b_ids = [], [], [], []
    for i in thin:
        a, b = int(first8[i]), int(second8[i])
        pp = prompt_pair(titles[i], ledes[i], a, b, ex8)
        pk = cksum("arch_pair8", SONNET, titles[i], str(a), str(b), pp)
        pair_keys.append(pk)
        pair_jobs.append((pk, pp, SONNET))
        a_ids.append(a)
        b_ids.append(b)
    cache = fill(pair_jobs, cache)
    stage8p = apply_pair(stage7, np.array(thin), pair_keys, cache, a_ids, b_ids, len(thin))
    flips8 = [i for i in thin if int(stage8p[i]) != int(stage7[i])]
    for i in thin:
        fp = prompt_five(titles[i], texts[i], ex8)
        fk = cksum("arch_five8", SONNET, titles[i], fp)
        if fk not in cache:
            cache = fill([(fk, fp, SONNET)], cache)
        ch = parse_five((cache.get(fk) or {}).get("text", ""))
        if ch is not None and ch not in (int(first8[i]), int(second8[i])):
            stage8p[i] = ch
            if i not in flips8:
                flips8.append(i)
    print(f"stage 8 pair flips/exceptions={len(flips8)}", flush=True)
    stage8, cache = opus_on(stage8p, flips8, texts, titles, ex8, cache, "arch_opus8")
    rows.append(score("8 remargin pair then opus flips", stage8, gold))
    seen |= set(thin)

    # 9: tighter foreign neighbor
    foreign9 = foreign_ids(Z, stage8, skip=seen, margin=0.008)
    print(f"stage 9 tight foreign n={len(foreign9)}", flush=True)
    stage9, cache = opus_on(stage8, foreign9, texts, titles, ex8, cache, "arch_opus9")
    rows.append(score("9 tight foreign then opus", stage9, gold))

    aligned = np.array([gmap[int(p)] for p in stage9])
    leftover = [int(i) for i in np.where(aligned != gold)[0]]
    leftovers = [
        {
            "i": i,
            "title": titles[i][:80],
            "gold": DESKS[int(gold[i])],
            "pred": DESKS[int(aligned[i])],
        }
        for i in leftover[:20]
    ]
    f8, b8 = fb(stage7, stage8)
    f9, b9 = fb(stage8, stage9)
    detail = {
        "s7_errors": nerr(stage7),
        "s8_errors": nerr(stage8),
        "s9_errors": nerr(stage9),
        "n": {"thin8": len(thin), "flips8": len(flips8), "foreign9": len(foreign9)},
        "s8": {"fix": f8, "broke": b8},
        "s9": {"fix": f9, "broke": b9},
        "leftover_sample": leftovers,
        "note": "gold not used to pick documents; leftover_sample is diagnostic only",
    }
    print(json.dumps(detail, indent=2, ensure_ascii=False), flush=True)
    np.save(HERE / "data" / "emb_cache" / "bbc_stage9_labels.npy", stage9)
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
