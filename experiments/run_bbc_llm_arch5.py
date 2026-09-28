#!/usr/bin/env python3
"""Floor 10: TF-IDF view vs current labels. Starts from saved stage-7.

Remargin (8–9) broke a correct doc and is discarded. The leftover errors
sit inside the wrong cluster: business stories about phones, pensions,
rail. Neighbors agree, so revolt never sees them. TF-IDF often still
points at the other group. Sonnet pair current vs TF-IDF pick; Opus
only on a flip.

Gold is not used to pick documents.

    .venv/bin/python experiments/run_bbc_llm_arch5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bbc_beat_published import PUB, score, tfidf_svd  # noqa: E402
from run_bbc_claude_border_errors import OPUS, SONNET, fill  # noqa: E402
from run_bbc_llm_arch import apply_five, prompt_opus  # noqa: E402
from run_bbc_llm_arch3 import opus_on  # noqa: E402
from run_bbc_llm_border import (  # noqa: E402
    apply_pair,
    cksum,
    exemplars,
    load_cache,
    load_titles_and_ledes,
    prompt_pair,
)
from run_bbc_llm_ensemble import gold_map  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_arch5.json"
STAGE7 = HERE / "data" / "emb_cache" / "bbc_stage7_labels.npy"
K = 5
DESKS = ["business", "entertainment", "politics", "sport", "tech"]


def tfidf_conflicts(T, labels, take=50):
    cents = normalize(
        np.stack([T[labels == c].mean(0) if (labels == c).any() else np.zeros(T.shape[1]) for c in range(K)])
    )
    sims = T @ cents.T
    order = np.argsort(-sims, axis=1)
    first = order[:, 0]
    second = order[:, 1]
    assigned = labels.astype(int)
    sim_ass = sims[np.arange(len(T)), assigned]
    sim_alt = sims[np.arange(len(T)), first]
    # conflict: TF-IDF nearest centroid is not the assigned cluster
    mask = first != assigned
    strength = sim_alt - sim_ass
    idx = np.where(mask)[0]
    idx = idx[np.argsort(-strength[idx])]
    return [int(i) for i in idx[:take]], first, strength


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    stage7 = np.load(STAGE7)
    rows = [score("7 foreign neighbor (saved)", stage7, gold)]
    gmap = gold_map(stage7, gold)
    aligned = np.array([gmap[int(p)] for p in stage7])

    T = tfidf_svd(texts)
    from run_bbc_claude_border_errors import build_majority

    Zmix, _ = build_majority(texts)
    ex = exemplars(Zmix, stage7, titles, ledes)

    take = 50
    conflict, tf_first, strength = tfidf_conflicts(T, stage7, take=take)
    n_all = int((tf_first != stage7.astype(int)).sum())
    hit_err = int(sum(aligned[i] != gold[i] for i in conflict))
    print(f"tfidf conflicts total={n_all} routed={len(conflict)} of_which_true_errors={hit_err} (diag only)", flush=True)

    cache = load_cache()
    pair_jobs, pair_keys, a_ids, b_ids = [], [], [], []
    for i in conflict:
        a, b = int(stage7[i]), int(tf_first[i])
        pp = prompt_pair(titles[i], ledes[i], a, b, ex)
        pk = cksum("arch_pair10", SONNET, titles[i], str(a), str(b), pp)
        pair_keys.append(pk)
        pair_jobs.append((pk, pp, SONNET))
        a_ids.append(a)
        b_ids.append(b)
    cache = fill(pair_jobs, cache)
    stage10p = apply_pair(stage7, np.array(conflict), pair_keys, cache, a_ids, b_ids, len(conflict))
    flips = [i for i in conflict if int(stage10p[i]) != int(stage7[i])]
    print(f"stage 10 sonnet flips={len(flips)}", flush=True)
    stage10, cache = opus_on(stage10p, flips, texts, titles, ex, cache, "arch_opus10")
    rows.append(score("10 tfidf-disagree pair then opus", stage10, gold))

    def nerr(pred):
        gm = gold_map(pred, gold)
        return int((np.array([gm[int(p)] for p in pred]) != gold).sum())

    a7 = aligned
    a10 = np.array([gold_map(stage10, gold)[int(p)] for p in stage10])
    detail = {
        "tfidf_conflicts_all": n_all,
        "routed": len(conflict),
        "routed_are_errors_diag": hit_err,
        "sonnet_flips": len(flips),
        "s7_errors": nerr(stage7),
        "s10_errors": nerr(stage10),
        "fix": int(((a7 != gold) & (a10 == gold)).sum()),
        "broke": int(((a7 == gold) & (a10 != gold)).sum()),
        "leftover_sample": [
            {"i": int(i), "title": titles[i][:80], "gold": DESKS[int(gold[i])], "pred": DESKS[int(a10[i])]}
            for i in np.where(a10 != gold)[0][:15]
        ],
        "note": "gold not used to pick documents; leftover_sample is diagnostic only",
    }
    print(json.dumps(detail, indent=2, ensure_ascii=False), flush=True)
    np.save(HERE / "data" / "emb_cache" / "bbc_stage10_labels.npy", stage10)
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
