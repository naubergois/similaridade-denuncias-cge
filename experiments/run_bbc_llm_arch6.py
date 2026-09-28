#!/usr/bin/env python3
"""Floor 11: closest-centroid interface, then Opus. Starts from stage 7.

Remargin and TF-IDF disagreement both net-negative. The leftover leak
is the two clusters whose centroids sit closest. Take members of each
that lean toward the other, send only those to Opus. Gold is not used
to pick documents.

    .venv/bin/python experiments/run_bbc_llm_arch6.py
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
from run_bbc_claude_border_errors import build_majority  # noqa: E402
from run_bbc_llm_arch3 import opus_on  # noqa: E402
from run_bbc_llm_border import exemplars, load_cache, load_titles_and_ledes  # noqa: E402
from run_bbc_llm_ensemble import gold_map  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_arch6.json"
STAGE7 = HERE / "data" / "emb_cache" / "bbc_stage7_labels.npy"
K = 5
DESKS = ["business", "entertainment", "politics", "sport", "tech"]
TAKE = 16


def closest_pair(Z, labels):
    cents = normalize(np.stack([Z[labels == c].mean(0) for c in range(K)]))
    S = cents @ cents.T
    np.fill_diagonal(S, -1.0)
    flat = int(np.argmax(S))
    a, b = divmod(flat, K)
    return a, b, cents, float(S[a, b])


def interface(Z, labels, src, dst, cents, take):
    members = np.where(labels == src)[0]
    sims = Z @ cents.T
    lean = sims[members, dst] - sims[members, src]
    return [int(i) for i in members[np.argsort(-lean)][:take]]


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    stage7 = np.load(STAGE7)
    rows = [score("7 foreign neighbor (saved)", stage7, gold)]
    Z, _ = build_majority(texts)
    a, b, cents, sim = closest_pair(Z, stage7)
    route = interface(Z, stage7, a, b, cents, TAKE) + interface(Z, stage7, b, a, cents, TAKE)
    route = list(dict.fromkeys(route))
    print(f"closest pair {a}-{b} sim={sim:.3f} routed={len(route)}", flush=True)

    ex = exemplars(Z, stage7, titles, ledes)
    cache = load_cache()
    stage11, cache = opus_on(stage7, route, texts, titles, ex, cache, "arch_opus11")
    rows.append(score("11 closest-pair interface then opus", stage11, gold))

    g7 = gold_map(stage7, gold)
    g11 = gold_map(stage11, gold)
    a7 = np.array([g7[int(p)] for p in stage7])
    a11 = np.array([g11[int(p)] for p in stage11])
    detail = {
        "closest_pair": [int(a), int(b)],
        "centroid_sim": sim,
        "routed": len(route),
        "s7_errors": int((a7 != gold).sum()),
        "s11_errors": int((a11 != gold).sum()),
        "fix": int(((a7 != gold) & (a11 == gold)).sum()),
        "broke": int(((a7 == gold) & (a11 != gold)).sum()),
        "leftover_sample": [
            {"i": int(i), "title": titles[i][:80], "gold": DESKS[int(gold[i])], "pred": DESKS[int(a11[i])]}
            for i in np.where(a11 != gold)[0][:12]
        ],
        "note": "gold not used to pick documents; leftover_sample is diagnostic only",
    }
    print(json.dumps(detail, indent=2, ensure_ascii=False), flush=True)
    np.save(HERE / "data" / "emb_cache" / "bbc_stage11_labels.npy", stage11)
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
