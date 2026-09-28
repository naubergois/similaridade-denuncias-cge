#!/usr/bin/env python3
"""Floor 13: second-closest centroid pair, rim exemplars, from stage 12.

    .venv/bin/python experiments/run_bbc_llm_arch8.py
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
from run_bbc_llm_arch6 import TAKE, interface  # noqa: E402
from run_bbc_llm_arch7 import rim_exemplars  # noqa: E402
from run_bbc_llm_border import load_cache, load_titles_and_ledes  # noqa: E402
from run_bbc_llm_ensemble import gold_map  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_arch8.json"
STAGE12 = HERE / "data" / "emb_cache" / "bbc_stage12_labels.npy"
K = 5


def second_pair(Z, labels):
    cents = normalize(np.stack([Z[labels == c].mean(0) for c in range(K)]))
    S = cents @ cents.T
    pairs = [(float(S[i, j]), i, j) for i in range(K) for j in range(i + 1, K)]
    pairs.sort(reverse=True)
    _, a1, b1 = pairs[0]
    sim, a2, b2 = pairs[1]
    return a2, b2, cents, sim, (a1, b1)


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    stage12 = np.load(STAGE12)
    rows = [score("12 rim-exemplars (saved)", stage12, gold)]
    Z, _ = build_majority(texts)
    a, b, cents, sim, first = second_pair(Z, stage12)
    route = interface(Z, stage12, a, b, cents, TAKE) + interface(Z, stage12, b, a, cents, TAKE)
    route = list(dict.fromkeys(route))
    print(f"first {first} second {a}-{b} sim={sim:.3f} routed={len(route)}", flush=True)
    ex = rim_exemplars(Z, stage12, titles, ledes)
    cache = load_cache()
    stage13, cache = opus_on(stage12, route, texts, titles, ex, cache, "arch_opus13")
    rows.append(score("13 second-pair rim then opus", stage13, gold))
    g12 = gold_map(stage12, gold)
    g13 = gold_map(stage13, gold)
    a12 = np.array([g12[int(p)] for p in stage12])
    a13 = np.array([g13[int(p)] for p in stage13])
    detail = {
        "second_pair": [int(a), int(b)],
        "first_pair": [int(first[0]), int(first[1])],
        "centroid_sim": sim,
        "routed": len(route),
        "s12_errors": int((a12 != gold).sum()),
        "s13_errors": int((a13 != gold).sum()),
        "fix": int(((a12 != gold) & (a13 == gold)).sum()),
        "broke": int(((a12 == gold) & (a13 != gold)).sum()),
        "note": "gold not used to pick documents",
    }
    print(json.dumps(detail, indent=2), flush=True)
    np.save(HERE / "data" / "emb_cache" / "bbc_stage13_labels.npy", stage13)
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
