#!/usr/bin/env python3
"""Floor 12: same closest-pair interface, but exemplars show the rim.

Two centroid titles made the money group look like banks-only, so Opus
kept telecom appointments in the gadget group. Here each group shows
its core plus the members that already lean toward another group.

    .venv/bin/python experiments/run_bbc_llm_arch7.py
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
from run_bbc_llm_arch6 import TAKE, closest_pair, interface  # noqa: E402
from run_bbc_llm_border import clip, load_cache, load_titles_and_ledes  # noqa: E402
from run_bbc_llm_ensemble import gold_map  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_arch7.json"
STAGE7 = HERE / "data" / "emb_cache" / "bbc_stage7_labels.npy"
K = 5
DESKS = ["business", "entertainment", "politics", "sport", "tech"]


def rim_exemplars(Z, pred, titles, ledes, n_core=2, n_rim=3) -> dict[int, list[str]]:
    cents = normalize(np.stack([Z[pred == c].mean(0) for c in range(K)]))
    sims = Z @ cents.T
    out: dict[int, list[str]] = {}
    for c in range(K):
        idx = np.where(pred == c)[0]
        own = sims[idx, c]
        core = idx[np.argsort(-own)[:n_core]]
        others = [d for d in range(K) if d != c]
        lean = sims[idx][:, others].max(axis=1) - own
        rim = []
        for i in idx[np.argsort(-lean)]:
            if i in set(core):
                continue
            rim.append(int(i))
            if len(rim) >= n_rim:
                break
        take = list(core) + rim
        out[c] = [f"- {titles[i]} — {clip(ledes[i], 220)}" for i in take]
    return out


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    stage7 = np.load(STAGE7)
    rows = [score("7 foreign neighbor (saved)", stage7, gold)]
    Z, _ = build_majority(texts)
    a, b, cents, sim = closest_pair(Z, stage7)
    route = interface(Z, stage7, a, b, cents, TAKE) + interface(Z, stage7, b, a, cents, TAKE)
    route = list(dict.fromkeys(route))
    ex = rim_exemplars(Z, stage7, titles, ledes)
    print(f"pair {a}-{b} routed={len(route)}", flush=True)
    for c, lines in ex.items():
        print(f"  group {c}:", flush=True)
        for line in lines:
            print(f"    {line[:110]}", flush=True)

    cache = load_cache()
    stage12, cache = opus_on(stage7, route, texts, titles, ex, cache, "arch_opus12")
    rows.append(score("12 rim-exemplars on closest pair", stage12, gold))

    g7 = gold_map(stage7, gold)
    g12 = gold_map(stage12, gold)
    a7 = np.array([g7[int(p)] for p in stage7])
    a12 = np.array([g12[int(p)] for p in stage12])
    detail = {
        "closest_pair": [int(a), int(b)],
        "centroid_sim": sim,
        "routed": len(route),
        "s7_errors": int((a7 != gold).sum()),
        "s12_errors": int((a12 != gold).sum()),
        "fix": int(((a7 != gold) & (a12 == gold)).sum()),
        "broke": int(((a7 == gold) & (a12 != gold)).sum()),
        "note": "gold not used to pick documents",
    }
    print(json.dumps(detail, indent=2), flush=True)
    np.save(HERE / "data" / "emb_cache" / "bbc_stage12_labels.npy", stage12)
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
