#!/usr/bin/env python3
"""Fourth and fifth floors on the BBC LLM stack. Not the CGE product.

Replays stages 1–3 from cache, then:

  4. Odd-one-out: Sonnet says whether a doc belongs with its assigned
     group's exemplars. Only a 'no' goes to Opus. Asked on the rest of
     the thin 160 (not already audited).
  5. Neighbor revolt: if 4 of 7 nearest neighbors have another label,
     Opus reassigns.

Gold is not used to choose documents.

    .venv/bin/python experiments/run_bbc_llm_arch2.py
"""
from __future__ import annotations

import json
import re
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
from run_bbc_llm_border import (  # noqa: E402
    apply_pair,
    cksum,
    clip,
    exemplars,
    load_cache,
    load_titles_and_ledes,
    margins,
    prompt_pair,
)
from run_bbc_llm_ensemble import gold_map  # noqa: E402
from run_bbc_nmi_lift import parse_five, prompt_five  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_arch2.json"
K = 5


def prompt_fit(title: str, body: str, group_lines: list[str]) -> str:
    return (
        "A clustering put this article in a group whose typical members are:\n"
        + "\n".join(group_lines)
        + "\n\nArticle:\nTitle: "
        + title
        + "\nText: "
        + clip(body, 700)
        + "\n\nDoes it belong in that group? Reply only yes or no."
    )


def parse_yesno(raw: str) -> bool | None:
    t = (raw or "").strip().lower()
    if re.match(r"yes\b", t):
        return True
    if re.match(r"no\b", t):
        return False
    return None


def replay_stages(texts, titles, gold):
    Z, majority = build_majority(texts)
    gap, first, second = margins(Z, majority)
    cents = normalize(
        np.stack([Z[majority == c].mean(0) if (majority == c).any() else np.zeros(Z.shape[1]) for c in range(K)])
    )
    sims = Z @ cents.T
    order3 = np.argsort(-sims, axis=1)
    second_c = order3[:, 1]
    third = order3[:, 2]
    gap23 = sims[np.arange(len(Z)), second_c] - sims[np.arange(len(Z)), third]
    order = np.argsort(gap)
    border80 = [int(i) for i in order[:80]]
    border160 = [int(i) for i in order[:160]]
    ex = exemplars(Z, majority, titles, titles)  # placeholder, overwritten
    return Z, majority, gap, first, second, gap23, order, border80, border160, cents


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    print("replay 1–3 …", flush=True)
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
    rows = [score("1 majority of 11", majority, gold)]

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
        fk = cksum("nmi_five", SONNET, titles[i], fp)
        ch = parse_five((cache.get(fk) or {}).get("text", ""))
        if ch is not None and ch not in (int(first[i]), int(second[i])):
            stage2[i] = ch
    rows.append(score("2 sonnet pair80 + exception", stage2, gold))

    flips = [i for i in border80 if int(stage2[i]) != int(majority[i])]
    close3 = [i for i in border80 if float(gap23[i]) < 0.02]
    route3 = sorted(set(flips) | set(close3))
    opus_keys = []
    for i in route3:
        op = prompt_opus(titles[i], texts[i], ex, int(stage2[i]))
        opus_keys.append(cksum("arch_opus", OPUS, titles[i], op))
    stage3 = apply_five(stage2, route3, opus_keys, cache)
    rows.append(score("3 opus audit flips+close3", stage3, gold))

    # --- stage 4: odd-one-out on thin 160 not yet audited ---
    ask4 = [i for i in border160 if i not in set(route3)]
    fit_jobs, fit_keys = [], []
    for i in ask4:
        g = int(stage3[i])
        pp = prompt_fit(titles[i], texts[i], ex[g])
        pk = cksum("arch_fit", SONNET, titles[i], str(g), pp)
        fit_keys.append(pk)
        fit_jobs.append((pk, pp, SONNET))
    cache = fill(fit_jobs, cache)
    nos = []
    for i, key in zip(ask4, fit_keys):
        ans = parse_yesno((cache.get(key) or {}).get("text", ""))
        if ans is False:
            nos.append(i)
    print(f"stage 4 asked={len(ask4)} no={len(nos)}", flush=True)
    o4_jobs, o4_keys = [], []
    for i in nos:
        op = prompt_opus(titles[i], texts[i], ex, int(stage3[i]))
        ok = cksum("arch_opus4", OPUS, titles[i], op)
        o4_keys.append(ok)
        o4_jobs.append((ok, op, OPUS))
    cache = fill(o4_jobs, cache)
    stage4 = apply_five(stage3, nos, o4_keys, cache)
    rows.append(score("4 odd-one-out then opus", stage4, gold))

    # --- stage 5: neighbor revolt on current labels ---
    nn = NearestNeighbors(n_neighbors=8, metric="cosine").fit(Z)
    _, idx = nn.kneighbors(Z)
    revolt = []
    for i in range(len(texts)):
        neigh = [int(j) for j in idx[i, 1:8]]
        labs = [int(stage4[j]) for j in neigh]
        top, ntop = Counter(labs).most_common(1)[0]
        if top != int(stage4[i]) and ntop >= 4:
            revolt.append(i)
    revolt = [i for i in revolt if i not in set(route3) and i not in set(nos)]
    print(f"stage 5 neighbor revolt n={len(revolt)}", flush=True)
    o5_jobs, o5_keys = [], []
    for i in revolt:
        op = prompt_opus(titles[i], texts[i], ex, int(stage4[i]))
        ok = cksum("arch_opus5", OPUS, titles[i], op)
        o5_keys.append(ok)
        o5_jobs.append((ok, op, OPUS))
    cache = fill(o5_jobs, cache)
    stage5 = apply_five(stage4, revolt, o5_keys, cache)
    rows.append(score("5 neighbor revolt then opus", stage5, gold))

    gmap = gold_map(majority, gold)

    def nerr(pred):
        return int((np.array([gmap[int(p)] for p in pred]) != gold).sum())

    a3 = np.array([gmap[int(p)] for p in stage3])
    a4 = np.array([gmap[int(p)] for p in stage4])
    a5 = np.array([gmap[int(p)] for p in stage5])
    detail = {
        "architecture": ["majority", "sonnet pair", "opus flips+close3", "odd-one-out", "neighbor revolt"],
        "stage4_asked": len(ask4),
        "stage4_no": len(nos),
        "stage5_revolt": len(revolt),
        "errors": {"s3": nerr(stage3), "s4": nerr(stage4), "s5": nerr(stage5)},
        "s4_fixed": int(((a3 != gold) & (a4 == gold)).sum()),
        "s4_broke": int(((a3 == gold) & (a4 != gold)).sum()),
        "s5_fixed": int(((a4 != gold) & (a5 == gold)).sum()),
        "s5_broke": int(((a4 == gold) & (a5 != gold)).sum()),
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
