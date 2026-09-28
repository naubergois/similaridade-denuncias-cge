#!/usr/bin/env python3
"""Several BBC clusterings vote. The LLM only breaks the ties.

Not the CGE product. Cluster IDs are aligned to a reference partition with
the Hungarian algorithm — gold labels are not used for that step.

  majority — take the aligned mode when it is unique
  llm_on_ties — Haiku sees exemplars of the competing groups plus the vote tally
  named_on_ties — same docs, but the model is told the five desk names
                (zero-shot; not a clustering number)

    .venv/bin/python experiments/run_bbc_llm_ensemble.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bbc_beat_published import (  # noqa: E402
    CACHE,
    PUB,
    consensus,
    factorial_kmeans,
    kmeans,
    load_titles_and_ledes,
    score,
    tfidf_svd,
)
from run_bbc_llm_border import (  # noqa: E402
    DESKS,
    MODEL,
    apply_pair,
    cksum,
    clip,
    exemplars,
    fill_llm,
    margins,
    oracle as gold_oracle,
    parse_desk,
    parse_pair,
    prompt_named as prompt_named_plain,
    prompt_pair,
)
from run_public_benchmarks import DATA, RNG, load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_ensemble.json"
# reuse the same on-disk cache file as the border run, but new keys
import run_bbc_llm_border as border_mod  # noqa: E402

K = 5


def align(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a = np.asarray(pred)
    b = np.asarray(ref)
    labs_a, labs_b = np.unique(a), np.unique(b)
    m = np.zeros((len(labs_a), len(labs_b)), dtype=int)
    ia = {c: i for i, c in enumerate(labs_a)}
    ib = {c: i for i, c in enumerate(labs_b)}
    for x, y in zip(a, b):
        m[ia[x], ib[y]] += 1
    r, c = linear_sum_assignment(m.max() - m)
    mapping = {int(labs_a[i]): int(labs_b[j]) for i, j in zip(r, c)}
    return np.array([mapping[int(x)] for x in a])


def gold_map(pred: np.ndarray, gold: np.ndarray) -> dict[int, int]:
    labs_p, labs_g = np.unique(pred), np.unique(gold)
    m = np.zeros((len(labs_p), len(labs_g)), dtype=int)
    ip = {c: i for i, c in enumerate(labs_p)}
    ig = {c: i for i, c in enumerate(labs_g)}
    for x, y in zip(pred, gold):
        m[ip[x], ig[y]] += 1
    r, c = linear_sum_assignment(m.max() - m)
    return {int(labs_p[i]): int(labs_g[j]) for i, j in zip(r, c)}


def parse_choice(raw: str, n: int) -> int | None:
    m = re.search(rf"[0-{n - 1}]", raw.strip())
    return int(m.group()) if m else None


def prompt_panel(title: str, lede: str, options: list[tuple[int, int, list[str]]]) -> str:
    blocks = []
    for i, (cid, votes, lines) in enumerate(options):
        body = "\n".join(lines) if lines else "- (empty)"
        blocks.append(f"Group {i} ({votes} model votes, internal id {cid}):\n{body}")
    return (
        "Several clustering models grouped this news article differently. "
        "Pick the existing group that fits. Do not invent a new group. "
        f"Reply with a single number from 0 to {len(options) - 1}.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nNew article:\nTitle: {title}\nLead: {clip(lede)}"
    )


def prompt_named(title: str, lede: str, tally: str) -> str:
    return (
        "Classify this BBC news article into exactly one desk: "
        "business, entertainment, politics, sport, tech.\n"
        f"Clustering models already voted as follows (aligned desk names): {tally}\n"
        "You may override them. Reply with one of the five words and nothing else.\n\n"
        f"Title: {title}\nLead: {clip(lede)}"
    )


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    Z_mp = normalize(np.load(CACHE / "pack-mpnet-BBC_News.npy").astype(np.float32))
    Z_bg = normalize(np.load(CACHE / "pack-bge-base-BBC_News.npy").astype(np.float32))
    T = tfidf_svd(texts)
    Z_mix = normalize(np.hstack([Z_mp, Z_bg, 0.20 * T]))
    Z_cat = normalize(np.hstack([Z_mp, Z_bg]))
    Z_tf = normalize(np.hstack([Z_mp, 0.25 * T]))
    Z_chunk = normalize(np.load(CACHE / "bbc-mpnet-chunk-mean.npy").astype(np.float32))
    Z_title = normalize(np.load(CACHE / "bbc-mpnet-titles.npy").astype(np.float32))
    Z_tt = normalize(np.hstack([Z_mp, 0.15 * Z_title]))

    print("fitting panel …", flush=True)
    raw: dict[str, np.ndarray] = {
        "mpnet": kmeans(Z_mp),
        "bge": kmeans(Z_bg),
        "concat": kmeans(Z_cat),
        "mix": kmeans(Z_mix),
        "mpnet+tfidf": kmeans(Z_tf),
        "chunk": kmeans(Z_chunk),
        "title": kmeans(Z_tt),
        "factorial_mix": factorial_kmeans(Z_mix),
        "consensus": consensus([Z_mp, Z_bg, T]),
        "gmm_mix": GaussianMixture(n_components=K, random_state=RNG, covariance_type="diag", max_iter=200).fit_predict(Z_mix),
        "spectral_knn": SpectralClustering(
            n_clusters=K,
            affinity="nearest_neighbors",
            n_neighbors=25,
            random_state=RNG,
            assign_labels="kmeans",
        ).fit_predict(Z_mix),
    }

    rows = []
    for name, pred in raw.items():
        rows.append(score(f"panel {name}", pred, gold))

    ref = raw["mix"]
    aligned = {name: (pred if name == "mix" else align(pred, ref)) for name, pred in raw.items()}
    names = list(aligned)
    stack = np.stack([aligned[n] for n in names], axis=1)
    n_models = stack.shape[1]

    majority = np.zeros(len(texts), dtype=int)
    max_votes = np.zeros(len(texts), dtype=int)
    n_unique = np.zeros(len(texts), dtype=int)
    disputes: list[int] = []
    for i in range(len(texts)):
        c = Counter(int(x) for x in stack[i])
        lab, votes = c.most_common(1)[0]
        majority[i] = lab
        max_votes[i] = votes
        n_unique[i] = len(c)
        if votes <= n_models // 2 or len(c) >= 3:
            disputes.append(i)

    rows.append(score("majority of 11 clusterings", majority, gold))
    print(
        f"panel={n_models} disputes={len(disputes)} "
        f"unanimous={(n_unique == 1).sum()} weak={(max_votes <= n_models // 2).sum()}",
        flush=True,
    )

    gmap = gold_map(ref, gold)
    aligned_gold = np.array([gmap[int(p)] for p in ref])
    maj_gold = np.array([gmap[int(p)] for p in majority])
    wrong_mix = set(np.where(aligned_gold != gold)[0].tolist())
    wrong_maj = set(np.where(maj_gold != gold)[0].tolist())
    print(
        f"errors mix={len(wrong_mix)} majority={len(wrong_maj)} "
        f"disputes that are mix-errors={len(wrong_mix & set(disputes))}",
        flush=True,
    )

    ex = exemplars(Z_mix, ref, titles, ledes)
    cache = border_mod.load_cache()
    panel_keys, panel_prompts = [], []
    named_keys, named_prompts = [], []
    option_ids: list[list[int]] = []
    for i in disputes:
        c = Counter(int(x) for x in stack[i])
        options = [(cid, votes, ex[cid]) for cid, votes in c.most_common()]
        option_ids.append([cid for cid, _, _ in options])
        pp = prompt_panel(titles[i], ledes[i], options)
        pk = cksum("ensemble_panel", MODEL, titles[i], str(c), pp)
        panel_keys.append(pk)
        panel_prompts.append(pp)
        tally = ", ".join(f"{DESKS[gmap[cid]]}={votes}" for cid, votes in c.most_common())
        np_ = prompt_named(titles[i], ledes[i], tally)
        nk = cksum("ensemble_named", MODEL, titles[i], tally, np_)
        named_keys.append(nk)
        named_prompts.append(np_)

    cache = fill_llm(list(zip(panel_keys, panel_prompts)) + list(zip(named_keys, named_prompts)), cache)

    llm_pred = majority.copy()
    named_pred = maj_gold.copy()
    used_panel = used_named = 0
    for i, pk, nk, opts in zip(disputes, panel_keys, named_keys, option_ids):
        choice = parse_choice((cache.get(pk) or {}).get("text", ""), len(opts))
        if choice is not None:
            llm_pred[i] = opts[choice]
            used_panel += 1
        desk = parse_desk((cache.get(nk) or {}).get("text", ""))
        if desk is not None:
            named_pred[i] = desk
            used_named += 1

    rows.append(score("majority + llm panel on ties", llm_pred, gold))
    rows.append(score("majority + llm named-desks on ties", named_pred, gold))

    # oracle: if any voter proposed a label that maps to gold, take it
    oracle = majority.copy()
    recoverable = 0
    for i in disputes:
        cands = {int(x) for x in stack[i]}
        hit = [c for c in cands if gmap[c] == int(gold[i])]
        if hit:
            oracle[i] = hit[0]
            recoverable += 1
    rows.append(score("oracle among panel labels on ties", oracle, gold))

    # Second stage: keep the majority, ask the LLM only about its own thin border.
    gap, first, second = margins(Z_mix, majority)
    order = np.argsort(gap)
    border_n = 80
    border = order[:border_n]
    ex_maj = exemplars(Z_mix, majority, titles, ledes)
    pair_keys, pair_prompts, a_ids, b_ids = [], [], [], []
    for i in border:
        i = int(i)
        a, b = int(first[i]), int(second[i])
        pp = prompt_pair(titles[i], ledes[i], a, b, ex_maj)
        pk = cksum("ensemble_border_pair", MODEL, titles[i], str(a), str(b), pp)
        pair_keys.append(pk)
        pair_prompts.append(pp)
        a_ids.append(a)
        b_ids.append(b)
    cache = fill_llm(list(zip(pair_keys, pair_prompts)), cache)
    for n in (40, 80):
        rows.append(score(f"majority + llm pair-border n={n}", apply_pair(majority, border, pair_keys, cache, a_ids, b_ids, n), gold))
        rows.append(score(f"oracle majority-border n={n}", gold_oracle(majority, gold, order, n), gold))

    close = [i for i in range(len(texts)) if max_votes[i] - (n_models - max_votes[i]) <= 2 and n_unique[i] > 1]
    print(f"close votes (top minus rest <=2)={len(close)}", flush=True)

    llm_gold = np.array([gmap[int(p)] for p in llm_pred])
    pair80 = apply_pair(majority, border, pair_keys, cache, a_ids, b_ids, 80)
    pair80_gold = np.array([gmap[int(p)] for p in pair80])
    detail = {
        "n_models": n_models,
        "models": names,
        "disputes": len(disputes),
        "unanimous": int((n_unique == 1).sum()),
        "errors_mix": int(len(wrong_mix)),
        "errors_majority": int(len(wrong_maj)),
        "errors_llm_panel": int((llm_gold != gold).sum()),
        "errors_named": int((named_pred != gold).sum()),
        "llm_panel_used": used_panel,
        "llm_named_used": used_named,
        "oracle_recoverable_ties": recoverable,
        "panel_fixed": int(((maj_gold != gold) & (llm_gold == gold)).sum()),
        "panel_broke": int(((maj_gold == gold) & (llm_gold != gold)).sum()),
        "named_fixed": int(((maj_gold != gold) & (named_pred == gold)).sum()),
        "named_broke": int(((maj_gold == gold) & (named_pred != gold)).sum()),
        "errors_majority_plus_pair80": int((pair80_gold != gold).sum()),
        "pair80_fixed": int(((maj_gold != gold) & (pair80_gold == gold)).sum()),
        "pair80_broke": int(((maj_gold == gold) & (pair80_gold != gold)).sum()),
        "model": MODEL,
        "protocol_note": "majority is clustering; panel/named on ties and pair-border are LLM stages",
    }
    print(json.dumps(detail), flush=True)
    payload = {
        "disclaimer": "Laboratory only. CGE production stays TF-IDF + DBSCAN at cosine 0.75.",
        "published": PUB,
        "n": len(texts),
        "detail": detail,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
