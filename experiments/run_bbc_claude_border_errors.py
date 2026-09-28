#!/usr/bin/env python3
"""Claude on the majority-border BBC articles that the panel got wrong.

Not the CGE product. Two readings, scored separately:

  gold_picked — only the border docs that disagree with gold are sent.
  That uses the label file to choose whom to ask. It is a diagnostic,
  not an unsupervised clustering number.

  fair_border — the same Sonnet pass on the 80 thinnest-margin docs,
  gold is not used to pick the set.

    .venv/bin/python experiments/run_bbc_claude_border_errors.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
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
    apply_pair,
    cksum,
    clip,
    exemplars,
    load_cache,
    margins,
    oracle as gold_oracle,
    parse_desk,
    parse_pair,
    prompt_named,
    prompt_pair,
    save_cache,
)
from run_bbc_llm_ensemble import align, gold_map  # noqa: E402
from run_public_benchmarks import RNG, load_bbc  # noqa: E402

OUT = HERE / "bbc_claude_border_errors.json"
SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-6"
WORKERS = 4
K = 5


def chat(prompt: str, model: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 24,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    last = ""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            return (data.get("content") or [{}])[0].get("text") or ""
        except urllib.error.HTTPError as exc:
            last = exc.read().decode()[:200]
            if exc.code in {429, 500, 502, 503, 529}:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(f"anthropic {exc.code} {last}") from exc
        except Exception as exc:
            last = str(exc)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"anthropic failed {last}")


def fill(jobs: list[tuple[str, str, str]], cache: dict) -> dict:
    pending = [(k, p, m) for k, p, m in jobs if k not in cache]
    print(f"llm cache hit={len(jobs) - len(pending)} miss={len(pending)}", flush=True)
    if not pending:
        return cache

    def one(item: tuple[str, str, str]) -> tuple[str, str, str]:
        key, prompt, model = item
        return key, model, chat(prompt, model)

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(one, item) for item in pending]
        for fut in as_completed(futs):
            key, model, text = fut.result()
            cache[key] = {"model": model, "text": text}
            done += 1
            if done % 10 == 0 or done == len(pending):
                save_cache(cache)
                print(f"llm {done}/{len(pending)}", flush=True)
    save_cache(cache)
    return cache


def apply_named_on(pred: np.ndarray, gold: np.ndarray, indices, keys, cache) -> np.ndarray:
    gmap = gold_map(pred, gold)
    out = np.array([gmap[int(p)] for p in pred])
    for i, key in zip(indices, keys):
        desk = parse_desk((cache.get(key) or {}).get("text", ""))
        if desk is not None:
            out[int(i)] = desk
    return out


def apply_pair_on(pred: np.ndarray, indices, keys, cache, a_ids, b_ids) -> np.ndarray:
    out = pred.copy()
    for i, key, a, b in zip(indices, keys, a_ids, b_ids):
        choice = parse_pair((cache.get(key) or {}).get("text", ""))
        if choice == 0:
            out[int(i)] = a
        elif choice == 1:
            out[int(i)] = b
    return out


def build_majority(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    Z_mp = normalize(np.load(CACHE / "pack-mpnet-BBC_News.npy").astype(np.float32))
    Z_bg = normalize(np.load(CACHE / "pack-bge-base-BBC_News.npy").astype(np.float32))
    T = tfidf_svd(texts)
    Z_mix = normalize(np.hstack([Z_mp, Z_bg, 0.20 * T]))
    Z_cat = normalize(np.hstack([Z_mp, Z_bg]))
    Z_tf = normalize(np.hstack([Z_mp, 0.25 * T]))
    Z_chunk = normalize(np.load(CACHE / "bbc-mpnet-chunk-mean.npy").astype(np.float32))
    Z_title = normalize(np.load(CACHE / "bbc-mpnet-titles.npy").astype(np.float32))
    Z_tt = normalize(np.hstack([Z_mp, 0.15 * Z_title]))
    raw = {
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
    ref = raw["mix"]
    aligned = {n: (p if n == "mix" else align(p, ref)) for n, p in raw.items()}
    stack = np.stack(list(aligned.values()), axis=1)
    majority = np.array([Counter(int(x) for x in stack[i]).most_common(1)[0][0] for i in range(len(texts))])
    return Z_mix, majority


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    print("fitting majority panel …", flush=True)
    Z, majority = build_majority(texts)
    rows = [score("majority of 11 clusterings", majority, gold)]
    gmap = gold_map(majority, gold)
    aligned = np.array([gmap[int(p)] for p in majority])
    wrong = np.where(aligned != gold)[0]
    gap, first, second = margins(Z, majority)
    order = np.argsort(gap)
    border80 = order[:80]
    erred = [int(i) for i in border80 if i in set(wrong.tolist())]
    print(f"majority errors={len(wrong)} border80={len(border80)} erred_in_border80={len(erred)}", flush=True)

    ex = exemplars(Z, majority, titles, ledes)
    examples = []
    for i in erred:
        examples.append(
            {
                "i": i,
                "title": titles[i],
                "gold": DESKS[int(gold[i])],
                "majority": DESKS[int(aligned[i])],
                "second": DESKS[gmap[int(second[i])]],
                "lede": clip(ledes[i], 180),
            }
        )
        print(f"ERR {DESKS[int(gold[i])]:13s} was {DESKS[int(aligned[i])]:13s} | {titles[i][:90]}", flush=True)

    cache = load_cache()
    jobs: list[tuple[str, str, str]] = []

    def pack_pair(indices, model: str):
        keys, prompts, a_ids, b_ids = [], [], [], []
        for i in indices:
            a, b = int(first[i]), int(second[i])
            pp = prompt_pair(titles[i], ledes[i], a, b, ex)
            pk = cksum("claude_pair", model, titles[i], str(a), str(b), pp)
            keys.append(pk)
            prompts.append(pp)
            a_ids.append(a)
            b_ids.append(b)
            jobs.append((pk, pp, model))
        return keys, a_ids, b_ids

    def pack_named(indices, model: str):
        keys, prompts = [], []
        for i in indices:
            np_ = prompt_named(titles[i], ledes[i])
            nk = cksum("claude_named", model, titles[i], np_)
            keys.append(nk)
            prompts.append(np_)
            jobs.append((nk, np_, model))
        return keys

    sonnet_err_pair, a_err, b_err = pack_pair(erred, SONNET)
    opus_err_pair, a_op, b_op = pack_pair(erred, OPUS)
    sonnet_err_named = pack_named(erred, SONNET)
    opus_err_named = pack_named(erred, OPUS)
    sonnet_fair_pair, a_fair, b_fair = pack_pair([int(i) for i in border80], SONNET)

    cache = fill(jobs, cache)

    sonnet_err = apply_pair_on(majority, erred, sonnet_err_pair, cache, a_err, b_err)
    opus_err = apply_pair_on(majority, erred, opus_err_pair, cache, a_op, b_op)
    sonnet_named = apply_named_on(majority, gold, erred, sonnet_err_named, cache)
    opus_named = apply_named_on(majority, gold, erred, opus_err_named, cache)
    sonnet_fair = apply_pair(majority, border80, sonnet_fair_pair, cache, a_fair, b_fair, 80)

    rows.append(score("sonnet pair on gold-picked border errors", sonnet_err, gold))
    rows.append(score("opus pair on gold-picked border errors", opus_err, gold))
    rows.append(score("sonnet named on gold-picked border errors", sonnet_named, gold))
    rows.append(score("opus named on gold-picked border errors", opus_named, gold))
    rows.append(score("sonnet pair on fair border n=80", sonnet_fair, gold))
    rows.append(score("oracle majority-border n=80", gold_oracle(majority, gold, order, 80), gold))

    def flips(before, after):
        bg = np.array([gmap[int(p)] for p in before]) if after.dtype == before.dtype and after.max() < 10 else before
        # after may already be gold-space (named)
        if after.max() <= 4 and (after == gold).sum() >= (np.array([gmap[int(p)] for p in after]) == gold).sum():
            ag = after if set(np.unique(after).tolist()) <= set(range(5)) and (after == aligned).mean() < 0.99 else np.array([gmap[int(p)] for p in after])
        else:
            ag = np.array([gmap[int(p)] for p in after])
        # simpler: always map cluster-space through gmap; named outputs are already desks
        return ag

    def errcount(pred, named=False):
        if named:
            return int((pred != gold).sum())
        return int((np.array([gmap[int(p)] for p in pred]) != gold).sum())

    detail = {
        "key_source": "ANTHROPIC_API_KEY already in the process environment; no key copied from another repo",
        "models": [SONNET, OPUS],
        "majority_errors": int(len(wrong)),
        "erred_in_border80": len(erred),
        "examples": examples,
        "sonnet_pair_errors": errcount(sonnet_err),
        "opus_pair_errors": errcount(opus_err),
        "sonnet_named_errors": errcount(sonnet_named, named=True),
        "opus_named_errors": errcount(opus_named, named=True),
        "sonnet_fair80_errors": errcount(sonnet_fair),
        "note": "gold-picked rows use the label file to choose the docs. fair_border does not.",
    }
    print(json.dumps({k: v for k, v in detail.items() if k != "examples"}, indent=2), flush=True)
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
