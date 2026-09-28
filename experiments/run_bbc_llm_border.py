#!/usr/bin/env python3
"""LLM reclassifies only the BBC documents that sit on a cluster border.

Not the CGE product. Two protocols, scored separately:

  pair_exemplars — no desk names. The model sees two typical articles from
  each of the two nearest clusters and answers 0 or 1. Still a clustering
  pipeline. This is the number to put next to GCC*.

  named_desks — the model is told the five BBC desks. That is zero-shot
  classification of the border, not unsupervised clustering.

    .venv/bin/python experiments/run_bbc_llm_border.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bbc_beat_published import (  # noqa: E402
    CACHE,
    PUB,
    hungarian_acc,
    load_titles_and_ledes,
    score,
    tfidf_svd,
)
from run_public_benchmarks import DATA, RNG, load_bbc  # noqa: E402

OUT = HERE / "bbc_llm_border.json"
CACHE_LLM = DATA / "bbc_llm_border_cache.json"
MODEL = "claude-haiku-4-5"
K = 5
N_BORDER = 200
WORKERS = 6
DESKS = ("business", "entertainment", "politics", "sport", "tech")


def chat(prompt: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": 16,
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
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
            return (data.get("content") or [{}])[0].get("text") or ""
        except urllib.error.HTTPError as exc:
            last = exc.read().decode()[:200]
            if exc.code in {429, 500, 502, 503, 529}:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"anthropic {exc.code} {last}") from exc
        except Exception as exc:
            last = str(exc)
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"anthropic failed {last}")


def cksum(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:20]


def clip(text: str, n: int = 420) -> str:
    return " ".join(text.split())[:n]


def load_cache() -> dict:
    if CACHE_LLM.exists():
        return json.loads(CACHE_LLM.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_LLM.parent.mkdir(parents=True, exist_ok=True)
    CACHE_LLM.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def best_view(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    Z_mp = normalize(np.load(CACHE / "pack-mpnet-BBC_News.npy").astype(np.float32))
    Z_bg = normalize(np.load(CACHE / "pack-bge-base-BBC_News.npy").astype(np.float32))
    T = tfidf_svd(texts)
    Z = normalize(np.hstack([Z_mp, Z_bg, 0.20 * T]))
    pred = KMeans(n_clusters=K, random_state=RNG, n_init=10).fit_predict(Z)
    return Z, pred


def margins(Z: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cents = np.stack([Z[pred == c].mean(axis=0) if (pred == c).any() else np.zeros(Z.shape[1]) for c in range(K)])
    cents = normalize(cents)
    sims = Z @ cents.T
    order = np.argsort(-sims, axis=1)
    first = order[:, 0]
    second = order[:, 1]
    gap = sims[np.arange(len(Z)), first] - sims[np.arange(len(Z)), second]
    return gap, first, second


def exemplars(Z: np.ndarray, pred: np.ndarray, titles: list[str], ledes: list[str], n: int = 2) -> dict[int, list[str]]:
    cents = np.stack([Z[pred == c].mean(axis=0) if (pred == c).any() else np.zeros(Z.shape[1]) for c in range(K)])
    cents = normalize(cents)
    out: dict[int, list[str]] = {}
    for c in range(K):
        idx = np.where(pred == c)[0]
        if len(idx) == 0:
            out[c] = []
            continue
        sim = Z[idx] @ cents[c]
        take = idx[np.argsort(-sim)[:n]]
        out[c] = [f"- {titles[i]} — {clip(ledes[i], 220)}" for i in take]
    return out


def prompt_pair(title: str, lede: str, a: int, b: int, ex: dict[int, list[str]]) -> str:
    return (
        "You are grouping news articles into existing clusters. "
        "Do not invent new groups. Reply with only 0 or 1.\n\n"
        f"Group 0 typical articles:\n{chr(10).join(ex[a]) or '- (empty)'}\n\n"
        f"Group 1 typical articles:\n{chr(10).join(ex[b]) or '- (empty)'}\n\n"
        f"New article:\nTitle: {title}\nLead: {clip(lede)}\n\n"
        "Which group does it belong to? 0 or 1."
    )


def prompt_named(title: str, lede: str) -> str:
    return (
        "Classify this BBC news article into exactly one desk: "
        "business, entertainment, politics, sport, tech.\n"
        "Reply with one of those five words and nothing else.\n\n"
        f"Title: {title}\nLead: {clip(lede)}"
    )


def parse_pair(raw: str) -> int | None:
    m = re.search(r"[01]", raw.strip())
    return int(m.group()) if m else None


def parse_desk(raw: str) -> int | None:
    t = raw.strip().lower()
    for i, name in enumerate(DESKS):
        if name in t:
            return i
    return None


def fill_llm(jobs: list[tuple[str, str]], cache: dict) -> dict:
    pending = [(k, p) for k, p in jobs if k not in cache]
    print(f"llm cache hit={len(jobs) - len(pending)} miss={len(pending)}", flush=True)
    if not pending:
        return cache

    def one(item: tuple[str, str]) -> tuple[str, str]:
        key, prompt = item
        return key, chat(prompt)

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(one, item) for item in pending]
        for fut in as_completed(futs):
            key, text = fut.result()
            cache[key] = {"model": MODEL, "text": text}
            done += 1
            if done % 20 == 0 or done == len(pending):
                save_cache(cache)
                print(f"llm {done}/{len(pending)}", flush=True)
    save_cache(cache)
    return cache


def apply_pair(pred: np.ndarray, order: np.ndarray, keys: list[str], cache: dict, a_ids: list[int], b_ids: list[int], take: int) -> np.ndarray:
    out = pred.copy()
    used = 0
    for i, key, a, b in zip(order, keys, a_ids, b_ids):
        if used >= take:
            break
        raw = (cache.get(key) or {}).get("text", "")
        choice = parse_pair(raw)
        if choice == 0:
            out[i] = a
            used += 1
        elif choice == 1:
            out[i] = b
            used += 1
        else:
            used += 1
    return out


def apply_named(pred: np.ndarray, gold: np.ndarray, order: np.ndarray, keys: list[str], cache: dict, take: int) -> np.ndarray:
    # Align current clusters to gold, then overwrite the border with named desks.
    aligned = pred.copy()
    # hungarian on the current pred so IDs match gold desks
    from scipy.optimize import linear_sum_assignment

    classes, clusters = np.unique(gold), np.unique(pred)
    m = np.zeros((len(clusters), len(classes)), dtype=int)
    ci = {c: i for i, c in enumerate(clusters)}
    ki = {c: i for i, c in enumerate(classes)}
    for a, b in zip(pred, gold):
        m[ci[a], ki[b]] += 1
    r, c = linear_sum_assignment(m.max() - m)
    mapping = {int(clusters[i]): int(classes[j]) for i, j in zip(r, c)}
    aligned = np.array([mapping[int(p)] for p in pred])
    used = 0
    for i, key in zip(order, keys):
        if used >= take:
            break
        desk = parse_desk((cache.get(key) or {}).get("text", ""))
        if desk is not None:
            aligned[i] = desk
        used += 1
    return aligned


def oracle(pred: np.ndarray, gold: np.ndarray, order: np.ndarray, take: int) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    classes, clusters = np.unique(gold), np.unique(pred)
    m = np.zeros((len(clusters), len(classes)), dtype=int)
    ci = {c: i for i, c in enumerate(clusters)}
    ki = {c: i for i, c in enumerate(classes)}
    for a, b in zip(pred, gold):
        m[ci[a], ki[b]] += 1
    r, c = linear_sum_assignment(m.max() - m)
    mapping = {int(clusters[i]): int(classes[j]) for i, j in zip(r, c)}
    aligned = np.array([mapping[int(p)] for p in pred])
    aligned[order[:take]] = gold[order[:take]]
    return aligned


def main() -> int:
    texts, gold, _ = load_bbc()
    titles, ledes = load_titles_and_ledes(texts)
    Z, pred = best_view(texts)
    gap, first, second = margins(Z, pred)
    order = np.argsort(gap)
    border = order[:N_BORDER]
    print(f"bbc n={len(texts)} border={N_BORDER} gap_p{int(100*N_BORDER/len(texts))}={gap[border[-1]]:.4f}", flush=True)

    # How many of the current errors sit on that border?
    from scipy.optimize import linear_sum_assignment

    classes, clusters = np.unique(gold), np.unique(pred)
    m = np.zeros((len(clusters), len(classes)), dtype=int)
    ci = {c: i for i, c in enumerate(clusters)}
    ki = {c: i for i, c in enumerate(classes)}
    for a, b in zip(pred, gold):
        m[ci[a], ki[b]] += 1
    r, c = linear_sum_assignment(m.max() - m)
    mapping = {int(clusters[i]): int(classes[j]) for i, j in zip(r, c)}
    aligned = np.array([mapping[int(p)] for p in pred])
    wrong = np.where(aligned != gold)[0]
    print(f"current errors={len(wrong)} of which in top-{N_BORDER} border={len(set(wrong) & set(border.tolist()))}", flush=True)

    ex = exemplars(Z, pred, titles, ledes)
    cache = load_cache()
    pair_keys, pair_prompts, a_ids, b_ids = [], [], [], []
    named_keys, named_prompts = [], []
    for i in border:
        i = int(i)
        a, b = int(first[i]), int(second[i])
        pp = prompt_pair(titles[i], ledes[i], a, b, ex)
        pk = cksum("pair", MODEL, titles[i], str(a), str(b), pp)
        pair_keys.append(pk)
        pair_prompts.append(pp)
        a_ids.append(a)
        b_ids.append(b)
        np_ = prompt_named(titles[i], ledes[i])
        nk = cksum("named", MODEL, titles[i], np_)
        named_keys.append(nk)
        named_prompts.append(np_)

    cache = fill_llm(list(zip(pair_keys, pair_prompts)) + list(zip(named_keys, named_prompts)), cache)

    rows = [score("mpnet+bge+tfidf64", pred, gold)]
    cuts = (40, 80, 120, 160, 200)
    for n in cuts:
        rows.append(score(f"oracle border n={n}", oracle(pred, gold, order, n), gold))
        rows.append(score(f"llm pair-exemplars n={n}", apply_pair(pred, border, pair_keys, cache, a_ids, b_ids, n), gold))
        rows.append(score(f"llm named-desks n={n}", apply_named(pred, gold, border, named_keys, cache, n), gold))

    # flips vs gold on the 200
    pair_pred = apply_pair(pred, border, pair_keys, cache, a_ids, b_ids, N_BORDER)
    named_pred = apply_named(pred, gold, border, named_keys, cache, N_BORDER)
    pair_aligned = np.array([mapping[int(p)] for p in pair_pred])
    detail = {
        "border": N_BORDER,
        "errors_before": int(len(wrong)),
        "errors_in_border": int(len(set(wrong) & set(border.tolist()))),
        "pair_flips": int((pair_pred[border] != pred[border]).sum()),
        "pair_errors_after": int((pair_aligned != gold).sum()),
        "named_errors_after": int((named_pred != gold).sum()),
        "named_fixed": int(((aligned != gold) & (named_pred == gold)).sum()),
        "named_broke": int(((aligned == gold) & (named_pred != gold)).sum()),
        "model": MODEL,
        "protocol_note": "pair-exemplars is clustering; named-desks is zero-shot on the border",
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
