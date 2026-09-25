#!/usr/bin/env python3
"""Replay the production Similaridade rule on downloaded public corpora.

Nothing here is a Ceará count. Every ρ and BCubed value is written only after
the clusterer finishes on texts that live under experiments/data/.

    .venv/bin/python experiments/run_public_benchmarks.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
OUT_JSON = HERE / "public_benchmarks.json"
OUT_FIG = ROOT / "paper" / "figures" / "fig6_public_benchmarks.png"
TAU = 0.75
EPS = 1.0 - TAU
MIN_SAMPLES = 2
RNG = 20260925
MAX_N = 2200


class UnionFind:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.p.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def components(self) -> dict[str, list[str]]:
        g: dict[str, list[str]] = defaultdict(list)
        for x in list(self.p):
            g[self.find(x)].append(x)
        return dict(g)


def bcubed(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float, float]:
    n = len(pred)
    if n == 0:
        return 0.0, 0.0, 0.0
    pred_g: dict[int, list[int]] = defaultdict(list)
    gold_g: dict[int, list[int]] = defaultdict(list)
    for i, (p, g) in enumerate(zip(pred, gold)):
        key_p = int(p) if int(p) >= 0 else -(i + 1)
        pred_g[key_p].append(i)
        gold_g[int(g)].append(i)
    p_map = {i: k for k, ids in pred_g.items() for i in ids}
    g_map = {i: k for k, ids in gold_g.items() for i in ids}
    prec = rec = 0.0
    for i in range(n):
        mates_p = pred_g[p_map[i]]
        mates_g = gold_g[g_map[i]]
        same = sum(1 for j in mates_p if g_map[j] == g_map[i])
        prec += same / len(mates_p)
        rec += same / len(mates_g)
    p, r = prec / n, rec / n
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return float(p), float(r), float(f1)


def mean_pairwise_cosine(X, idx: list[int]) -> float:
    if len(idx) < 2:
        return 0.0
    block = cosine_similarity(X[idx])
    iu = np.triu_indices(len(idx), k=1)
    return float(block[iu].mean())


def run_stack(texts: list[str], tau: float = TAU) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9, stop_words="english")
    X = vec.fit_transform(texts)
    raw = DBSCAN(eps=1.0 - tau, min_samples=MIN_SAMPLES, metric="cosine", n_jobs=-1).fit_predict(X)
    labels = np.array(raw, dtype=int)
    accepted = labels.copy()
    for lab in set(labels.tolist()):
        if lab < 0:
            continue
        idx = np.where(labels == lab)[0].tolist()
        if len(idx) < 2 or mean_pairwise_cosine(X, idx) < tau:
            accepted[idx] = -1
    grouped = int((accepted >= 0).sum())
    n = len(texts)
    return {
        "n": n,
        "grouped": grouped,
        "rho": grouped / n if n else 0.0,
        "n_clusters": int(len({int(x) for x in accepted if x >= 0})),
        "leftovers": n - grouped,
        "labels": accepted,
    }


def clean(text: str) -> str:
    return " ".join(text.split())


def load_20ng() -> tuple[list[str], np.ndarray, str]:
    cats = (
        "rec.autos",
        "sci.space",
        "talk.politics.mideast",
        "misc.forsale",
        "comp.graphics",
        "sci.med",
    )
    data = fetch_20newsgroups(
        subset="train",
        categories=list(cats),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RNG,
    )
    by: dict[int, list[str]] = defaultdict(list)
    for text, y in zip(data.data, data.target):
        t = clean(text)
        if len(t) >= 40:
            by[int(y)].append(t)
    texts, gold = [], []
    for lab, items in sorted(by.items()):
        take = items[:120]
        texts.extend(take)
        gold.extend([lab] * len(take))
    note = f"sklearn fetch_20newsgroups train, 6 groups, first 120 texts/group after length filter, seed={RNG}"
    return texts, np.array(gold, dtype=int), note


def load_bbc() -> tuple[list[str], np.ndarray, str]:
    zpath = DATA / "bbc-fulltext.zip"
    if not zpath.exists():
        raise FileNotFoundError(zpath)
    texts, gold, names = [], [], []
    with zipfile.ZipFile(zpath) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".txt") or name.endswith("/"):
                continue
            parts = Path(name).parts
            if len(parts) < 3:
                continue
            topic = parts[1]
            raw = zf.read(name).decode("latin-1", errors="replace")
            t = clean(raw)
            if len(t) < 40:
                continue
            if topic not in names:
                names.append(topic)
            texts.append(t)
            gold.append(names.index(topic))
    return texts, np.array(gold, dtype=int), f"bbc-fulltext.zip, {len(names)} topics, all kept files"


def load_reuters() -> tuple[list[str], np.ndarray, str]:
    tpath = DATA / "reuters21578.tar.gz"
    if not tpath.exists():
        raise FileNotFoundError(tpath)
    topic_re = re.compile(r"<TOPICS>(.*?)</TOPICS>", re.S)
    d_re = re.compile(r"<D>(.*?)</D>", re.S)
    title_re = re.compile(r"<TITLE>(.*?)</TITLE>", re.S)
    body_re = re.compile(r"<BODY>(.*?)</BODY>", re.S)
    texts, gold_names = [], []
    with tarfile.open(tpath, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("reut2-") and m.name.endswith(".sgm")]
        for m in sorted(members, key=lambda x: x.name):
            raw = tar.extractfile(m).read().decode("latin-1", errors="replace")
            for block in raw.split("<REUTERS")[1:]:
                topics = topic_re.search(block)
                if not topics:
                    continue
                labs = [x.strip() for x in d_re.findall(topics.group(1)) if x.strip()]
                if len(labs) != 1:
                    continue
                title = title_re.search(block)
                body = body_re.search(block)
                t = clean(((title.group(1) if title else "") + " " + (body.group(1) if body else "")))
                if len(t) < 40:
                    continue
                texts.append(t)
                gold_names.append(labs[0])
    # keep the most frequent single topics so n stays measurable on this Mac
    counts = defaultdict(int)
    for g in gold_names:
        counts[g] += 1
    keep = {k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:8]}
    texts2, gold2, names = [], [], []
    rng = np.random.RandomState(RNG)
    by = defaultdict(list)
    for t, g in zip(texts, gold_names):
        if g in keep:
            by[g].append(t)
    for g in sorted(keep):
        items = by[g]
        rng.shuffle(items)
        take = items[: min(200, len(items))]
        if g not in names:
            names.append(g)
        texts2.extend(take)
        gold2.extend([names.index(g)] * len(take))
    note = (
        f"reuters21578.tar.gz, single-topic BODY+TITLE, top-8 topics, "
        f"up to 200 docs/topic, seed={RNG}, topics={names}"
    )
    return texts2, np.array(gold2, dtype=int), note


def load_qqp() -> tuple[list[str], np.ndarray, str]:
    zpath = DATA / "QQP.zip"
    if not zpath.exists():
        raise FileNotFoundError(zpath)
    uf = UnionFind()
    text_of: dict[str, str] = {}
    with zipfile.ZipFile(zpath) as zf:
        with zf.open("QQP/train.tsv") as fh:
            rows = csv.DictReader((line.decode("utf-8", errors="replace") for line in fh), delimiter="\t")
            for row in rows:
                if row.get("is_duplicate") != "1":
                    continue
                a, b = row["qid1"], row["qid2"]
                uf.union(a, b)
                text_of.setdefault(a, clean(row.get("question1") or ""))
                text_of.setdefault(b, clean(row.get("question2") or ""))
    comps = [ids for ids in uf.components().values() if len(ids) >= 2]
    comps.sort(key=lambda c: (-len(c), min(c)))
    texts, gold = [], []
    for i, ids in enumerate(comps):
        if len(texts) >= MAX_N:
            break
        for qid in ids:
            t = text_of.get(qid, "")
            if len(t) < 8:
                continue
            texts.append(t)
            gold.append(i)
    note = (
        f"QQP/train.tsv duplicate=1 pairs only, connected components, "
        f"first families until n~{MAX_N}, seed unused for pair filter"
    )
    return texts, np.array(gold, dtype=int), note


def load_cqadupstack() -> tuple[list[str], np.ndarray, str]:
    folder = DATA / "cqadupstack-webmasters"
    corpus_p = folder / "corpus.jsonl"
    queries_p = folder / "queries.jsonl"
    qrels_p = folder / "qrels.tsv"
    for p in (corpus_p, queries_p, qrels_p):
        if not p.exists():
            raise FileNotFoundError(p)
    texts_by: dict[str, str] = {}
    for path in (corpus_p, queries_p):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            tid = str(obj.get("_id") or obj.get("id"))
            t = clean((obj.get("title") or "") + " " + (obj.get("text") or ""))
            if t:
                texts_by[tid] = t
    uf = UnionFind()
    with qrels_p.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            q, d = str(row["query-id"]), str(row["corpus-id"])
            if int(float(row.get("score") or 0)) <= 0:
                continue
            uf.union(q, d)
    comps = [ids for ids in uf.components().values() if len(ids) >= 2]
    comps.sort(key=lambda c: (-len(c), min(c)))
    texts, gold = [], []
    missing = 0
    for i, ids in enumerate(comps):
        for tid in ids:
            t = texts_by.get(tid, "")
            if len(t) < 8:
                missing += 1
                continue
            texts.append(t)
            gold.append(i)
    note = (
        f"mteb/cqadupstack-webmasters corpus+queries+qrels, "
        f"only ids in a duplicate component, missing_text={missing}"
    )
    return texts, np.array(gold, dtype=int), note


def summarize(name: str, gold_kind: str, texts, gold, note: str) -> dict:
    print(f"\n=== {name}  n={len(texts)}  gold={gold_kind} ===", flush=True)
    print(note, flush=True)
    stack = run_stack(texts)
    p, r, f1 = bcubed(stack["labels"], gold)
    row = {
        "name": name,
        "gold_kind": gold_kind,
        "n": stack["n"],
        "grouped": stack["grouped"],
        "rho": round(stack["rho"], 4),
        "n_clusters": stack["n_clusters"],
        "leftovers": stack["leftovers"],
        "gold_families": int(len(set(gold.tolist()))),
        "bcubed_p": round(p, 4),
        "bcubed_r": round(r, 4),
        "bcubed_f1": round(f1, 4),
        "tau": TAU,
        "eps": EPS,
        "note": note,
        "source": "measured",
    }
    print(
        f"MEASURED  ρ={row['rho']:.4f}  grouped={row['grouped']}/{row['n']}  "
        f"clusters={row['n_clusters']}  leftovers={row['leftovers']}  "
        f"BCubed P/R/F1={row['bcubed_p']:.4f}/{row['bcubed_r']:.4f}/{row['bcubed_f1']:.4f}",
        flush=True,
    )
    return row


def render(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "paper" / "figures" / ".mpl"))
    labels = [r["name"] for r in rows]
    rhos = [100 * r["rho"] for r in rows]
    f1s = [100 * r["bcubed_f1"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    y = np.arange(len(rows))
    h = 0.36
    ax.barh(y + h / 2, rhos, h, color="#2c5aa0", label="Grouping rate ρ (%)")
    ax.barh(y - h / 2, f1s, h, color="#1f6f63", label="BCubed F1 vs that gold (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percent")
    ax.set_title("Measured production rule on downloaded public sets")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    plt.close()


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    jobs = (
        ("20 Newsgroups", "theme", load_20ng),
        ("BBC News", "theme", load_bbc),
        ("Reuters-21578", "theme", load_reuters),
        ("Quora QQP dups", "duplicate-question", load_qqp),
        ("CQADupStack webmasters", "duplicate-question", load_cqadupstack),
    )
    rows = []
    failed = []
    for name, kind, loader in jobs:
        try:
            texts, gold, note = loader()
        except Exception as exc:
            print(f"SKIP {name}: {exc}", flush=True)
            failed.append({"name": name, "error": str(exc)})
            continue
        if len(texts) < 20:
            print(f"SKIP {name}: too few texts ({len(texts)})", flush=True)
            failed.append({"name": name, "error": f"n={len(texts)}"})
            continue
        rows.append(summarize(name, kind, texts, gold, note))
    payload = {
        "rule": {
            "encoder": "TF-IDF unigrams–trigrams, L2",
            "clusterer": "DBSCAN, cosine distance",
            "eps": EPS,
            "min_samples": MIN_SAMPLES,
            "accept": f"|C|>=2 and mean pairwise cosine >= {TAU}",
        },
        "source": "measured on files in experiments/data and sklearn 20 Newsgroups",
        "not_run": [
            "Events 2012 (tweet-id hydration; no Twitter API in this run)",
        ],
        "failed": failed,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        render(rows)
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
