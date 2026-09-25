#!/usr/bin/env python3
"""Pre-registered alternatives toward published scores. Not the CGE product.

Same five packs (BCubed, our subset — not a leaderboard):
  encoders: MiniLM, mpnet, bge-base
  theme:    KMeans(k=#classes), agglomerative average cosine
  duplicate: HDBSCAN(min_cluster_size=2), cosine graph at τ=0.80

Protocol-shaped checks (still not a submission to the official board):
  20 Newsgroups test, all 20 classes, KMeans(k=20), NMI
  QQP train pairs, balanced sample, τ tuned on 20% and scored on 80%
  QQP same split, logistic regression on pair features (supervised head)
  CQADupStack webmasters retrieval, nDCG@10

    .venv/bin/python experiments/run_sota_alternatives.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_public_benchmarks import (  # noqa: E402
    DATA,
    RNG,
    UnionFind,
    bcubed,
    clean,
    load_20ng,
    load_bbc,
    load_cqadupstack,
    load_qqp,
    load_reuters,
)

OUT_JSON = HERE / "sota_alternatives.json"
CACHE = DATA / "emb_cache"
DUP_TAU = 0.80
EMBEDDERS = (
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", ""),
    ("mpnet", "sentence-transformers/all-mpnet-base-v2", ""),
    ("bge-base", "BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
)


def device() -> str:
    # MPS on this Mac stalled inside a GPU→CPU copy during bge encoding (9 GB, ~1% CPU).
    return "cpu"


def embed_texts(model_name: str, texts: list[str], cache_key: str, prefix: str = "") -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{cache_key}.npy"
    if path.exists():
        Z = np.load(path)
        if Z.shape[0] == len(texts):
            print(f"cache {cache_key} {Z.shape}", flush=True)
            return Z
    from sentence_transformers import SentenceTransformer

    print(f"encode {cache_key} n={len(texts)} …", flush=True)
    model = SentenceTransformer(model_name, device=device())
    payload = [prefix + t for t in texts] if prefix else texts
    Z = model.encode(
        payload,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    Z = np.asarray(Z, dtype=np.float32)
    np.save(path, Z)
    del model
    return Z


def score_partition(labels: np.ndarray, gold: np.ndarray) -> dict:
    p, r, f1 = bcubed(labels, gold)
    return {
        "bcubed_p": round(p, 4),
        "bcubed_r": round(r, 4),
        "bcubed_f1": round(f1, 4),
        "ami": round(float(adjusted_mutual_info_score(gold, labels)), 4),
        "nmi": round(float(normalized_mutual_info_score(gold, labels)), 4),
    }


def cluster_theme(Z: np.ndarray, gold: np.ndarray, how: str) -> np.ndarray:
    k = int(len(set(gold.tolist())))
    if how == "kmeans":
        return KMeans(n_clusters=k, random_state=RNG, n_init=10).fit_predict(Z).astype(int)
    if how == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(Z).astype(int)
    raise ValueError(how)


def cluster_hdbscan(Z: np.ndarray) -> np.ndarray:
    import hdbscan

    raw = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(Z)
    labels = np.asarray(raw, dtype=int)
    bc = labels.copy()
    next_id = int(bc.max()) + 1 if (bc >= 0).any() else 0
    for i, lab in enumerate(bc):
        if lab < 0:
            bc[i] = next_id
            next_id += 1
    return bc


def cluster_cosine_graph(Z: np.ndarray, tau: float = DUP_TAU) -> np.ndarray:
    n = Z.shape[0]
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Chunk the Gram matrix so n=2225 stays in memory.
    step = 512
    for i0 in range(0, n, step):
        block = Z[i0 : i0 + step] @ Z.T
        hits = np.argwhere(block >= tau)
        for r, c in hits:
            i = i0 + int(r)
            j = int(c)
            if i < j:
                union(i, j)
    labels = np.array([find(i) for i in range(n)], dtype=int)
    # Compact ids; leave true singletons as their own id (full partition).
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(int)


def run_same_packs() -> list[dict]:
    jobs = (
        ("20 Newsgroups", "theme", load_20ng),
        ("BBC News", "theme", load_bbc),
        ("Reuters-21578", "theme", load_reuters),
        ("Quora QQP dups", "duplicate-question", load_qqp),
        ("CQADupStack webmasters", "duplicate-question", load_cqadupstack),
    )
    loaded = []
    for name, kind, loader in jobs:
        texts, gold, note = loader()
        loaded.append((name, kind, texts, gold, note))
        print(f"loaded {name} n={len(texts)}", flush=True)

    rows = []
    for enc_name, model_name, _prefix in EMBEDDERS:
        for name, kind, texts, gold, note in loaded:
            Z = embed_texts(model_name, texts, f"pack-{enc_name}-{name.replace(' ', '_')}")
            methods = ("kmeans", "agglomerative") if kind == "theme" else ("hdbscan", "cosine_graph_0.80")
            for how in methods:
                if how == "kmeans":
                    labels = cluster_theme(Z, gold, "kmeans")
                elif how == "agglomerative":
                    labels = cluster_theme(Z, gold, "agglomerative")
                elif how == "hdbscan":
                    labels = cluster_hdbscan(Z)
                else:
                    labels = cluster_cosine_graph(Z, DUP_TAU)
                row = {
                    "probe": "same_packs_bcubed",
                    "pack": name,
                    "gold_kind": kind,
                    "encoder": enc_name,
                    "method": how,
                    "n": len(texts),
                    "note": note,
                    **score_partition(labels, gold),
                    "source": "measured",
                }
                print(
                    f"{name} {enc_name} {how}  F1={row['bcubed_f1']:.4f} NMI={row['nmi']:.4f}",
                    flush=True,
                )
                rows.append(row)
    return rows


def run_20ng_full() -> list[dict]:
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RNG,
    )
    texts, gold = [], []
    for text, y in zip(data.data, data.target):
        t = clean(text)
        if not t:
            continue
        texts.append(t)
        gold.append(int(y))
    gold_a = np.array(gold, dtype=int)
    rows = []
    for enc_name, model_name, _prefix in EMBEDDERS:
        Z = embed_texts(model_name, texts, f"20ng-test-{enc_name}")
        labels = KMeans(n_clusters=20, random_state=RNG, n_init=10).fit_predict(Z).astype(int)
        row = {
            "probe": "20ng_test_k20",
            "encoder": enc_name,
            "method": "kmeans_k20",
            "n": len(texts),
            "n_classes": 20,
            "note": "sklearn 20 Newsgroups test, headers/footers/quotes removed, empty dropped",
            **score_partition(labels, gold_a),
            "source": "measured",
        }
        print(f"20NG test {enc_name} NMI={row['nmi']:.4f} F1={row['bcubed_f1']:.4f}", flush=True)
        rows.append(row)
    return rows


def load_qqp_pairs(n_pos: int = 4000, n_neg: int = 4000) -> tuple[list[str], list[str], np.ndarray]:
    rng = np.random.RandomState(RNG)
    pos, neg = [], []
    zpath = DATA / "QQP.zip"
    import zipfile

    # Reservoir over the whole train file, then keep the requested counts.
    seen_pos = seen_neg = 0
    with zipfile.ZipFile(zpath) as zf:
        with zf.open("QQP/train.tsv") as fh:
            rows = csv.DictReader((line.decode("utf-8", errors="replace") for line in fh), delimiter="\t")
            for row in rows:
                q1 = clean(row.get("question1") or "")
                q2 = clean(row.get("question2") or "")
                if len(q1) < 8 or len(q2) < 8:
                    continue
                if row.get("is_duplicate") == "1":
                    seen_pos += 1
                    if len(pos) < n_pos:
                        pos.append((q1, q2))
                    else:
                        j = int(rng.randint(0, seen_pos))
                        if j < n_pos:
                            pos[j] = (q1, q2)
                else:
                    seen_neg += 1
                    if len(neg) < n_neg:
                        neg.append((q1, q2))
                    else:
                        j = int(rng.randint(0, seen_neg))
                        if j < n_neg:
                            neg[j] = (q1, q2)
    a = [p[0] for p in pos] + [p[0] for p in neg]
    b = [p[1] for p in pos] + [p[1] for p in neg]
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=int)
    return a, b, y


def pair_f1(score: np.ndarray, y: np.ndarray, idx_tune: np.ndarray, idx_test: np.ndarray) -> dict:
    grid = np.round(np.arange(0.50, 0.96, 0.01), 2)
    best_t, best_f = 0.80, -1.0
    for t in grid:
        pred = (score[idx_tune] >= t).astype(int)
        f = f1_score(y[idx_tune], pred, zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    pred = (score[idx_test] >= best_t).astype(int)
    return {
        "tau": best_t,
        "tune_f1": round(best_f, 4),
        "test_f1": round(float(f1_score(y[idx_test], pred, zero_division=0)), 4),
        "test_acc": round(float(accuracy_score(y[idx_test], pred)), 4),
    }


def run_qqp_pairs() -> list[dict]:
    a, b, y = load_qqp_pairs()
    idx = np.arange(len(y))
    idx_tune, idx_test = train_test_split(idx, test_size=0.8, random_state=RNG, stratify=y)
    rows = []
    for enc_name, model_name, prefix in EMBEDDERS:
        # Pair task: same prefix on both sides. bge uses its retrieval instruction.
        Za = embed_texts(model_name, a, f"qqp-a-{enc_name}", prefix=prefix)
        Zb = embed_texts(model_name, b, f"qqp-b-{enc_name}", prefix=prefix)
        score = np.sum(Za * Zb, axis=1)
        metrics = pair_f1(score, y, idx_tune, idx_test)
        row = {
            "probe": "qqp_pair_cosine",
            "encoder": enc_name,
            "method": "cosine_tau_tuned_on_20pct",
            "n_pairs": int(len(y)),
            "n_pos": int(y.sum()),
            "note": "QQP train.tsv balanced sample, not the GLUE test set",
            **metrics,
            "source": "measured",
        }
        print(f"QQP cosine {enc_name} test F1={row['test_f1']:.4f} acc={row['test_acc']:.4f}", flush=True)
        rows.append(row)
        if enc_name == "bge-base":
            feat = np.hstack([Za, Zb, np.abs(Za - Zb), Za * Zb])
            clf = LogisticRegression(max_iter=400, random_state=RNG)
            clf.fit(feat[idx_tune], y[idx_tune])
            pred = clf.predict(feat[idx_test])
            sup = {
                "probe": "qqp_pair_logreg",
                "encoder": enc_name,
                "method": "logreg_on_pair_features_fit_on_20pct",
                "n_pairs": int(len(y)),
                "test_f1": round(float(f1_score(y[idx_test], pred, zero_division=0)), 4),
                "test_acc": round(float(accuracy_score(y[idx_test], pred)), 4),
                "note": "QQP train.tsv balanced sample, not the GLUE test set",
                "source": "measured",
            }
            print(f"QQP logreg {enc_name} test F1={sup['test_f1']:.4f}", flush=True)
            rows.append(sup)
    return rows


def ndcg_at_10(ranked_rel: np.ndarray) -> float:
    k = 10
    rel = ranked_rel[:k].astype(np.float64)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discounts).sum())
    ideal = np.sort(ranked_rel)[::-1][:k]
    idcg = float((ideal * (1.0 / np.log2(np.arange(2, ideal.size + 2)))).sum())
    return 0.0 if idcg == 0 else dcg / idcg


def run_cqa_retrieval() -> list[dict]:
    folder = DATA / "cqadupstack-webmasters"
    corpus_ids, corpus_texts = [], []
    for line in (folder / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        tid = str(obj.get("_id") or obj.get("id"))
        t = clean((obj.get("title") or "") + " " + (obj.get("text") or ""))
        if t:
            corpus_ids.append(tid)
            corpus_texts.append(t)
    queries = []
    for line in (folder / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        qid = str(obj.get("_id") or obj.get("id"))
        t = clean((obj.get("title") or "") + " " + (obj.get("text") or ""))
        if t:
            queries.append((qid, t))
    qrels: dict[str, dict[str, int]] = {}
    with (folder / "qrels.tsv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if int(float(row.get("score") or 0)) <= 0:
                continue
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = 1
    id_index = {tid: i for i, tid in enumerate(corpus_ids)}
    rows = []
    for enc_name, model_name, prefix in EMBEDDERS:
        Zc = embed_texts(model_name, corpus_texts, f"cqa-corpus-{enc_name}")
        Zq = embed_texts(model_name, [t for _, t in queries], f"cqa-queries-{enc_name}", prefix=prefix)
        scores = []
        for i, (qid, _) in enumerate(queries):
            rel_map = qrels.get(qid)
            if not rel_map:
                continue
            sims = Zq[i] @ Zc.T
            order = np.argsort(-sims)
            ranked_rel = np.array(
                [1 if corpus_ids[j] in rel_map else 0 for j in order[:50]],
                dtype=np.int8,
            )
            # ideal uses the full relevant count, not only the top 50
            n_rel = sum(1 for cid in rel_map if cid in id_index)
            full = np.zeros(max(n_rel, ranked_rel.size), dtype=np.float64)
            full[: ranked_rel.size] = ranked_rel
            # Put the remaining relevant mass into the ideal only: ndcg_at_10
            # needs the ideal ordering of all judgments. If a relevant doc is
            # outside the top 50, DCG misses it and IDCG still counts it.
            if n_rel > int(ranked_rel.sum()):
                extra = n_rel - int(ranked_rel.sum())
                tail = np.ones(extra, dtype=np.float64)
                judged = np.concatenate([ranked_rel.astype(np.float64), tail])
            else:
                judged = ranked_rel.astype(np.float64)
            scores.append(ndcg_at_10(judged))
        nd = float(np.mean(scores)) if scores else 0.0
        row = {
            "probe": "cqa_webmasters_ndcg10",
            "encoder": enc_name,
            "method": "cosine_retrieval",
            "n_queries": len(scores),
            "n_corpus": len(corpus_texts),
            "ndcg_at_10": round(nd, 4),
            "note": "one CQADupStack forum, not the 12-forum MTEB average",
            "source": "measured",
        }
        print(f"CQA nDCG@10 {enc_name} {row['ndcg_at_10']:.4f} queries={len(scores)}", flush=True)
        rows.append(row)
    return rows


def main() -> int:
    payload = {
        "disclaimer": (
            "Laboratory only. CGE production stays TF-IDF+DBSCAN. "
            "None of these probes is an official GLUE or MTEB submission."
        ),
        "published_reference": {
            "qqp_glue": "pair classification; recent fine-tunes near 92% accuracy and 89 F1 on GLUE-style QQP",
            "20ng_clustering": "SBERT trained for clustering reported NMI 0.725 on 20 Newsgroups test (Learn the Big Picture, REPL4NLP 2021, SBERT-COB)",
            "cqa_mteb": "retrieval nDCG@10 averaged over 12 forums, not BCubed",
        },
        "rows": [],
    }
    payload["rows"].extend(run_same_packs())
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["rows"].extend(run_20ng_full())
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["rows"].extend(run_qqp_pairs())
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["rows"].extend(run_cqa_retrieval())
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
