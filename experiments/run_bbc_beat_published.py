#!/usr/bin/env python3
"""Laboratory shots at published GCC* on BBC News. Not the CGE product.

Published (Keraghel & Nadif, arXiv:2412.14867, Dec 2024):
  GCC* NMI=0.9512 ARI=0.9666 ACC=0.9761
  Their KMeans on OpenAI text-embedding-3-small: ACC=0.9301 ARI=0.9101

What they do that we have not yet measured here:
  1. Full-document embeddings (their model keeps 8191 tokens; mpnet cuts at 384).
  2. Linear graph autoencoder + factorial k-means (joint W, G, F), not vanilla KMeans.
  3. Residual one-hop mix on a sparse entity graph, not T^p diffusion.
  4. Title/lede emphasis (BBC files start with a headline).
  5. Consensus of several views, then self-training on confident docs.

    .venv/bin/python experiments/run_bbc_beat_published.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, diags
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_public_benchmarks import DATA, RNG, clean, load_bbc  # noqa: E402
from run_bbc_gcc_links import synsets  # noqa: E402

OUT = HERE / "bbc_beat_published.json"
CACHE = DATA / "emb_cache"
NER_PATH = DATA / "bbc_ner.json"
PUB = {"nmi": 0.9512, "ari": 0.9666, "acc": 0.9761, "paper": "arXiv:2412.14867 Dec 2024 GCC*"}


def hungarian_acc(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    classes, clusters = np.unique(y), np.unique(pred)
    m = np.zeros((len(clusters), len(classes)), dtype=int)
    ci = {c: i for i, c in enumerate(clusters)}
    ki = {c: i for i, c in enumerate(classes)}
    for a, b in zip(pred, y):
        m[ci[a], ki[b]] += 1
    r, c = linear_sum_assignment(m.max() - m)
    return float(m[r, c].sum() / len(y))


def score(name: str, pred, gold) -> dict:
    row = {
        "method": name,
        "nmi": round(float(normalized_mutual_info_score(gold, pred)), 4),
        "ari": round(float(adjusted_rand_score(gold, pred)), 4),
        "acc": round(hungarian_acc(gold, pred), 4),
        "source": "measured",
    }
    beat = row["nmi"] >= PUB["nmi"] and row["ari"] >= PUB["ari"]
    mark = "BEAT" if beat else ("CLOSE" if row["nmi"] >= 0.93 else "")
    print(
        f"{name:48s} NMI={row['nmi']:.4f} ARI={row['ari']:.4f} ACC={row['acc']:.4f} {mark}",
        flush=True,
    )
    return row


def kmeans(Z: np.ndarray, k: int = 5, n_init: int = 10, seed: int = RNG) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit_predict(Z)


def load_titles_and_ledes(texts: list[str]) -> tuple[list[str], list[str]]:
    zpath = DATA / "bbc-fulltext.zip"
    titles, ledes = [], []
    with zipfile.ZipFile(zpath) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".txt") or name.endswith("/"):
                continue
            parts = Path(name).parts
            if len(parts) < 3:
                continue
            raw = zf.read(name).decode("latin-1", errors="replace")
            t = clean(raw)
            if len(t) < 40:
                continue
            first = raw.split("\n", 1)[0].strip()
            titles.append(clean(first) if first else t[:160])
            words = t.split()
            ledes.append(" ".join(words[:80]))
    if len(titles) != len(texts):
        raise RuntimeError(f"title count {len(titles)} != texts {len(texts)}")
    return titles, ledes


def word_chunks(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
        i += size - overlap
    return out


def encode(texts: list[str], cache_key: str, max_len: int = 384) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{cache_key}.npy"
    if path.exists():
        Z = np.load(path)
        if Z.shape[0] == len(texts):
            print(f"cache {cache_key} {Z.shape}", flush=True)
            return np.asarray(Z, dtype=np.float32)
    from sentence_transformers import SentenceTransformer

    print(f"encode {cache_key} n={len(texts)} maxlen={max_len}", flush=True)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    model.max_seq_length = max_len
    Z = np.asarray(
        model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    np.save(path, Z)
    del model
    return Z


def encode_mean_chunks(texts: list[str], cache_key: str) -> np.ndarray:
    path = CACHE / f"{cache_key}.npy"
    if path.exists():
        Z = np.load(path)
        if Z.shape[0] == len(texts):
            print(f"cache {cache_key} {Z.shape}", flush=True)
            return np.asarray(Z, dtype=np.float32)
    flat, owners = [], []
    for i, text in enumerate(texts):
        pieces = word_chunks(text)
        for p in pieces:
            flat.append(p)
            owners.append(i)
    print(f"chunk encode docs={len(texts)} chunks={len(flat)}", flush=True)
    Zc = encode(flat, f"{cache_key}-flat")
    dim = Zc.shape[1]
    acc = np.zeros((len(texts), dim), dtype=np.float64)
    cnt = np.zeros(len(texts), dtype=np.float64)
    for vec, i in zip(Zc, owners):
        acc[i] += vec
        cnt[i] += 1
    Z = acc / np.maximum(cnt[:, None], 1.0)
    Z = normalize(Z).astype(np.float32)
    np.save(path, Z)
    return Z


def entity_matrix(entities: list[list[str]], kinds: set[str], min_pairs: int) -> tuple[np.ndarray, csr_matrix]:
    filtered = [[x for x in doc if x.split(":", 1)[0] in kinds] for doc in entities]
    groups = synsets(filtered)
    df: dict[str, int] = defaultdict(int)
    for g in groups:
        for s in g:
            df[s] += 1
    inv: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        for s in g:
            if df[s] >= 2:
                inv[s].append(i)
    shared: dict[tuple[int, int], int] = defaultdict(int)
    for docs in inv.values():
        if len(docs) > 80:
            continue
        for a in range(len(docs)):
            for b in range(a + 1, len(docs)):
                i, j = docs[a], docs[b]
                key = (i, j) if i < j else (j, i)
                shared[key] += 1
    strong = {pair: k for pair, k in shared.items() if k >= min_pairs}
    used = set()
    for i, j in strong:
        used |= groups[i] & groups[j]
    vocab = {s: k for k, s in enumerate(sorted(used))}
    rows, cols, data = [], [], []
    for i, g in enumerate(groups):
        for s in g:
            j = vocab.get(s)
            if j is not None:
                rows.append(i)
                cols.append(j)
                data.append(1.0)
    n = len(entities)
    X = csr_matrix((data, (rows, cols)), shape=(n, max(len(vocab), 1)))
    n_comp = min(32, max(2, X.shape[1] - 1))
    E = normalize(TruncatedSVD(n_components=n_comp, random_state=RNG).fit_transform(TfidfTransformer().fit_transform(X)))
    A = np.zeros((n, n), dtype=np.float32)
    for (i, j), w in strong.items():
        A[i, j] = A[j, i] = float(w)
    return E.astype(np.float32), csr_matrix(A)


def residual_smooth(Z: np.ndarray, A: csr_matrix, alpha: float) -> np.ndarray:
    A = A.tocsr()
    A = A + A.T
    A.setdiag(0)
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.reciprocal(np.sqrt(np.maximum(deg, 1e-9)))
    S = diags(dinv) @ A @ diags(dinv)
    mix = (1.0 - alpha) * Z + alpha * (S @ Z)
    return normalize(mix).astype(np.float32)


def factorial_kmeans(Y: np.ndarray, k: int = 5, rounds: int = 8) -> np.ndarray:
    """Alternating G, F, W as in GCC eq. (9), without graph powering."""
    d = Y.shape[1]
    e = min(k, d)
    U, _, _ = np.linalg.svd(Y, full_matrices=False)
    # randomized-PCA style init of W from SVD of Y
    _, _, Vt = np.linalg.svd(Y, full_matrices=False)
    W = Vt[:e].T
    pred = kmeans(Y @ W, k=k)
    for _ in range(rounds):
        G = np.zeros((len(Y), k), dtype=np.float64)
        G[np.arange(len(Y)), pred] = 1.0
        Z = Y @ W
        F = np.zeros((k, e), dtype=np.float64)
        for c in range(k):
            idx = pred == c
            if idx.any():
                F[c] = Z[idx].mean(axis=0)
        # orthogonal Procrustes: Y^T G F = U Σ V^T, W ← U V^T
        M = Y.T @ (G @ F)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        W = (U @ Vt)[:, :e]
        pred = kmeans(Y @ W, k=k)
    return pred


def self_train(Z: np.ndarray, gold_k: int = 5, keep: float = 0.70, rounds: int = 6) -> np.ndarray:
    pred = kmeans(Z, k=gold_k)
    for _ in range(rounds):
        cents = np.stack([Z[pred == c].mean(axis=0) if (pred == c).any() else np.zeros(Z.shape[1]) for c in range(gold_k)])
        cents = normalize(cents)
        sims = Z @ cents.T
        assigned = sims[np.arange(len(Z)), pred]
        second = np.partition(sims, -2, axis=1)[:, -2]
        conf = assigned - second
        thr = np.quantile(conf, 1.0 - keep)
        confident = conf >= thr
        if confident.sum() < gold_k * 8:
            break
        cents = np.stack(
            [Z[confident & (pred == c)].mean(axis=0) if (confident & (pred == c)).any() else cents[c] for c in range(gold_k)]
        )
        cents = normalize(cents)
        pred = (Z @ cents.T).argmax(axis=1)
    return pred


def consensus(views: list[np.ndarray], k: int = 5) -> np.ndarray:
    n = views[0].shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    for Z in views:
        pred = kmeans(Z, k=k)
        for c in range(k):
            idx = np.where(pred == c)[0]
            if len(idx) > 1:
                C[np.ix_(idx, idx)] += 1.0
    C /= max(len(views), 1)
    np.fill_diagonal(C, 0.0)
    dist = np.clip(1.0 - C, 0.0, 1.0)
    return AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(dist)


def mutual_knn_spectral(Z: np.ndarray, k: int = 5, knn: int = 12) -> np.ndarray:
    sims = Z @ Z.T
    np.fill_diagonal(sims, -1.0)
    neigh = np.argpartition(-sims, knn, axis=1)[:, :knn]
    A = np.zeros((len(Z), len(Z)), dtype=np.float32)
    for i, js in enumerate(neigh):
        for j in js:
            A[i, int(j)] = 1.0
    mutual = np.minimum(A, A.T)
    return SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        random_state=RNG,
        assign_labels="kmeans",
    ).fit_predict(mutual)


def tfidf_svd(texts: list[str], n_comp: int = 64) -> np.ndarray:
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.7, stop_words="english", max_features=20000).fit_transform(texts)
    return normalize(TruncatedSVD(n_components=n_comp, random_state=RNG).fit_transform(X)).astype(np.float32)


def main() -> int:
    texts, gold, _ = load_bbc()
    rows: list[dict] = []
    Z_mp = normalize(np.load(CACHE / "pack-mpnet-BBC_News.npy").astype(np.float32))
    Z_bg = normalize(np.load(CACHE / "pack-bge-base-BBC_News.npy").astype(np.float32))
    print(f"bbc n={len(texts)} mpnet={Z_mp.shape} bge={Z_bg.shape}", flush=True)

    rows.append(score("mpnet KMeans", kmeans(Z_mp), gold))
    rows.append(score("bge KMeans", kmeans(Z_bg), gold))
    rows.append(score("concat mpnet+bge KMeans", kmeans(normalize(np.hstack([Z_mp, Z_bg]))), gold))

    T = tfidf_svd(texts)
    rows.append(score("mpnet+tfidf64 w=0.25", kmeans(normalize(np.hstack([Z_mp, 0.25 * T]))), gold))
    rows.append(score("mpnet+bge+tfidf64", kmeans(normalize(np.hstack([Z_mp, Z_bg, 0.20 * T]))), gold))

    for keep in (0.60, 0.75, 0.85):
        rows.append(score(f"mpnet self-train keep={keep}", self_train(Z_mp, keep=keep), gold))

    rows.append(score("mpnet factorial-kmeans", factorial_kmeans(Z_mp), gold))
    rows.append(score("concat factorial-kmeans", factorial_kmeans(normalize(np.hstack([Z_mp, Z_bg]))), gold))

    try:
        rows.append(score("mpnet mutual-kNN12 spectral", mutual_knn_spectral(Z_mp, knn=12), gold))
        rows.append(score("mpnet mutual-kNN20 spectral", mutual_knn_spectral(Z_mp, knn=20), gold))
    except Exception as exc:
        print(f"spectral skipped: {exc}", flush=True)

    rows.append(score("consensus mpnet+bge+tfidf", consensus([Z_mp, Z_bg, T]), gold))

    titles, ledes = load_titles_and_ledes(texts)
    Z_title = encode(titles, "bbc-mpnet-titles", max_len=64)
    Z_lede = encode(ledes, "bbc-mpnet-ledes", max_len=128)
    for w in (0.15, 0.30, 0.45):
        rows.append(score(f"mpnet+title w={w}", kmeans(normalize(np.hstack([Z_mp, w * Z_title]))), gold))
        rows.append(score(f"mpnet+lede w={w}", kmeans(normalize(np.hstack([Z_mp, w * Z_lede]))), gold))
    rows.append(score("title only KMeans", kmeans(Z_title), gold))
    rows.append(score("lede only KMeans", kmeans(Z_lede), gold))
    mix = normalize(0.55 * Z_mp + 0.25 * Z_title + 0.20 * Z_lede)
    rows.append(score("blend mpnet/title/lede", kmeans(mix), gold))

    Z_chunk = encode_mean_chunks(texts, "bbc-mpnet-chunk-mean")
    rows.append(score("mpnet chunk-mean KMeans", kmeans(Z_chunk), gold))
    rows.append(score("chunk-mean self-train keep=0.75", self_train(Z_chunk, keep=0.75), gold))
    rows.append(score("chunk-mean factorial-kmeans", factorial_kmeans(Z_chunk), gold))
    rows.append(score("chunk+title w=0.30", kmeans(normalize(np.hstack([Z_chunk, 0.30 * Z_title]))), gold))
    rows.append(score("chunk+mpnet concat", kmeans(normalize(np.hstack([Z_mp, Z_chunk]))), gold))

    entities = json.loads(NER_PATH.read_text())
    E, A = entity_matrix(entities, {"PER", "ORG"}, min_pairs=2)
    print(f"entity features {E.shape} edges={(A.nnz) // 2}", flush=True)
    rows.append(score("mpnet+PER/ORG w=0.30", kmeans(normalize(np.hstack([Z_mp, 0.30 * E]))), gold))
    rows.append(score("chunk+PER/ORG w=0.30", kmeans(normalize(np.hstack([Z_chunk, 0.30 * E]))), gold))
    for alpha in (0.05, 0.10, 0.20):
        Zs = residual_smooth(Z_mp, A, alpha)
        rows.append(score(f"mpnet residual-SGC α={alpha}", kmeans(Zs), gold))
        rows.append(score(f"mpnet residual-SGC α={alpha} factorial", factorial_kmeans(Zs), gold))
        Zc = residual_smooth(Z_chunk, A, alpha)
        rows.append(score(f"chunk residual-SGC α={alpha}", kmeans(Zc), gold))

    best_rep = normalize(np.hstack([Z_chunk, 0.30 * Z_title, 0.20 * E]))
    rows.append(score("chunk+title+entity KMeans", kmeans(best_rep), gold))
    rows.append(score("chunk+title+entity self-train", self_train(best_rep, keep=0.75), gold))
    rows.append(score("chunk+title+entity factorial", factorial_kmeans(best_rep), gold))
    rows.append(score("chunk+title+entity residual0.10", kmeans(residual_smooth(best_rep, A, 0.10)), gold))

    rows.sort(key=lambda r: (-r["nmi"], -r["ari"], -r["acc"]))
    payload = {
        "disclaimer": "Laboratory only. CGE production stays TF-IDF + DBSCAN at cosine 0.75.",
        "published": PUB,
        "n": len(texts),
        "best": rows[0],
        "beat_published": rows[0]["nmi"] >= PUB["nmi"] and rows[0]["ari"] >= PUB["ari"],
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    print(
        f"BEST {rows[0]['method']} NMI={rows[0]['nmi']:.4f} vs published {PUB['nmi']:.4f} "
        f"beat={payload['beat_published']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
