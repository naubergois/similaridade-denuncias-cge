#!/usr/bin/env python3
"""BBC vs GCC*. Keep mpnet. Link docs the way the paper does, as a low-weight feature.

Paper rule (arXiv 2412.14867): same entity type, cosine > 0.9, at least three such pairs.
The link is concatenated with the document vector. It does not average neighbor vectors.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_public_benchmarks import RNG, load_bbc  # noqa: E402

NER_PATH = HERE / "data" / "bbc_deberta_ner.json"
MODEL = "lambdavi/deberta-v3-base_conll03"


def chunks(text: str, size: int = 1200, overlap: int = 150):
    if len(text) <= size:
        return [text]
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        if i + size >= len(text):
            break
        i += size - overlap
    return out


def extract(texts: list[str]) -> list[list[str]]:
    if NER_PATH.exists():
        ents = json.loads(NER_PATH.read_text())
        print(f"loaded {NER_PATH} n={len(ents)}", flush=True)
        return ents
    from transformers import pipeline

    print(f"loading {MODEL}", flush=True)
    ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple", device=-1)
    flat, owners = [], []
    for di, text in enumerate(texts):
        for piece in chunks(text):
            flat.append(piece)
            owners.append(di)
    found = [set() for _ in texts]
    for i in range(0, len(flat), 8):
        outs = ner(flat[i : i + 8])
        for doc, di in zip(outs, owners[i : i + 8]):
            for ent in doc:
                group = ent["entity_group"]
                if group not in {"PER", "ORG", "LOC"}:
                    continue
                name = " ".join(ent["word"].replace("##", "").split()).lower().strip(" .")
                if len(name) < 3 or name in {"bbc", "uk", "us", "un"}:
                    continue
                found[di].add(f"{group}:{name}")
        if i and i % 400 == 0:
            print(f"ner chunks {i}/{len(flat)}", flush=True)
    ents = [sorted(s) for s in found]
    NER_PATH.parent.mkdir(parents=True, exist_ok=True)
    NER_PATH.write_text(json.dumps(ents))
    print(f"saved {NER_PATH}", flush=True)
    return ents


def acc(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    classes, clusters = np.unique(y), np.unique(pred)
    m = np.zeros((len(clusters), len(classes)), dtype=int)
    ci = {c: i for i, c in enumerate(clusters)}
    ki = {c: i for i, c in enumerate(classes)}
    for a, b in zip(pred, y):
        m[ci[a], ki[b]] += 1
    r, c = linear_sum_assignment(m.max() - m)
    return float(m[r, c].sum() / len(y))


def synsets(entities: list[list[str]]) -> list[set[str]]:
    """Same type and mpnet cosine > 0.9 become one link token."""
    from sentence_transformers import SentenceTransformer

    by_type: dict[str, dict[str, int]] = defaultdict(dict)
    for doc in entities:
        for item in doc:
            kind, name = item.split(":", 1)
            by_type[kind].setdefault(name, len(by_type[kind]))
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for kind, names in by_type.items():
        keys = list(names)
        if len(keys) < 2:
            continue
        vec = model.encode(keys, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)
        print(f"entity cosine {kind} n={len(keys)}", flush=True)
        block = 256
        for i0 in range(0, len(keys), block):
            sims = vec[i0 : i0 + block] @ vec.T
            hits = np.argwhere(sims > 0.9)
            for r, c in hits:
                i = i0 + int(r)
                j = int(c)
                if i < j:
                    union(f"{kind}:{keys[i]}", f"{kind}:{keys[j]}")
    groups = [set() for _ in entities]
    for di, doc in enumerate(entities):
        for item in doc:
            groups[di].add(find(item))
    return groups


def main() -> int:
    texts, gold, _ = load_bbc()
    Z = normalize(np.load(HERE / "data" / "emb_cache" / "pack-mpnet-BBC_News.npy").astype(np.float32))
    entities = extract(texts)
    groups = synsets(entities)
    # document feature = synsets that take part in at least one 3-pair link
    df = defaultdict(int)
    for g in groups:
        for s in g:
            df[s] += 1
    # count pairwise shared synsets via inverted index, keep those with count >= 3
    inv = defaultdict(list)
    for i, g in enumerate(groups):
        for s in g:
            if df[s] >= 2:
                inv[s].append(i)
    shared = defaultdict(int)
    for docs in inv.values():
        if len(docs) > 80:
            continue
        for a in range(len(docs)):
            for b in range(a + 1, len(docs)):
                i, j = docs[a], docs[b]
                key = (i, j) if i < j else (j, i)
                shared[key] += 1
    strong = {pair for pair, k in shared.items() if k >= 3}
    print(f"doc pairs with >=3 entity links: {len(strong)}", flush=True)
    # features: synsets that participate in a strong pair, idf-like
    used = set()
    for i, j in strong:
        used |= groups[i] & groups[j]
    vocab = {s: k for k, s in enumerate(sorted(used))}
    print(f"synsets used as features: {len(vocab)}", flush=True)
    rows, cols, data = [], [], []
    for i, g in enumerate(groups):
        for s in g:
            j = vocab.get(s)
            if j is not None:
                rows.append(i)
                cols.append(j)
                data.append(1.0)
    X = csr_matrix((data, (rows, cols)), shape=(len(texts), max(len(vocab), 1)))
    n_comp = min(32, max(2, X.shape[1] - 1))
    E = normalize(TruncatedSVD(n_components=n_comp, random_state=RNG).fit_transform(TfidfTransformer().fit_transform(X))).astype(np.float32)

    def acc(y, pred) -> float:
        y, pred = np.asarray(y), np.asarray(pred)
        classes, clusters = np.unique(y), np.unique(pred)
        m = np.zeros((len(clusters), len(classes)), dtype=int)
        ci = {c: i for i, c in enumerate(clusters)}
        ki = {c: i for i, c in enumerate(classes)}
        for a, b in zip(pred, y):
            m[ci[a], ki[b]] += 1
        r, c = linear_sum_assignment(m.max() - m)
        return float(m[r, c].sum() / len(y))

    def report(name, pred):
        nmi = float(normalized_mutual_info_score(gold, pred))
        ari = float(adjusted_rand_score(gold, pred))
        a = acc(gold, pred)
        print(f"{name} NMI={nmi:.4f} ARI={ari:.4f} ACC={a:.4f}", flush=True)

    report("mpnet only", KMeans(n_clusters=5, random_state=RNG, n_init=10).fit_predict(Z))
    for w in (0.15, 0.30):
        mix = normalize(np.hstack([Z, w * E]))
        pred = KMeans(n_clusters=5, random_state=RNG, n_init=10).fit_predict(mix)
        report(f"mpnet + 3-link features w={w}", pred)
    print("published GCC* NMI=0.9512 ARI=0.9666 ACC=0.9761", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
