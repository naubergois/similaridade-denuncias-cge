#!/usr/bin/env python3
"""Game-theoretic partitions on the cached bge vectors for 20 Newsgroups.

  dominant sets: replicator dynamics (evolutionary stable clusters)
  hedonic Nash: each text moves to the cluster that pays it more
  Shapley reweight: feature weight = how much that axis wins a linear game on the train split

Test labels are not used to fit. Target NMI is 0.725.

    .venv/bin/python experiments/run_game_theory_20ng.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestCentroid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_public_benchmarks import RNG  # noqa: E402
from run_trained_alternatives import load_20ng_split  # noqa: E402

CACHE = HERE / "data" / "emb_cache"
OUT = HERE / "game_theory_20ng.json"
TARGET = 0.725


def nmi(gold, labels) -> float:
    return float(normalized_mutual_info_score(gold, np.asarray(labels)))


def dominant_sets(Z: np.ndarray, max_sets: int = 40) -> np.ndarray:
    """Peel evolutionary stable subsets from a nonnegative similarity graph."""
    n = Z.shape[0]
    sim = np.clip(Z @ Z.T, 0, None).astype(np.float32)
    np.fill_diagonal(sim, 0)
    labels = np.full(n, -1, dtype=int)
    alive = np.ones(n, dtype=bool)
    cid = 0
    for _ in range(max_sets):
        idx = np.flatnonzero(alive)
        if len(idx) < 8:
            break
        A = sim[np.ix_(idx, idx)]
        x = np.full(len(idx), 1.0 / len(idx), dtype=np.float64)
        for _it in range(80):
            Ax = A @ x
            x = x * Ax
            s = x.sum()
            if s <= 1e-12:
                break
            x /= s
            if np.nanmax(np.abs(x * (Ax / s) - x)) < 1e-4:
                pass
        support = x > (1.0 / len(idx))
        if int(support.sum()) < 5:
            support = x >= np.quantile(x, 0.8)
        if int(support.sum()) < 5:
            break
        labels[idx[support]] = cid
        alive[idx[support]] = False
        cid += 1
        print(f"dominant set {cid} size {int(support.sum())} left {int(alive.sum())}", flush=True)
    # Leftover texts join the nearest extracted coalition.
    if alive.any() and (labels >= 0).any():
        means = []
        ids = sorted(set(labels[labels >= 0].tolist()))
        for c in ids:
            means.append(Z[labels == c].mean(axis=0))
        M = np.vstack(means)
        M /= np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-8)
        left = np.flatnonzero(alive)
        choice = (Z[left] @ M.T).argmax(axis=1)
        labels[left] = np.array(ids)[choice]
    return labels


def hedonic(Z: np.ndarray, k: int = 20, rounds: int = 15, rival: float = 0.5) -> np.ndarray:
    """Each text is a player. Payoff is closeness to its coalition minus closeness to the rival."""
    labels = KMeans(n_clusters=k, random_state=RNG, n_init=10).fit_predict(Z)
    for r in range(rounds):
        means = []
        for c in range(k):
            members = Z[labels == c]
            means.append(members.mean(axis=0) if len(members) else np.zeros(Z.shape[1]))
        M = np.vstack(means)
        M /= np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-8)
        sims = Z @ M.T
        own = sims[np.arange(len(Z)), labels]
        rival_pay = sims.copy()
        rival_pay[np.arange(len(Z)), labels] = -1.0
        payoff = sims - rival * rival_pay.max(axis=1, keepdims=True)
        new_labels = payoff.argmax(axis=1)
        moved = int((new_labels != labels).sum())
        labels = new_labels
        print(f"hedonic round {r + 1} moved {moved}", flush=True)
        if moved == 0:
            break
    return labels


def shapley_reweight(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo Shapley over feature blocks, using a linear classifier as the game's value."""
    rng = np.random.RandomState(RNG)
    n_blocks = 16
    dim = Ztr.shape[1]
    edges = np.linspace(0, dim, n_blocks + 1, dtype=int)
    base = LogisticRegression(max_iter=200)
    base.fit(Ztr, ytr)
    base_acc = float(base.score(Ztr[:2000], ytr[:2000]))
    weights = np.zeros(dim, dtype=np.float64)
    perms = 6
    for p in range(perms):
        order = rng.permutation(n_blocks)
        acc_so_far = 0.25  # chance-ish prior on 20 classes, measured as the empty coalition
        for b in order:
            lo, hi = int(edges[b]), int(edges[b + 1])
            mask_cols = []
            for prev in order[: list(order).index(b) + 1]:
                a, c = int(edges[prev]), int(edges[prev + 1])
                mask_cols.extend(range(a, c))
            acc = float(LogisticRegression(max_iter=120).fit(Ztr[:, mask_cols], ytr).score(Ztr[:2000, mask_cols], ytr[:2000]))
            weights[lo:hi] += acc - acc_so_far
            acc_so_far = acc
        print(f"shapley perm {p + 1}/{perms} empty-to-full {base_acc:.3f}", flush=True)
    weights /= perms
    weights = np.clip(weights, 0, None)
    if weights.sum() == 0:
        weights[:] = 1.0
    scale = np.sqrt(weights + 1e-8).astype(np.float32)
    A = Ztr * scale
    B = Zte * scale
    A /= np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-8)
    B /= np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-8)
    proto = NearestCentroid().fit(A, ytr).predict(B)
    km = KMeans(n_clusters=20, random_state=RNG, n_init=10).fit_predict(B)
    return proto, km


def main() -> int:
    _, train_y = load_20ng_split("train")
    _, test_y = load_20ng_split("test")
    Ztr = np.load(CACHE / "20ng-train-bge-base.npy").astype(np.float32)
    Zte = np.load(CACHE / "20ng-test-bge-base.npy").astype(np.float32)
    Ztr /= np.linalg.norm(Ztr, axis=1, keepdims=True).clip(min=1e-8)
    Zte /= np.linalg.norm(Zte, axis=1, keepdims=True).clip(min=1e-8)
    rows = []

    def add(name, labels, **extra):
        row = {
            "name": name,
            "nmi": round(nmi(test_y, labels), 4),
            "n_clusters": int(len(set(np.asarray(labels).tolist()))),
            "published_nmi": TARGET,
            "source": "measured",
            **extra,
        }
        rows.append(row)
        print(f"{name} NMI={row['nmi']:.4f} clusters={row['n_clusters']}", flush=True)
        OUT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    add("hedonic_nash_bge", hedonic(Zte))
    add("dominant_set_replicator_bge", dominant_sets(Zte))
    proto, km = shapley_reweight(Ztr, train_y, Zte)
    add("shapley_reweight_nearest_centroid", proto, uses_train_labels=True)
    add("shapley_reweight_kmeans", km, uses_train_labels=True)
    best = max(rows, key=lambda r: r["nmi"])
    print(f"BEST {best['name']} NMI={best['nmi']} target={TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
