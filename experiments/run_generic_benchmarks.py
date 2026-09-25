#!/usr/bin/env python3
"""Lab methods for public packs. Not the CGE production stack.

Production (Ouvidoria) stays in run_public_benchmarks.py:
  TF-IDF 1–3 + DBSCAN ε=0.25 + accept mean cosine ≥ 0.75.

Generic lab stacks (task-aware, another gold → another pipeline):
  theme:     MiniLM embeddings → KMeans with k = #gold classes
  duplicate: MiniLM embeddings → HDBSCAN(min_cluster_size=2)

    .venv/bin/python experiments/run_generic_benchmarks.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_public_benchmarks import (  # noqa: E402
    DATA,
    RNG,
    bcubed,
    load_20ng,
    load_bbc,
    load_cqadupstack,
    load_qqp,
    load_reuters,
    run_stack as run_production,
)

OUT_JSON = HERE / "generic_benchmarks.json"
OUT_FIG = HERE.parent / "paper" / "figures" / "fig7_generic_vs_production.png"

# Pinned before the run. Not tuned on these packs after seeing scores.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DUP_MIN_CLUSTER = 2
_MODEL = None


def embed(texts: list[str]) -> np.ndarray:
    global _MODEL
    from sentence_transformers import SentenceTransformer

    if _MODEL is None:
        print(f"loading {EMBED_MODEL} …", flush=True)
        _MODEL = SentenceTransformer(EMBED_MODEL)
    Z = _MODEL.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(Z, dtype=np.float32)


def run_generic_theme(texts: list[str], gold: np.ndarray) -> dict:
    """Theme gold: recover the known number of topics (standard unsupervised setup)."""
    k = int(len(set(gold.tolist())))
    Z = embed(texts)
    labels = KMeans(n_clusters=k, random_state=RNG, n_init=10).fit_predict(Z)
    labels = np.asarray(labels, dtype=int)
    return {
        "method": "generic-theme",
        "pipeline": f"MiniLM→KMeans(k={k})",
        "n": len(texts),
        "labels": labels,
        "n_clusters": k,
        "grouped": len(texts),
        "rho": 1.0,
        "leftovers": 0,
        "params": {"k": k, "encoder": EMBED_MODEL},
    }


def run_generic_duplicate(texts: list[str], gold: np.ndarray) -> dict:
    """Duplicate-question gold: density clusters on MiniLM (HDBSCAN), no fixed τ."""
    del gold
    import hdbscan

    Z = embed(texts)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=DUP_MIN_CLUSTER,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    raw = clusterer.fit_predict(Z)
    labels = np.asarray(raw, dtype=int)
    n = len(texts)
    # Full partition for BCubed: noise points become singletons
    bc = labels.copy()
    next_id = int(bc.max()) + 1 if (bc >= 0).any() else 0
    for i, lab in enumerate(bc):
        if lab < 0:
            bc[i] = next_id
            next_id += 1
    leftovers = int((labels < 0).sum())
    grouped = n - leftovers
    n_cl = int(len({int(x) for x in labels if x >= 0}))
    return {
        "method": "generic-duplicate",
        "pipeline": f"MiniLM→HDBSCAN(min_cluster_size={DUP_MIN_CLUSTER})",
        "n": n,
        "labels": bc,
        "labels_rho": labels,
        "n_clusters": n_cl,
        "grouped": grouped,
        "rho": grouped / n if n else 0.0,
        "leftovers": leftovers,
        "params": {"min_cluster_size": DUP_MIN_CLUSTER, "encoder": EMBED_MODEL},
    }


def score(stack: dict, gold: np.ndarray, use_rho_labels: bool = False) -> dict:
    labels = stack["labels_rho"] if use_rho_labels and "labels_rho" in stack else stack["labels"]
    # BCubed always against the full partition for generic-duplicate
    bc_labels = stack["labels"]
    p, r, f1 = bcubed(bc_labels, gold)
    ami = float(adjusted_mutual_info_score(gold, bc_labels))
    nmi = float(normalized_mutual_info_score(gold, bc_labels))
    return {
        "bcubed_p": round(p, 4),
        "bcubed_r": round(r, 4),
        "bcubed_f1": round(f1, 4),
        "ami": round(ami, 4),
        "nmi": round(nmi, 4),
        "rho": round(float(stack["rho"]), 4),
        "grouped": int(stack["grouped"]),
        "n_clusters": int(stack["n_clusters"]),
        "leftovers": int(stack["leftovers"]),
    }


def run_generic(texts: list[str], gold: np.ndarray, gold_kind: str) -> dict:
    if gold_kind == "theme":
        return run_generic_theme(texts, gold)
    if gold_kind == "duplicate-question":
        return run_generic_duplicate(texts, gold)
    raise ValueError(f"unknown gold_kind={gold_kind}")


def summarize_pair(name: str, gold_kind: str, texts, gold, note: str) -> dict:
    print(f"\n=== {name}  n={len(texts)}  gold={gold_kind} ===", flush=True)
    print(note, flush=True)

    prod = run_production(texts)
    # production BCubed uses accepted labels (noise as singletons inside bcubed)
    prod_scores = {
        "method": "production-cge",
        "pipeline": "TF-IDF(1-3)→DBSCAN(ε=0.25)→accept mean≥0.75",
        **{k: v for k, v in score({"labels": prod["labels"], "rho": prod["rho"], "grouped": prod["grouped"], "n_clusters": prod["n_clusters"], "leftovers": prod["leftovers"]}, gold).items()},
    }
    print(
        f"PRODUCTION  ρ={prod_scores['rho']:.4f}  BCubed F1={prod_scores['bcubed_f1']:.4f}  "
        f"AMI={prod_scores['ami']:.4f}",
        flush=True,
    )

    gen = run_generic(texts, gold, gold_kind)
    gen_scores = {
        "method": gen["method"],
        "pipeline": gen["pipeline"],
        "params": gen["params"],
        **score(gen, gold),
    }
    print(
        f"GENERIC     ρ={gen_scores['rho']:.4f}  BCubed F1={gen_scores['bcubed_f1']:.4f}  "
        f"AMI={gen_scores['ami']:.4f}  [{gen['pipeline']}]",
        flush=True,
    )

    return {
        "name": name,
        "gold_kind": gold_kind,
        "n": len(texts),
        "gold_families": int(len(set(gold.tolist()))),
        "note": note,
        "production": prod_scores,
        "generic": gen_scores,
        "source": "measured",
    }


def render(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ.setdefault("MPLCONFIGDIR", str(HERE.parent / "paper" / "figures" / ".mpl"))
    names = [r["name"] for r in rows]
    prod_f1 = [100 * r["production"]["bcubed_f1"] for r in rows]
    gen_f1 = [100 * r["generic"]["bcubed_f1"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(rows))
    h = 0.36
    ax.barh(y + h / 2, prod_f1, h, color="#8a8a8a", label="Production CGE (BCubed F1 %)")
    ax.barh(y - h / 2, gen_f1, h, color="#1f6f63", label="Generic lab (BCubed F1 %)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("BCubed F1 vs that pack's gold (%)")
    ax.set_title("Lab generic method vs CGE production rule (product unchanged)")
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
            failed.append({"name": name, "error": f"n={len(texts)}"})
            continue
        rows.append(summarize_pair(name, kind, texts, gold, note))

    payload = {
        "disclaimer": (
            "Generic stacks are laboratory only. "
            "Do not change CGE Ouvidoria production (TF-IDF+DBSCAN τ=0.75)."
        ),
        "generic_theme": {
            "pipeline": "MiniLM→KMeans(k=#classes)",
            "encoder": EMBED_MODEL,
            "seed": RNG,
        },
        "generic_duplicate": {
            "pipeline": "MiniLM→HDBSCAN(min_cluster_size=2)",
            "encoder": EMBED_MODEL,
            "min_cluster_size": DUP_MIN_CLUSTER,
        },
        "failed": failed,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        render(rows)
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
