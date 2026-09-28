#!/usr/bin/env python3
"""Next laboratory shots at the published numbers. Not the CGE product.

  20 Newsgroups test, KMeans(k=20), NMI
    MiniLM contrastive on the train split (same class = positive), 256 tokens
  CQADupStack webmasters, nDCG@10
    e5-base with query/passage prefixes, then MiniLM cross-encoder rerank of the top 20
  Quora pair F1
    roberta-base cross-encoder, one epoch, natural duplicate rate, holdout of train.tsv

    .venv/bin/python experiments/run_next_alternatives.py
"""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_public_benchmarks import DATA, RNG, clean  # noqa: E402
from run_trained_alternatives import load_20ng_split, load_cqa, ndcg_from_scores  # noqa: E402

OUT_JSON = HERE / "next_alternatives.json"
PUBLISHED = {"20ng_nmi": 0.725, "qqp_f1": 0.89, "cqa_note": "twelve-forum average, not this forum"}


def load_payload() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {
        "disclaimer": "Laboratory only. CGE production stays TF-IDF + DBSCAN at cosine 0.75.",
        "published_reference": PUBLISHED,
        "rows": [],
    }


def save(payload: dict) -> None:
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}", flush=True)


def have(payload: dict, probe: str, encoder: str) -> bool:
    return any(r.get("probe") == probe and r.get("encoder") == encoder for r in payload["rows"])


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = a @ p.T / temperature
    target = torch.arange(a.size(0))
    return F.cross_entropy(logits, target)


def train_contrastive(model, texts: list[str], gold: np.ndarray, epochs: int, batch: int, max_len: int) -> None:
    by: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(gold.tolist()):
        by[int(y)].append(i)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.max_seq_length = max_len
    model.train()
    rng = np.random.RandomState(RNG)
    n = len(texts)
    for epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        steps = 0
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            if len(idx) < 4:
                continue
            pos = []
            for i in idx:
                mates = by[int(gold[i])]
                j = int(mates[int(rng.randint(0, len(mates)))])
                if j == int(i) and len(mates) > 1:
                    j = int(mates[int(rng.randint(0, len(mates)))])
                pos.append(j)
            anchors = [texts[int(i)][:6000] for i in idx]
            positives = [texts[j][:6000] for j in pos]
            ea = model(model.tokenize(anchors))["sentence_embedding"]
            ep = model(model.tokenize(positives))["sentence_embedding"]
            loss = info_nce(ea, ep)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
            if steps % 40 == 0:
                print(f"20NG contrastive epoch {epoch + 1} step {steps} loss {total / steps:.4f}", flush=True)
        print(f"20NG contrastive epoch {epoch + 1} mean loss {total / max(steps, 1):.4f}", flush=True)
    model.eval()


def embed(model, texts: list[str], max_len: int) -> np.ndarray:
    model.eval()
    model.max_seq_length = max_len
    return np.asarray(
        model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )


def run_20ng(payload: dict) -> None:
    if have(payload, "20ng_test_k20", "minilm-contrastive"):
        print("skip 20NG contrastive", flush=True)
        return
    from sentence_transformers import SentenceTransformer

    train_x, train_y = load_20ng_split("train")
    test_x, test_y = load_20ng_split("test")
    print(f"20NG contrastive train={len(train_x)} test={len(test_x)}", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    best = {"nmi": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, 4):
        train_contrastive(model, train_x, train_y, epochs=1, batch=32, max_len=256)
        Z = embed(model, test_x, 256)
        pred = KMeans(n_clusters=20, random_state=RNG, n_init=10).fit_predict(Z)
        nmi = float(normalized_mutual_info_score(test_y, pred))
        print(f"20NG contrastive epoch {epoch} NMI={nmi:.4f}", flush=True)
        if nmi > best["nmi"]:
            best = {
                "nmi": nmi,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "Z": Z,
            }
        if nmi >= PUBLISHED["20ng_nmi"]:
            break
    model.load_state_dict(best["state"])
    Z_test = best["Z"]
    Z_train = embed(model, train_x, 256)
    probe = LogisticRegression(max_iter=400)
    probe.fit(Z_train, train_y)
    guessed = probe.predict(Z_test)
    row = {
        "probe": "20ng_test_k20",
        "encoder": "minilm-contrastive",
        "method": f"inbatch_infonce_best_epoch_{best['epoch']}_maxlen_256",
        "nmi": round(best["nmi"], 4),
        "linear_probe_nmi": round(float(normalized_mutual_info_score(test_y, guessed)), 4),
        "linear_probe_acc": round(float(accuracy_score(test_y, guessed)), 4),
        "n_train": len(train_x),
        "n_test": len(test_x),
        "published_nmi": PUBLISHED["20ng_nmi"],
        "note": "labels used only on the train split; KMeans NMI is the clustering number",
        "source": "measured",
    }
    print(
        f"20NG best KMeans NMI={row['nmi']:.4f} probe NMI={row['linear_probe_nmi']:.4f} "
        f"acc={row['linear_probe_acc']:.4f}",
        flush=True,
    )
    payload["rows"].append(row)
    save(payload)
    del model


def run_cqa(payload: dict) -> None:
    if have(payload, "cqa_webmasters_ndcg10", "e5-base"):
        print("skip e5", flush=True)
        return
    from sentence_transformers import SentenceTransformer

    corpus_ids, corpus_texts, queries, qrels, id_index = load_cqa()
    model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
    print(f"CQA e5 corpus={len(corpus_texts)}", flush=True)
    Zc = embed(model, ["passage: " + t for t in corpus_texts], 256)
    Zq = embed(model, ["query: " + t for _, t in queries], 256)
    dense = Zq @ Zc.T
    nd = ndcg_from_scores(dense, corpus_ids, queries, qrels, id_index)
    row = {
        "probe": "cqa_webmasters_ndcg10",
        "encoder": "e5-base",
        "method": "bi_encoder_query_passage_prefix",
        "ndcg_at_10": round(nd, 4),
        "n_queries": len(queries),
        "n_corpus": len(corpus_texts),
        "note": "one forum; published score averages twelve",
        "source": "measured",
    }
    print(f"CQA e5 nDCG@10 {row['ndcg_at_10']:.4f}", flush=True)
    payload["rows"].append(row)
    save(payload)

    if have(payload, "cqa_webmasters_ndcg10", "e5-base+minilm-cross"):
        del model
        return
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    topk = 20
    reranked = np.full_like(dense, -1e9)
    for i, (_, qtext) in enumerate(queries):
        order = np.argsort(-dense[i])[:topk]
        pairs = [(qtext[:1500], corpus_texts[int(j)][:1500]) for j in order]
        scores = np.asarray(reranker.predict(pairs, batch_size=16, show_progress_bar=False), dtype=np.float32)
        reranked[i, order] = scores
        if i and i % 100 == 0:
            print(f"CQA rerank {i}/{len(queries)}", flush=True)
    nd_r = ndcg_from_scores(reranked, corpus_ids, queries, qrels, id_index)
    row_r = {
        "probe": "cqa_webmasters_ndcg10",
        "encoder": "e5-base+minilm-cross",
        "method": "rerank_top20",
        "ndcg_at_10": round(nd_r, 4),
        "n_queries": len(queries),
        "n_corpus": len(corpus_texts),
        "note": "one forum; cross-encoder is the public MS MARCO MiniLM, not fine-tuned on CQA",
        "source": "measured",
    }
    print(f"CQA rerank nDCG@10 {row_r['ndcg_at_10']:.4f}", flush=True)
    payload["rows"].append(row_r)
    save(payload)
    del model, reranker


def load_qqp_natural(n: int = 15000) -> tuple[list[str], list[str], np.ndarray]:
    rng = np.random.RandomState(RNG + 11)
    kept: list[tuple[str, str, int]] = []
    seen = 0
    with zipfile.ZipFile(DATA / "QQP.zip") as zf:
        with zf.open("QQP/train.tsv") as fh:
            rows = csv.DictReader((line.decode("utf-8", errors="replace") for line in fh), delimiter="\t")
            for row in rows:
                q1 = clean(row.get("question1") or "")
                q2 = clean(row.get("question2") or "")
                if len(q1) < 8 or len(q2) < 8:
                    continue
                label = 1 if row.get("is_duplicate") == "1" else 0
                seen += 1
                item = (q1, q2, label)
                if len(kept) < n:
                    kept.append(item)
                else:
                    j = int(rng.randint(0, seen))
                    if j < n:
                        kept[j] = item
    a = [p[0] for p in kept]
    b = [p[1] for p in kept]
    y = np.array([p[2] for p in kept], dtype=int)
    return a, b, y


def run_qqp(payload: dict) -> None:
    if have(payload, "qqp_pair_finetune", "roberta-base-cross"):
        print("skip roberta", flush=True)
        return
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    a, b, y = load_qqp_natural(15000)
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=RNG, stratify=y)
    print(f"QQP natural n={len(y)} pos_rate={y.mean():.3f} fit={len(idx_train)} test={len(idx_test)}", flush=True)
    name = "distilroberta-base"
    # distilroberta-base is the model that fits a 16 GB CPU for one epoch of this sample.
    tok = AutoTokenizer.from_pretrained(name)
    clf = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    clf.train()
    opt = torch.optim.AdamW(clf.parameters(), lr=2e-5)
    rng = np.random.RandomState(RNG + 3)
    order = rng.permutation(len(idx_train))
    batch = 8
    total = 0.0
    steps = 0
    for start in range(0, len(order), batch):
        sel = idx_train[order[start : start + batch]]
        enc = tok(
            [a[i] for i in sel],
            [b[i] for i in sel],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        labels = torch.tensor(y[sel], dtype=torch.long)
        out = clf(**enc, labels=labels)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        total += float(out.loss.detach())
        steps += 1
        if steps % 40 == 0:
            print(f"QQP roberta step {steps} loss {total / steps:.4f}", flush=True)
    print(f"QQP roberta mean loss {total / max(steps, 1):.4f}", flush=True)
    clf.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(idx_test), 16):
            sel = idx_test[start : start + 16]
            enc = tok(
                [a[i] for i in sel],
                [b[i] for i in sel],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            pred = clf(**enc).logits.argmax(dim=-1).cpu().numpy()
            preds.append(pred)
    pred = np.concatenate(preds)
    row = {
        "probe": "qqp_pair_finetune",
        "encoder": "roberta-base-cross",
        "method": "distilroberta_base_1epoch_natural_rate",
        "n_pairs": int(len(y)),
        "n_fit": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "pos_rate": round(float(y.mean()), 4),
        "test_f1": round(float(f1_score(y[idx_test], pred, zero_division=0)), 4),
        "test_acc": round(float(accuracy_score(y[idx_test], pred)), 4),
        "published_f1": PUBLISHED["qqp_f1"],
        "note": "holdout of QQP train.tsv at the file's duplicate rate, not the GLUE test; backbone is distilroberta-base",
        "source": "measured",
    }
    print(f"QQP distilroberta F1={row['test_f1']:.4f} acc={row['test_acc']:.4f}", flush=True)
    payload["rows"].append(row)
    save(payload)


def main() -> int:
    torch.set_num_threads(4)
    payload = load_payload()
    run_20ng(payload)
    run_cqa(payload)
    run_qqp(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
