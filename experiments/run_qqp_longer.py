#!/usr/bin/env python3
"""Longer cross-encoder on the same QQP holdout. Not the GLUE test set.

Four epochs. The epoch kept for the test score is the one with the best F1
on the tune slice.

    .venv/bin/python experiments/run_qqp_longer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_public_benchmarks import RNG  # noqa: E402
from run_trained_alternatives import load_qqp  # noqa: E402

OUT_JSON = HERE / "trained_alternatives.json"


def train_epoch(model, clf, opt, tok, a, b, y, idx_fit, batch=32, max_len=96) -> float:
    backbone = model[0].auto_model
    model.train()
    rng = np.random.RandomState(RNG + 7)
    order = rng.permutation(len(idx_fit))
    y_i = y.astype(np.int64)
    total = 0.0
    steps = 0
    for start in range(0, len(order), batch):
        sel = idx_fit[order[start : start + batch]]
        enc = tok(
            [a[i] for i in sel],
            [b[i] for i in sel],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        out = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).type_as(hidden)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = torch.nn.functional.cross_entropy(clf(emb), torch.tensor(y_i[sel], dtype=torch.long))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.detach())
        steps += 1
    return total / max(steps, 1)


def predict(model, clf, tok, a, b, idx, max_len=96) -> np.ndarray:
    backbone = model[0].auto_model
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(idx), 64):
            sel = idx[start : start + 64]
            enc = tok(
                [a[i] for i in sel],
                [b[i] for i in sel],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            out = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).type_as(hidden)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            preds.append(clf(emb).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def main() -> int:
    from sentence_transformers import SentenceTransformer

    a, b, y = load_qqp(8000, 8000)
    y_i = y.astype(int)
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=RNG, stratify=y_i)
    rng = np.random.RandomState(RNG)
    shuffled = idx_train.copy()
    rng.shuffle(shuffled)
    cut = int(len(idx_train) * 0.1)
    idx_tune, idx_fit = shuffled[:cut], shuffled[cut:]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    dim = model.get_embedding_dimension()
    clf = torch.nn.Linear(dim, 2)
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 2e-5},
            {"params": clf.parameters(), "lr": 1e-3},
        ]
    )
    tok = model.tokenizer
    best = {"tune_f1": -1.0, "epoch": 0, "state": None, "clf": None}
    for epoch in range(1, 5):
        loss = train_epoch(model, clf, opt, tok, a, b, y, idx_fit)
        pred_tune = predict(model, clf, tok, a, b, idx_tune)
        tune_f1 = float(f1_score(y_i[idx_tune], pred_tune, zero_division=0))
        print(f"epoch {epoch} loss {loss:.4f} tune F1 {tune_f1:.4f}", flush=True)
        if tune_f1 > best["tune_f1"]:
            best = {
                "tune_f1": tune_f1,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "clf": {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()},
            }
    model.load_state_dict(best["state"])
    clf.load_state_dict(best["clf"])
    pred = predict(model, clf, tok, a, b, idx_test)
    row = {
        "probe": "qqp_pair_finetune",
        "encoder": "minilm-cross-encoder",
        "method": "pair_sequence_softmax_4epochs_best_tune",
        "n_pairs": int(len(y)),
        "n_fit": int(len(idx_fit)),
        "n_test": int(len(idx_test)),
        "chosen_epoch": best["epoch"],
        "tune_f1": round(best["tune_f1"], 4),
        "test_f1": round(float(f1_score(y_i[idx_test], pred, zero_division=0)), 4),
        "test_acc": round(float(accuracy_score(y_i[idx_test], pred)), 4),
        "published_f1": 0.89,
        "note": "balanced sample of QQP train.tsv, epoch chosen on the tune slice, not the GLUE test set",
        "source": "measured",
    }
    print(f"QQP longer test F1={row['test_f1']:.4f} epoch={row['chosen_epoch']}", flush=True)
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    payload["rows"].append(row)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
