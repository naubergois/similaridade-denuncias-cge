#!/usr/bin/env python3
"""Score saved BBC partitions. No LLM. Not CGE production.

    .venv/bin/python experiments/run_bbc_final_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bbc_beat_published import PUB, score  # noqa: E402
from run_public_benchmarks import load_bbc  # noqa: E402

LABELS = HERE / "results" / "bbc_labels"
OUT = HERE / "bbc_final_scores.json"


def main() -> int:
    _, gold, _ = load_bbc()
    rows = []
    for path in sorted(LABELS.glob("bbc_stage*_labels.npy")):
        rows.append({**score(path.stem, np.load(path), gold), "file": str(path.relative_to(HERE.parent))})
    final = next(r for r in rows if r["method"] == "bbc_stage15_labels")
    payload = {
        "disclaimer": "Laboratory only. CGE production stays TF-IDF + DBSCAN at cosine 0.75.",
        "published": PUB,
        "final": final,
        "beats_published": final["nmi"] >= PUB["nmi"] and final["ari"] >= PUB["ari"] and final["acc"] >= PUB["acc"],
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} beats={payload['beats_published']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
