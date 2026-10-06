#!/usr/bin/env python3
"""Retrieval eval for on-the-fly-rag indexes (any model, any modality).

Questions file (JSON list); each item:

    {"id": "t1", "set": "drive", "type": "text->text",
     "index": "drive_gemma",            # key into --index NAME=DIR (or "a+b" to merge)
     "query": "..."  |  "query_file": "path/to/img.jpg",
     "modality": "image",               # optional --modality filter
     "expected": ["docx/policy.docx"],  # hit if a result path starts with one of these
     "must_contain": "ninety",          # optional: hit chunk text must contain this too
     "all_of": true}                    # optional multi-hop: every expected doc in top-k

Reports hit@1 / hit@5 / MRR@10 per (set, type) and per-question top results.

    python scripts/eval_multimodal.py --questions q.json \
        --index drive_gemma=/path/idx1 --index drive_granite=/path/idx2 \
        [--compare drive_gemma:drive_granite,drive_granite_small] --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from on_the_fly_rag.search import _embedder_for_index, search  # noqa: E402
from on_the_fly_rag.store import VectorStore  # noqa: E402

K = 10


def _merged_store(stores: Dict[str, VectorStore], names: List[str]) -> VectorStore:
    """In-memory union of same-model indexes; paths get a '<name>/' prefix."""
    from dataclasses import replace

    first = stores[names[0]]
    mids = {stores[n].config.get("model_id") for n in names}
    dims = {stores[n].vectors.shape[1] for n in names}
    if len(mids) != 1 or len(dims) != 1:
        raise SystemExit(f"cannot merge indexes with different models/dims: {mids} {dims}")
    vs = VectorStore(first.index_dir)
    vs.config = dict(first.config)
    vs.chunks, vecs = [], []
    for n in names:
        s = stores[n]
        for c in s.chunks:
            vs.chunks.append(replace(c, path=f"{n}/{c.path}",
                                     source=f"{n}/{c.source}" if c.source else None))
        vecs.append(s.vectors)
    vs.vectors = np.vstack(vecs)
    return vs


def _match(path: str, expected: List[str]) -> int:
    for i, e in enumerate(expected):
        if path == e or path.startswith(e) or path.split("/", 1)[-1].startswith(e):
            return i
    return -1


def score_question(q: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    exp = q["expected"]
    must = (q.get("must_contain") or "").lower()
    ranks: Dict[int, int] = {}
    for r, h in enumerate(hits, 1):
        i = _match(h["path"], exp)
        if i < 0:
            continue
        if must and must not in (h.get("text") or "").lower():
            continue
        ranks.setdefault(i, r)
    if q.get("all_of"):
        found5 = sum(1 for r in ranks.values() if r <= 5)
        first = min(ranks.values()) if ranks else None
        return {
            "hit1": int(first == 1),
            "hit5": int(found5 == len(exp)),
            "rr": 1.0 / first if first else 0.0,
            "rank": first,
            "found_in_top5": f"{found5}/{len(exp)}",
        }
    first = min(ranks.values()) if ranks else None
    return {"hit1": int(first == 1), "hit5": int(bool(first and first <= 5)),
            "rr": 1.0 / first if first else 0.0, "rank": first}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, action="append")
    ap.add_argument("--index", action="append", default=[], help="NAME=DIR (repeatable)")
    ap.add_argument("--compare", action="append", default=[],
                    help="SRC:ALT1,ALT2 - rerun questions on index SRC against ALT indexes (text only)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    questions: List[Dict[str, Any]] = []
    for qf in args.questions:
        questions.extend(json.loads(Path(qf).read_text()))
    dirs = dict(x.split("=", 1) for x in args.index)
    stores: Dict[str, VectorStore] = {}
    for n, d in dirs.items():
        s = VectorStore(d)
        s.load()
        stores[n] = s

    runs = []  # (question, index_name)
    for q in questions:
        runs.append((q, q["index"]))
    for spec in args.compare:
        src, alts = spec.split(":", 1)
        for q in questions:
            if q["index"] == src and q["type"].startswith("text->text"):
                for alt in alts.split(","):
                    runs.append((q, alt))

    embedders: Dict[str, Any] = {}
    gemma = None

    def store_for(name: str) -> VectorStore:
        if name not in stores:
            stores[name] = _merged_store(stores, name.split("+"))
        return stores[name]

    def embedder_for(name: str, s: VectorStore):
        nonlocal gemma
        if s.config.get("backend") == "torch":
            if gemma is None:
                from on_the_fly_rag.embed_torch import GemmaEmbedder
                from on_the_fly_rag.registry import get_preset

                t0 = time.time()
                gemma = GemmaEmbedder(get_preset("embeddinggemma-2").model_dir,
                                      dim=int(s.config.get("dim") or 768),
                                      modalities=("text", "image", "video", "audio"),
                                      threads=args.threads)
                print(f"[eval] loaded embeddinggemma-2 full in {time.time()-t0:.1f}s", file=sys.stderr)
            return gemma
        key = s.config.get("model_id") or name
        if key not in embedders:
            embedders[key] = _embedder_for_index(s)
        return embedders[key]

    results = []
    t_all = time.time()
    for q, name in runs:
        s = store_for(name)
        emb = embedder_for(name, s)
        t0 = time.time()
        hits = search(q.get("query"), index_dir=s.index_dir, top_k=K, modality=q.get("modality"),
                      query_file=q.get("query_file"), embedder=emb, store=s)
        sc = score_question(q, hits)
        results.append({
            "id": q["id"], "set": q["set"], "type": q["type"], "index": name,
            "model": s.config.get("model_id"), "query": q.get("query") or Path(q["query_file"]).name,
            "modality_filter": q.get("modality"), **sc, "latency_s": round(time.time() - t0, 2),
            "top": [{"path": h["path"], "modality": h.get("modality", "text"),
                     "score": round(h["score"], 4),
                     **({"start_sec": h["start_sec"], "end_sec": h["end_sec"]}
                        if h.get("start_sec") is not None else {})} for h in hits[:3]],
        })
        print(f"[{q['id']:>5}] {name:22s} rank={sc['rank']} {results[-1]['query'][:60]!r} -> "
              f"{hits[0]['path'] if hits else None} ({hits[0]['score']:.3f})", file=sys.stderr)

    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[(r["set"], r["type"], r["index"])].append(r)
    table = []
    for (st, ty, ix), rs in sorted(groups.items()):
        table.append({"set": st, "type": ty, "index": ix, "model": rs[0]["model"], "n": len(rs),
                      "hit@1": round(float(np.mean([r["hit1"] for r in rs])), 3),
                      "hit@5": round(float(np.mean([r["hit5"] for r in rs])), 3),
                      "mrr@10": round(float(np.mean([r["rr"] for r in rs])), 3)})
    print(f"\n{'set':7s} {'type':22s} {'index':26s} {'n':>3s} {'hit@1':>6s} {'hit@5':>6s} {'MRR':>6s}")
    for t in table:
        print(f"{t['set']:7s} {t['type']:22s} {t['index']:26s} {t['n']:3d} {t['hit@1']:6.3f} "
              f"{t['hit@5']:6.3f} {t['mrr@10']:6.3f}")
    print(f"\n{len(results)} queries in {time.time()-t_all:.1f}s", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(json.dumps({"table": table, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
