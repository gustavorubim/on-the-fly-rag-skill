"""Query a local vector index (single-threaded)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .embed import MiniLMEmbedder
from .paths import DEFAULT_MODEL_ONNX, DEFAULT_TOKENIZER
from .store import VectorStore


def _hits_to_dicts(hits) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for score, chunk in hits:
        results.append(
            {
                "score": round(float(score), 6),
                "path": chunk.path,
                "chunk_id": chunk.chunk_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                "text": chunk.text,
            }
        )
    return results


def search(
    query: str,
    *,
    index_dir: Path | str,
    top_k: int = 5,
    model_path: Path | str = DEFAULT_MODEL_ONNX,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    path_contains: Optional[str] = None,
    path_glob: Optional[str] = None,
    path_prefix: Optional[str] = None,
    embedder: Optional[MiniLMEmbedder] = None,
    store: Optional[VectorStore] = None,
) -> List[Dict[str, Any]]:
    if store is None:
        store = VectorStore(index_dir)
        store.load()
    emb = embedder or MiniLMEmbedder(
        model_path=model_path, tokenizer_path=tokenizer_path
    )
    qvec = emb.encode_one(query)
    hits = store.search(
        qvec,
        top_k=top_k,
        path_contains=path_contains,
        path_glob=path_glob,
        path_prefix=path_prefix,
    )
    return _hits_to_dicts(hits)


def multi_search(
    queries: Sequence[Union[str, Dict[str, Any]]],
    *,
    index_dir: Path | str,
    top_k: int = 5,
    model_path: Path | str = DEFAULT_MODEL_ONNX,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
) -> List[Dict[str, Any]]:
    """Run multiple retrievals, reusing one embedder + loaded index.

    Each item is either a query string or a dict with keys:
    ``query`` (required), ``top_k``, ``path_contains``, ``path_glob``,
    ``path_prefix``, ``id`` (optional label for the hop).
    """
    store = VectorStore(index_dir)
    store.load()
    emb = MiniLMEmbedder(model_path=model_path, tokenizer_path=tokenizer_path)
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(queries):
        if isinstance(item, str):
            q = item
            opts: Dict[str, Any] = {}
            hop_id = f"q{i}"
        else:
            q = str(item["query"])
            opts = {
                k: item[k]
                for k in ("top_k", "path_contains", "path_glob", "path_prefix")
                if k in item and item[k] is not None
            }
            hop_id = str(item.get("id") or item.get("label") or f"q{i}")
        k = int(opts.pop("top_k", top_k))
        hits = search(
            q,
            index_dir=index_dir,
            top_k=k,
            embedder=emb,
            store=store,
            **opts,
        )
        out.append({"id": hop_id, "query": q, "filters": opts, "hits": hits})
    return out


# MiniLM cosine on short technical queries often lands ~0.20–0.55 for real hits.
THIN_SCORE = 0.20


def coverage_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple verify helper: hit counts, unique paths, thin-hop flags.

    A hop is thin if it has no hits or ``top_score < THIN_SCORE`` (default 0.20).
    Agents should re-retrieve thin hops (broader query or looser path filter).
    """
    hops = []
    all_paths = set()
    for r in results:
        paths = {h["path"] for h in r.get("hits", [])}
        all_paths |= paths
        n = len(r.get("hits", []))
        top = r["hits"][0]["score"] if r.get("hits") else 0.0
        hops.append(
            {
                "id": r.get("id"),
                "query": r.get("query"),
                "hit_count": n,
                "top_score": top,
                "paths": sorted(paths),
                "thin": n == 0 or top < THIN_SCORE,
            }
        )
    return {
        "hops": hops,
        "unique_paths": sorted(all_paths),
        "thin_hops": [h["id"] for h in hops if h["thin"]],
        "ok": all(not h["thin"] for h in hops),
    }


def format_results(results: List[Dict[str, Any]], *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(results, indent=2, ensure_ascii=False)
    if not results:
        return "(no hits)"
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        preview = r["text"].replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:237] + "..."
        lines.append(
            f"{i}. score={r['score']:.4f}  {r['path']}  [{r['chunk_id']}]\n   {preview}"
        )
    return "\n".join(lines)


def format_multi(
    results: List[Dict[str, Any]], *, as_json: bool = False, with_coverage: bool = True
) -> str:
    payload: Dict[str, Any] = {"results": results}
    if with_coverage:
        payload["coverage"] = coverage_stats(results)
    if as_json:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    lines: List[str] = []
    for block in results:
        lines.append(f"## {block['id']}: {block['query']}")
        if block.get("filters"):
            lines.append(f"filters={block['filters']}")
        lines.append(format_results(block["hits"]))
        lines.append("")
    if with_coverage:
        cov = payload["coverage"]
        lines.append(
            f"coverage: ok={cov['ok']} thin={cov['thin_hops']} paths={cov['unique_paths']}"
        )
    return "\n".join(lines).rstrip() + "\n"
