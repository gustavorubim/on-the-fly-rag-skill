"""Query a local vector index (single-threaded)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .embed import OnnxEmbedder
from .paths import DEFAULT_MODEL_ONNX, DEFAULT_TOKENIZER
from .registry import load_spec_from_index_config, resolve_model
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


def _embedder_for_index(
    store: VectorStore,
    *,
    model: Optional[Union[str, Path]] = None,
    model_path: Optional[Path | str] = None,
    tokenizer_path: Optional[Path | str] = None,
) -> OnnxEmbedder:
    """Build an embedder matching the index (or an explicit override)."""
    if model is not None:
        return OnnxEmbedder.from_resolve(model, tokenizer=tokenizer_path)
    if model_path is not None:
        return OnnxEmbedder.from_resolve(
            model_path, tokenizer=tokenizer_path or DEFAULT_TOKENIZER
        )

    cfg = store.config or {}
    spec = load_spec_from_index_config(cfg)
    if spec is not None:
        try:
            return OnnxEmbedder.from_spec(spec)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Index was built with model {spec.id!r} but weights are unavailable.\n{e}"
            ) from e

    # Legacy indexes: fall back to recorded paths or MiniLM default
    mp = cfg.get("model") or DEFAULT_MODEL_ONNX
    tp = cfg.get("tokenizer") or tokenizer_path or DEFAULT_TOKENIZER
    try:
        return OnnxEmbedder.from_resolve(mp, tokenizer=tp)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not load embedder for index {store.index_dir}: {e}"
        ) from e


def search(
    query: str,
    *,
    index_dir: Path | str,
    top_k: int = 5,
    model: Optional[Union[str, Path]] = None,
    model_path: Optional[Path | str] = None,
    tokenizer_path: Optional[Path | str] = None,
    path_contains: Optional[str] = None,
    path_glob: Optional[str] = None,
    path_prefix: Optional[str] = None,
    embedder: Optional[OnnxEmbedder] = None,
    store: Optional[VectorStore] = None,
) -> List[Dict[str, Any]]:
    if store is None:
        store = VectorStore(index_dir)
        store.load()
    emb = embedder or _embedder_for_index(
        store, model=model, model_path=model_path, tokenizer_path=tokenizer_path
    )
    # Dim check: avoid silent garbage if wrong model used
    if store.vectors is not None and store.vectors.shape[1] != emb.dim:
        raise ValueError(
            f"Model dim {emb.dim} does not match index dim {store.vectors.shape[1]}. "
            f"Re-ingest with the same model, or pass --model matching index config "
            f"(model_id={ (store.config or {}).get('model_id', '?') })."
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
    model: Optional[Union[str, Path]] = None,
    model_path: Optional[Path | str] = None,
    tokenizer_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Run multiple retrievals, reusing one embedder + loaded index.

    Each item is either a query string or a dict with keys:
    ``query`` (required), ``top_k``, ``path_contains``, ``path_glob``,
    ``path_prefix``, ``id`` (optional label for the hop).
    """
    store = VectorStore(index_dir)
    store.load()
    emb = _embedder_for_index(
        store, model=model, model_path=model_path, tokenizer_path=tokenizer_path
    )
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
