"""Query a local vector index (single-threaded)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embed import MiniLMEmbedder
from .paths import DEFAULT_MODEL_ONNX, DEFAULT_TOKENIZER
from .store import VectorStore


def search(
    query: str,
    *,
    index_dir: Path | str,
    top_k: int = 5,
    model_path: Path | str = DEFAULT_MODEL_ONNX,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    path_contains: Optional[str] = None,
) -> List[Dict[str, Any]]:
    store = VectorStore(index_dir)
    store.load()
    emb = MiniLMEmbedder(model_path=model_path, tokenizer_path=tokenizer_path)
    qvec = emb.encode_one(query)
    hits = store.search(qvec, top_k=top_k, path_contains=path_contains)
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
