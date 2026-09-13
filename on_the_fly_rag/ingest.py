"""Ingest a folder: chunk + embed + persist (parallel CPU by default)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chunk import Chunk, chunk_path, iter_files, load_tokenizer
from .embed import MiniLMEmbedder
from .paths import DEFAULT_MODEL_ONNX, DEFAULT_TOKENIZER
from .store import StoredChunk, VectorStore

# Module-level state for worker processes (set by initializer).
_WORKER_TOKENIZER = None
_WORKER_EMBEDDER: Optional[MiniLMEmbedder] = None
_WORKER_ROOT: Optional[Path] = None
_WORKER_MAX_TOKENS = 200
_WORKER_OVERLAP = 40
_WORKER_MODEL: Optional[str] = None
_WORKER_TOKENIZER_PATH: Optional[str] = None


def _default_workers() -> int:
    n = os.cpu_count() or 1
    return max(1, n)


def _init_chunk_worker(
    tokenizer_path: str,
    root: str,
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    global _WORKER_TOKENIZER, _WORKER_ROOT, _WORKER_MAX_TOKENS, _WORKER_OVERLAP
    _WORKER_TOKENIZER = load_tokenizer(Path(tokenizer_path))
    _WORKER_ROOT = Path(root)
    _WORKER_MAX_TOKENS = max_tokens
    _WORKER_OVERLAP = overlap_tokens


def _chunk_one_file(path_str: str) -> List[Dict[str, Any]]:
    assert _WORKER_TOKENIZER is not None and _WORKER_ROOT is not None
    chunks = chunk_path(
        Path(path_str),
        root=_WORKER_ROOT,
        tokenizer=_WORKER_TOKENIZER,
        max_tokens=_WORKER_MAX_TOKENS,
        overlap_tokens=_WORKER_OVERLAP,
    )
    return [asdict(c) for c in chunks]


def _init_embed_worker(model_path: str, tokenizer_path: str) -> None:
    global _WORKER_EMBEDDER, _WORKER_MODEL, _WORKER_TOKENIZER_PATH
    _WORKER_MODEL = model_path
    _WORKER_TOKENIZER_PATH = tokenizer_path
    _WORKER_EMBEDDER = MiniLMEmbedder(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )


def _embed_batch(payload: Tuple[int, List[str]]) -> Tuple[int, np.ndarray]:
    """Embed a batch; returns (batch_index, vectors) to preserve order."""
    assert _WORKER_EMBEDDER is not None
    idx, texts = payload
    vecs = _WORKER_EMBEDDER.encode(texts, batch_size=len(texts))
    return idx, vecs


def _chunk_files_parallel(
    files: Sequence[Path],
    *,
    root: Path,
    tokenizer_path: Path,
    max_tokens: int,
    overlap_tokens: int,
    workers: int,
) -> List[Chunk]:
    if not files:
        return []
    if workers <= 1 or len(files) == 1:
        tok = load_tokenizer(tokenizer_path)
        out: List[Chunk] = []
        for f in files:
            out.extend(
                chunk_path(
                    f,
                    root=root,
                    tokenizer=tok,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
        return out

    chunks: List[Chunk] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_chunk_worker,
        initargs=(str(tokenizer_path), str(root), max_tokens, overlap_tokens),
    ) as pool:
        futures = {pool.submit(_chunk_one_file, str(f)): f for f in files}
        for fut in as_completed(futures):
            for d in fut.result():
                chunks.append(Chunk(**d))
    # Stable order by path then chunk index encoded in chunk_id
    chunks.sort(key=lambda c: (c.path, c.chunk_id))
    return chunks


def _embed_parallel(
    texts: Sequence[str],
    *,
    model_path: Path,
    tokenizer_path: Path,
    workers: int,
    batch_size: int = 32,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    batches: List[Tuple[int, List[str]]] = []
    for i in range(0, len(texts), batch_size):
        batches.append((len(batches), list(texts[i : i + batch_size])))

    if workers <= 1 or len(batches) == 1:
        emb = MiniLMEmbedder(model_path=model_path, tokenizer_path=tokenizer_path)
        return emb.encode(list(texts), batch_size=batch_size)

    results: Dict[int, np.ndarray] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_embed_worker,
        initargs=(str(model_path), str(tokenizer_path)),
    ) as pool:
        futures = [pool.submit(_embed_batch, b) for b in batches]
        for fut in as_completed(futures):
            idx, vecs = fut.result()
            results[idx] = vecs

    ordered = [results[i] for i in range(len(batches))]
    return np.vstack(ordered)


def ingest(
    source: Path | str,
    *,
    index_dir: Path | str,
    model_path: Path | str = DEFAULT_MODEL_ONNX,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    max_tokens: int = 200,
    overlap_tokens: int = 40,
    workers: Optional[int] = None,
    batch_size: int = 32,
    extensions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Chunk + embed ``source`` into ``index_dir`` using ``workers`` CPU processes."""
    source = Path(source).resolve()
    index_dir = Path(index_dir)
    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path)
    if workers is None:
        workers = _default_workers()
    workers = max(1, int(workers))

    files = list(iter_files(source, extensions=extensions))
    root = source if source.is_dir() else source.parent

    chunks = _chunk_files_parallel(
        files,
        root=root,
        tokenizer_path=tokenizer_path,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        workers=workers,
    )

    texts = [c.text for c in chunks]
    vectors = _embed_parallel(
        texts,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        workers=workers,
        batch_size=batch_size,
    )

    stored = [
        StoredChunk(
            chunk_id=c.chunk_id,
            path=c.path,
            text=c.text,
            start_char=c.start_char,
            end_char=c.end_char,
            token_count=c.token_count,
        )
        for c in chunks
    ]
    store = VectorStore(index_dir)
    config = {
        "source": str(source),
        "model": str(model_path),
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "workers": workers,
        "files": len(files),
        "chunks": len(stored),
    }
    store.save(stored, vectors, config=config)
    return config
