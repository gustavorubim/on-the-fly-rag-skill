"""Ingest a folder: chunk + embed + persist (parallel CPU by default)."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .chunk import ALL_EXTENSIONS, Chunk, chunk_path, iter_files, load_tokenizer
from .embed import OnnxEmbedder, load_embedder
from .media import MediaOptions, is_media, media_items_for, modality_of
from .paths import DEFAULT_TOKENIZER
from .registry import (
    DEFAULT_MODEL_ID,
    check_index_compat,
    get_preset,
    resolve_model,
    spec_to_config,
    validate_dim,
)
from .store import StoredChunk, VectorStore

# Module-level state for worker processes (set by initializer).
_WORKER_TOKENIZER = None
_WORKER_EMBEDDER: Optional[OnnxEmbedder] = None
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


def _init_embed_worker(
    model_path: str,
    tokenizer_path: str,
    max_seq_length: int,
    dim: int,
    pooling: str,
    model_id: str,
) -> None:
    global _WORKER_EMBEDDER, _WORKER_MODEL, _WORKER_TOKENIZER_PATH
    _WORKER_MODEL = model_path
    _WORKER_TOKENIZER_PATH = tokenizer_path
    _WORKER_EMBEDDER = OnnxEmbedder(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_seq_length=max_seq_length,
        dim=dim,
        pooling=pooling,
        model_id=model_id,
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
    max_seq_length: int,
    dim: int,
    pooling: str,
    model_id: str,
    workers: int,
    batch_size: int = 32,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    batches: List[Tuple[int, List[str]]] = []
    for i in range(0, len(texts), batch_size):
        batches.append((len(batches), list(texts[i : i + batch_size])))

    if workers <= 1 or len(batches) == 1:
        emb = OnnxEmbedder(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_seq_length=max_seq_length,
            dim=dim,
            pooling=pooling,
            model_id=model_id,
        )
        return emb.encode(list(texts), batch_size=batch_size)

    results: Dict[int, np.ndarray] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_embed_worker,
        initargs=(
            str(model_path),
            str(tokenizer_path),
            max_seq_length,
            dim,
            pooling,
            model_id,
        ),
    ) as pool:
        futures = [pool.submit(_embed_batch, b) for b in batches]
        for fut in as_completed(futures):
            idx, vecs = fut.result()
            results[idx] = vecs

    ordered = [results[i] for i in range(len(batches))]
    return np.vstack(ordered)


def _log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


def doc_title(chunk_path: str) -> str:
    """Document-prompt title: file name, plus the slide label for PPTX sections."""
    base, _, frag = chunk_path.partition("#")
    name = Path(base).name
    if frag.startswith("slide-"):
        return f"{name} slide {frag[len('slide-'):]}"
    return name


def _read_existing_config(index_dir: Path) -> Optional[Dict[str, Any]]:
    cfg = index_dir / "config.json"
    if not cfg.is_file() or not (index_dir / "vectors.npy").is_file():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _needed_modalities(media_files: Sequence[Path], multimodal: str) -> Tuple[str, ...]:
    if multimodal == "on":
        return ("text", "image", "video", "audio")
    mods = {"text"} | {modality_of(f) for f in media_files}
    return tuple(sorted(mods))


def _embed_media_torch(
    emb: Any,
    media_files: Sequence[Path],
    *,
    root: Path,
    opts: MediaOptions,
    quiet: bool,
) -> Tuple[List[StoredChunk], List[np.ndarray], List[Dict[str, str]]]:
    stored: List[StoredChunk] = []
    vecs: List[np.ndarray] = []
    errors: List[Dict[str, str]] = []
    for n, f in enumerate(media_files, 1):
        rel = str(f.resolve().relative_to(root.resolve()))
        t0 = time.time()
        try:
            items = media_items_for(f, root=root, opts=opts)
            v = emb.encode_media([it.model_input() for it in items])
        except Exception as e:  # noqa: BLE001 - one bad file must not abort ingest
            errors.append({"path": rel, "error": f"{type(e).__name__}: {e}"[:300]})
            _log(f"  [media {n}/{len(media_files)}] SKIP {rel}: {e}", quiet=quiet)
            continue
        for k, it in enumerate(items):
            stored.append(
                StoredChunk(
                    chunk_id=f"{it.path}::{k}",
                    path=it.path,
                    text=it.description,
                    start_char=0,
                    end_char=0,
                    token_count=0,
                    modality=it.modality,
                    source=it.source,
                    start_sec=it.start_sec,
                    end_sec=it.end_sec,
                    frame_start=it.frame_start,
                    frame_end=it.frame_end,
                    title=Path(it.source).name,
                )
            )
        vecs.append(v)
        _log(
            f"  [media {n}/{len(media_files)}] {rel}: {len(items)} {items[0].modality} "
            f"segment(s) in {time.time() - t0:.1f}s",
            quiet=quiet,
        )
    return stored, vecs, errors


def ingest(
    source: Path | str,
    *,
    index_dir: Path | str,
    model: Optional[Union[str, Path]] = None,
    model_path: Optional[Path | str] = None,
    tokenizer_path: Optional[Path | str] = None,
    max_tokens: int = 200,
    overlap_tokens: int = 40,
    workers: Optional[int] = None,
    batch_size: int = 32,
    extensions: Optional[Sequence[str]] = None,
    dim: Optional[int] = None,
    multimodal: str = "auto",
    append: bool = False,
    dtype: Optional[str] = None,
    torch_threads: Optional[int] = None,
    media_options: Optional[MediaOptions] = None,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Chunk + embed ``source`` into ``index_dir`` using ``workers`` CPU processes.

    ``model`` may be a preset id (``minilm``, ``granite-small``, ``granite``,
    ``embeddinggemma-2``) or an ONNX path. Legacy ``model_path`` /
    ``tokenizer_path`` still work.

    ``dim`` (Matryoshka, embeddinggemma-2 only): 768/512/256/128.
    ``multimodal``: ``auto`` (load vision/audio encoders only if the corpus has
    such files), ``on`` (always full model), ``off`` (text only; skip media).
    ``append``: add/replace files in an existing index; refuses a different
    model or dim (:class:`IndexCompatError`).
    """
    if multimodal not in ("auto", "on", "off"):
        raise ValueError("multimodal must be auto|on|off")
    source = Path(source).resolve()
    index_dir = Path(index_dir)

    existing_cfg = _read_existing_config(index_dir) if append else None
    if append and existing_cfg is None:
        _log(f"--append: no existing index at {index_dir}; creating a new one", quiet=quiet)

    if model is None and model_path is None and existing_cfg and existing_cfg.get("model_id"):
        model = existing_cfg["model_id"]  # append defaults to the index's model
    if dim is None and existing_cfg and existing_cfg.get("dim"):
        try:
            if get_preset(str(model or DEFAULT_MODEL_ID)).id == existing_cfg.get("model_id"):
                dim = int(existing_cfg["dim"])
        except KeyError:
            pass

    if model is not None:
        spec, onnx, tok = resolve_model(model, tokenizer=tokenizer_path)
    elif model_path is not None:
        spec, onnx, tok = resolve_model(
            model_path, tokenizer=tokenizer_path or DEFAULT_TOKENIZER
        )
    else:
        spec, onnx, tok = resolve_model(
            DEFAULT_MODEL_ID, tokenizer=tokenizer_path
        )

    out_dim = validate_dim(spec, dim)
    if existing_cfg is not None:
        check_index_compat(existing_cfg, model_id=spec.id, dim=out_dim, action="append")

    if workers is None:
        workers = _default_workers()
    workers = max(1, int(workers))

    all_files = list(iter_files(source, extensions=extensions or ALL_EXTENSIONS))
    root = source if source.is_dir() else source.parent
    text_files = [f for f in all_files if not is_media(f)]
    media_files = [f for f in all_files if is_media(f)]
    skipped_media: List[str] = []
    if media_files and (spec.backend != "torch" or multimodal == "off"):
        skipped_media = [str(f.resolve().relative_to(root.resolve())) for f in media_files]
        reason = (
            f"model {spec.id!r} is text-only" if spec.backend != "torch" else "--multimodal off"
        )
        print(
            f"Notice: skipping {len(media_files)} image/video/audio file(s) ({reason}). "
            "Use --model embeddinggemma-2 to index them.",
            file=sys.stderr,
        )
        media_files = []

    # Cap chunk size under the model context window (leave room for specials).
    # The model limit is only a ceiling; the default stays RAG-sized (~200 tok).
    effective_max = min(max_tokens, max(32, spec.max_seq_length - 8))

    t_start = time.time()
    chunks = _chunk_files_parallel(
        text_files,
        root=root,
        tokenizer_path=tok,
        max_tokens=effective_max,
        overlap_tokens=overlap_tokens,
        workers=workers,
    )
    t_chunked = time.time()

    texts = [c.text for c in chunks]
    titles = [doc_title(c.path) for c in chunks]
    media_stored: List[StoredChunk] = []
    media_vecs: List[np.ndarray] = []
    media_errors: List[Dict[str, str]] = []
    modalities_loaded: Tuple[str, ...] = ("text",)
    if spec.backend == "torch":
        modalities_loaded = _needed_modalities(media_files, multimodal)
        threads = torch_threads if torch_threads is not None else workers
        t_load = time.time()
        emb = load_embedder(
            spec, dim=out_dim, modalities=modalities_loaded, dtype=dtype, threads=threads
        )
        _log(
            f"Loaded {spec.id} ({emb.num_parameters / 1e6:.0f}M params, "
            f"modalities={','.join(emb.modalities)}, dtype={emb.dtype_name}, "
            f"threads={threads}) in {time.time() - t_load:.1f}s",
            quiet=quiet,
        )
        parts: List[np.ndarray] = []
        step = max(1, batch_size)
        log_every = max(step, (len(texts) // 10 // step) * step)
        for i in range(0, len(texts), step):
            parts.append(emb.encode_documents(texts[i : i + step], titles[i : i + step], batch_size=step))
            done = min(i + step, len(texts))
            if done == len(texts) or done % log_every == 0:
                _log(f"  [text] {done}/{len(texts)} chunks", quiet=quiet)
        vectors = np.vstack(parts) if parts else np.zeros((0, out_dim), dtype=np.float32)
        if media_files:
            media_stored, media_vecs, media_errors = _embed_media_torch(
                emb, media_files, root=root, opts=media_options or MediaOptions(), quiet=quiet
            )
        dtype_name = emb.dtype_name
    else:
        vectors = _embed_parallel(
            texts,
            model_path=onnx,
            tokenizer_path=tok,
            max_seq_length=spec.max_seq_length,
            dim=spec.dim,
            pooling=spec.pooling,
            model_id=spec.id,
            workers=workers,
            batch_size=batch_size,
        )
        dtype_name = None
    t_embedded = time.time()

    stored = [
        StoredChunk(
            chunk_id=c.chunk_id,
            path=c.path,
            text=c.text,
            start_char=c.start_char,
            end_char=c.end_char,
            token_count=c.token_count,
            title=titles[i] if spec.backend == "torch" else None,
        )
        for i, c in enumerate(chunks)
    ]
    stored.extend(media_stored)
    if media_vecs:
        vectors = np.vstack([vectors] + media_vecs) if len(vectors) else np.vstack(media_vecs)

    store = VectorStore(index_dir)
    new_files = len(text_files) + len(media_files)
    sources = [str(source)]
    if existing_cfg is not None:
        store.load()
        new_rel = {str(f.resolve().relative_to(root.resolve())) for f in text_files + media_files}
        keep = [
            i for i, c in enumerate(store.chunks)
            if (c.source or c.path.split("#", 1)[0]) not in new_rel
        ]
        old_vecs = store.vectors[keep] if store.vectors is not None and keep else None
        stored = [store.chunks[i] for i in keep] + stored
        if old_vecs is not None and len(old_vecs):
            vectors = np.vstack([old_vecs, vectors]) if len(vectors) else old_vecs
        sources = list(dict.fromkeys(list(existing_cfg.get("sources") or [existing_cfg.get("source")]) + sources))

    config: Dict[str, Any] = {
        "source": str(source),
        "max_tokens": effective_max,
        "overlap_tokens": overlap_tokens,
        "workers": workers,
        "files": new_files,
        "chunks": len(stored),
    }
    config.update(spec_to_config(spec))
    config["dim"] = out_dim
    if spec.backend == "torch":
        mod_counts: Dict[str, int] = {}
        for c in stored:
            mod_counts[c.modality or "text"] = mod_counts.get(c.modality or "text", 0) + 1
        config.update(
            {
                "index_format": 2,
                "matryoshka": out_dim != spec.dim,
                "dtype": dtype_name,
                "modalities_loaded": list(modalities_loaded),
                "modality_counts": mod_counts,
                "media_files": len(media_files),
                "media_errors": media_errors,
                "media_options": (media_options or MediaOptions()).validated().__dict__
                if media_files else None,
                "timing_s": {
                    "chunk": round(t_chunked - t_start, 2),
                    "embed": round(t_embedded - t_chunked, 2),
                },
            }
        )
    if skipped_media:
        config["skipped_media"] = len(skipped_media)
    if existing_cfg is not None:
        config["appended"] = True
        config["sources"] = sources
    # Legacy key already set by spec_to_config as "model"
    store.save(stored, vectors, config=config)
    return config
