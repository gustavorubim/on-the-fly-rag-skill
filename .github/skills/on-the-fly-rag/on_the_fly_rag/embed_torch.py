"""PyTorch backend: EmbeddingGemma 2 via sentence-transformers (optional extras).

Imported lazily so the base (ONNX-only) install never needs torch. Install with
``pip install -r requirements-gemma.txt``.

Rules implemented here (from the model card):

* prompts – queries: ``task: search result | query: {q}``;
  documents: ``title: {title|none} | text: {chunk}``; media gets **no** prefix.
* precision – FP32 (default, best on most CPUs) or BF16. FP16 is refused: the
  model's activations overflow float16 and silently produce NaNs.
* Matryoshka – embed at 768, keep the leading ``dim`` values, then L2-normalize.
* selective loading – text-only (~271M params) unless images/video/audio are
  needed: ``config_kwargs={"vision_config": None, "audio_config": None}``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT_FMT = "title: {title} | text: "
NATIVE_DIM = 768
MATRYOSHKA_DIMS = (768, 512, 256, 128)
_FORBIDDEN_DTYPES = {"float16", "fp16", "half", "torch.float16"}
_DTYPE_ALIASES = {
    "float32": "float32", "fp32": "float32", "f32": "float32", "torch.float32": "float32",
    "bfloat16": "bfloat16", "bf16": "bfloat16", "torch.bfloat16": "bfloat16",
}


class PrecisionError(ValueError):
    """Raised when FP16 (unsupported by EmbeddingGemma 2) is requested."""


def normalize_dtype(dtype: Optional[str]) -> str:
    """Map a user dtype string to float32/bfloat16; refuse float16."""
    raw = (dtype or os.environ.get("ON_THE_FLY_RAG_DTYPE") or "float32").strip().lower()
    if raw in _FORBIDDEN_DTYPES:
        raise PrecisionError(
            "EmbeddingGemma 2 must run in float32 or bfloat16; float16 overflows and "
            "returns NaN / degraded embeddings (see model card). Use --dtype fp32 or bf16."
        )
    if raw not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype {dtype!r}; use fp32 or bf16")
    return _DTYPE_ALIASES[raw]


def format_query(q: str) -> str:
    return f"{QUERY_PROMPT}{q}"


def format_document(text: str, title: Optional[str] = None) -> str:
    t = (title or "").strip().replace("\n", " ") or "none"
    return f"{DOC_PROMPT_FMT.format(title=t)}{text}"


def matryoshka_truncate(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Keep the first ``dim`` components and re-L2-normalize each row."""
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    if dim > v.shape[1]:
        raise ValueError(f"cannot truncate {v.shape[1]}-d vectors to {dim}")
    v = v[:, :dim]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.clip(norms, 1e-12, None)).astype(np.float32)


def config_kwargs_for(modalities: Iterable[str]) -> dict:
    """Selective encoder loading (text tower is always loaded)."""
    mods = set(modalities)
    need_vision = bool(mods & {"image", "video"})
    need_audio = "audio" in mods
    ck: dict = {}
    if not need_vision:
        ck["vision_config"] = None
    if not need_audio:
        ck["audio_config"] = None
    return ck


def _set_threads(threads: Optional[int]) -> None:
    import torch

    env = os.environ.get("ON_THE_FLY_RAG_TORCH_THREADS")
    n = threads if threads is not None else (int(env) if env else None)
    if n is not None and n > 0:
        torch.set_num_threads(int(n))


class GemmaEmbedder:
    """EmbeddingGemma 2 (local vendored folder) on CPU via sentence-transformers."""

    backend = "torch"

    def __init__(
        self,
        model_dir: Path | str,
        *,
        dim: int = NATIVE_DIM,
        modalities: Sequence[str] = ("text",),
        dtype: Optional[str] = None,
        threads: Optional[int] = None,
        model_id: str = "embeddinggemma-2",
        max_seq_length: int = 8192,
        device: str = "cpu",
    ) -> None:
        if int(dim) not in MATRYOSHKA_DIMS:
            raise ValueError(f"dim must be one of {MATRYOSHKA_DIMS}, got {dim}")
        self.dtype_name = normalize_dtype(dtype)
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on env
            raise ImportError(
                "embeddinggemma-2 needs the optional PyTorch extras: "
                "pip install -r requirements-gemma.txt"
            ) from e

        model_dir = Path(model_dir)
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"EmbeddingGemma 2 weights not found at {model_dir / 'model.safetensors'}. "
                "Unshard first: python -m on_the_fly_rag unshard --model embeddinggemma-2"
            )
        _set_threads(threads)
        if not os.environ.get("ON_THE_FLY_RAG_VERBOSE"):
            try:  # keep stderr readable: no per-tensor "Loading weights" bar
                from transformers.utils import logging as hf_logging

                hf_logging.disable_progress_bar()
                hf_logging.set_verbosity_error()
            except Exception:  # noqa: BLE001
                pass
        self.model_dir = model_dir
        self.model_id = model_id
        self.dim = int(dim)
        self.modalities = tuple(sorted(set(modalities) | {"text"}))
        torch_dtype = getattr(torch, self.dtype_name)
        self.model = SentenceTransformer(
            str(model_dir),
            device=device,
            local_files_only=True,
            model_kwargs={"dtype": torch_dtype},
            config_kwargs=config_kwargs_for(self.modalities),
        )
        self.model.max_seq_length = int(max_seq_length)
        self._guard_precision()
        self.max_seq_length = int(max_seq_length)

    # ------------------------------------------------------------------ guards
    def _guard_precision(self) -> None:
        import torch

        bad = {str(p.dtype) for p in self.model.parameters() if p.dtype == torch.float16}
        if bad:
            raise PrecisionError(
                "EmbeddingGemma 2 loaded with float16 parameters; refusing (NaN risk). "
                "Use fp32 or bf16."
            )

    @property
    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def _finish(self, vecs: Any) -> np.ndarray:
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if not np.isfinite(arr).all():
            raise FloatingPointError(
                "EmbeddingGemma 2 produced non-finite embeddings (NaN/Inf). "
                f"dtype={self.dtype_name}; never run this model in float16."
            )
        return matryoshka_truncate(arr, self.dim)

    def _encode(self, inputs: List[Any], *, batch_size: int) -> np.ndarray:
        if not inputs:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._finish(vecs)

    # ------------------------------------------------------------------ API
    def encode_queries(self, queries: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        return self._encode([format_query(q) for q in queries], batch_size=batch_size)

    def encode_documents(
        self,
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        *,
        batch_size: int = 16,
    ) -> np.ndarray:
        titles = list(titles) if titles is not None else [None] * len(texts)
        docs = [format_document(t, ti) for t, ti in zip(texts, titles)]
        return self._encode(docs, batch_size=batch_size)

    def encode_media(self, items: Sequence[Any], *, batch_size: int = 1) -> np.ndarray:
        """Embed model-ready media inputs (PIL image / audio dict / video dict)."""
        mods = set()
        for it in items:
            if isinstance(it, dict) and "sampling_rate" in it:
                mods.add("audio")
            elif isinstance(it, dict):
                mods.add("video")
            else:
                mods.add("image")
        have = set(self.modalities)
        if (mods & {"image", "video"}) and not (have & {"image", "video"}):
            raise RuntimeError(
                "Embedder was loaded without the vision encoder; re-create it with "
                "image/video modalities (or --multimodal on)."
            )
        if "audio" in mods and "audio" not in have:
            raise RuntimeError(
                "Embedder was loaded without the audio encoder; re-create it with the "
                "audio modality (or --multimodal on)."
            )
        out = [self._encode([it], batch_size=1) for it in items] if batch_size <= 1 else [
            self._encode(list(items[i : i + batch_size]), batch_size=batch_size)
            for i in range(0, len(items), batch_size)
        ]
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    # Back-compat with OnnxEmbedder-style call sites
    def encode(self, texts: Sequence[str], *, batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        return self.encode_documents(texts, batch_size=batch_size)

    def encode_one(self, text: str, *, normalize: bool = True) -> np.ndarray:
        return self.encode_queries([text])[0]
