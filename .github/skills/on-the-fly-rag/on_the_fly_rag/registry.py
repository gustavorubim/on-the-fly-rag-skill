"""Bundled embedding model registry (MiniLM + IBM Granite R2 + EmbeddingGemma 2).

Two backends:

* ``onnx``  – MiniLM / Granite via onnxruntime (base install, unchanged).
* ``torch`` – EmbeddingGemma 2 via sentence-transformers / transformers on PyTorch
  (optional extras: ``requirements-gemma.txt`` or ``pip install .[gemma]``).
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .paths import SKILL_ROOT
from .shard import unshard_file

DEFAULT_MODEL_ID = "minilm"

# Preset ids accepted by CLI `--model`
PRESET_IDS = ("minilm", "granite-small", "granite", "embeddinggemma-2")

# Presets that need the optional PyTorch extras.
TORCH_PRESET_IDS = ("embeddinggemma-2",)
GEMMA_INSTALL_HINT = (
    "pip install -r requirements-gemma.txt   (or: pip install '.[gemma]' from the skill folder)"
)


@dataclass(frozen=True)
class ModelSpec:
    """One vendored (or path-resolved) embedder."""

    id: str
    label: str
    hf_id: str
    model_dir: Path
    dim: int
    max_seq_length: int
    pooling: str  # mean | cls | sentence_embedding
    params_m: int
    context_note: str
    retrieval_ballpark: str
    license: str
    onnx_name: str = "model.onnx"
    tokenizer_name: str = "tokenizer.json"
    needs_unshard_hint: bool = False
    backend: str = "onnx"  # onnx | torch
    modalities: Tuple[str, ...] = ("text",)
    matryoshka_dims: Tuple[int, ...] = ()
    ram_note: str = ""

    @property
    def onnx_path(self) -> Path:
        """Path of the weights file (``model.onnx`` or ``model.safetensors``)."""
        return self.model_dir / self.onnx_name

    @property
    def weights_path(self) -> Path:
        return self.onnx_path

    @property
    def is_multimodal(self) -> bool:
        return len(self.modalities) > 1

    def allowed_dims(self) -> Tuple[int, ...]:
        return self.matryoshka_dims or (self.dim,)

    @property
    def tokenizer_path(self) -> Path:
        return self.model_dir / self.tokenizer_name

    @property
    def shard_prefix(self) -> Path:
        return Path(str(self.onnx_path) + ".part")

    @property
    def shard_manifest(self) -> Path:
        return Path(str(self.shard_prefix) + ".manifest.json")


def _bundled() -> Dict[str, ModelSpec]:
    models = SKILL_ROOT / "models"
    return {
        "minilm": ModelSpec(
            id="minilm",
            label="MiniLM (all-MiniLM-L6-v2)",
            hf_id="sentence-transformers/all-MiniLM-L6-v2",
            model_dir=models / "all-MiniLM-L6-v2",
            dim=384,
            max_seq_length=256,
            pooling="mean",
            params_m=22,
            context_note="256 tok",
            retrieval_ballpark="good general baseline; tiny/offline-fast",
            license="Apache-2.0",
            needs_unshard_hint=False,
        ),
        "granite-small": ModelSpec(
            id="granite-small",
            label="Granite Small English R2",
            hf_id="ibm-granite/granite-embedding-small-english-r2",
            model_dir=models / "granite-embedding-small-english-r2",
            dim=384,
            max_seq_length=8192,
            pooling="sentence_embedding",
            params_m=47,
            context_note="8k tok",
            retrieval_ballpark="stronger retrieval than MiniLM; ~94MiB ONNX",
            license="Apache-2.0",
            needs_unshard_hint=False,
        ),
        "granite": ModelSpec(
            id="granite",
            label="Granite English R2",
            hf_id="ibm-granite/granite-embedding-english-r2",
            model_dir=models / "granite-embedding-english-r2",
            dim=768,
            max_seq_length=8192,
            pooling="sentence_embedding",
            params_m=149,
            context_note="8k tok",
            retrieval_ballpark="best quality of the three; sharded ~303MiB ONNX",
            license="Apache-2.0",
            needs_unshard_hint=True,
        ),
        "embeddinggemma-2": ModelSpec(
            id="embeddinggemma-2",
            label="EmbeddingGemma 2 (multimodal)",
            hf_id="google/embeddinggemma-2",
            model_dir=models / "embeddinggemma-2",
            dim=768,
            max_seq_length=8192,
            pooling="mean",
            params_m=740,
            context_note="8k tok shared across modalities",
            retrieval_ballpark=(
                "text + images + video + audio in one 768-d space (Matryoshka 512/256/128); "
                "PyTorch backend, needs gemma extras; sharded ~1.49GB safetensors"
            ),
            license="Apache-2.0",
            onnx_name="model.safetensors",
            needs_unshard_hint=True,
            backend="torch",
            modalities=("text", "image", "video", "audio"),
            matryoshka_dims=(768, 512, 256, 128),
            ram_note=(
                "CPU FP32: text-only load 271M params (~2.1GB peak RSS; ~1.0GB with --dtype bf16); "
                "full multimodal 744M params (~4.8GB peak RSS). Much slower than the ONNX "
                "models, especially for images (~15s each) and video (~95s per 16-frame segment) on 2 CPU threads"
            ),
        ),
    }


def list_presets() -> List[ModelSpec]:
    return [ _bundled()[k] for k in PRESET_IDS ]


def get_preset(model_id: str) -> ModelSpec:
    key = model_id.strip().lower()
    # aliases
    aliases = {
        "mini-lm": "minilm",
        "all-minilm-l6-v2": "minilm",
        "granite-embedding-small-english-r2": "granite-small",
        "granite-small-r2": "granite-small",
        "granite-embedding-english-r2": "granite",
        "granite-r2": "granite",
        "granite-english": "granite",
        "embeddinggemma": "embeddinggemma-2",
        "embeddinggemma2": "embeddinggemma-2",
        "embedding-gemma-2": "embeddinggemma-2",
        "gemma": "embeddinggemma-2",
        "gemma2": "embeddinggemma-2",
        "google/embeddinggemma-2": "embeddinggemma-2",
    }
    key = aliases.get(key, key)
    specs = _bundled()
    if key not in specs:
        raise KeyError(
            f"Unknown model preset {model_id!r}. "
            f"Choose one of: {', '.join(PRESET_IDS)}"
        )
    return specs[key]



def _rel(path: Path) -> str:
    """Prefer repo-relative paths in user-facing commands."""
    try:
        return str(path.resolve().relative_to(SKILL_ROOT.resolve()))
    except ValueError:
        return str(path)


def model_file_status(spec: ModelSpec) -> Dict[str, Any]:
    """Return readiness info for a bundled model (onnx present / needs unshard)."""
    onnx = spec.onnx_path
    manifest = spec.shard_manifest
    parts = sorted(spec.model_dir.glob(spec.onnx_name + ".part[0-9]*"))
    has_onnx = onnx.is_file()
    has_shards = bool(parts) or manifest.is_file()
    if has_onnx:
        state = "ready"
    elif has_shards:
        state = "needs_unshard"
    else:
        state = "missing"
    return {
        "id": spec.id,
        "label": spec.label,
        "onnx_path": str(onnx),
        "tokenizer_path": str(spec.tokenizer_path),
        "state": state,
        "has_onnx": has_onnx,
        "has_shards": has_shards,
        "shard_prefix": str(spec.shard_prefix) if has_shards else None,
        "unshard_command": (
            f"python -m on_the_fly_rag unshard --input {_rel(spec.shard_prefix)}"
            if has_shards and not has_onnx
            else None
        ),
        "dim": spec.dim,
        "max_seq_length": spec.max_seq_length,
        "params_m": spec.params_m,
        "license": spec.license,
        "backend": spec.backend,
        "modalities": list(spec.modalities),
        "dims": list(spec.allowed_dims()),
        "extras_installed": torch_extras_available() if spec.backend == "torch" else True,
        "install_command": GEMMA_INSTALL_HINT if spec.backend == "torch" else None,
        "ram_note": spec.ram_note or None,
    }


def ensure_onnx(spec: ModelSpec, *, auto_unshard: bool = False) -> Path:
    """Return path to usable ONNX, optionally unsharding first.

    Raises ``FileNotFoundError`` with an actionable message when weights are
    missing or still sharded.
    """
    if spec.onnx_path.is_file():
        return spec.onnx_path
    if spec.shard_manifest.is_file() or list(
        spec.model_dir.glob(spec.onnx_name + ".part[0-9]*")
    ):
        if auto_unshard:
            return unshard_file(spec.shard_prefix, output_path=spec.onnx_path)
        raise FileNotFoundError(
            f"Model {spec.id} weights are sharded at {spec.shard_prefix}*. "
            f"Unshard before use:\n"
            f"  python -m on_the_fly_rag unshard --input {_rel(spec.shard_prefix)}"
        )
    raise FileNotFoundError(
        f"ONNX model not found for {spec.id} at {spec.onnx_path}. "
        "Restore the skill package models/ folder or re-copy the on-the-fly-rag skill."
        if spec.backend == "onnx"
        else f"Model weights not found for {spec.id} at {spec.onnx_path}. "
        "Restore the skill package models/ folder or re-copy the on-the-fly-rag skill."
    )


def resolve_model(
    model: Optional[str | Path] = None,
    *,
    tokenizer: Optional[str | Path] = None,
) -> Tuple[ModelSpec, Path, Path]:
    """Resolve CLI ``--model`` (preset id or ONNX path) → spec, onnx, tokenizer.

    Default preset is MiniLM. When ``model`` is a filesystem path to an ONNX
    file, a synthetic spec is built (pooling inferred by the embedder).
    """
    if model is None or (isinstance(model, str) and model.strip() == ""):
        model = DEFAULT_MODEL_ID

    raw = str(model).strip()
    raw_lower = raw.lower()

    # 1) Preset id (and aliases)
    try:
        spec = get_preset(raw_lower)
        ensure_onnx(spec)
        tok = Path(tokenizer) if tokenizer else spec.tokenizer_path
        return spec, spec.onnx_path, tok
    except KeyError:
        pass

    # 2) Path to ONNX or model directory
    path = Path(raw).expanduser()
    if path.is_dir():
        for spec in list_presets():
            if path.resolve() == spec.model_dir.resolve():
                ensure_onnx(spec)
                tok = Path(tokenizer) if tokenizer else spec.tokenizer_path
                return spec, spec.onnx_path, tok
        onnx = path / "model.onnx"
        tok = Path(tokenizer) if tokenizer else path / "tokenizer.json"
    else:
        onnx = path
        tok = Path(tokenizer) if tokenizer else onnx.parent / "tokenizer.json"
        for spec in list_presets():
            if onnx.resolve() == spec.onnx_path.resolve() or (
                onnx.parent.resolve() == spec.model_dir.resolve()
                and onnx.name == spec.onnx_name
            ):
                ensure_onnx(spec)
                return (
                    spec,
                    spec.onnx_path,
                    Path(tokenizer) if tokenizer else spec.tokenizer_path,
                )

    if not onnx.is_file():
        prefix = Path(str(onnx) + ".part")
        if Path(str(prefix) + ".manifest.json").is_file() or list(
            onnx.parent.glob(onnx.name + ".part[0-9]*")
        ):
            raise FileNotFoundError(
                f"ONNX model not found at {onnx} (sharded). Unshard with:\n"
                f"  python -m on_the_fly_rag unshard --input {prefix}"
            )
        raise FileNotFoundError(f"ONNX model not found at {onnx}")

    synthetic = ModelSpec(
        id="custom",
        label=f"custom ({onnx.name})",
        hf_id="custom",
        model_dir=onnx.parent,
        dim=384,
        max_seq_length=256,
        pooling="mean",
        params_m=0,
        context_note="custom",
        retrieval_ballpark="custom ONNX path",
        license="unknown",
        onnx_name=onnx.name,
        tokenizer_name=Path(tok).name,
    )
    return synthetic, onnx, Path(tok)



def spec_to_config(spec: ModelSpec) -> Dict[str, Any]:
    """Fields stored in index config.json so search can reload the same model."""
    return {
        "model_id": spec.id,
        "model_label": spec.label,
        "model_hf_id": spec.hf_id,
        "model": str(spec.onnx_path),
        "tokenizer": str(spec.tokenizer_path),
        "dim": spec.dim,
        "native_dim": spec.dim,
        "max_seq_length": spec.max_seq_length,
        "pooling": spec.pooling,
        "backend": spec.backend,
    }


def load_spec_from_index_config(config: Dict[str, Any]) -> Optional[ModelSpec]:
    """Rebuild a ModelSpec from ingest config when possible."""
    mid = config.get("model_id")
    if mid and mid in PRESET_IDS:
        return get_preset(str(mid))
    model_path = config.get("model")
    if model_path:
        try:
            spec, _, _ = resolve_model(model_path, tokenizer=config.get("tokenizer"))
            return spec
        except (FileNotFoundError, KeyError, OSError):
            return None
    return None


def format_choice_outline() -> str:
    """Agent-facing text for the first embed/index ask (model picker)."""
    lines = [
        "Choose an embedding model before the first ingest (default: MiniLM):",
        "",
    ]
    for spec in list_presets():
        st = model_file_status(spec)
        ready = {
            "ready": "ready",
            "needs_unshard": "needs unshard before first use",
            "missing": "weights missing",
        }[st["state"]]
        default = " **[default]**" if spec.id == DEFAULT_MODEL_ID else ""
        lines.append(
            f"- `{spec.id}` — {spec.label}{default}\n"
            f"  {spec.params_m}M params · {spec.dim}-d · ctx {spec.context_note} · "
            f"{spec.retrieval_ballpark}\n"
            + (f"  backend: {spec.backend} · modalities: {', '.join(spec.modalities)}"
               + (f" · --dim {'/'.join(str(d) for d in spec.allowed_dims())}" if spec.matryoshka_dims else "")
               + "\n" if spec.backend != "onnx" else "")
            + (f"  cost: {spec.ram_note}\n" if spec.ram_note else "")
            + (f"  extras: {'installed' if st.get('extras_installed') else 'NOT installed → ' + GEMMA_INSTALL_HINT}\n"
               if spec.backend == "torch" else "")
            + f"  status: {ready}"
            + (
                f" → `{st['unshard_command']}`"
                if st.get("unshard_command")
                else ""
            )
        )
    lines.append("")
    lines.append(
        "Reply with `minilm`, `granite-small`, `granite`, or `embeddinggemma-2` "
        "(or confirm default). Pick `embeddinggemma-2` when the corpus has images, "
        "video or audio, or when cross-modal search matters."
    )
    return "\n".join(lines)


def all_models_status() -> List[Dict[str, Any]]:
    return [model_file_status(s) for s in list_presets()]


def torch_extras_available() -> bool:
    """True when the optional PyTorch backend (gemma extras) is importable."""
    return all(
        importlib.util.find_spec(m) is not None
        for m in ("torch", "transformers", "sentence_transformers")
    )


class IndexCompatError(ValueError):
    """Raised when a query/append would mix embedding models or dimensions."""


def check_index_compat(
    config: Dict[str, Any],
    *,
    model_id: Optional[str] = None,
    dim: Optional[int] = None,
    action: str = "search",
) -> None:
    """Refuse to mix models / Matryoshka dims within one index.

    ``config`` is the index ``config.json``. ``model_id`` / ``dim`` are what the
    caller wants to use (``None`` = not specified → accept the index's value).
    Legacy indexes without ``model_id`` are only checked on ``dim``.
    """
    idx_model = config.get("model_id")
    idx_dim = config.get("dim")
    if model_id is not None and idx_model:
        try:
            want = get_preset(str(model_id)).id
        except KeyError:
            want = str(model_id)
        if want != idx_model:
            verb = "append to" if action == "append" else "search"
            raise IndexCompatError(
                f"Index was built with model {idx_model!r} (dim={idx_dim}); refusing to "
                f"{verb} it with model {want!r}. Vectors from different models are not "
                f"comparable. Use --model {idx_model}, or ingest into a new --index."
            )
    if dim is not None and idx_dim is not None and int(dim) != int(idx_dim):
        verb = "append to" if action == "append" else "query"
        raise IndexCompatError(
            f"Index dim is {idx_dim} (model {idx_model or '?'}); refusing to {verb} it at "
            f"dim {dim}. Queries and documents must share one Matryoshka dimension. "
            f"Use --dim {idx_dim}, or re-ingest into a new --index with --dim {dim}."
        )


def validate_dim(spec: ModelSpec, dim: Optional[int]) -> int:
    """Return the effective output dim for ``spec`` (Matryoshka-aware)."""
    if dim is None:
        return spec.dim
    dim = int(dim)
    allowed: Sequence[int] = spec.allowed_dims()
    if dim not in allowed:
        raise ValueError(
            f"--dim {dim} is not supported by {spec.id}; choose one of "
            f"{', '.join(str(d) for d in allowed)}"
            + ("" if spec.matryoshka_dims else " (no Matryoshka support)")
        )
    return dim
