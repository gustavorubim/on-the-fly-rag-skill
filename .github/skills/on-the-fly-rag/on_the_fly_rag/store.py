"""Local vector store: numpy matrix + JSONL metadata."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StoredChunk:
    chunk_id: str
    path: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    # Multimodal metadata (defaults keep old text-only indexes loadable).
    modality: str = "text"  # text | image | video | audio
    source: Optional[str] = None  # source file path (no #fragment)
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    title: Optional[str] = None


_STORED_FIELDS = set(StoredChunk.__dataclass_fields__)  # type: ignore[attr-defined]


def chunk_to_json(c: StoredChunk) -> Dict[str, Any]:
    """Serialize, omitting multimodal fields that are unset (old format for text)."""
    d = asdict(c)
    for k in ("source", "start_sec", "end_sec", "frame_start", "frame_end", "title"):
        if d.get(k) is None:
            d.pop(k, None)
    if d.get("modality") == "text":
        d.pop("modality", None)
    return d


def parse_modalities(value: Optional[Iterable[str] | str]) -> Optional[set]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [v.strip().lower() for v in value.replace("|", ",").split(",")]
    else:
        parts = [str(v).strip().lower() for v in value]
    mods = {p for p in parts if p}
    valid = {"text", "image", "video", "audio"}
    bad = mods - valid
    if bad:
        raise ValueError(f"unknown modality {sorted(bad)}; choose from {sorted(valid)}")
    return mods or None


def _path_matches(
    path: str,
    *,
    path_contains: Optional[str] = None,
    path_glob: Optional[str] = None,
    path_prefix: Optional[str] = None,
) -> bool:
    # Paths may include a section fragment (e.g. report.pptx#slide-2).
    norm = path.replace("\\", "/")
    base = norm.split("#", 1)[0]
    base_name = Path(base).name
    if path_contains and path_contains.lower() not in norm.lower():
        return False
    if path_prefix:
        pref = path_prefix.replace("\\", "/")
        if not (norm.startswith(pref) or base.startswith(pref)):
            return False
    if path_glob:
        patterns = [p.strip() for p in path_glob.split("|") if p.strip()]
        if patterns and not any(
            fnmatch.fnmatch(norm, pat)
            or fnmatch.fnmatch(base, pat)
            or fnmatch.fnmatch(base_name, pat)
            or fnmatch.fnmatch(Path(norm).name, pat)
            for pat in patterns
        ):
            return False
    return True


class VectorStore:
    def __init__(self, index_dir: Path | str) -> None:
        self.index_dir = Path(index_dir)
        self.meta_path = self.index_dir / "chunks.jsonl"
        self.vectors_path = self.index_dir / "vectors.npy"
        self.config_path = self.index_dir / "config.json"
        self.chunks: List[StoredChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.config: Dict[str, Any] = {}

    def save(
        self,
        chunks: Sequence[StoredChunk],
        vectors: np.ndarray,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with self.meta_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(chunk_to_json(c), ensure_ascii=False) + "\n")
        np.save(self.vectors_path, vectors.astype(np.float32))
        cfg = dict(config or {})
        cfg.setdefault("count", len(chunks))
        cfg.setdefault("dim", int(vectors.shape[1]) if len(vectors) else 0)
        self.config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self.chunks = list(chunks)
        self.vectors = vectors.astype(np.float32)
        self.config = cfg

    def load(self) -> None:
        if not self.meta_path.is_file() or not self.vectors_path.is_file():
            raise FileNotFoundError(
                f"No index at {self.index_dir}. Run ingest first."
            )
        chunks: List[StoredChunk] = []
        with self.meta_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(StoredChunk(**{k: v for k, v in obj.items() if k in _STORED_FIELDS}))
        self.chunks = chunks
        self.vectors = np.load(self.vectors_path)
        if self.config_path.is_file():
            try:
                self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.config = {}
        else:
            self.config = {}

    def search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 5,
        path_contains: Optional[str] = None,
        path_glob: Optional[str] = None,
        path_prefix: Optional[str] = None,
        modality: Optional[Iterable[str] | str] = None,
    ) -> List[Tuple[float, StoredChunk]]:
        if self.vectors is None or not self.chunks:
            self.load()
        mods = parse_modalities(modality)
        assert self.vectors is not None
        q = query_vec.astype(np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self.vectors @ q
        indices = list(range(len(self.chunks)))
        if path_contains or path_glob or path_prefix or mods:
            indices = [
                i
                for i in indices
                if (not mods or (self.chunks[i].modality or "text") in mods)
                and _path_matches(
                    self.chunks[i].path,
                    path_contains=path_contains,
                    path_glob=path_glob,
                    path_prefix=path_prefix,
                )
            ]
            if not indices:
                return []
            scores_view = scores[indices]
            order = np.argsort(-scores_view)[:top_k]
            return [(float(scores_view[j]), self.chunks[indices[j]]) for j in order]
        order = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.chunks[i]) for i in order]
