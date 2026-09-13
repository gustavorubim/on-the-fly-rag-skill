"""Local vector store: numpy matrix + JSONL metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StoredChunk:
    chunk_id: str
    path: str
    text: str
    start_char: int
    end_char: int
    token_count: int


class VectorStore:
    def __init__(self, index_dir: Path | str) -> None:
        self.index_dir = Path(index_dir)
        self.meta_path = self.index_dir / "chunks.jsonl"
        self.vectors_path = self.index_dir / "vectors.npy"
        self.config_path = self.index_dir / "config.json"
        self.chunks: List[StoredChunk] = []
        self.vectors: Optional[np.ndarray] = None

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
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(self.vectors_path, vectors.astype(np.float32))
        cfg = dict(config or {})
        cfg.setdefault("count", len(chunks))
        cfg.setdefault("dim", int(vectors.shape[1]) if len(vectors) else 0)
        self.config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self.chunks = list(chunks)
        self.vectors = vectors.astype(np.float32)

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
                chunks.append(StoredChunk(**obj))
        self.chunks = chunks
        self.vectors = np.load(self.vectors_path)

    def search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 5,
        path_contains: Optional[str] = None,
    ) -> List[Tuple[float, StoredChunk]]:
        if self.vectors is None or not self.chunks:
            self.load()
        assert self.vectors is not None
        q = query_vec.astype(np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self.vectors @ q
        indices = list(range(len(self.chunks)))
        if path_contains:
            needle = path_contains.lower()
            indices = [i for i in indices if needle in self.chunks[i].path.lower()]
            if not indices:
                return []
            scores_view = scores[indices]
            order = np.argsort(-scores_view)[:top_k]
            return [(float(scores_view[j]), self.chunks[indices[j]]) for j in order]
        order = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.chunks[i]) for i in order]
