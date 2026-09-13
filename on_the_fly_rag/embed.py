"""ONNX MiniLM embedding (mean-pool + L2 normalize)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .paths import DEFAULT_MODEL_ONNX, DEFAULT_TOKENIZER

MAX_SEQ_LENGTH = 256


class MiniLMEmbedder:
    """Local all-MiniLM-L6-v2 embedder via onnxruntime (no HF download)."""

    def __init__(
        self,
        model_path: Union[Path, str] = DEFAULT_MODEL_ONNX,
        tokenizer_path: Union[Path, str] = DEFAULT_TOKENIZER,
        max_seq_length: int = MAX_SEQ_LENGTH,
    ) -> None:
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "If you only have shards, run: python -m on_the_fly_rag unshard "
                f"--input {model_path}.part"
            )
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_length = max_seq_length
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_seq_length)
        self.tokenizer.enable_padding(length=max_seq_length)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Intra-op threads left to ORT; multi-process ingest owns inter-batch parallelism.
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.dim = 384

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            vectors.append(self._encode_batch(batch, normalize=normalize))
        return np.vstack(vectors)

    def encode_one(self, text: str, *, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]

    def _encode_batch(self, texts: List[str], *, normalize: bool) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self.session.run(None, feeds)
        hidden = outputs[0]
        mask = attention_mask.astype(np.float32)[:, :, None]
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts
        if normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, 1e-12, None)
        return pooled.astype(np.float32)
