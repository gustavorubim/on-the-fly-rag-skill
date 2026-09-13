"""ONNX embedding (MiniLM mean-pool or Granite sentence_embedding / CLS)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .paths import DEFAULT_MODEL_ONNX, DEFAULT_TOKENIZER
from .registry import ModelSpec, ensure_onnx, get_preset, resolve_model

MAX_SEQ_LENGTH = 256  # MiniLM default; Granite specs override


class OnnxEmbedder:
    """Local ONNX embedder via onnxruntime (no HF download at runtime)."""

    def __init__(
        self,
        model_path: Union[Path, str] = DEFAULT_MODEL_ONNX,
        tokenizer_path: Union[Path, str] = DEFAULT_TOKENIZER,
        max_seq_length: int = MAX_SEQ_LENGTH,
        *,
        dim: Optional[int] = None,
        pooling: Optional[str] = None,
        model_id: Optional[str] = None,
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
        self.model_id = model_id or "custom"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Truncation only; padding is applied per-batch (dynamic) below.
        self.tokenizer.enable_truncation(max_length=max_seq_length)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._output_names = [o.name for o in self.session.get_outputs()]

        if pooling:
            self.pooling = pooling
        elif "sentence_embedding" in self._output_names:
            self.pooling = "sentence_embedding"
        else:
            self.pooling = "mean"

        if dim is not None:
            self.dim = int(dim)
        else:
            # Prefer sentence_embedding shape; else last_hidden_state hidden size
            for o in self.session.get_outputs():
                if o.name == "sentence_embedding" and len(o.shape) >= 2:
                    try:
                        self.dim = int(o.shape[-1])
                        break
                    except (TypeError, ValueError):
                        pass
            else:
                for o in self.session.get_outputs():
                    if o.name == "last_hidden_state" and len(o.shape) >= 3:
                        try:
                            self.dim = int(o.shape[-1])
                            break
                        except (TypeError, ValueError):
                            pass
                else:
                    self.dim = 384

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> "OnnxEmbedder":
        ensure_onnx(spec)
        return cls(
            model_path=spec.onnx_path,
            tokenizer_path=spec.tokenizer_path,
            max_seq_length=spec.max_seq_length,
            dim=spec.dim,
            pooling=spec.pooling,
            model_id=spec.id,
        )

    @classmethod
    def from_model_id(cls, model_id: str = "minilm") -> "OnnxEmbedder":
        return cls.from_spec(get_preset(model_id))

    @classmethod
    def from_resolve(
        cls,
        model: Optional[str | Path] = None,
        tokenizer: Optional[str | Path] = None,
    ) -> "OnnxEmbedder":
        spec, onnx, tok = resolve_model(model, tokenizer=tokenizer)
        return cls(
            model_path=onnx,
            tokenizer_path=tok,
            max_seq_length=spec.max_seq_length,
            dim=spec.dim if spec.id != "custom" else None,
            pooling=spec.pooling if spec.id != "custom" else None,
            model_id=spec.id,
        )

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
        # Encode without fixed padding, then pad to batch max (capped by trunc).
        encodings = self.tokenizer.encode_batch(texts)
        lengths = [len(e.ids) for e in encodings]
        max_len = max(lengths) if lengths else 1
        max_len = min(max_len, self.max_seq_length)

        batch = len(texts)
        input_ids = np.zeros((batch, max_len), dtype=np.int64)
        attention_mask = np.zeros((batch, max_len), dtype=np.int64)
        for i, e in enumerate(encodings):
            ids = e.ids[:max_len]
            mask = e.attention_mask[:max_len]
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(mask)] = mask

        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self.session.run(None, feeds)
        by_name = {name: arr for name, arr in zip(self._output_names, outputs)}

        if self.pooling == "sentence_embedding" and "sentence_embedding" in by_name:
            pooled = by_name["sentence_embedding"]
        elif self.pooling in ("cls", "sentence_embedding"):
            hidden = by_name.get("last_hidden_state", outputs[0])
            pooled = hidden[:, 0, :]
        else:
            # mean pool
            hidden = by_name.get("last_hidden_state", outputs[0])
            mask = attention_mask.astype(np.float32)[:, :, None]
            summed = (hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = summed / counts

        pooled = np.asarray(pooled, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, 1e-12, None)
        return pooled.astype(np.float32)


# Back-compat alias used by older call sites / tests
class MiniLMEmbedder(OnnxEmbedder):
    """Local all-MiniLM-L6-v2 embedder via onnxruntime (no HF download)."""

    def __init__(
        self,
        model_path: Union[Path, str] = DEFAULT_MODEL_ONNX,
        tokenizer_path: Union[Path, str] = DEFAULT_TOKENIZER,
        max_seq_length: int = MAX_SEQ_LENGTH,
    ) -> None:
        super().__init__(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_seq_length=max_seq_length,
            dim=384,
            pooling="mean",
            model_id="minilm",
        )
