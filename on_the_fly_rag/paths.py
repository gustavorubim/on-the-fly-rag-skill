from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "all-MiniLM-L6-v2"
DEFAULT_MODEL_ONNX = DEFAULT_MODEL_DIR / "model.onnx"
DEFAULT_TOKENIZER = DEFAULT_MODEL_DIR / "tokenizer.json"
