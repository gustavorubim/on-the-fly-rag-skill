from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Skill package root: .github/skills/on-the-fly-rag (or ~/.copilot/skills/on-the-fly-rag)
SKILL_ROOT = PACKAGE_ROOT.parent
# Back-compat alias: code historically called this REPO_ROOT; models live next to the package.
REPO_ROOT = SKILL_ROOT
DEFAULT_MODEL_DIR = SKILL_ROOT / "models" / "all-MiniLM-L6-v2"
DEFAULT_MODEL_ONNX = DEFAULT_MODEL_DIR / "model.onnx"
DEFAULT_TOKENIZER = DEFAULT_MODEL_DIR / "tokenizer.json"
GRANITE_SMALL_DIR = SKILL_ROOT / "models" / "granite-embedding-small-english-r2"
GRANITE_DIR = SKILL_ROOT / "models" / "granite-embedding-english-r2"
