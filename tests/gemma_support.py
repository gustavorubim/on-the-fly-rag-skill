"""Shared availability check for the real-model EmbeddingGemma 2 tests."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".github" / "skills" / "on-the-fly-rag"
sys.path.insert(0, str(SKILL_ROOT))

from on_the_fly_rag.media import ffmpeg_exe  # noqa: E402
from on_the_fly_rag.registry import get_preset, torch_extras_available  # noqa: E402


def _reason() -> str:
    if os.environ.get("ON_THE_FLY_RAG_SKIP_GEMMA") == "1":
        return "ON_THE_FLY_RAG_SKIP_GEMMA=1"
    if not torch_extras_available():
        return "gemma extras missing: pip install -r requirements-gemma.txt"
    if not get_preset("embeddinggemma-2").weights_path.is_file():
        return "embeddinggemma-2 weights not unsharded: python -m on_the_fly_rag unshard --model embeddinggemma-2"
    if not ffmpeg_exe():
        return "ffmpeg binary not found (system ffmpeg or pip install imageio-ffmpeg)"
    return ""


GEMMA_SKIP_REASON = _reason()
GEMMA_READY = not GEMMA_SKIP_REASON
