"""pytest glue: the ``gemma`` marker auto-skips when the model or extras are missing."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "skills" / "on-the-fly-rag"))

from gemma_support import GEMMA_SKIP_REASON  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if not GEMMA_SKIP_REASON:
        return
    skip = pytest.mark.skip(reason=GEMMA_SKIP_REASON)
    for item in items:
        if "gemma" in item.keywords:
            item.add_marker(skip)
