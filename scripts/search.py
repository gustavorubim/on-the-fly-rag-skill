#!/usr/bin/env python3
"""Wrapper: python scripts/search.py "query" [args...]"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from on_the_fly_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["search", *sys.argv[1:]]))
