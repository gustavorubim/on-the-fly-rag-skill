"""Token-aware chunking for MiniLM (max 256 tokens)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from tokenizers import Tokenizer

from .extract import BINARY_EXTENSIONS, try_load_document

logger = logging.getLogger(__name__)

# Leave room for [CLS]/[SEP] and stay under MiniLM's 256 ctx.
DEFAULT_MAX_TOKENS = 200
DEFAULT_OVERLAP_TOKENS = 40
SPECIAL_TOKEN_OVERHEAD = 2  # CLS + SEP

# Text/code extensions read as UTF-8; binary formats go through extractors.
TEXT_EXTENSIONS = (
    ".md",
    ".txt",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".sh",
    ".sql",
    ".html",
    ".css",
)

DEFAULT_EXTENSIONS = TEXT_EXTENSIONS + tuple(sorted(BINARY_EXTENSIONS))


@dataclass
class Chunk:
    text: str
    path: str
    start_char: int
    end_char: int
    chunk_id: str
    token_count: int


def load_tokenizer(tokenizer_path: Path | str) -> Tokenizer:
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()
    return tok


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def chunk_text(
    text: str,
    *,
    path: str = "",
    tokenizer: Tokenizer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> List[Chunk]:
    """Split text into overlapping chunks under ``max_tokens`` (incl. specials).

    Encode the full document without specials, take sliding windows of token
    ids, and slice the **original** text via character offsets so snippets
    match the source exactly.
    """
    if not text or not text.strip():
        return []

    max_content = max(16, max_tokens - SPECIAL_TOKEN_OVERHEAD)
    overlap_tokens = max(0, min(overlap_tokens, max_content - 1))
    step = max(1, max_content - overlap_tokens)

    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = enc.ids
    offsets = enc.offsets
    if not ids:
        return []

    chunks: List[Chunk] = []
    start = 0
    while start < len(ids):
        end = min(start + max_content, len(ids))
        # Skip empty-offset tokens at edges when possible
        start_char = offsets[start][0]
        end_char = offsets[end - 1][1]
        # Expand to include any zero-width tokens inside the window
        for i in range(start, end):
            a, b = offsets[i]
            if b > a:
                start_char = min(start_char, a) if start_char < b else a
                end_char = max(end_char, b)
        # Recompute cleanly:
        nonempty = [(a, b) for a, b in offsets[start:end] if b > a]
        if nonempty:
            start_char = nonempty[0][0]
            end_char = nonempty[-1][1]
            piece = text[start_char:end_char]
        else:
            piece = ""
        piece = piece.strip()
        if piece:
            cid = f"{path}::{len(chunks)}" if path else f"chunk-{len(chunks)}"
            chunks.append(
                Chunk(
                    text=piece,
                    path=path,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_id=cid,
                    token_count=(end - start) + SPECIAL_TOKEN_OVERHEAD,
                )
            )
        if end >= len(ids):
            break
        start += step

    return chunks



_SLIDE_SPLIT = re.compile(r"(?=\[Slide \d+\])")


def _sections_for_chunking(text: str) -> list[tuple[str, str]]:
    """Split extractor output into labeled sections when slide markers exist.

    Returns list of (path_suffix, section_text). path_suffix is "" for a single
    body, or "#slide-N" for PPTX-style sections so multi-hop filters/debug can
    see which slide a hit came from while ``path`` basename filters still work.
    """
    text = text.strip()
    if not text:
        return []
    if "[Slide " not in text:
        return [("", text)]
    parts = [p.strip() for p in _SLIDE_SPLIT.split(text) if p.strip()]
    out: list[tuple[str, str]] = []
    for part in parts:
        m = re.match(r"\[Slide (\d+)\]", part)
        suffix = f"#slide-{m.group(1)}" if m else ""
        out.append((suffix, part))
    return out if out else [("", text)]


def iter_files(
    root: Path,
    *,
    extensions: Optional[Sequence[str]] = None,
) -> Iterator[Path]:
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    root = root.resolve()
    if root.is_file():
        yield root
        return
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".rag_index",
        "models",
        ".tox",
        "dist",
        "build",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() in ext_set:
            yield path


def chunk_path(
    path: Path,
    *,
    root: Path,
    tokenizer: Tokenizer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> List[Chunk]:
    """Load via ``load_document`` (extractors for PDF/DOCX/PPTX), then chunk.

    Unreadable/unparseable files are logged and skipped (empty list) so parallel
    ingest does not abort the whole run. PPTX slide markers become separate
    sections (``path#slide-N``) so multi-hop retrieval can target topics.
    """
    text = try_load_document(path)
    if text is None:
        return []
    rel = str(path.resolve().relative_to(root.resolve()))
    chunks: List[Chunk] = []
    for suffix, section in _sections_for_chunking(text):
        section_path = f"{rel}{suffix}"
        part_chunks = chunk_text(
            section,
            path=section_path,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunks.extend(part_chunks)
    return chunks
