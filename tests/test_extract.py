"""Offline tests for PDF/DOCX/PPTX extractors and ingest path."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from on_the_fly_rag.chunk import chunk_path, iter_files, load_tokenizer
from on_the_fly_rag.extract import (
    ExtractError,
    extract_docx,
    extract_pdf,
    extract_pptx,
    load_document,
    try_load_document,
)
from on_the_fly_rag.ingest import ingest
from on_the_fly_rag.paths import DEFAULT_TOKENIZER
from on_the_fly_rag.search import search

OFFICE = ROOT / "fixtures" / "office"
EVAL = ROOT / "fixtures" / "eval_corpus"


class TestExtractors(unittest.TestCase):
    def test_pdf_text(self):
        text = extract_pdf(OFFICE / "tiny.pdf")
        self.assertIn("embedding retrieval", text.lower())

    def test_docx_text(self):
        text = extract_docx(OFFICE / "tiny.docx")
        self.assertIn("semantic search", text.lower())

    def test_pptx_includes_notes(self):
        text = extract_pptx(OFFICE / "tiny.pptx")
        self.assertIn("Office extractors", text)
        self.assertIn("Speaker notes", text)
        self.assertIn("[Slide 1]", text)

    def test_load_document_dispatch(self):
        self.assertIn("Tiny PDF", load_document(OFFICE / "tiny.pdf"))
        self.assertIn("Tiny DOCX", load_document(OFFICE / "tiny.docx"))
        md = ROOT / "fixtures" / "sample_docs" / "intro.md"
        self.assertTrue(len(load_document(md)) > 0)

    def test_bad_pdf_raises_or_skips(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "broken.pdf"
            bad.write_bytes(b"%PDF-1.4 not a real pdf junk")
            with self.assertRaises(ExtractError):
                extract_pdf(bad)
            self.assertIsNone(try_load_document(bad))

    def test_iter_files_includes_office(self):
        paths = {p.name for p in iter_files(OFFICE)}
        self.assertIn("tiny.pdf", paths)
        self.assertIn("tiny.docx", paths)
        self.assertIn("tiny.pptx", paths)

    def test_chunk_path_uses_extractor(self):
        tok = load_tokenizer(DEFAULT_TOKENIZER)
        chunks = chunk_path(OFFICE / "tiny.pdf", root=OFFICE, tokenizer=tok)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("embedding", chunks[0].text.lower())

    def test_ingest_skips_bad_file_without_crash(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "mix"
            src.mkdir()
            (src / "ok.txt").write_text("hello retrieval world about chunks", encoding="utf-8")
            (src / "bad.pdf").write_bytes(b"not-a-pdf")
            index = Path(td) / "idx"
            with self.assertLogs("on_the_fly_rag.extract", level="WARNING") as cm:
                cfg = ingest(src, index_dir=index, workers=1)
            self.assertGreaterEqual(cfg["chunks"], 1)
            self.assertTrue(any("bad.pdf" in r for r in cm.output))


class TestEvalCorpusIngest(unittest.TestCase):
    def test_eval_corpus_office_ingest_search(self):
        with tempfile.TemporaryDirectory() as td:
            index = Path(td) / "idx"
            cfg = ingest(EVAL, index_dir=index, workers=2, batch_size=8)
            self.assertGreaterEqual(cfg["files"], 4)
            self.assertGreater(cfg["chunks"], 0)
            hits = search(
                "NovaSync latency SLA milliseconds",
                index_dir=index,
                top_k=5,
                path_glob="*.pdf",
            )
            self.assertGreaterEqual(len(hits), 1)
            self.assertTrue(hits[0]["path"].endswith(".pdf"))
            joined = " ".join(h["text"] for h in hits).lower()
            self.assertTrue("50" in joined or "latency" in joined)


if __name__ == "__main__":
    unittest.main()
