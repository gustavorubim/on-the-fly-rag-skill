"""Active-index defaults, state file, and -i override (offline)."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from on_the_fly_rag.active import (
    default_index_dir,
    load_active,
    resolve_index_dir,
    save_active,
)
from on_the_fly_rag.cli import main
from on_the_fly_rag.ingest import ingest

SAMPLE = ROOT / "fixtures" / "sample_docs"


class TestDefaultIndexPath(unittest.TestCase):
    def test_beside_source_folder(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "corpus"
            src.mkdir()
            self.assertEqual(default_index_dir(src), (src / ".rag_index").resolve())

    def test_file_source_uses_parent(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "notes.md"
            f.write_text("hi", encoding="utf-8")
            self.assertEqual(default_index_dir(f), (Path(td) / ".rag_index").resolve())


class TestActiveState(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "state.json"
            src = Path(td) / "docs"
            idx = Path(td) / "docs" / ".rag_index"
            src.mkdir()
            with mock.patch.dict(os.environ, {"ON_THE_FLY_RAG_STATE": str(state)}):
                payload = save_active(source=src, index_dir=idx)
                self.assertEqual(payload["source"], str(src.resolve()))
                self.assertEqual(payload["index_dir"], str(idx.resolve()))
                self.assertIn("updated_at", payload)
                loaded = load_active()
                self.assertEqual(loaded["index_dir"], payload["index_dir"])

    def test_resolve_explicit_override(self):
        with tempfile.TemporaryDirectory() as td:
            explicit = Path(td) / "custom_idx"
            # Create minimal index markers
            explicit.mkdir()
            (explicit / "vectors.npy").write_bytes(b"")
            (explicit / "chunks.jsonl").write_text("", encoding="utf-8")
            got = resolve_index_dir(explicit, require_exists=True)
            self.assertEqual(got, explicit)

    def test_resolve_active_when_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "state.json"
            idx = Path(td) / "idx"
            idx.mkdir()
            (idx / "vectors.npy").write_bytes(b"")
            (idx / "chunks.jsonl").write_text("", encoding="utf-8")
            with mock.patch.dict(os.environ, {"ON_THE_FLY_RAG_STATE": str(state)}):
                save_active(source=Path(td) / "src", index_dir=idx)
                got = resolve_index_dir(None, require_exists=True)
                self.assertEqual(got.resolve(), idx.resolve())

    def test_resolve_errors_without_active(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "missing.json"
            with mock.patch.dict(os.environ, {"ON_THE_FLY_RAG_STATE": str(state)}):
                with mock.patch("on_the_fly_rag.active.Path.cwd", return_value=Path(td)):
                    with self.assertRaises(FileNotFoundError):
                        resolve_index_dir(None, require_exists=True)


class TestCliActiveDefaults(unittest.TestCase):
    def test_ingest_sets_active_and_search_uses_it(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            state = td_path / "active.json"
            # Copy-ish: point ingest at fixtures but write index under td
            index = td_path / "corpus" / ".rag_index"
            # Put a symlink or just ingest fixtures with -o
            with mock.patch.dict(os.environ, {"ON_THE_FLY_RAG_STATE": str(state)}):
                err = io.StringIO()
                out = io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    rc = main(
                        [
                            "ingest",
                            str(SAMPLE),
                            "-o",
                            str(index),
                            "-j",
                            "1",
                            "--batch-size",
                            "8",
                        ]
                    )
                self.assertEqual(rc, 0)
                cfg = json.loads(out.getvalue())
                self.assertIn("active", cfg)
                self.assertEqual(Path(cfg["active"]["index_dir"]), index.resolve())
                self.assertTrue(state.is_file())

                # status
                out2 = io.StringIO()
                with redirect_stdout(out2), redirect_stderr(io.StringIO()):
                    rc = main(["status", "--json"])
                self.assertEqual(rc, 0)
                st = json.loads(out2.getvalue())
                self.assertEqual(Path(st["index_dir"]), index.resolve())

                # search without -i uses active
                out3 = io.StringIO()
                with redirect_stdout(out3), redirect_stderr(io.StringIO()):
                    rc = main(["search", "parallel CPU workers", "-k", "2", "--json"])
                self.assertEqual(rc, 0)
                hits = json.loads(out3.getvalue())
                self.assertGreaterEqual(len(hits), 1)

    def test_search_i_override_ignores_active(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            state = td_path / "active.json"
            idx_a = td_path / "a"
            idx_b = td_path / "b"
            ingest(SAMPLE, index_dir=idx_a, workers=1, batch_size=8)
            ingest(SAMPLE, index_dir=idx_b, workers=1, batch_size=8)
            with mock.patch.dict(os.environ, {"ON_THE_FLY_RAG_STATE": str(state)}):
                save_active(source=SAMPLE, index_dir=idx_a)
                out = io.StringIO()
                with redirect_stdout(out), redirect_stderr(io.StringIO()):
                    rc = main(
                        [
                            "search",
                            "authentication",
                            "-i",
                            str(idx_b),
                            "-k",
                            "1",
                            "--json",
                        ]
                    )
                self.assertEqual(rc, 0)
                hits = json.loads(out.getvalue())
                self.assertGreaterEqual(len(hits), 1)
                # Active still points at A
                self.assertEqual(Path(load_active()["index_dir"]), idx_a.resolve())

    def test_use_switches_active(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            state = td_path / "active.json"
            idx = td_path / "switched"
            ingest(SAMPLE, index_dir=idx, workers=1, batch_size=8)
            with mock.patch.dict(os.environ, {"ON_THE_FLY_RAG_STATE": str(state)}):
                out = io.StringIO()
                with redirect_stdout(out), redirect_stderr(io.StringIO()):
                    rc = main(["use", str(idx), "--source", str(SAMPLE)])
                self.assertEqual(rc, 0)
                payload = json.loads(out.getvalue())
                self.assertEqual(Path(payload["index_dir"]), idx.resolve())
                self.assertEqual(Path(load_active()["index_dir"]), idx.resolve())

    def test_ingest_default_path_beside_source(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            state = td_path / "active.json"
            # Copy sample docs into a temp corpus so default index lands there
            corpus = td_path / "my_docs"
            corpus.mkdir()
            for f in SAMPLE.iterdir():
                if f.is_file():
                    (corpus / f.name).write_bytes(f.read_bytes())
            with mock.patch.dict(os.environ, {"ON_THE_FLY_RAG_STATE": str(state)}):
                out = io.StringIO()
                with redirect_stdout(out), redirect_stderr(io.StringIO()):
                    rc = main(
                        ["ingest", str(corpus), "-j", "1", "--batch-size", "8"]
                    )
                self.assertEqual(rc, 0)
                expected = corpus / ".rag_index"
                self.assertTrue((expected / "vectors.npy").is_file())
                cfg = json.loads(out.getvalue())
                self.assertEqual(Path(cfg["active"]["index_dir"]), expected.resolve())


if __name__ == "__main__":
    unittest.main()
