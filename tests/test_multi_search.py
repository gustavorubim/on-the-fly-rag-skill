"""Path filters + multi_search / coverage helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from on_the_fly_rag.cli import main
from on_the_fly_rag.ingest import ingest
from on_the_fly_rag.search import coverage_stats, multi_search, search

EVAL = ROOT / "fixtures" / "eval_corpus"
SAMPLE = ROOT / "fixtures" / "sample_docs"


class TestPathFiltersAndMulti(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.index = Path(cls._td.name) / "idx"
        ingest(EVAL, index_dir=cls.index, workers=2, batch_size=8)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_path_glob_pdf_only(self):
        hits = search(
            "authentication OAuth2",
            index_dir=self.index,
            top_k=5,
            path_glob="*.pdf",
        )
        self.assertTrue(hits)
        self.assertTrue(all(h["path"].endswith(".pdf") for h in hits))

    def test_path_contains_pricing(self):
        hits = search(
            "Pro tier cost dollars",
            index_dir=self.index,
            top_k=5,
            path_contains="pricing",
        )
        self.assertTrue(hits)
        self.assertTrue(all("pricing" in h["path"].lower() for h in hits))

    def test_multi_search_compare_docs(self):
        queries = [
            {
                "id": "spec_latency",
                "query": "NovaSync p99 latency SLA under 50 milliseconds",
                "path_glob": "*product_spec*",
            },
            {
                "id": "ops_latency",
                "query": "NovaSync observed p99 latency production",
                "path_glob": "*ops_status*",
            },
            {
                "id": "pricing_pro",
                "query": "Pro tier price and requests per minute",
                "path_contains": "pricing",
            },
        ]
        results = multi_search(queries, index_dir=self.index, top_k=3)
        self.assertEqual(len(results), 3)
        cov = coverage_stats(results)
        self.assertTrue(cov["ok"], cov)
        # Spec should surface 50ms; Ops 120ms
        spec_text = " ".join(h["text"] for h in results[0]["hits"])
        ops_text = " ".join(h["text"] for h in results[1]["hits"])
        self.assertIn("50", spec_text)
        self.assertIn("120", ops_text)


    def test_pptx_slide_paths(self):
        hits = search(
            "API keys AND OAuth2 both accepted production",
            index_dir=self.index,
            top_k=3,
            path_glob="*.pptx",
        )
        self.assertTrue(hits)
        self.assertIn("#slide-2", hits[0]["path"])
        self.assertIn("API keys", hits[0]["text"])

    def test_cli_multi_search_json(self):
        qfile = Path(self._td.name) / "q.json"
        qfile.write_text(
            json.dumps(
                [
                    {"id": "a", "query": "OAuth2 only authentication", "path_glob": "*.pdf"},
                    {"id": "b", "query": "API keys accepted in production", "path_glob": "*.pptx"},
                ]
            ),
            encoding="utf-8",
        )
        # Capture stdout
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(
                [
                    "multi-search",
                    str(qfile),
                    "--index",
                    str(self.index),
                    "--json",
                    "-k",
                    "3",
                ]
            )
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertIn("results", payload)
        self.assertIn("coverage", payload)
        self.assertEqual(len(payload["results"]), 2)


if __name__ == "__main__":
    unittest.main()
