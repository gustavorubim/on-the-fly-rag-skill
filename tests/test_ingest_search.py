import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".github" / "skills" / "on-the-fly-rag"
sys.path.insert(0, str(SKILL_ROOT))

from on_the_fly_rag.ingest import ingest
from on_the_fly_rag.search import search

FIXTURES = ROOT / "fixtures" / "sample_docs"


class TestIngestSearch(unittest.TestCase):
    def test_ingest_and_search_offline(self):
        with tempfile.TemporaryDirectory() as td:
            index = Path(td) / "idx"
            cfg = ingest(FIXTURES, index_dir=index, workers=2, batch_size=8)
            self.assertGreater(cfg["chunks"], 0)
            self.assertTrue((index / "vectors.npy").is_file())
            self.assertTrue((index / "chunks.jsonl").is_file())

            hits = search("parallel CPU workers for ingest", index_dir=index, top_k=3)
            self.assertGreaterEqual(len(hits), 1)
            self.assertIn("score", hits[0])
            self.assertTrue(hits[0]["path"])
            # Top hit should relate to ingest/API docs
            joined = " ".join(h["text"].lower() for h in hits)
            self.assertTrue(
                "ingest" in joined or "worker" in joined or "parallel" in joined
            )

    def test_workers_flag_single(self):
        with tempfile.TemporaryDirectory() as td:
            index = Path(td) / "idx1"
            cfg = ingest(FIXTURES, index_dir=index, workers=1)
            self.assertEqual(cfg["workers"], 1)
            meta = json.loads((index / "config.json").read_text())
            self.assertEqual(meta["workers"], 1)


if __name__ == "__main__":
    unittest.main()
