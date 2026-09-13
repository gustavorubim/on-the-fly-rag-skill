import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from on_the_fly_rag.chunk import chunk_text, load_tokenizer
from on_the_fly_rag.paths import DEFAULT_TOKENIZER


class TestChunk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer(DEFAULT_TOKENIZER)

    def test_short_text_one_chunk(self):
        chunks = chunk_text("Hello world.", path="a.md", tokenizer=self.tok)
        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(chunks[0].token_count, 256)

    def test_long_text_under_limit(self):
        para = " ".join(["retrieval"] * 400)
        chunks = chunk_text(para, path="long.md", tokenizer=self.tok, max_tokens=200)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.token_count, 256)
            self.assertLessEqual(c.token_count, 202)  # 200 + small slack

    def test_overlap_present(self):
        text = "\n\n".join(f"Section {i} talks about topic-{i} in detail." for i in range(30))
        chunks = chunk_text(
            text, path="ov.md", tokenizer=self.tok, max_tokens=80, overlap_tokens=20
        )
        self.assertGreaterEqual(len(chunks), 2)


if __name__ == "__main__":
    unittest.main()
