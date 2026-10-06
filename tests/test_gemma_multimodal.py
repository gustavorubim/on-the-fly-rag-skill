"""Real-model EmbeddingGemma 2 retrieval tests, one class per query type.

Uses the tiny CC0/synthetic fixtures in fixtures/multimodal (see SOURCES.md).
Auto-skipped unless the extras are installed and the weights are unsharded:

    pip install -r .github/skills/on-the-fly-rag/requirements-gemma.txt
    (cd .github/skills/on-the-fly-rag && python -m on_the_fly_rag unshard --model embeddinggemma-2)
    python -m pytest -m gemma -v          # from the repo root

Set ON_THE_FLY_RAG_TORCH_THREADS to control CPU threads.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import pytest

    pytestmark = pytest.mark.gemma
except ImportError:  # plain `python -m unittest` still works
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma_support import GEMMA_READY, GEMMA_SKIP_REASON, SKILL_ROOT  # noqa: E402

sys.path.insert(0, str(SKILL_ROOT))

MM = ROOT / "fixtures" / "multimodal"
CORPUS = MM / "corpus"
QUERIES = MM / "queries"
THREADS = int(os.environ.get("ON_THE_FLY_RAG_TORCH_THREADS") or min(4, os.cpu_count() or 1))


def _base(path: str) -> str:
    return path.split("#", 1)[0]


@unittest.skipUnless(GEMMA_READY, GEMMA_SKIP_REASON)
class GemmaFixtureIndex(unittest.TestCase):
    """One mixed-modality index over the whole fixture corpus (built once)."""

    _td = None
    index = None
    config = None
    embedder = None

    @classmethod
    def setUpClass(cls):
        if GemmaFixtureIndex.index is not None:
            return
        from on_the_fly_rag.embed_torch import GemmaEmbedder
        from on_the_fly_rag.ingest import ingest
        from on_the_fly_rag.registry import get_preset

        GemmaFixtureIndex._td = tempfile.TemporaryDirectory()
        GemmaFixtureIndex.index = Path(GemmaFixtureIndex._td.name) / "mm_idx"
        GemmaFixtureIndex.config = ingest(
            CORPUS,
            index_dir=GemmaFixtureIndex.index,
            model="embeddinggemma-2",
            workers=1,
            batch_size=8,
            torch_threads=THREADS,
        )
        spec = get_preset("embeddinggemma-2")
        GemmaFixtureIndex.embedder = GemmaEmbedder(
            spec.model_dir,
            dim=768,
            modalities=("text", "image", "video", "audio"),
            threads=THREADS,
        )

    def ask(self, query=None, *, modality=None, query_file=None, top_k=5):
        from on_the_fly_rag.search import search

        return search(
            query,
            index_dir=self.index,
            top_k=top_k,
            modality=modality,
            query_file=query_file,
            embedder=self.embedder,
        )

    def assertTop1(self, hits, expected_base):
        self.assertTrue(hits, "no hits")
        got = [(_base(h["path"]), round(h["score"], 3)) for h in hits]
        self.assertEqual(_base(hits[0]["path"]), expected_base, got)


class TestIndexShape(GemmaFixtureIndex):
    def test_config_records_modalities_and_dim(self):
        cfg = self.config
        self.assertEqual(cfg["model_id"], "embeddinggemma-2")
        self.assertEqual(cfg["dim"], 768)
        self.assertEqual(cfg["dtype"], "float32")
        self.assertEqual(cfg["media_errors"], [])
        counts = cfg["modality_counts"]
        for mod in ("text", "image", "audio", "video"):
            self.assertGreater(counts.get(mod, 0), 0, counts)
        saved = json.loads((self.index / "config.json").read_text())
        self.assertEqual(saved["dim"], 768)


class TestTextToText(GemmaFixtureIndex):
    def test_text_query_finds_notes(self):
        hits = self.ask("How often are the tomato beds watered in July?", modality="text")
        self.assertTop1(hits, "notes.md")


class TestTextToImage(GemmaFixtureIndex):
    def test_objects(self):
        for q, want in [
            ("a bunch of yellow bananas", "images/bananas.jpg"),
            ("a red bicycle parked on the grass", "images/red_bicycle.jpg"),
            ("a giraffe among trees", "images/giraffe.jpg"),
            ("a pizza with cheese and tomato sauce", "images/pizza.jpg"),
            ("a sailboat on the water", "images/sailboat.jpg"),
        ]:
            with self.subTest(q=q):
                self.assertTop1(self.ask(q, modality="image"), want)


class TestImageToImage(GemmaFixtureIndex):
    def test_query_image_finds_same_object(self):
        for qf, want in [
            (QUERIES / "pizza_query.jpg", "images/pizza.jpg"),
            (QUERIES / "giraffe_query.jpg", "images/giraffe.jpg"),
        ]:
            with self.subTest(q=qf.name):
                self.assertTop1(self.ask(query_file=qf, modality="image"), want)


class TestTextToAudio(GemmaFixtureIndex):
    """Short (<2 s) sound effects are the weakest modality (see docs/eval.md).

    Assertions are therefore: exact top-1 for the clearly distinct clips, and
    category-level top-1 (an animal clip) + exact top-2 for the dog bark, whose
    0.9 s fixture scores within ~0.01 of the cat meow.
    """

    ANIMALS = ("audio/dog_bark.wav", "audio/cat_meow.wav")

    def test_distinct_sounds_top1(self):
        for q, want in [
            ("a cat meowing", "audio/cat_meow.wav"),
            ("piano music", "audio/piano.flac"),
        ]:
            with self.subTest(q=q):
                self.assertTop1(self.ask(q, modality="audio"), want)

    def test_dog_bark_animal_top1_exact_top2(self):
        hits = self.ask("a dog barking", modality="audio", top_k=2)
        got = [_base(h["path"]) for h in hits]
        self.assertIn(got[0], self.ANIMALS, got)
        self.assertIn("audio/dog_bark.wav", got)


class TestTextToVideo(GemmaFixtureIndex):
    def test_clips(self):
        for q, want in [
            ("a burning candle flame in the dark", "video/candle.mp4"),
            ("a waterfall in a forest gorge", "video/waterfall.mp4"),
        ]:
            with self.subTest(q=q):
                hits = self.ask(q, modality="video")
                self.assertTop1(hits, want)
                self.assertEqual(hits[0]["modality"], "video")
                self.assertIsNotNone(hits[0].get("start_sec"))


class TestMixedIndex(GemmaFixtureIndex):
    """No modality filter: the query must pick the right item *and* modality."""

    def test_animal_sound_query_ranks_audio_first(self):
        hits = self.ask("the sound of a dog barking", top_k=2)
        self.assertEqual(hits[0]["modality"], "audio", hits[0])
        self.assertIn(_base(hits[0]["path"]), TestTextToAudio.ANIMALS)

    def test_cross_modal_top1(self):
        for q, want, mod in [
            ("a red bicycle", "images/red_bicycle.jpg", "image"),
            ("piano music", "audio/piano.flac", "audio"),
            ("video of a candle burning", "video/candle.mp4", "video"),
            ("drip irrigation schedule and rain sensor", "notes.md", "text"),
        ]:
            with self.subTest(q=q):
                hits = self.ask(q)
                self.assertTop1(hits, want)
                self.assertEqual(hits[0]["modality"], mod)


@unittest.skipUnless(GEMMA_READY, GEMMA_SKIP_REASON)
class TestTextOnlyLoadAndDim(unittest.TestCase):
    def test_text_corpus_uses_text_tower_at_dim_256(self):
        from on_the_fly_rag.ingest import ingest
        from on_the_fly_rag.registry import IndexCompatError
        from on_the_fly_rag.search import search
        from on_the_fly_rag.store import VectorStore

        with tempfile.TemporaryDirectory() as td:
            idx = Path(td) / "t"
            cfg = ingest(CORPUS / "notes.md", index_dir=idx, model="gemma", dim=256,
                         workers=1, torch_threads=THREADS)
            self.assertEqual(cfg["modalities_loaded"], ["text"])
            self.assertEqual(cfg["dim"], 256)
            vs = VectorStore(idx)
            vs.load()
            self.assertEqual(vs.vectors.shape[1], 256)
            hits = search("rain sensor pauses watering", index_dir=idx, top_k=1,
                          threads=THREADS)
            self.assertEqual(_base(hits[0]["path"]), "notes.md")
            with self.assertRaises(IndexCompatError):
                search("rain", index_dir=idx, dim=768)
            with self.assertRaises(IndexCompatError):
                ingest(CORPUS / "notes.md", index_dir=idx, model="gemma", dim=512,
                       append=True, workers=1)


if __name__ == "__main__":
    unittest.main()
