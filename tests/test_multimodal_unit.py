"""EmbeddingGemma 2 plumbing that does NOT need the model or torch.

Covers prompts, precision guard, Matryoshka truncation, index model/dim
compatibility refusals, the modality filter, store back-compat, media
segmenting (needs an ffmpeg binary) and the ONNX "skip media" notice.
"""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".github" / "skills" / "on-the-fly-rag"
sys.path.insert(0, str(SKILL_ROOT))

from on_the_fly_rag.embed_torch import (  # noqa: E402
    PrecisionError,
    format_document,
    format_query,
    matryoshka_truncate,
    normalize_dtype,
)
from on_the_fly_rag.ingest import doc_title, ingest  # noqa: E402
from on_the_fly_rag.media import (  # noqa: E402
    MediaOptions,
    ffmpeg_exe,
    media_items_for,
    modality_of,
)
from on_the_fly_rag.registry import (  # noqa: E402
    IndexCompatError,
    check_index_compat,
    get_preset,
    validate_dim,
)
from on_the_fly_rag.search import search  # noqa: E402
from on_the_fly_rag.store import StoredChunk, VectorStore  # noqa: E402

SAMPLE = ROOT / "fixtures" / "sample_docs"
MM = ROOT / "fixtures" / "multimodal" / "corpus"


class TestPromptsAndPrecision(unittest.TestCase):
    def test_prompts(self):
        self.assertEqual(format_query("red bike"), "task: search result | query: red bike")
        self.assertEqual(
            format_document("body", "deck.pptx slide 3"),
            "title: deck.pptx slide 3 | text: body",
        )
        self.assertEqual(format_document("body", None), "title: none | text: body")
        self.assertEqual(format_document("body", "  "), "title: none | text: body")

    def test_doc_title(self):
        self.assertEqual(doc_title("docs/deck.pptx#slide-4"), "deck.pptx slide 4")
        self.assertEqual(doc_title("a/b/report.pdf"), "report.pdf")
        self.assertEqual(doc_title("notes.md#2"), "notes.md")

    def test_fp16_refused(self):
        for bad in ("fp16", "float16", "half"):
            with self.assertRaises(PrecisionError):
                normalize_dtype(bad)
        self.assertEqual(normalize_dtype(None), "float32")
        self.assertEqual(normalize_dtype("bf16"), "bfloat16")
        self.assertEqual(normalize_dtype("fp32"), "float32")

    def test_matryoshka_truncate_renormalizes(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(5, 768)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        for d in (768, 512, 256, 128):
            t = matryoshka_truncate(v, d)
            self.assertEqual(t.shape, (5, d))
            np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, rtol=1e-5)
            # Truncation keeps the leading coordinates' direction.
            np.testing.assert_allclose(
                t[0], v[0, :d] / np.linalg.norm(v[0, :d]), rtol=1e-5, atol=1e-6
            )
        with self.assertRaises(ValueError):
            matryoshka_truncate(v, 1000)


class TestRegistryDims(unittest.TestCase):
    def test_validate_dim(self):
        g = get_preset("embeddinggemma-2")
        self.assertEqual(validate_dim(g, None), 768)
        self.assertEqual(validate_dim(g, 256), 256)
        with self.assertRaises(ValueError):
            validate_dim(g, 300)
        m = get_preset("minilm")
        self.assertEqual(validate_dim(m, 384), 384)
        with self.assertRaises(ValueError):
            validate_dim(m, 128)  # ONNX presets have no Matryoshka

    def test_gemma_preset(self):
        g = get_preset("gemma")
        self.assertEqual(g.id, "embeddinggemma-2")
        self.assertEqual(g.backend, "torch")
        self.assertEqual(set(g.modalities), {"text", "image", "video", "audio"})
        self.assertEqual(get_preset("granite").backend, "onnx")


class TestIndexCompat(unittest.TestCase):
    CFG = {"model_id": "embeddinggemma-2", "dim": 256}

    def test_same_model_and_dim_ok(self):
        check_index_compat(self.CFG, model_id="embeddinggemma-2", dim=256)
        check_index_compat(self.CFG, model_id="gemma", dim=None)  # alias
        check_index_compat(self.CFG)

    def test_other_model_refused(self):
        with self.assertRaises(IndexCompatError) as ctx:
            check_index_compat(self.CFG, model_id="granite")
        self.assertIn("refusing to search", str(ctx.exception))
        with self.assertRaises(IndexCompatError) as ctx:
            check_index_compat(self.CFG, model_id="minilm", action="append")
        self.assertIn("append", str(ctx.exception))

    def test_other_dim_refused(self):
        with self.assertRaises(IndexCompatError) as ctx:
            check_index_compat(self.CFG, dim=768)
        self.assertIn("--dim 256", str(ctx.exception))

    def test_search_and_append_refusals_on_real_index(self):
        with tempfile.TemporaryDirectory() as td:
            idx = Path(td) / "idx"
            ingest(SAMPLE, index_dir=idx, model="minilm", workers=1)
            with self.assertRaises(IndexCompatError):
                search("ingest workers", index_dir=idx, model="granite-small")
            with self.assertRaises(IndexCompatError):
                search("ingest workers", index_dir=idx, dim=256)
            with self.assertRaises(IndexCompatError):
                ingest(SAMPLE, index_dir=idx, model="granite-small", workers=1, append=True)
            # Same model appends fine and keeps one copy per file.
            before = json.loads((idx / "config.json").read_text())["chunks"]
            cfg = ingest(SAMPLE, index_dir=idx, workers=1, append=True)
            self.assertEqual(cfg["model_id"], "minilm")
            self.assertEqual(cfg["chunks"], before)
            self.assertTrue(search("ingest workers", index_dir=idx, top_k=1))


class TestStoreModality(unittest.TestCase):
    def test_filter_and_backcompat(self):
        with tempfile.TemporaryDirectory() as td:
            vs = VectorStore(td)
            chunks = [
                StoredChunk("a::0", "a.md", "alpha", 0, 5, 1),
                StoredChunk(
                    "v.mp4#t=0.0-16.0::0", "v.mp4#t=0.0-16.0", "[video]", 0, 0, 0,
                    modality="video", source="v.mp4", start_sec=0.0, end_sec=16.0,
                    frame_start=0, frame_end=15,
                ),
                StoredChunk("i.png::0", "i.png", "[image]", 0, 0, 0, modality="image", source="i.png"),
            ]
            vecs = np.eye(3, 4, dtype=np.float32)
            vs.save(chunks, vecs, config={"model_id": "embeddinggemma-2", "dim": 4})
            lines = (Path(td) / "chunks.jsonl").read_text().splitlines()
            self.assertNotIn("modality", json.loads(lines[0]))  # text rows keep old format
            vs2 = VectorStore(td)
            vs2.load()
            q = np.array([0.1, 0.9, 0.5, 0], dtype=np.float32)
            self.assertEqual(vs2.search(q, top_k=1)[0][1].modality, "video")
            hits = vs2.search(q, top_k=3, modality="image")
            self.assertEqual([h[1].modality for h in hits], ["image"])
            hits = vs2.search(q, top_k=3, modality="text,image")
            self.assertEqual({h[1].modality for h in hits}, {"text", "image"})
            with self.assertRaises(ValueError):
                vs2.search(q, modality="smell")

    def test_legacy_rows_load(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "chunks.jsonl").write_text(
                json.dumps({"chunk_id": "x", "path": "x.md", "text": "t", "start_char": 0,
                            "end_char": 1, "token_count": 1, "future_field": 1}) + "\n"
            )
            np.save(Path(td, "vectors.npy"), np.ones((1, 3), dtype=np.float32))
            vs = VectorStore(td)
            vs.load()
            self.assertEqual(vs.chunks[0].modality, "text")


class TestOnnxSkipsMedia(unittest.TestCase):
    def test_minilm_skips_media_with_notice(self):
        with tempfile.TemporaryDirectory() as td:
            err = io.StringIO()
            with redirect_stderr(err):
                cfg = ingest(MM, index_dir=Path(td) / "idx", model="minilm", workers=1)
            self.assertIn("skipping", err.getvalue())
            self.assertGreater(cfg["skipped_media"], 0)
            vs = VectorStore(Path(td) / "idx")
            vs.load()
            self.assertTrue(all(c.modality == "text" for c in vs.chunks))
            self.assertTrue(any(c.path.endswith("notes.md") for c in vs.chunks))
            with self.assertRaises(ValueError):
                search(None, index_dir=Path(td) / "idx",
                       query_file=ROOT / "fixtures/multimodal/queries/pizza_query.jpg")


@unittest.skipUnless(ffmpeg_exe(), "ffmpeg binary not available")
class TestMediaSegmenting(unittest.TestCase):
    def test_modality_of(self):
        self.assertEqual(modality_of("a/B.JPG"), "image")
        self.assertEqual(modality_of("x.webm"), "video")
        self.assertEqual(modality_of("x.flac"), "audio")
        self.assertEqual(modality_of("x.md"), "text")

    def test_image_item(self):
        items = media_items_for(MM / "images" / "pizza.jpg", root=MM)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].modality, "image")
        self.assertEqual(items[0].path, "images/pizza.jpg")

    def test_audio_windows_offsets(self):
        items = media_items_for(MM / "audio" / "piano.flac", root=MM,
                                opts=MediaOptions(audio_window_s=2.0))
        self.assertEqual([round(i.start_sec, 1) for i in items], [0.0, 2.0, 4.0])
        self.assertTrue(items[1].path.startswith("audio/piano.flac#t=2.0-4.0"))
        arr = items[0].model_input()
        self.assertEqual(arr["sampling_rate"], 16000)
        self.assertEqual(arr["array"].dtype, np.float32)
        # Window is clamped to the processor's 280-token (11.2 s) audio cap.
        self.assertLessEqual(MediaOptions(audio_window_s=60).validated().audio_window_s, 11.2)

    def test_short_audio_single_item(self):
        items = media_items_for(MM / "audio" / "speech.wav", root=MM)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].path, "audio/speech.wav")

    def test_video_segments(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "clip.mp4"
            subprocess.run(
                [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "testsrc=size=64x48:rate=10:duration=9", "-pix_fmt", "yuv420p", str(out)],
                check=True,
            )
            items = media_items_for(out, root=Path(td),
                                    opts=MediaOptions(video_fps=1.0, segment_frames=4))
            self.assertGreaterEqual(len(items), 2)
            self.assertEqual(items[0].path, "clip.mp4#t=0.0-4.0")
            self.assertEqual((items[1].frame_start, items[1].start_sec), (4, 4.0))
            inp = items[0].model_input()
            self.assertEqual(inp["array"].shape[1:], (48, 64, 3))
            self.assertIn("video_metadata", inp)
            # Segment length is clamped to the processor's 32-frame cap.
            self.assertEqual(MediaOptions(segment_frames=100).validated().segment_frames, 32)


if __name__ == "__main__":
    unittest.main()
