"""Model registry, config recording, and unshard wiring (offline)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".github" / "skills" / "on-the-fly-rag"
sys.path.insert(0, str(SKILL_ROOT))

from on_the_fly_rag.ingest import ingest
from on_the_fly_rag.registry import (
    DEFAULT_MODEL_ID,
    PRESET_IDS,
    all_models_status,
    format_choice_outline,
    get_preset,
    model_file_status,
    resolve_model,
    spec_to_config,
)
from on_the_fly_rag.shard import shard_file, unshard_file
from on_the_fly_rag.store import VectorStore

SAMPLE = ROOT / "fixtures" / "sample_docs"


class TestRegistry(unittest.TestCase):
    def test_presets_exist(self):
        self.assertEqual(DEFAULT_MODEL_ID, "minilm")
        self.assertEqual(PRESET_IDS, ("minilm", "granite-small", "granite"))
        for pid in PRESET_IDS:
            spec = get_preset(pid)
            self.assertEqual(spec.id, pid)
            self.assertTrue(spec.tokenizer_path.is_file(), msg=spec.tokenizer_path)

    def test_minilm_ready(self):
        st = model_file_status(get_preset("minilm"))
        self.assertEqual(st["state"], "ready")

    def test_granite_small_ready(self):
        st = model_file_status(get_preset("granite-small"))
        self.assertEqual(st["state"], "ready")
        self.assertEqual(st["dim"], 384)

    def test_granite_needs_unshard_or_ready(self):
        st = model_file_status(get_preset("granite"))
        self.assertIn(st["state"], {"ready", "needs_unshard"})
        self.assertEqual(st["dim"], 768)
        if st["state"] == "needs_unshard":
            self.assertIn("unshard", st["unshard_command"])

    def test_resolve_minilm_default(self):
        spec, onnx, tok = resolve_model(None)
        self.assertEqual(spec.id, "minilm")
        self.assertTrue(onnx.is_file())
        self.assertTrue(tok.is_file())

    def test_choice_outline_mentions_all(self):
        text = format_choice_outline()
        for pid in PRESET_IDS:
            self.assertIn(f"`{pid}`", text)

    def test_all_models_status(self):
        rows = all_models_status()
        self.assertEqual(len(rows), 3)


class TestConfigRecordsModel(unittest.TestCase):
    def test_ingest_writes_model_id(self):
        with tempfile.TemporaryDirectory() as td:
            idx = Path(td) / "idx"
            cfg = ingest(SAMPLE, index_dir=idx, model="minilm", workers=1, batch_size=8)
            self.assertEqual(cfg["model_id"], "minilm")
            self.assertEqual(cfg["dim"], 384)
            self.assertEqual(cfg["pooling"], "mean")
            stored = json.loads((idx / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["model_id"], "minilm")
            store = VectorStore(idx)
            store.load()
            self.assertEqual(store.config.get("model_id"), "minilm")

    def test_spec_to_config_keys(self):
        spec = get_preset("granite-small")
        cfg = spec_to_config(spec)
        self.assertEqual(cfg["model_id"], "granite-small")
        self.assertEqual(cfg["dim"], 384)
        self.assertEqual(cfg["pooling"], "sentence_embedding")


class TestUnshardWiring(unittest.TestCase):
    def test_fake_shard_roundtrip_and_status(self):
        """Tiny fake weight + manifest; status reports needs_unshard then ready."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model_dir = td / "fake-model"
            model_dir.mkdir()
            src = model_dir / "model.onnx"
            # tiny fake "onnx" payload
            src.write_bytes(b"FAKEONNX" + bytes(range(256)) * 400)
            parts = shard_file(src, shard_size=20_000)
            self.assertGreaterEqual(len(parts), 2)
            src.unlink()
            self.assertFalse(src.is_file())

            # Mimic registry status logic
            from on_the_fly_rag.registry import ModelSpec

            spec = ModelSpec(
                id="fake",
                label="fake",
                hf_id="fake",
                model_dir=model_dir,
                dim=8,
                max_seq_length=32,
                pooling="mean",
                params_m=0,
                context_note="test",
                retrieval_ballpark="test",
                license="MIT",
            )
            st = model_file_status(spec)
            self.assertEqual(st["state"], "needs_unshard")
            self.assertIsNotNone(st["unshard_command"])

            out = unshard_file(spec.shard_prefix, output_path=spec.onnx_path)
            self.assertTrue(out.is_file())
            st2 = model_file_status(spec)
            self.assertEqual(st2["state"], "ready")


if __name__ == "__main__":
    unittest.main()
