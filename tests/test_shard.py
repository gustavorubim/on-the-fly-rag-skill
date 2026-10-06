import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".github" / "skills" / "on-the-fly-rag"
sys.path.insert(0, str(SKILL_ROOT))

from on_the_fly_rag.cli import main as cli_main
from on_the_fly_rag.shard import (
    ChecksumError,
    sha256_file,
    shard_file,
    unshard_file,
    verify_shards,
)


def _make(td: Path, n: int = 2000) -> Path:
    src = td / "weights.bin"
    src.write_bytes(bytes(range(256)) * n)  # ~512KB
    return src


class TestShard(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _make(td)
            parts = shard_file(src, output_prefix=td / "weights.bin.part", shard_size=50_000)
            self.assertGreaterEqual(len(parts), 2)
            for p in parts:
                self.assertLessEqual(p.stat().st_size, 50_000)
            out = td / "weights.out"
            got = unshard_file(td / "weights.bin.part", output_path=out)
            self.assertEqual(src.read_bytes(), got.read_bytes())

    def test_manifest_has_whole_and_part_checksums(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _make(td)
            parts = shard_file(src, output_prefix=td / "weights.bin.part", shard_size=50_000)
            m = json.loads((td / "weights.bin.part.manifest.json").read_text())
            self.assertEqual(m["sha256"], sha256_file(src))
            self.assertEqual(len(m["part_sha256"]), len(parts))
            for p in parts:
                self.assertEqual(sha256_file(p), m["part_sha256"][p.name])
                self.assertEqual(p.stat().st_size, m["part_bytes"][p.name])
            summary = verify_shards(td / "weights.bin.part")
            self.assertEqual(summary["parts"], len(parts))

    def test_corrupt_part_fails_loudly_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _make(td)
            parts = shard_file(src, output_prefix=td / "weights.bin.part", shard_size=50_000)
            raw = bytearray(parts[1].read_bytes())
            raw[10] ^= 0xFF  # same size, different bytes
            parts[1].write_bytes(bytes(raw))
            out = td / "weights.out"
            with self.assertRaises(ChecksumError) as ctx:
                unshard_file(td / "weights.bin.part", output_path=out)
            self.assertIn(parts[1].name, str(ctx.exception))
            self.assertFalse(out.exists())
            self.assertEqual(list(td.glob("*.unshard-tmp")), [])

    def test_truncated_part_fails(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _make(td)
            parts = shard_file(src, output_prefix=td / "weights.bin.part", shard_size=50_000)
            parts[0].write_bytes(parts[0].read_bytes()[:-1])
            with self.assertRaises(ChecksumError):
                verify_shards(td / "weights.bin.part")

    def test_missing_part_fails(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _make(td)
            parts = shard_file(src, output_prefix=td / "weights.bin.part", shard_size=50_000)
            parts[-1].unlink()
            with self.assertRaises(FileNotFoundError):
                unshard_file(td / "weights.bin.part", output_path=td / "out")

    def test_whole_file_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _make(td)
            shard_file(src, output_prefix=td / "weights.bin.part", shard_size=50_000)
            mp = td / "weights.bin.part.manifest.json"
            m = json.loads(mp.read_text())
            m["sha256"] = "0" * 64
            mp.write_text(json.dumps(m))
            out = td / "weights.out"
            with self.assertRaises(ChecksumError) as ctx:
                unshard_file(td / "weights.bin.part", output_path=out)
            self.assertIn("Checksum mismatch", str(ctx.exception))
            self.assertFalse(out.exists())

    def test_legacy_manifest_without_part_checksums(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _make(td)
            shard_file(src, output_prefix=td / "weights.bin.part", shard_size=50_000)
            mp = td / "weights.bin.part.manifest.json"
            m = json.loads(mp.read_text())
            m.pop("part_sha256")
            m.pop("part_bytes")
            mp.write_text(json.dumps(m))
            got = unshard_file(td / "weights.bin.part", output_path=td / "out")
            self.assertEqual(got.read_bytes(), src.read_bytes())

    def test_cli_verify_only_and_exit_code(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _make(td)
            parts = shard_file(src, output_prefix=td / "weights.bin.part", shard_size=50_000)
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(io.StringIO()):
                rc = cli_main(["unshard", "--input", str(td / "weights.bin.part"), "--verify-only"])
            self.assertEqual(rc, 0)
            raw = bytearray(parts[0].read_bytes())
            raw[0] ^= 0x01
            parts[0].write_bytes(bytes(raw))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = cli_main(["unshard", "--input", str(td / "weights.bin.part"), "--verify-only"])
            self.assertEqual(rc, 2)


class TestBundledManifests(unittest.TestCase):
    """The committed manifests carry per-part checksums matching the committed parts' sizes."""

    def test_manifests_consistent(self):
        for mp in (SKILL_ROOT / "models").glob("*/*.part.manifest.json"):
            m = json.loads(mp.read_text())
            self.assertEqual(len(m["sha256"]), 64, mp)
            self.assertIn("part_sha256", m, mp)
            self.assertEqual(sorted(m["part_sha256"]), sorted(m["parts"]), mp)
            self.assertEqual(sum(m["part_bytes"].values()), m["total_bytes"], mp)
            for p, n in m["part_bytes"].items():
                self.assertLessEqual(n, 95 * 1024 * 1024, p)  # GitHub 100 MB file cap
                part = mp.parent / p
                if part.exists():
                    self.assertEqual(part.stat().st_size, n, p)


if __name__ == "__main__":
    unittest.main()
