import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".github" / "skills" / "on-the-fly-rag"
sys.path.insert(0, str(SKILL_ROOT))

from on_the_fly_rag.shard import shard_file, unshard_file


class TestShard(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "weights.bin"
            src.write_bytes(bytes(range(256)) * 2000)  # ~512KB
            parts = shard_file(src, output_prefix=td / "weights.bin.part", shard_size=50_000)
            self.assertGreaterEqual(len(parts), 2)
            for p in parts:
                self.assertLessEqual(p.stat().st_size, 50_000)
            out = td / "weights.out"
            got = unshard_file(td / "weights.bin.part", output_path=out)
            self.assertEqual(src.read_bytes(), got.read_bytes())


if __name__ == "__main__":
    unittest.main()
