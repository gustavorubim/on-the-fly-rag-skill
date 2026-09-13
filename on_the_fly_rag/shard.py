"""Split / reassemble large model weight files for GitHub's 100MB limit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

DEFAULT_SHARD_SIZE = 90 * 1024 * 1024  # 90 MiB


def shard_file(
    input_path: Path | str,
    *,
    output_prefix: Path | str | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> List[Path]:
    src = Path(input_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    prefix = Path(output_prefix) if output_prefix else Path(str(src) + ".part")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    for old in sorted(prefix.parent.glob(prefix.name + "*")):
        suffix = old.name[len(prefix.name) :]
        if suffix.isdigit():
            old.unlink()

    parts: List[Path] = []
    idx = 0
    with src.open("rb") as f:
        while True:
            data = f.read(shard_size)
            if not data:
                break
            part_path = Path(f"{prefix}{idx:02d}")
            part_path.write_bytes(data)
            parts.append(part_path)
            idx += 1

    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    manifest = Path(str(prefix) + ".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "source": src.name,
                "sha256": digest,
                "shard_size": shard_size,
                "parts": [p.name for p in parts],
                "total_bytes": src.stat().st_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return parts


def unshard_file(
    prefix: Path | str,
    *,
    output_path: Path | str | None = None,
) -> Path:
    prefix = Path(prefix)
    if prefix.name.endswith("00") and not Path(str(prefix) + ".manifest.json").exists():
        base = str(prefix)
        while base and base[-1].isdigit():
            base = base[:-1]
        prefix = Path(base)

    manifest_path = Path(str(prefix) + ".manifest.json")
    parent = prefix.parent

    if manifest_path.is_file():
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        part_names = meta["parts"]
        expected = meta.get("sha256")
        out_name = meta.get("source")
    else:
        parts = sorted(parent.glob(prefix.name + "[0-9]*"))
        part_names = [p.name for p in parts if p.name[len(prefix.name) :].isdigit()]
        expected = None
        out_name = None
        if not part_names:
            raise FileNotFoundError(f"No shard parts found for prefix {prefix}")

    if output_path is None:
        if out_name:
            output_path = parent / out_name
        elif str(prefix).endswith(".part"):
            output_path = Path(str(prefix)[: -len(".part")])
        else:
            output_path = Path(str(prefix) + ".reassembled")
    else:
        output_path = Path(output_path)

    with output_path.open("wb") as out:
        for name in part_names:
            part = parent / name
            if not part.is_file():
                raise FileNotFoundError(part)
            out.write(part.read_bytes())

    if expected:
        got = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if got != expected:
            output_path.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch after unshard: expected {expected}, got {got}"
            )
    return output_path
