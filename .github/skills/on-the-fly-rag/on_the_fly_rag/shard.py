"""Split / reassemble large model weight files for GitHub's 100MB limit.

Manifest format (``<prefix>.manifest.json``)::

    {
      "source": "model.safetensors",
      "sha256": "<sha256 of the whole file>",
      "total_bytes": 1488915288,
      "shard_size": 94371840,
      "parts": ["model.safetensors.part00", ...],
      "part_sha256": {"model.safetensors.part00": "<sha256>", ...},
      "part_bytes": {"model.safetensors.part00": 94371840, ...}
    }

``unshard_file`` verifies every part (when ``part_sha256`` is present) *before*
writing, then verifies the reassembled file against ``sha256``. Any mismatch
raises :class:`ChecksumError` and leaves no partial output behind.
Older manifests without ``part_sha256`` still unshard (whole-file check only).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# 90 MiB = 94,371,840 bytes: under GitHub's 100 MiB hard limit and <= 95 MB.
DEFAULT_SHARD_SIZE = 90 * 1024 * 1024
_IO_BLOCK = 8 * 1024 * 1024


class ChecksumError(ValueError):
    """Raised when a shard or the reassembled file fails SHA-256 verification."""


def sha256_file(path: Path | str, *, block: int = _IO_BLOCK) -> str:
    """Stream a file through SHA-256 (constant memory, safe for multi-GB weights)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            data = f.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def manifest_path_for(prefix: Path | str) -> Path:
    return Path(str(prefix) + ".manifest.json")


def shard_file(
    input_path: Path | str,
    *,
    output_prefix: Path | str | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> List[Path]:
    """Split ``input_path`` into ``<prefix>NN`` parts plus a SHA-256 manifest."""
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
    part_sha: Dict[str, str] = {}
    part_bytes: Dict[str, int] = {}
    whole = hashlib.sha256()
    idx = 0
    with src.open("rb") as f:
        while True:
            remaining = shard_size
            part_path = Path(f"{prefix}{idx:02d}")
            ph = hashlib.sha256()
            written = 0
            with part_path.open("wb") as out:
                while remaining > 0:
                    data = f.read(min(_IO_BLOCK, remaining))
                    if not data:
                        break
                    out.write(data)
                    ph.update(data)
                    whole.update(data)
                    written += len(data)
                    remaining -= len(data)
            if written == 0:
                part_path.unlink()
                break
            parts.append(part_path)
            part_sha[part_path.name] = ph.hexdigest()
            part_bytes[part_path.name] = written
            idx += 1
            if written < shard_size:
                break

    manifest = manifest_path_for(prefix)
    manifest.write_text(
        json.dumps(
            {
                "source": src.name,
                "sha256": whole.hexdigest(),
                "total_bytes": src.stat().st_size,
                "shard_size": shard_size,
                "parts": [p.name for p in parts],
                "part_sha256": part_sha,
                "part_bytes": part_bytes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return parts


def _normalize_prefix(prefix: Path) -> Path:
    if prefix.name.endswith("00") and not manifest_path_for(prefix).exists():
        base = str(prefix)
        while base and base[-1].isdigit():
            base = base[:-1]
        return Path(base)
    return prefix


def _per_part(meta: Dict[str, object], key: str) -> Dict[str, object]:
    """Per-part values keyed by part name (accepts a dict or a list parallel to ``parts``)."""
    val = meta.get(key) or {}
    if isinstance(val, list):
        return dict(zip(meta["parts"], val))  # type: ignore[arg-type]
    return dict(val)  # type: ignore[arg-type]


def verify_shards(prefix: Path | str) -> Dict[str, object]:
    """Check every part listed in the manifest (size + SHA-256) without writing.

    Returns a summary dict; raises :class:`ChecksumError` / ``FileNotFoundError``.
    """
    prefix = _normalize_prefix(Path(prefix))
    manifest_path = manifest_path_for(prefix)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No shard manifest at {manifest_path}")
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent = prefix.parent
    part_sha = _per_part(meta, "part_sha256")
    part_bytes = _per_part(meta, "part_bytes")
    checked = 0
    for name in meta["parts"]:
        part = parent / name
        if not part.is_file():
            raise FileNotFoundError(part)
        if name in part_bytes and part.stat().st_size != int(part_bytes[name]):
            raise ChecksumError(
                f"Shard size mismatch for {part}: expected {part_bytes[name]} bytes, "
                f"got {part.stat().st_size}. Re-download / git checkout the shard."
            )
        if name in part_sha:
            got = sha256_file(part)
            if got != part_sha[name]:
                raise ChecksumError(
                    f"SHA-256 mismatch for shard {part}: expected {part_sha[name]}, got {got}. "
                    "The shard is corrupt or truncated (e.g. a Git LFS pointer or partial clone)."
                )
            checked += 1
    return {
        "manifest": str(manifest_path),
        "parts": len(meta["parts"]),
        "parts_checksummed": checked,
        "sha256": meta.get("sha256"),
        "total_bytes": meta.get("total_bytes"),
    }


def unshard_file(
    prefix: Path | str,
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Reassemble parts → file, verifying per-part and whole-file SHA-256.

    Raises :class:`ChecksumError` (a ``ValueError``) on any mismatch; no partial
    output file is left behind.
    """
    prefix = _normalize_prefix(Path(prefix))
    manifest_path = manifest_path_for(prefix)
    parent = prefix.parent

    expected: Optional[str]
    if manifest_path.is_file():
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        part_names = meta["parts"]
        expected = meta.get("sha256")
        out_name = meta.get("source")
        verify_shards(prefix)  # per-part checks first (fail before writing)
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

    tmp = output_path.with_name(output_path.name + ".unshard-tmp")
    h = hashlib.sha256()
    try:
        with tmp.open("wb") as out:
            for name in part_names:
                part = parent / name
                if not part.is_file():
                    raise FileNotFoundError(part)
                with part.open("rb") as f:
                    while True:
                        data = f.read(_IO_BLOCK)
                        if not data:
                            break
                        out.write(data)
                        h.update(data)
        got = h.hexdigest()
        if expected and got != expected:
            raise ChecksumError(
                f"Checksum mismatch after unshard: expected {expected}, got {got}"
            )
        os.replace(tmp, output_path)
    finally:
        tmp.unlink(missing_ok=True)
    return output_path
