"""Active index state: one default corpus index for day-to-day search.

Convention
----------
* Index files live beside the corpus: ``<source>/.rag_index/``.
* After ingest, the workspace state file records that index as **active**.
* ``search`` / ``multi-search`` use the active index when ``-i`` is omitted.
* Pass ``-i`` / ``--index`` (or ``use``) to juggle multiple vector stores.

State file (first match wins for reads; writes prefer cwd):
  1. ``$ON_THE_FLY_RAG_STATE`` if set
  2. ``./.on-the-fly-rag.json`` (workspace)
  3. ``$XDG_CACHE_HOME/on-the-fly-rag/active.json`` or ``~/.cache/...``
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import all_models_status, load_spec_from_index_config


STATE_FILENAME = ".on-the-fly-rag.json"
INDEX_DIRNAME = ".rag_index"


def default_index_dir(source: Path | str) -> Path:
    """Prefer ``<source>/.rag_index`` (file → parent folder)."""
    source = Path(source).resolve()
    root = source if source.is_dir() else source.parent
    return root / INDEX_DIRNAME


def _xdg_cache_state() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "on-the-fly-rag" / "active.json"
    return Path.home() / ".cache" / "on-the-fly-rag" / "active.json"


def state_path(*, for_write: bool = False) -> Path:
    """Resolve where to read/write active-index state."""
    env = os.environ.get("ON_THE_FLY_RAG_STATE")
    if env:
        return Path(env).expanduser()
    cwd_state = Path.cwd() / STATE_FILENAME
    if for_write:
        return cwd_state
    if cwd_state.is_file():
        return cwd_state
    cache = _xdg_cache_state()
    if cache.is_file():
        return cache
    return cwd_state


def load_active() -> Optional[Dict[str, Any]]:
    path = state_path(for_write=False)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("index_dir"):
        return None
    return data


def save_active(
    *,
    source: Path | str,
    index_dir: Path | str,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Persist active index pointer; returns the written payload."""
    payload: Dict[str, Any] = {
        "source": str(Path(source).resolve()),
        "index_dir": str(Path(index_dir).resolve()),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = path if path is not None else state_path(for_write=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def resolve_index_dir(
    explicit: Optional[Path | str] = None,
    *,
    require_exists: bool = False,
) -> Path:
    """Return explicit ``-i`` path, else active index, else cwd ``.rag_index``.

    Raises ``FileNotFoundError`` when nothing usable is found (and
    ``require_exists`` is True, or no fallback path exists as a directory /
    active pointer).
    """
    if explicit is not None:
        p = Path(explicit)
        if require_exists and not _looks_like_index(p):
            raise FileNotFoundError(f"Index not found: {p}")
        return p

    active = load_active()
    if active:
        p = Path(active["index_dir"])
        if require_exists and not _looks_like_index(p):
            raise FileNotFoundError(
                f"Active index missing or incomplete: {p} "
                f"(re-run ingest or pass -i/--index)"
            )
        return p

    fallback = Path.cwd() / INDEX_DIRNAME
    if _looks_like_index(fallback):
        return fallback

    raise FileNotFoundError(
        "No active index. Ingest a folder first "
        "(sets active state), or pass -i/--index."
    )


def _looks_like_index(path: Path) -> bool:
    return (path / "vectors.npy").is_file() and (path / "chunks.jsonl").is_file()


def _index_model_info(index_dir: Path) -> Dict[str, Any]:
    """Read model_id / readiness from an index config.json if present."""
    cfg_path = index_dir / "config.json"
    info: Dict[str, Any] = {"model_id": None, "model_label": None, "dim": None}
    if not cfg_path.is_file():
        return info
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return info
    info["model_id"] = cfg.get("model_id")
    info["model_label"] = cfg.get("model_label")
    info["dim"] = cfg.get("dim")
    info["model_path"] = cfg.get("model")
    spec = load_spec_from_index_config(cfg)
    if spec is not None:
        from .registry import model_file_status

        st = model_file_status(spec)
        info["model_state"] = st["state"]
        info["unshard_command"] = st.get("unshard_command")
    return info


def format_status(active: Optional[Dict[str, Any]] = None) -> str:
    if active is None:
        active = load_active()
    model_lines: List[str] = []
    for st in all_models_status():
        flag = st["state"]
        extra = ""
        if st.get("unshard_command"):
            extra = f" → {st['unshard_command']}"
        model_lines.append(
            f"  - {st['id']}: {flag} ({st['params_m']}M, {st['dim']}-d){extra}"
        )
    bundled = "bundled models:\n" + "\n".join(model_lines)

    if not active:
        cwd_fb = Path.cwd() / INDEX_DIRNAME
        if _looks_like_index(cwd_fb):
            minfo = _index_model_info(cwd_fb)
            return (
                "No active-index state file.\n"
                f"Fallback cwd index exists: {cwd_fb.resolve()}\n"
                f"index model: {minfo.get('model_id') or '?'} "
                f"(state={minfo.get('model_state', '?')})\n"
                "Tip: run `ingest` (sets active) or `use <index_dir>`.\n"
                + bundled
            )
        return (
            "No active index.\n"
            "Ingest a corpus folder to set one, or `use <index_dir>`.\n"
            + bundled
        )
    idx = Path(active["index_dir"])
    exists = _looks_like_index(idx)
    minfo = _index_model_info(idx) if exists else {}
    lines = [
        f"source:     {active.get('source', '')}",
        f"index_dir:  {idx}",
        f"updated_at: {active.get('updated_at', '')}",
        f"exists:     {exists}",
        f"state_file: {state_path(for_write=False)}",
        f"model_id:   {minfo.get('model_id') or '(unknown)'}",
        f"model:      {minfo.get('model_label') or minfo.get('model_path') or '?'}",
        f"model_state:{minfo.get('model_state') or ('ready' if exists else 'n/a')}",
    ]
    if minfo.get("unshard_command"):
        lines.append(f"unshard:    {minfo['unshard_command']}")
    lines.append(bundled)
    return "\n".join(lines)
