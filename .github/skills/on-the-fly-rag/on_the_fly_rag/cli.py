"""argparse CLI for ingest / search / multi-search / status / use / shard / unshard."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .active import (
    default_index_dir,
    format_status,
    load_active,
    resolve_index_dir,
    save_active,
)
from .ingest import _default_workers, ingest
from .registry import (
    DEFAULT_MODEL_ID,
    PRESET_IDS,
    all_models_status,
    format_choice_outline,
)
from .search import format_multi, format_results, multi_search, search
from .media import MediaOptions
from .registry import get_preset
from .shard import ChecksumError, shard_file, unshard_file, verify_shards


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Embedding model preset or ONNX path. Presets: "
            f"{', '.join(PRESET_IDS)} (default: {DEFAULT_MODEL_ID}). "
            "Search defaults to the model recorded in the index config."
        ),
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Path to tokenizer.json (optional when using a preset)",
    )


def _add_torch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dtype",
        choices=["fp32", "bf16"],
        default=None,
        help="embeddinggemma-2 precision (default fp32; bf16 halves RAM). fp16 is "
        "refused: the model produces NaNs in float16",
    )
    p.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="torch intra-op threads for embeddinggemma-2 (ingest default: -j; "
        "env ON_THE_FLY_RAG_TORCH_THREADS). On a busy CPU, 1-2 can be faster",
    )


def _add_dim_arg(p: argparse.ArgumentParser, *, for_search: bool) -> None:
    p.add_argument(
        "--dim",
        type=int,
        choices=[768, 512, 256, 128],
        default=None,
        help=(
            "Matryoshka output dim (embeddinggemma-2 only). Truncate then L2-normalize; "
            "recorded in the index config"
            if not for_search
            else "Assert the index dim (queries always use the index's dim; a mismatch is refused)"
        ),
    )


def _add_modality_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--modality",
        type=str,
        default=None,
        help="Only return chunks of these modalities: text,image,video,audio (comma-separated)",
    )


def _add_path_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--path-contains",
        type=str,
        default=None,
        help="Only chunks whose path contains this substring (case-insensitive)",
    )
    p.add_argument(
        "--path",
        "--glob",
        dest="path_glob",
        type=str,
        default=None,
        help="fnmatch glob on chunk path or basename (e.g. '*.pdf' or 'docs/a.md'); "
        "use | to OR multiple patterns",
    )
    p.add_argument(
        "--path-prefix",
        type=str,
        default=None,
        help="Only chunks whose relative path starts with this prefix",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="on_the_fly_rag",
        description="On-the-fly RAG: chunk, embed, and search a local folder.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Chunk + embed a folder into a local index")
    p_ing.add_argument("source", type=Path, help="Folder (or file) to index")
    p_ing.add_argument(
        "--index",
        "-o",
        type=Path,
        default=None,
        help="Output index directory (default: <source>/.rag_index)",
    )
    p_ing.add_argument(
        "--no-active",
        action="store_true",
        help="Do not update the workspace active-index state after ingest",
    )
    p_ing.add_argument(
        "--workers",
        "-j",
        type=int,
        default=None,
        help=f"CPU worker processes (default: os.cpu_count()={_default_workers()})",
    )
    p_ing.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Chunk size in model tokens (default 200; the model context is only a ceiling)",
    )
    p_ing.add_argument("--overlap-tokens", type=int, default=40)
    p_ing.add_argument("--batch-size", type=int, default=None,
                       help="Embedding batch size (default 32 ONNX / 8 embeddinggemma-2)")
    p_ing.add_argument(
        "--append",
        action="store_true",
        help="Add/replace files in an existing index (refuses a different model or --dim)",
    )
    p_ing.add_argument(
        "--multimodal",
        choices=["auto", "on", "off"],
        default="auto",
        help="embeddinggemma-2: auto = load vision/audio encoders only if the corpus has "
        "images/video/audio (text-only load otherwise); on = full model; off = skip media",
    )
    p_ing.add_argument("--video-fps", type=float, default=1.0, help="Video frame sampling rate")
    p_ing.add_argument("--video-segment-frames", type=int, default=16,
                       help="Frames per embedded video segment (<=32; 140 tok/frame)")
    p_ing.add_argument("--max-video-frames", type=int, default=600,
                       help="Cap on sampled frames per video file")
    p_ing.add_argument("--audio-window", type=float, default=10.0,
                       help="Seconds per embedded audio window (<=11.2: processor caps 280 audio tokens)")
    p_ing.add_argument("--max-audio-seconds", type=float, default=1800.0,
                       help="Cap on decoded seconds per audio file")
    _add_dim_arg(p_ing, for_search=False)
    _add_torch_args(p_ing)
    _add_model_args(p_ing)

    p_se = sub.add_parser("search", help="Semantic search over an index")
    p_se.add_argument("query", type=str, nargs="?", default=None, help="Query string")
    p_se.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="Use an image/audio/video file as the query (embeddinggemma-2 indexes), "
        "or a text file whose contents are the query",
    )
    p_se.add_argument(
        "--index",
        "-i",
        type=Path,
        default=None,
        help="Index directory (default: active index from .on-the-fly-rag.json)",
    )
    p_se.add_argument("--top-k", "-k", type=int, default=5)
    _add_path_filter_args(p_se)
    p_se.add_argument("--json", action="store_true", help="Emit JSON")
    _add_modality_arg(p_se)
    _add_dim_arg(p_se, for_search=True)
    _add_torch_args(p_se)
    _add_model_args(p_se)

    p_ms = sub.add_parser(
        "multi-search",
        help="Batch multiple retrieval queries (JSON file or stdin) for multi-hop agents",
    )
    p_ms.add_argument(
        "queries_file",
        type=Path,
        nargs="?",
        default=None,
        help="JSON file: list of strings or {query,id,path_contains,path_glob,path_prefix,top_k}. "
        "Omit to read stdin.",
    )
    p_ms.add_argument(
        "--index",
        "-i",
        type=Path,
        default=None,
        help="Index directory (default: active index from .on-the-fly-rag.json)",
    )
    p_ms.add_argument("--top-k", "-k", type=int, default=5)
    p_ms.add_argument("--json", action="store_true", help="Emit JSON (default for agents)")
    _add_modality_arg(p_ms)
    _add_dim_arg(p_ms, for_search=True)
    _add_model_args(p_ms)

    p_st = sub.add_parser("status", help="Show the active (default) index + model readiness")
    p_st.add_argument("--json", action="store_true", help="Emit JSON")

    p_use = sub.add_parser(
        "use",
        help="Set the active index to an existing index directory",
    )
    p_use.add_argument(
        "index_dir",
        type=Path,
        help="Existing index directory (must contain vectors.npy + chunks.jsonl)",
    )
    p_use.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional corpus path recorded in state (default: index parent or prior)",
    )

    p_models = sub.add_parser(
        "models",
        help="List bundled embedding model presets and readiness (unshard needed?)",
    )
    p_models.add_argument("--json", action="store_true", help="Emit JSON")
    p_models.add_argument(
        "--choice",
        action="store_true",
        help="Print the first-ask model choice outline for agents",
    )

    p_sh = sub.add_parser("shard", help="Split a large weight file into <100MB parts")
    p_sh.add_argument("input", type=Path)
    p_sh.add_argument("--prefix", type=Path, default=None)
    p_sh.add_argument("--shard-size", type=int, default=90 * 1024 * 1024)

    p_us = sub.add_parser(
        "unshard",
        help="Reassemble sharded weight parts (verifies per-part + whole-file SHA-256)",
    )
    p_us.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Prefix of parts (e.g. model.onnx.part or model.safetensors.part)",
    )
    p_us.add_argument(
        "--model",
        type=str,
        default=None,
        help="Preset id instead of --input (e.g. granite, embeddinggemma-2)",
    )
    p_us.add_argument("--output", "-o", type=Path, default=None)
    p_us.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check shard checksums against the manifest; write nothing",
    )

    return parser


def _resolve_search_index(parser: argparse.ArgumentParser, explicit: Path | None) -> Path:
    try:
        return resolve_index_dir(explicit, require_exists=True)
    except FileNotFoundError as e:
        parser.error(str(e))
        raise  # unreachable; keeps type checkers happy


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        workers = args.workers if args.workers is not None else _default_workers()
        index_dir = args.index if args.index is not None else default_index_dir(args.source)
        model = args.model if args.model is not None else DEFAULT_MODEL_ID
        print(
            f"Ingesting {args.source} → {index_dir} with {workers} worker(s) "
            f"(cpu_count={os.cpu_count()}, model={model})...",
            file=sys.stderr,
        )
        try:
            spec_backend = get_preset(model).backend
        except KeyError:
            spec_backend = "onnx"
        batch = args.batch_size or (8 if spec_backend == "torch" else 32)
        try:
            cfg = ingest(
                args.source,
                index_dir=index_dir,
                model=None if (args.append and args.model is None) else model,
                tokenizer_path=args.tokenizer,
                max_tokens=args.max_tokens,
                overlap_tokens=args.overlap_tokens,
                workers=workers,
                batch_size=batch,
                dim=args.dim,
                multimodal=args.multimodal,
                append=args.append,
                dtype=args.dtype,
                torch_threads=args.torch_threads,
                media_options=MediaOptions(
                    video_fps=args.video_fps,
                    segment_frames=args.video_segment_frames,
                    max_video_frames=args.max_video_frames,
                    audio_window_s=args.audio_window,
                    max_audio_s=args.max_audio_seconds,
                ),
                quiet=False,
            )
        except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 1
        if not args.no_active:
            active = save_active(source=args.source, index_dir=index_dir)
            cfg["active"] = active
        print(json.dumps(cfg, indent=2))
        return 0

    if args.command == "search":
        if not args.query and args.query_file is None:
            parser.error("search needs a query string or --query-file")
        index_dir = _resolve_search_index(parser, args.index)
        try:
            results = search(
                args.query,
                index_dir=index_dir,
                top_k=args.top_k,
                model=args.model,
                tokenizer_path=args.tokenizer,
                path_contains=args.path_contains,
                path_glob=args.path_glob,
                path_prefix=args.path_prefix,
                modality=args.modality,
                dim=args.dim,
                query_file=args.query_file,
                dtype=args.dtype,
                threads=args.torch_threads,
            )
        except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 1
        print(format_results(results, as_json=args.json))
        return 0

    if args.command == "multi-search":
        index_dir = _resolve_search_index(parser, args.index)
        if args.queries_file is not None:
            raw = args.queries_file.read_text(encoding="utf-8")
        else:
            raw = sys.stdin.read()
        queries = json.loads(raw)
        if not isinstance(queries, list):
            parser.error("multi-search input must be a JSON list")
        try:
            results = multi_search(
                queries,
                index_dir=index_dir,
                top_k=args.top_k,
                model=args.model,
                tokenizer_path=args.tokenizer,
                modality=args.modality,
                dim=args.dim,
            )
        except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 1
        print(format_multi(results, as_json=args.json))
        return 0

    if args.command == "status":
        active = load_active()
        if args.json:
            payload = {
                "active": active,
                "models": all_models_status(),
            }
            if active and active.get("index_dir"):
                from .active import _index_model_info

                payload["index_model"] = _index_model_info(Path(active["index_dir"]))
            print(json.dumps(payload, indent=2))
        else:
            print(format_status(active))
        return 0

    if args.command == "models":
        if args.choice:
            print(format_choice_outline())
            return 0
        if args.json:
            print(json.dumps(all_models_status(), indent=2))
        else:
            print(format_choice_outline())
        return 0

    if args.command == "use":
        index_dir = Path(args.index_dir).resolve()
        if not (index_dir / "vectors.npy").is_file() or not (
            index_dir / "chunks.jsonl"
        ).is_file():
            parser.error(
                f"Not a usable index (need vectors.npy + chunks.jsonl): {index_dir}"
            )
        source = args.source
        if source is None:
            prior = load_active()
            if prior and Path(prior.get("index_dir", "")).resolve() == index_dir:
                source = Path(prior["source"])
            else:
                # Prefer parent when index is <source>/.rag_index
                source = (
                    index_dir.parent
                    if index_dir.name == ".rag_index"
                    else index_dir.parent
                )
        active = save_active(source=source, index_dir=index_dir)
        print(json.dumps(active, indent=2))
        return 0

    if args.command == "shard":
        parts = shard_file(
            args.input, output_prefix=args.prefix, shard_size=args.shard_size
        )
        for p in parts:
            print(p)
        return 0

    if args.command == "unshard":
        prefix = args.input
        if prefix is None:
            if not args.model:
                parser.error("unshard needs --input <prefix> or --model <preset>")
            try:
                prefix = get_preset(args.model).shard_prefix
            except KeyError as e:
                parser.error(str(e))
        try:
            if args.verify_only:
                print(json.dumps(verify_shards(prefix), indent=2))
                return 0
            out = unshard_file(prefix, output_path=args.output)
        except ChecksumError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        except FileNotFoundError as e:
            print(f"ERROR: missing file: {e}", file=sys.stderr)
            return 1
        print(f"{out}  (sha256 verified)")
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
