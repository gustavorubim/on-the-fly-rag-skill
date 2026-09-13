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
from .shard import shard_file, unshard_file


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
    p_ing.add_argument("--max-tokens", type=int, default=200)
    p_ing.add_argument("--overlap-tokens", type=int, default=40)
    p_ing.add_argument("--batch-size", type=int, default=32)
    _add_model_args(p_ing)

    p_se = sub.add_parser("search", help="Semantic search over an index")
    p_se.add_argument("query", type=str, help="Query string")
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

    p_us = sub.add_parser("unshard", help="Reassemble sharded weight parts")
    p_us.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Prefix of parts (e.g. model.onnx.part)",
    )
    p_us.add_argument("--output", "-o", type=Path, default=None)

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
            cfg = ingest(
                args.source,
                index_dir=index_dir,
                model=model,
                tokenizer_path=args.tokenizer,
                max_tokens=args.max_tokens,
                overlap_tokens=args.overlap_tokens,
                workers=workers,
                batch_size=args.batch_size,
            )
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            return 1
        if not args.no_active:
            active = save_active(source=args.source, index_dir=index_dir)
            cfg["active"] = active
        print(json.dumps(cfg, indent=2))
        return 0

    if args.command == "search":
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
            )
        except (FileNotFoundError, ValueError) as e:
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
            )
        except (FileNotFoundError, ValueError) as e:
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
        out = unshard_file(args.input, output_path=args.output)
        print(out)
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
