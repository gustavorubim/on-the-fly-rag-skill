"""argparse CLI for ingest / search / multi-search / shard / unshard."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .ingest import _default_workers, ingest
from .paths import DEFAULT_MODEL_ONNX, DEFAULT_TOKENIZER
from .search import format_multi, format_results, multi_search, search
from .shard import shard_file, unshard_file


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_ONNX,
        help="Path to ONNX model (default: bundled MiniLM)",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Path to tokenizer.json",
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
        default=Path(".rag_index"),
        help="Output index directory (default: .rag_index)",
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
        default=Path(".rag_index"),
        help="Index directory",
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
        default=Path(".rag_index"),
        help="Index directory",
    )
    p_ms.add_argument("--top-k", "-k", type=int, default=5)
    p_ms.add_argument("--json", action="store_true", help="Emit JSON (default for agents)")
    _add_model_args(p_ms)

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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        workers = args.workers if args.workers is not None else _default_workers()
        print(
            f"Ingesting {args.source} with {workers} worker(s) "
            f"(cpu_count={os.cpu_count()})...",
            file=sys.stderr,
        )
        cfg = ingest(
            args.source,
            index_dir=args.index,
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            workers=workers,
            batch_size=args.batch_size,
        )
        print(json.dumps(cfg, indent=2))
        return 0

    if args.command == "search":
        results = search(
            args.query,
            index_dir=args.index,
            top_k=args.top_k,
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            path_contains=args.path_contains,
            path_glob=args.path_glob,
            path_prefix=args.path_prefix,
        )
        print(format_results(results, as_json=args.json))
        return 0

    if args.command == "multi-search":
        if args.queries_file is not None:
            raw = args.queries_file.read_text(encoding="utf-8")
        else:
            raw = sys.stdin.read()
        queries = json.loads(raw)
        if not isinstance(queries, list):
            parser.error("multi-search input must be a JSON list")
        results = multi_search(
            queries,
            index_dir=args.index,
            top_k=args.top_k,
            model_path=args.model,
            tokenizer_path=args.tokenizer,
        )
        print(format_multi(results, as_json=args.json))
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
