---
name: on-the-fly-rag
description: >-
  On-the-fly RAG over a local folder of docs or a codebase. Use when the user
  asks to search, retrieve, or answer questions from a document corpus without
  a hosted vector DB. Chooses between lightweight grep/ripgrep and local
  embedding RAG with a bundled MiniLM ONNX model.
license: MIT
---

# On-the-fly RAG

Help the user retrieve context from a **local folder or codebase** using this
repository's scripts. Prefer working software over abstractions.

## Decision guide: grep vs embed

| Prefer **lightweight search** (rg / grep / path filters) when… | Prefer **embedding RAG** when… |
| --- | --- |
| Exact symbol, error string, API name, or filename is known | Question is conceptual / paraphrased |
| Need call sites, imports, or structural navigation | Semantic similarity across many docs |
| Corpus is huge and a quick pass is enough | User asked to "index" / "RAG" / "retrieve chunks" |
| One-off lookup | Repeated questions over the same folder |

Optional cheap structure pass: list files, skim READMEs, build a rough
import/link graph with rg — then decide.

## Setup (once per machine)

From the clone of this skill repo (or a checkout that includes `on_the_fly_rag/`
and `models/`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Default model is **vendored** at `models/all-MiniLM-L6-v2/model.onnx` (~87MB).
No Hugging Face download is required at runtime.

If weights were shipped as shards (`model.onnx.part00`, …):

```bash
python -m on_the_fly_rag unshard --input models/all-MiniLM-L6-v2/model.onnx.part
```

## Ingest (parallel CPU)

Index a folder. By default uses **all CPU cores** via process workers
(`os.cpu_count()`). Override with `-j` / `--workers`:

```bash
python -m on_the_fly_rag ingest /path/to/docs --index .rag_index
python -m on_the_fly_rag ingest /path/to/docs --index .rag_index -j 4
# or
python scripts/ingest.py /path/to/docs -o .rag_index -j 4
```

Chunking stays under MiniLM's **256-token** window (~200 tokens + overlap).

## Search

```bash
python -m on_the_fly_rag search "how is auth handled?" --index .rag_index -k 5
python scripts/search.py "how is auth handled?" -i .rag_index --json
```

Search is single-threaded. Cite returned `path` + snippet when answering.

## Workflow for agents

1. Clarify the corpus root (folder / repo path).
2. Choose grep vs embed using the table above.
3. If embedding: ingest (if no fresh `.rag_index`), then search top-k.
4. Answer using retrieved chunks; quote paths. Do not invent file contents.
5. Never commit secrets; indexes under `.rag_index/` are gitignored.

## Swapping a larger model

Replace `models/.../model.onnx` (and tokenizer) or pass `--model` / `--tokenizer`.
If a weight file exceeds GitHub's 100MB limit, shard before commit:

```bash
python -m on_the_fly_rag shard path/to/big.onnx --shard-size 90000000
```

Document unshard-on-first-use for consumers.

