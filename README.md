# on-the-fly-rag-skill

Self-contained **GitHub Copilot / coding-agent skill** for **on-the-fly RAG** over a local folder of docs or a codebase.

- **Lightweight search** — ripgrep / regex / path filters / cheap structure
- **On-the-fly embedding RAG** — chunk + embed with **bundled** ONNX models (MiniLM default + IBM Granite English R2), persist a local vector store, search top-k chunks
- **Office/PDF ingest** — `.pdf` / `.docx` / `.pptx` → plain text via lightweight extractors, then the same chunk → embed → store path
- **Multi-hop retrieval** — path/glob filters + `multi-search` + skill instructions (scratchpad / verify loop) for compare & synthesize

No hosted vector DB. No Hugging Face download after clone (weights are vendored). Works offline once Python deps are installed. **First ingest:** pick MiniLM vs Granite (see [Embedding models](#embedding-models)).

## What’s included

**Distributable skill unit** (copy one folder):

| Path | Purpose |
| --- | --- |
| `.github/skills/on-the-fly-rag/` | **Canonical self-contained skill package** |
| `…/SKILL.md` | Copilot skill instructions |
| `…/on_the_fly_rag/` | Python package: extract, chunk, embed, ingest, search, multi-search, active index, shard |
| `…/scripts/` | Thin CLI wrappers (`ingest.py`, `search.py`, `multi_search.py`, `shard.py`, `unshard.py`) |
| `…/models/all-MiniLM-L6-v2/` | Default MiniLM ONNX + tokenizer (**~87MB**) |
| `…/models/granite-embedding-*-english-r2/` | IBM Granite R2 ONNX (small ready; english **sharded**, unshard first) |
| `…/requirements.txt` / `pyproject.toml` | Pinned deps + install metadata |

**Development repo root** (not required for skill install):

| Path | Purpose |
| --- | --- |
| `fixtures/sample_docs/` | Tiny text corpus for offline tests |
| `fixtures/office/` | Minimal PDF/DOCX/PPTX fixtures |
| `fixtures/eval_corpus/` | Multi-doc NovaSync corpus (pdf+docx+pptx+md) for multi-hop eval |
| `docs/eval.md` | Multi-hop eval notes / queries / outcomes |
| `tests/` | Chunk / extract / ingest+search / multi-search / active-index / shard smoke tests |
| `skill/on-the-fly-rag/` | Pointer README → canonical package above |

## Install (Copilot skill)

Copy **one folder** — the entire self-contained package:

### Personal skill (all projects)

```bash
git clone https://github.com/gustavorubim/on-the-fly-rag-skill.git
mkdir -p ~/.copilot/skills
cp -R on-the-fly-rag-skill/.github/skills/on-the-fly-rag ~/.copilot/skills/on-the-fly-rag
cd ~/.copilot/skills/on-the-fly-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: pip install -e .
```

Reload skills in Copilot CLI: `/skills reload`, then `/skills info on-the-fly-rag`.

### Project skill (this or any repo)

Copy the whole package (not only `SKILL.md`) so the tree is:

```text
.github/skills/on-the-fly-rag/
  SKILL.md
  on_the_fly_rag/
  models/
  scripts/
  requirements.txt
  pyproject.toml
```

### Python deps (required for embed path)

Run from the **skill package directory** (after copy, or from the clone path below):

```bash
cd .github/skills/on-the-fly-rag   # or ~/.copilot/skills/on-the-fly-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| Package | Role |
| --- | --- |
| `onnxruntime`, `tokenizers`, `numpy` | Embed + chunk |
| `pypdf` | `.pdf` text extraction |
| `python-docx` | `.docx` paragraphs/tables |
| `python-pptx` | `.pptx` slide text + speaker notes |

Unparseable Office/PDF files are **logged and skipped** (ingest continues).

## Embedding models

| Preset (`--model`) | HF id | Params | Dim | Ctx | On disk | Retrieval ballpark |
| --- | --- | --- | --- | --- | --- | --- |
| `minilm` **(default)** | `sentence-transformers/all-MiniLM-L6-v2` | 22M | 384 | 256 | ~87MB ONNX | Fast/tiny offline baseline |
| `granite-small` | `ibm-granite/granite-embedding-small-english-r2` | 47M | 384 | 8k | ~94MB ONNX | Stronger retrieval, still small |
| `granite` | `ibm-granite/granite-embedding-english-r2` | 149M | 768 | 8k | ~303MB ONNX (**sharded**) | Best quality of the three |

All three run on **onnxruntime CPU** (no PyTorch required at runtime). Granite weights are Apache-2.0 (IBM); MiniLM is Apache-2.0 (sentence-transformers).

### First-question model choice (agents)

On the **first** embed/index ask, the agent must outline which model will be used and let the user pick (or confirm MiniLM). Example:

```text
/on-the-fly-rag index this folder
```

Agent should present MiniLM vs Granite Small R2 vs Granite English R2 (size/quality + unshard need), then unshard if needed, ingest with that model, and record `model_id` in the index `config.json` so search matches.

```bash
# From .github/skills/on-the-fly-rag (or after pip install -e .)
python -m on_the_fly_rag models          # choice outline + readiness
python -m on_the_fly_rag models --json
```

### Unshard Granite English R2 (required once)

`granite` ships as shards under GitHub’s 100MB limit:

```bash
python -m on_the_fly_rag unshard --input models/granite-embedding-english-r2/model.onnx.part
```

`granite-small` is a single &lt;100MB file — no unshard. MiniLM likewise.

### CLI `--model` presets

```bash
python -m on_the_fly_rag ingest ./docs --model minilm
python -m on_the_fly_rag ingest ./docs --model granite-small
python -m on_the_fly_rag unshard --input models/granite-embedding-english-r2/model.onnx.part
python -m on_the_fly_rag ingest ./docs --model granite
python -m on_the_fly_rag status   # active index model + whether shards need unshard
```

Ingest writes `model_id` / paths into `<index>/config.json`. Search loads that model automatically; it errors clearly if weights are missing or still sharded. Re-ingest after switching models (dimensions differ: 384 vs 768).

## Prompt cookbook

After the skill is installed, invoke it in Copilot CLI with `/on-the-fly-rag` (skill name from `SKILL.md`), then ask in natural language. Copilot may also auto-select the skill when your ask matches its description.

**Default UX:** index a folder once → that becomes the **active** corpus. Day-to-day questions do **not** mention `.rag_index`. Different file groups can use different indexes; name an index path only when you need the multi-store escape hatch.

Copy-paste examples (what you type):

### 1. Ingest a folder (sets the default / active index)

On first index, the agent asks which model to use (MiniLM default vs Granite).

```text
/on-the-fly-rag index ./docs and make it the default corpus
```

```text
/on-the-fly-rag index ./docs with granite-small
```

```text
/on-the-fly-rag ingest ./fixtures/sample_docs with default parallel workers
```

### 2. Semantic search / Q&A (no index path)

```text
/on-the-fly-rag what discusses authentication? cite file paths
```

```text
/on-the-fly-rag how does ingest pick CPU workers? quote the top chunks
```

### 3. Grep vs embed (lightweight first)

```text
/on-the-fly-rag I need the exact string ProcessPoolExecutor — use ripgrep, not embeddings
```

```text
/on-the-fly-rag should I grep or embed for "where do we talk about latency SLAs across the docs?" — pick one and do it
```

### 4. Compare 2–3 docs (path filters)

```text
/on-the-fly-rag compare product_spec vs ops_status on p99 latency; path-filter each hop and cite both sides
```

```text
/on-the-fly-rag multi-search: Spec auth policy vs Ops auth in production — filter by *.pdf and *.pptx
```

### 5. Multi-hop + scratchpad / verify

```text
/on-the-fly-rag Pro pricing vs Spec rate limits: plan hops, keep a scratchpad, re-retrieve if coverage is thin, then answer with citations
```

```text
/on-the-fly-rag FAQ → Spec → Ops on NovaSync latency: dependent facts across docs; verify before answering
```

### 6. Office / PDF mixed corpus

```text
/on-the-fly-rag ingest ./fixtures/eval_corpus (pdf/docx/pptx/md), then summarize auth drift between Spec and Ops
```

### 7. Override (multi-store escape hatch)

```text
/on-the-fly-rag search using index /tmp/legal_corpus/.rag_index for retention policy; cite paths
```

For the NovaSync multi-doc eval (Spec 50ms vs Ops 120ms, OAuth2 vs API keys, Pro pricing), see [docs/eval.md](docs/eval.md). CLI equivalents live in [CLI usage](#cli-usage) below.

## CLI usage

Run from `.github/skills/on-the-fly-rag` (skill root), or `pip install -e` that folder / set `PYTHONPATH` to it. Fixture paths below are relative to the **repo root**.

**Defaults:** ingest writes `<source>/.rag_index` and sets workspace active state
(`.on-the-fly-rag.json`). `search` / `multi-search` use that active index when
`-i` / `--index` is omitted. Pass `-i` to override; `status` / `use` inspect or
switch the active pointer.

```bash
# Ingest (parallel across CPU cores by default) — includes PDF/DOCX/PPTX
# → ./fixtures/sample_docs/.rag_index + sets active
python -m on_the_fly_rag ingest ./fixtures/sample_docs
python -m on_the_fly_rag ingest ./fixtures/eval_corpus -j 4

# Active index helpers
python -m on_the_fly_rag status
python -m on_the_fly_rag use ./fixtures/eval_corpus/.rag_index

# Search (active index; no -i needed)
python -m on_the_fly_rag search "parallel CPU workers" -k 5
python -m on_the_fly_rag search "latency SLA" --path-contains product_spec
python -m on_the_fly_rag search "observed latency" --glob '*.pptx'
python -m on_the_fly_rag search "auth secrets" --json
# Override when juggling several stores:
python -m on_the_fly_rag search "auth secrets" -i /tmp/other/.rag_index --json

# Multi-hop batch (compare 2 docs) — uses active index
cat > /tmp/nova_hops.json <<'JSON'
[
  {"id":"spec","query":"p99 latency SLA milliseconds","path_glob":"*product_spec*"},
  {"id":"ops","query":"observed p99 latency milliseconds","path_glob":"*ops_status*"}
]
JSON
python -m on_the_fly_rag multi-search /tmp/nova_hops.json --json -k 3

# Same via scripts/ from the skill package (explicit -i/-o still supported)
python scripts/ingest.py ./docs -o .rag_index -j 8
python scripts/search.py "how does ingest work?" -i .rag_index
python scripts/multi_search.py /tmp/nova_hops.json -i .rag_index --json
```

### Path filters

| Flag | Meaning |
| --- | --- |
| `--path-contains SUB` | Only chunks whose path contains `SUB` (case-insensitive) |
| `--path` / `--glob PAT` | `fnmatch` on relative path or basename; `\|` ORs patterns |
| `--path-prefix PRE` | Only paths starting with `PRE` |

### Parallel ingest

Ingest uses **`ProcessPoolExecutor`** so chunking and embedding batches run across CPU cores (embedding is CPU-bound under ONNX Runtime).

| Flag | Default | Meaning |
| --- | --- | --- |
| `-j` / `--workers` | `os.cpu_count()` | Number of worker **processes** |

Search / multi-search stay single-threaded / simple.

## Multi-hop agent usage

The skill (`SKILL.md`) defines an explicit **plan → retrieve → scratchpad → verify → answer** loop. Use it when comparing documents, finding contradictions, or synthesizing across files.

1. Plan 2+ retrieval queries (path-filter per doc when comparing).
2. Call `search` / `multi-search`.
3. Keep a scratchpad of facts, gaps, and contradictions.
4. If `coverage.thin_hops` is non-empty (or scores look weak), re-retrieve.
5. Answer with citations only from retrieved chunks.

See [docs/eval.md](docs/eval.md) for the NovaSync multi-doc eval (Spec 50ms vs Ops 120ms, OAuth2 vs API keys, Pro pricing).

## Architecture

```text
docs/ (md/txt/code + pdf/docx/pptx)
       ──► extract (Office/PDF → text) or UTF-8 read
       ──► chunk (token-aware, ~200 tok + overlap)
       ──► embed (ONNX MiniLM mean-pool or Granite sentence_embedding + L2)
       ──► <source>/.rag_index/{vectors.npy, chunks.jsonl, config.json}
       ──► .on-the-fly-rag.json  (active: {source, index_dir, updated_at})
query ──► embed ──► cosine top-k (+ optional path/glob) ──► paths + snippets
multi ──► N queries (batch) ──► coverage stats for verify step
```

- **Models:** MiniLM (default, 384-d) or IBM Granite English R2 small/full (384/768-d, 8k ctx); see table above
- **Chunking:** token-aware via `tokenizers`; stays under 256 (default max ~200 + CLS/SEP)
- **Extractors:** `.github/skills/on-the-fly-rag/on_the_fly_rag/extract.py` is the only place binary formats are handled
- **Store:** numpy float32 matrix + JSONL metadata (no SQLite required)
- **Active index:** one default corpus pointer in workspace `.on-the-fly-rag.json` (override with `-i` / `use`)
- **Shard/unshard:** split oversized weights for GitHub’s 100MB file limit

```bash
python -m on_the_fly_rag shard path/to/big.onnx
python -m on_the_fly_rag unshard --input models/granite-embedding-english-r2/model.onnx.part
```

### Custom ONNX path

Pass `--model /path/to/model.onnx` (and `--tokenizer` if needed). Prefer presets (`minilm` / `granite-small` / `granite`) for the bundled models.

## Decision guide: grep vs embed

| Use **rg / grep** | Use **embedding RAG** |
| --- | --- |
| Exact symbols, errors, filenames | Conceptual / paraphrased questions |
| Import / call-site navigation | “What discusses X?” across many docs |
| Huge trees, one-off lookup | User asked to index / retrieve chunks |
| | Multi-doc compare → multi-hop + path filters |

## Tests

From the **repo root** (tests + fixtures stay here; the package lives under `.github/skills/…`):

```bash
cd .github/skills/on-the-fly-rag && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ../../..   # back to repo root
source .github/skills/on-the-fly-rag/.venv/bin/activate
python -m unittest discover -s tests -v
```

All tests are offline (vendored model + fixtures).

## Limits

- MiniLM context is **256 tokens**; Granite R2 supports up to **8192** (chunk defaults stay ~200 unless you raise `--max-tokens`).
- Default model is MiniLM (fast); use `granite-small` / `granite` for stronger retrieval.
- `granite` must be **unsharded** once after clone before ingest/search.
- Index is local cosine search (brute-force); fine for small/medium corpora, not millions of chunks.
- Non-allow-listed binaries are skipped; PDF/DOCX/PPTX go through extractors (`chunk.iter_files` + `extract.load_document`).
- GitHub blocks files ≥100MB — keep each committed weight file under that limit (default ONNX is ~87MB).

## License

MIT — see [LICENSE](LICENSE).

Model weights:

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Apache-2.0
- [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) — Apache-2.0
- [ibm-granite/granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) — Apache-2.0

Vendored Granite ONNX (fp16) via [onnx-community](https://huggingface.co/onnx-community) exports of the IBM checkpoints.
