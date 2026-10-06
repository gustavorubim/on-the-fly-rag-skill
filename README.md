# on-the-fly-rag-skill

Self-contained **GitHub Copilot / coding-agent skill** for **on-the-fly RAG** over a local folder of docs or a codebase.

- **Lightweight search** — ripgrep / regex / path filters / cheap structure
- **On-the-fly embedding RAG** — chunk + embed with **bundled** ONNX models (MiniLM default + IBM Granite English R2), persist a local vector store, search top-k chunks
- **Office/PDF ingest** — `.pdf` / `.docx` / `.pptx` → plain text via lightweight extractors, then the same chunk → embed → store path
- **Multi-hop retrieval** — path/glob filters + `multi-search` + skill instructions (scratchpad / verify loop) for compare & synthesize
- **Multimodal RAG (optional)** — bundled **Google EmbeddingGemma 2** (740M, Apache-2.0) embeds text, images, video and audio into one 768-d space: text→image/video/audio search, image→image search, `--modality` filters, Matryoshka `--dim`. Needs the optional PyTorch extras (`requirements-gemma.txt`) and `ffmpeg`.

No hosted vector DB. No Hugging Face download after clone (weights are vendored). Works offline once Python deps are installed. **First ingest:** pick MiniLM vs Granite vs EmbeddingGemma 2 (see [Embedding models](#embedding-models)).

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
| `…/models/embeddinggemma-2/` | Google EmbeddingGemma 2 safetensors (**16 shards, 1.49GB**, SHA-256 manifest; unshard first) |
| `…/requirements.txt` / `pyproject.toml` | Pinned deps + install metadata |
| `…/requirements-gemma.txt` / `.[gemma]` extra | Optional PyTorch backend for EmbeddingGemma 2 |

**Development repo root** (not required for skill install):

| Path | Purpose |
| --- | --- |
| `fixtures/sample_docs/` | Tiny text corpus for offline tests |
| `fixtures/office/` | Minimal PDF/DOCX/PPTX fixtures |
| `fixtures/eval_corpus/` | Multi-doc NovaSync corpus (pdf+docx+pptx+md) for multi-hop eval |
| `fixtures/multimodal/` | Tiny CC0 images / audio / video + synthetic text for the multimodal tests ([sources](fixtures/multimodal/SOURCES.md)) |
| `docs/eval.md` | Eval notes: multimodal pass/fail table, Granite comparison, NovaSync multi-hop |
| `tests/` | Chunk / extract / ingest+search / multi-search / active-index / shard / multimodal tests (`-m gemma` = real model) |
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

**Optional — EmbeddingGemma 2 (multimodal) backend.** Only needed for `--model embeddinggemma-2`; MiniLM/Granite never import PyTorch.

```bash
pip install -r requirements-gemma.txt --extra-index-url https://download.pytorch.org/whl/cpu
# or: pip install '.[gemma]' --extra-index-url https://download.pytorch.org/whl/cpu
python -m on_the_fly_rag unshard --model embeddinggemma-2     # once; verifies SHA-256
```

| Package (pinned) | Role |
| --- | --- |
| `torch` 2.14.1 (CPU wheel), `torchvision` 0.29.1 | Model runtime + video processor |
| `transformers` 5.19.0, `sentence-transformers` 6.1.0, `safetensors` 0.8.0 | EmbeddingGemma 2 model / processor |
| `pillow` 12.3.0, `soundfile` 0.14.0 | Image / audio I/O |
| `imageio-ffmpeg` 0.6.0 | Bundled `ffmpeg` fallback for video/audio decoding (a system `ffmpeg` on PATH is used first) |

## Embedding models

| Preset (`--model`) | HF id | Params | Dim | Ctx | On disk | Retrieval ballpark |
| --- | --- | --- | --- | --- | --- | --- |
| `minilm` **(default)** | `sentence-transformers/all-MiniLM-L6-v2` | 22M | 384 | 256 | ~87MB ONNX | Fast/tiny offline baseline |
| `granite-small` | `ibm-granite/granite-embedding-small-english-r2` | 47M | 384 | 8k | ~94MB ONNX | Stronger retrieval, still small |
| `granite` | `ibm-granite/granite-embedding-english-r2` | 149M | 768 | 8k | ~303MB ONNX (**sharded**) | Best quality of the ONNX three |
| `embeddinggemma-2` | `google/embeddinggemma-2` | 740M (270M text-only) | 768 (Matryoshka 512/256/128) | 8k | ~1.49GB safetensors (**16 shards**) | Text + image + video + audio in one space; much heavier |

MiniLM and Granite run on **onnxruntime CPU** (no PyTorch). `embeddinggemma-2` runs on **PyTorch CPU** via `sentence-transformers` (optional extras). Granite weights are Apache-2.0 (IBM); MiniLM is Apache-2.0 (sentence-transformers); EmbeddingGemma 2 is Apache-2.0 (Google DeepMind).

**EmbeddingGemma 2 cost (measured on this repo's 8-vCPU dev box, shared/contended, 2 torch threads):**

| Load | Params | Peak RSS | Load time |
| --- | --- | --- | --- |
| text-only (auto for text corpora and all text queries) | 271M | ~2.1GB FP32 / ~1.0GB `--dtype bf16` | ~10s |
| full multimodal (corpus has images/video/audio, or `--multimodal on`) | 744M | ~4.8GB FP32 | ~10s |

Ingest throughput on the same box: images ~14s each, audio ~2.6× real time, video ~93s per 16-frame (16s @ 1fps) segment, text ≈1.1 chunks/s (200-token chunks). Details: [docs/eval.md](docs/eval.md#cost). Expect 10–100× slower than the ONNX models; prefer `granite` for text-only corpora unless you need the shared multimodal space.

### First-question model choice (agents)

On the **first** embed/index ask, the agent must outline which model will be used and let the user pick (or confirm MiniLM). Example:

```text
/on-the-fly-rag index this folder
```

Agent should present MiniLM vs Granite Small R2 vs Granite English R2 vs EmbeddingGemma 2 (size/quality, unshard need, RAM/speed cost, extras; recommend EmbeddingGemma 2 only when the folder has images/video/audio or the user wants cross-modal search), then unshard if needed, ingest with that model, and record `model_id` in the index `config.json` so search matches.

```bash
# From .github/skills/on-the-fly-rag (or after pip install -e .)
python -m on_the_fly_rag models          # choice outline + readiness
python -m on_the_fly_rag models --json
```

### Unshard Granite English R2 / EmbeddingGemma 2 (required once)

`granite` and `embeddinggemma-2` ship as shards under GitHub’s 100MB limit. Each manifest stores a SHA-256 per part **and** for the whole file; `unshard` verifies both and fails loudly (exit 2, no output file) on any mismatch:

```bash
python -m on_the_fly_rag unshard --model granite
python -m on_the_fly_rag unshard --model embeddinggemma-2
python -m on_the_fly_rag unshard --model embeddinggemma-2 --verify-only   # check shards, write nothing
python -m on_the_fly_rag unshard --input models/granite-embedding-english-r2/model.onnx.part  # explicit form
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

Ingest writes `model_id`, `dim` and paths into `<index>/config.json`. Search loads that model automatically; it errors clearly if weights are missing or still sharded. Every index is keyed to **one model and one dim**: `search` / `--append` with a different `--model` or `--dim` is **refused** (`IndexCompatError`) instead of returning garbage. Re-ingest into a new `--index` to switch.

### EmbeddingGemma 2: multimodal ingest & search

```bash
# Mixed folder: text + images + video + audio (vision/audio encoders load only if needed)
python -m on_the_fly_rag ingest ./media_docs --model embeddinggemma-2
# Smaller vectors (Matryoshka: truncate + L2-renormalize; recorded in config, queries follow it)
python -m on_the_fly_rag ingest ./docs --model embeddinggemma-2 --dim 256
# Add/replace files later (same model + dim enforced)
python -m on_the_fly_rag ingest ./media_docs/new --model embeddinggemma-2 --append --index ./media_docs/.rag_index

python -m on_the_fly_rag search "a red bicycle" --modality image
python -m on_the_fly_rag search "a dog barking" --modality audio
python -m on_the_fly_rag search "candle burning in the dark" --modality video   # hits show [video 00:00.0–00:16.0 frames 0–15]
python -m on_the_fly_rag search --query-file ./query.jpg --modality image          # image → image
python -m on_the_fly_rag search "drip irrigation schedule"                           # mixed: ranks all modalities together
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--multimodal {auto,on,off}` | `auto` | `auto`: load vision/audio encoders only if the corpus has such files; `off`: text tower only, skip media |
| `--dim {768,512,256,128}` | 768 | Matryoshka output dim (128 degrades multimodal quality per the model card) |
| `--dtype {fp32,bf16}` | fp32 | `fp16` is refused (the model overflows in float16) |
| `--torch-threads N` | `-j` | torch intra-op threads (on a busy machine fewer can be faster) |
| `--video-fps` / `--video-segment-frames` / `--max-video-frames` | 1 / 16 / 600 | Frame sampling; ≤32 frames per segment (140 tokens/frame, processor cap) |
| `--audio-window` / `--max-audio-seconds` | 10 / 1800 | Audio windows ≤11.2s (processor caps one item at 280 audio tokens) |
| `search --modality text,image,…` | all | Filter hits by modality |
| `search --query-file F` | – | Query with an image / audio / video file (or a text file) |

Images: png/jpg/jpeg/webp/gif · Video: mp4/mov/webm · Audio: wav/mp3/m4a/flac/ogg. Text chunks are embedded as `title: <file name or "deck.pptx slide N"> | text: …`; queries as `task: search result | query: …` (model-card prompts). Media metadata (modality, source, start/end seconds, frame range) is stored per chunk. With MiniLM/Granite, media files are **skipped with a notice**.

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

### 7. Images, video, audio (EmbeddingGemma 2)

```text
/on-the-fly-rag index ./field_notes (it has photos, voice memos and clips) with embeddinggemma-2
```

```text
/on-the-fly-rag find the photo of the red bicycle; show the file path
```

```text
/on-the-fly-rag where in the videos does a candle appear? give the timestamp range
```

```text
/on-the-fly-rag which recordings contain a dog barking? audio only
```

```text
/on-the-fly-rag find images similar to ./query.jpg
```

### 8. Override (multi-store escape hatch)

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

**Real-model multimodal tests** (EmbeddingGemma 2: text, text→image, image→image, text→audio, text→video, mixed index, text-only load + `--dim`). They are marked `gemma` and auto-skip unless the extras are installed and the weights are unsharded:

```bash
pip install pytest -r .github/skills/on-the-fly-rag/requirements-gemma.txt --extra-index-url https://download.pytorch.org/whl/cpu
(cd .github/skills/on-the-fly-rag && python -m on_the_fly_rag unshard --model embeddinggemma-2)
python -m pytest -m gemma -v          # ~4 min with 3 torch threads on a busy 8-vCPU box; ~9GB peak RAM; ON_THE_FLY_RAG_TORCH_THREADS=N to tune
python -m pytest                      # everything (gemma tests included when available)
```

## Limits

- MiniLM context is **256 tokens**; Granite R2 supports up to **8192** (chunk defaults stay ~200 unless you raise `--max-tokens`).
- Default model is MiniLM (fast); use `granite-small` / `granite` for stronger retrieval.
- `granite` must be **unsharded** once after clone before ingest/search.
- Index is local cosine search (brute-force); fine for small/medium corpora, not millions of chunks.
- Non-allow-listed binaries are skipped; PDF/DOCX/PPTX go through extractors (`chunk.iter_files` + `extract.load_document`).
- GitHub blocks files ≥100MB — keep each committed weight file under that limit (default ONNX is ~87MB).
- EmbeddingGemma 2 is CPU-heavy (see cost table); text→audio retrieval on very short (<2s) sound clips is weak (see [docs/eval.md](docs/eval.md)). One query embeds only the first segment of a media `--query-file`.

## License

MIT — see [LICENSE](LICENSE).

Model weights:

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Apache-2.0
- [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) — Apache-2.0
- [ibm-granite/granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) — Apache-2.0
- [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) — Apache-2.0 (vendored unchanged from revision `914f7f8`, safetensors split into 16 parts)

Vendored Granite ONNX (fp16) via [onnx-community](https://huggingface.co/onnx-community) exports of the IBM checkpoints.
