---
name: on-the-fly-rag
description: >-
  On-the-fly RAG over a local folder of docs or a codebase. Use when the user
  asks to search, retrieve, or answer questions from a document corpus without
  a hosted vector DB. Chooses between lightweight grep/ripgrep and local
  embedding RAG with bundled MiniLM / IBM Granite R2 ONNX models, or the optional
  multimodal Google EmbeddingGemma 2 (text, images, video, audio in one space).
  Supports PDF/DOCX/PPTX ingest, image/video/audio search and multi-hop retrieval
  with a scratchpad.
license: MIT
---

# On-the-fly RAG

Help the user retrieve context from a **local folder or codebase** using this
**self-contained skill package** (this folder). Prefer working software over abstractions.

## Default UX: one main index

**Day-to-day:** the user slash-calls this skill and asks to embed/index a folder.
That corpus becomes the **active/default** index. Later questions do **not**
mention `.rag_index` — search uses the active index automatically.

| Rule | Detail |
| --- | --- |
| One corpus → one index | Prefer writing the store at `<source>/.rag_index` |
| Active pointer | After ingest, workspace `.on-the-fly-rag.json` stores `{source, index_dir, updated_at}` |
| Multi-store escape hatch | Different file groups → different indexes; pass `-i` / `--index` or `use` to switch |

Agents: when the user names a folder to index, ingest it (sets active). For
ordinary Q&A, run `search` / `multi-search` **without** `-i`. Only pass an
explicit index path when the user is juggling several vector stores.

## First embed: choose a model (required)

On the **first** embed/index ask in a session (or when no index exists yet), you
**must outline which model will be used** and let the user pick or confirm the
default **before** ingesting.

| Preset | When to suggest | Size / dim / ctx | Notes |
| --- | --- | --- | --- |
| `minilm` **(default)** | Tiny/offline-fast path, quick trials | ~87MB · 384-d · 256 tok | Ready after clone |
| `granite-small` | Better retrieval, still small | ~94MB · 384-d · 8k | Ready after clone |
| `granite` | Best text quality (ONNX) | ~303MB · 768-d · 8k | **Unshard first** |
| `embeddinggemma-2` | Folder has **images / video / audio**, or user wants cross-modal / image→image search | ~1.49GB · 768-d (Matryoshka 512/256/128) · 8k | **Unshard first** + optional PyTorch extras (`requirements-gemma.txt`) + `ffmpeg`. CPU cost: text-only load ~2.1GB RAM, full multimodal ~4.8GB; images ~15s each, video ~95s per 16s segment on 2 threads |

Tell the user the RAM/speed cost before choosing `embeddinggemma-2`. Do not
suggest it for plain text folders unless asked: Granite is far cheaper there.
With MiniLM/Granite, images/video/audio are skipped with a notice.

```bash
python -m on_the_fly_rag models          # print choice outline + readiness
python -m on_the_fly_rag status          # active index model + shard state
```

Unshard once if the user picks `granite` or `embeddinggemma-2` (SHA-256 verified per
part and for the whole file; exit code 2 and no output on mismatch):

```bash
python -m on_the_fly_rag unshard --model granite
python -m on_the_fly_rag unshard --model embeddinggemma-2      # --verify-only to just check
pip install -r requirements-gemma.txt --extra-index-url https://download.pytorch.org/whl/cpu   # gemma only
```

Then ingest with `--model minilm|granite-small|granite|embeddinggemma-2`. The index
`config.json` stores `model_id` and `dim`; later search uses that model automatically
(errors if weights are missing/unsharded). Search or `--append` with a different
model or `--dim` is **refused** — re-ingest into a new index to switch.

Example first prompts:

```text
/on-the-fly-rag index this folder
```

→ Agent states/asks: MiniLM (default) vs Granite Small vs Granite English R2 vs EmbeddingGemma 2 (if media present) →
unshards if needed → ingest → active index records the model.

```text
/on-the-fly-rag index ./docs with granite-small
```

→ Skip the picker when the user already named a preset; still unshard if required.

## Decision guide: grep vs embed

| Prefer **lightweight search** (rg / grep / path filters) when… | Prefer **embedding RAG** when… |
| --- | --- |
| Exact symbol, error string, API name, or filename is known | Question is conceptual / paraphrased |
| Need call sites, imports, or structural navigation | Semantic similarity across many docs |
| Corpus is huge and a quick pass is enough | User asked to "index" / "RAG" / "retrieve chunks" |
| One-off lookup | Repeated questions over the same folder |
| Multi-doc compare / contradiction / synthesis | Use **multi-hop embed** below |

Optional cheap structure pass: list files, skim READMEs, build a rough
import/link graph with rg — then decide.

## Setup (once per machine)

This folder **is** the skill package (`SKILL.md` + `on_the_fly_rag/` + `models/` +
`scripts/` + deps). Install by copying **this entire directory**:

```bash
cp -R .github/skills/on-the-fly-rag ~/.copilot/skills/on-the-fly-rag
cd ~/.copilot/skills/on-the-fly-rag   # or wherever you copied it
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional: pip install -e .
```

Run all CLIs **from the skill root** (this folder), or set `PYTHONPATH=.`, or use
`pip install -e .` so `python -m on_the_fly_rag` works from anywhere.

Deps include **pypdf**, **python-docx**, **python-pptx** so `.pdf` / `.docx` /
`.pptx` are extracted to plain text during ingest (same chunk → embed → store
path as markdown/code).

Default / fallback model is **vendored** MiniLM at
`models/all-MiniLM-L6-v2/model.onnx` (~87MB) next to this package. Also vendored:
Granite Small R2 (ready) and Granite English R2 (**sharded**). No Hugging Face
download at runtime.

```bash
# From skill root — Granite English R2 (149M); required once before --model granite
python -m on_the_fly_rag unshard --input models/granite-embedding-english-r2/model.onnx.part
```

## Ingest (parallel CPU)

Index a folder. Default index path is **`<source>/.rag_index`**; ingest also
writes the **active** pointer (`.on-the-fly-rag.json` in the workspace cwd).
By default uses **all CPU cores** via process workers (`os.cpu_count()`).
Override workers with `-j` / `--workers`:

```bash
# Default: MiniLM, index beside the corpus + set active
python -m on_the_fly_rag ingest /path/to/docs
python -m on_the_fly_rag ingest /path/to/docs --model granite-small -j 4
python -m on_the_fly_rag ingest /path/to/docs --model granite

# Optional: custom index location (still sets active unless --no-active)
python -m on_the_fly_rag ingest /path/to/docs -o /tmp/other_index
python scripts/ingest.py /path/to/docs -o .rag_index -j 4 --model minilm
```

Supported sources: text/code extensions **plus** `.pdf`, `.docx`, `.pptx`
(slide body + speaker notes). Unparseable binaries are **logged and skipped**;
ingest does not abort.

Chunking defaults stay ~200 tokens + overlap (safe for MiniLM's 256 window; Granite allows larger `--max-tokens` up to 8k).

## Multimodal (EmbeddingGemma 2 only)

```bash
python -m on_the_fly_rag ingest /path/to/folder --model embeddinggemma-2            # text + media
python -m on_the_fly_rag ingest /path/to/docs --model embeddinggemma-2 --dim 256    # Matryoshka
python -m on_the_fly_rag ingest /path/to/more --append -o /path/to/folder/.rag_index
python -m on_the_fly_rag search "a red bicycle" --modality image
python -m on_the_fly_rag search "where is the candle?" --modality video   # hit shows start–end time + frames
python -m on_the_fly_rag search "dog barking" --modality audio
python -m on_the_fly_rag search --query-file photo.jpg --modality image   # image → image
```

- Media: png/jpg/jpeg/webp/gif, mp4/mov/webm (1 fps, ≤32-frame segments, default 16),
  wav/mp3/m4a/flac/ogg (≤11.2s windows, default 10s). Decoding uses `ffmpeg`.
- `--multimodal auto` (default) loads the vision/audio encoders only when the corpus
  has such files; text corpora and text queries use the 271M text tower only.
- Cite media hits with path + offset, e.g. `video/demo.mp4#t=16.0-32.0` (00:16–00:32).
  Never describe media content beyond what the user can verify; say it is a
  similarity match, not a transcript.
- Short (<2s) sound effects retrieve weakly by text; treat audio hits as candidates.
- `--dtype bf16` halves RAM; fp16 is refused (NaN risk per model card).

## Active index: status / use

```bash
python -m on_the_fly_rag status
python -m on_the_fly_rag status --json
# Point active at an existing store (multi-store escape hatch)
python -m on_the_fly_rag use /path/to/some/.rag_index
python -m on_the_fly_rag use /path/to/other_index --source /path/to/docs
```

## Search (single query)

Omit `-i` to use the **active** index (normal day-to-day). Pass `-i` only to
override:

```bash
python -m on_the_fly_rag search "how is auth handled?" -k 5
python -m on_the_fly_rag search "latency SLA" --path-contains product_spec
python -m on_the_fly_rag search "observed latency" --glob '*.pptx'
# Override when juggling several indexes:
python -m on_the_fly_rag search "how is auth handled?" -i /path/to/other/.rag_index --json
python scripts/search.py "how is auth handled?" -i .rag_index --json
```

Path filters (restrict retrieval to named files / subfolders):

| Flag | Meaning |
| --- | --- |
| `--path-contains SUB` | path substring (case-insensitive) |
| `--path` / `--glob PAT` | fnmatch on path or basename; `\|` ORs patterns |
| `--path-prefix PRE` | relative path prefix |

## Multi-search (batch hops)

```bash
python -m on_the_fly_rag multi-search queries.json --json -k 5
# or pipe JSON list on stdin
echo '[{"id":"a","query":"...","path_glob":"*.pdf"}]' \
  | python -m on_the_fly_rag multi-search --json
# Override:
python -m on_the_fly_rag multi-search queries.json -i /path/to/other/.rag_index --json
```

Each list item: string **or** object with `query`, optional `id`, `top_k`,
`path_contains`, `path_glob`, `path_prefix`. Response includes `coverage`
(`thin_hops`, `unique_paths`, `ok`).

## Multi-hop agent loop (required for compare / synthesize / contradict)

Do **not** answer complex cross-doc questions with a single search. Loop:

1. **Clarify** corpus root + whether an index exists; ingest if needed (sets active).
2. **Plan hops** — write 2–N retrieval queries. **When comparing Doc A vs Doc B
   (or any named files), always attach path filters** (`--glob` / `--path-contains`
   / `--path-prefix`) per hop. Unfiltered search often conflates docs that
   cross-reference each other (e.g. Ops notes quoting Spec numbers).
3. **Retrieve** via `search` and/or `multi-search` (active index; no `-i` unless override).
4. **Scratchpad** — append structured notes after each hop (template below).
5. **Verify coverage** — if a hop is thin (`coverage.thin_hops` / low scores /
   missing an expected doc), replan and re-retrieve (new query or looser filter).
6. **Answer** only from scratchpad + cited paths. Call out contradictions
   explicitly. Never invent file contents.

### Scratchpad template

```markdown
## Scratchpad
### Goal
<user question in one line>

### Plan
- hop1: <query> [filter: ...]
- hop2: <query> [filter: ...]

### Notes
| hop | path(s) | score | fact | gap? |
| --- | --- | --- | --- | --- |
| 1 | ... | 0.72 | Spec p99=50ms | |
| 2 | ... | 0.68 | Ops p99=120ms | contradicts hop1 |

### Gaps / contradictions
- ...

### Verify
- [ ] each sub-question has ≥1 solid hit
- [ ] named docs actually retrieved (or explained missing)
- [ ] ready to answer / need another hop: <yes/no + query>
```


### Path filters are mandatory for compares

Eval showed that a global query mentioning Spec numbers can rank **Ops** first
when Ops speaker notes quote the Spec (50ms). Always scope hops with
`--glob '*product_spec*'` / `--glob '*ops_status*'` (etc.). PPTX hits may
appear as `file.pptx#slide-N` (one chunk per slide); globs like `*.pptx` still
match. Treat hops with no hits or MiniLM top score &lt; ~0.20 as **thin** and
re-retrieve.

### Example: compare two docs

```bash
# After ingesting the corpus (active index set):
cat > /tmp/q.json <<'JSON'
[
  {"id":"spec","query":"p99 latency SLA","path_glob":"*product_spec*"},
  {"id":"ops","query":"observed p99 latency","path_glob":"*ops_status*"}
]
JSON
python -m on_the_fly_rag multi-search /tmp/q.json --json -k 3
```

Then fill the scratchpad, note Spec 50ms vs Ops 120ms, and answer with both
citations. If either hop is thin, retry with a broader query or drop the glob.

## Workflow for agents (simple questions)

1. Clarify the corpus root (folder / repo path).
2. Choose grep vs embed using the table above.
3. If embedding: **first** confirm model (MiniLM / Granite Small / Granite / EmbeddingGemma 2 for media),
   unshard if needed, ingest once (sets active + records `model_id`), then search
   **without** naming the index path.
4. For multi-doc / compare / contradiction → use the **multi-hop loop**.
5. Answer using retrieved chunks; quote paths. Do not invent file contents.
6. Never commit secrets; indexes under `.rag_index/` and `.on-the-fly-rag.json`
   are gitignored.

## Example user prompts

Users invoke this skill in Copilot with `/on-the-fly-rag` then a natural-language ask.
**Prefer prompts that do not mention `.rag_index`.** Typical prompts:

```text
/on-the-fly-rag index ./docs and make it the default corpus
```

```text
/on-the-fly-rag index this folder
```

```text
/on-the-fly-rag index ./docs with granite-small
```

```text
/on-the-fly-rag what discusses authentication? cite file paths
```

```text
/on-the-fly-rag index ./field_notes (photos, voice memos, clips) with embeddinggemma-2, then find the red bicycle photo
```

```text
/on-the-fly-rag where in the videos does a candle appear? give timestamps
```

```text
/on-the-fly-rag compare product_spec vs ops_status on p99 latency; path-filter each hop and cite both sides
```

```text
/on-the-fly-rag Pro pricing vs Spec rate limits: plan hops, keep a scratchpad, re-retrieve if coverage is thin, then answer with citations
```

Override example (multi-store):

```text
/on-the-fly-rag search using index /tmp/legal_corpus/.rag_index for retention policy; cite paths
```

More copy-paste examples live in the development repo root README **Prompt cookbook** (fixtures/tests stay at repo root; this folder is the distributable skill).

## Models & sharding

Presets: `minilm` | `granite-small` | `granite` | `embeddinggemma-2`. Custom ONNX: `--model path/to.onnx`.
Granite English R2 and EmbeddingGemma 2 are shipped sharded; consumers unshard once (command above).
To shard a new weight before commit:

```bash
python -m on_the_fly_rag shard path/to/big.onnx --shard-size 90000000
```
