---
name: on-the-fly-rag
description: >-
  On-the-fly RAG over a local folder of docs or a codebase. Use when the user
  asks to search, retrieve, or answer questions from a document corpus without
  a hosted vector DB. Chooses between lightweight grep/ripgrep and local
  embedding RAG with a bundled MiniLM ONNX model. Supports PDF/DOCX/PPTX
  ingest and multi-hop retrieval with a scratchpad.
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
| Multi-doc compare / contradiction / synthesis | Use **multi-hop embed** below |

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

Deps include **pypdf**, **python-docx**, **python-pptx** so `.pdf` / `.docx` /
`.pptx` are extracted to plain text during ingest (same chunk → embed → store
path as markdown/code).

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

Supported sources: text/code extensions **plus** `.pdf`, `.docx`, `.pptx`
(slide body + speaker notes). Unparseable binaries are **logged and skipped**;
ingest does not abort.

Chunking stays under MiniLM's **256-token** window (~200 tokens + overlap).

## Search (single query)

```bash
python -m on_the_fly_rag search "how is auth handled?" --index .rag_index -k 5
python -m on_the_fly_rag search "latency SLA" -i .rag_index --path-contains product_spec
python -m on_the_fly_rag search "observed latency" -i .rag_index --glob '*.pptx'
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
python -m on_the_fly_rag multi-search queries.json -i .rag_index --json -k 5
# or pipe JSON list on stdin
echo '[{"id":"a","query":"...","path_glob":"*.pdf"}]' \
  | python -m on_the_fly_rag multi-search -i .rag_index --json
```

Each list item: string **or** object with `query`, optional `id`, `top_k`,
`path_contains`, `path_glob`, `path_prefix`. Response includes `coverage`
(`thin_hops`, `unique_paths`, `ok`).

## Multi-hop agent loop (required for compare / synthesize / contradict)

Do **not** answer complex cross-doc questions with a single search. Loop:

1. **Clarify** corpus root + whether an index exists; ingest if needed.
2. **Plan hops** — write 2–N retrieval queries. **When comparing Doc A vs Doc B
   (or any named files), always attach path filters** (`--glob` / `--path-contains`
   / `--path-prefix`) per hop. Unfiltered search often conflates docs that
   cross-reference each other (e.g. Ops notes quoting Spec numbers).
3. **Retrieve** via `search` and/or `multi-search`.
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
# After ingesting the corpus:
cat > /tmp/q.json <<'JSON'
[
  {"id":"spec","query":"p99 latency SLA","path_glob":"*product_spec*"},
  {"id":"ops","query":"observed p99 latency","path_glob":"*ops_status*"}
]
JSON
python -m on_the_fly_rag multi-search /tmp/q.json -i .rag_index --json -k 3
```

Then fill the scratchpad, note Spec 50ms vs Ops 120ms, and answer with both
citations. If either hop is thin, retry with a broader query or drop the glob.

## Workflow for agents (simple questions)

1. Clarify the corpus root (folder / repo path).
2. Choose grep vs embed using the table above.
3. If embedding: ingest (if no fresh `.rag_index`), then search top-k.
4. For multi-doc / compare / contradiction → use the **multi-hop loop**.
5. Answer using retrieved chunks; quote paths. Do not invent file contents.
6. Never commit secrets; indexes under `.rag_index/` are gitignored.

## Example user prompts

Users invoke this skill in Copilot with `/on-the-fly-rag` then a natural-language ask. Typical prompts:

```text
/on-the-fly-rag ingest this folder ./docs into .rag_index
```

```text
/on-the-fly-rag compare product_spec vs ops_status on p99 latency; path-filter each hop and cite both sides
```

```text
/on-the-fly-rag Pro pricing vs Spec rate limits: plan hops, keep a scratchpad, re-retrieve if coverage is thin, then answer with citations
```

More copy-paste examples (grep vs embed, Office/PDF corpus, simple Q&A) live in the repo README **Prompt cookbook**.

## Swapping a larger model

Replace `models/.../model.onnx` (and tokenizer) or pass `--model` / `--tokenizer`.
If a weight file exceeds GitHub's 100MB limit, shard before commit:

```bash
python -m on_the_fly_rag shard path/to/big.onnx --shard-size 90000000
```

Document unshard-on-first-use for consumers.
