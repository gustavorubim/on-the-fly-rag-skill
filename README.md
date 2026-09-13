# on-the-fly-rag-skill

Self-contained **GitHub Copilot / coding-agent skill** for **on-the-fly RAG** over a local folder of docs or a codebase.

- **Lightweight search** — ripgrep / regex / path filters / cheap structure
- **On-the-fly embedding RAG** — chunk + embed with a **bundled** MiniLM ONNX model, persist a local vector store, search top-k chunks

No hosted vector DB. No Hugging Face download for the default model after clone. Works offline once Python deps are installed.

## What’s included

| Path | Purpose |
| --- | --- |
| `.github/skills/on-the-fly-rag/SKILL.md` | Copilot project skill (also under `skill/on-the-fly-rag/`) |
| `on_the_fly_rag/` | Python package: chunk, embed, ingest, search, shard |
| `scripts/` | Thin CLI wrappers (`ingest.py`, `search.py`, `shard.py`, `unshard.py`) |
| `models/all-MiniLM-L6-v2/` | Vendored ONNX weights + tokenizer (**~87MB**, &lt;100MB) |
| `fixtures/sample_docs/` | Tiny corpus for offline tests |
| `tests/` | Chunk / ingest+search / shard smoke tests |

## Install (Copilot skill)

### Personal skill (all projects)

```bash
git clone https://github.com/gustavorubim/on-the-fly-rag-skill.git
mkdir -p ~/.copilot/skills
cp -R on-the-fly-rag-skill/skill/on-the-fly-rag ~/.copilot/skills/
# Keep the clone — scripts + models are referenced from the skill instructions.
# Or add the clone path and rely on .github/skills inside it as a project skill.
```

Reload skills in Copilot CLI: `/skills reload`, then `/skills info on-the-fly-rag`.

### Project skill (this or any repo)

Copy or submodule so the skill lives at:

```text
.github/skills/on-the-fly-rag/SKILL.md
```

### Python deps (required for embed path)

```bash
cd on-the-fly-rag-skill
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## CLI usage

```bash
# Ingest (parallel across CPU cores by default)
python -m on_the_fly_rag ingest ./fixtures/sample_docs --index .rag_index
python -m on_the_fly_rag ingest ./docs --index .rag_index -j 4   # --workers / -j

# Search
python -m on_the_fly_rag search "parallel CPU workers" --index .rag_index -k 5
python -m on_the_fly_rag search "auth secrets" -i .rag_index --json

# Same via scripts/
python scripts/ingest.py ./docs -o .rag_index -j 8
python scripts/search.py "how does ingest work?" -i .rag_index
```

### Parallel ingest

Ingest uses **`ProcessPoolExecutor`** so chunking and embedding batches run across CPU cores (embedding is CPU-bound under ONNX Runtime).

| Flag | Default | Meaning |
| --- | --- | --- |
| `-j` / `--workers` | `os.cpu_count()` | Number of worker **processes** |

Search stays single-threaded / simple.

## Architecture

```text
docs/ ──► chunk (token-aware, ~200 tok + overlap)
       ──► embed (ONNX all-MiniLM-L6-v2, mean-pool + L2)
       ──► .rag_index/{vectors.npy, chunks.jsonl, config.json}
query ──► embed ──► cosine top-k ──► paths + snippets
```

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` exported ONNX, 384-dim, 256 ctx
- **Chunking:** token-aware via `tokenizers`; stays under 256 (default max ~200 + CLS/SEP)
- **Store:** numpy float32 matrix + JSONL metadata (no SQLite required)
- **Shard/unshard:** split oversized weights for GitHub’s 100MB file limit

```bash
python -m on_the_fly_rag shard models/all-MiniLM-L6-v2/model.onnx
python -m on_the_fly_rag unshard --input models/all-MiniLM-L6-v2/model.onnx.part
```

### Swapping a larger model (e.g. bge-small)

1. Export or download an ONNX encoder + matching `tokenizer.json`.
2. If any file is ≥100MB, `shard` it before pushing; document `unshard` on first use.
3. Pass `--model` / `--tokenizer` to ingest and search (or replace files under `models/`).

## Decision guide: grep vs embed

| Use **rg / grep** | Use **embedding RAG** |
| --- | --- |
| Exact symbols, errors, filenames | Conceptual / paraphrased questions |
| Import / call-site navigation | “What discusses X?” across many docs |
| Huge trees, one-off lookup | User asked to index / retrieve chunks |

## Tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

All tests are offline (vendored model + fixtures).

## Limits

- MiniLM context is **256 tokens** — do not use huge chunks.
- Default model is small/general; domain jargon may need a stronger encoder.
- Index is local cosine search (brute-force); fine for small/medium corpora, not millions of chunks.
- Binary and non-text files are skipped; extension allow-list is in `chunk.iter_files`.
- GitHub blocks files ≥100MB — keep each committed weight file under that limit (default ONNX is ~87MB).

## License

MIT — see [LICENSE](LICENSE).

Model weights: originally from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0).
