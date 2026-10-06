# on-the-fly-rag (Copilot skill package)

**Self-contained** skill unit: copy this entire folder into your Copilot skills directory.

```bash
# From a clone of https://github.com/gustavorubim/on-the-fly-rag-skill
cp -R .github/skills/on-the-fly-rag ~/.copilot/skills/on-the-fly-rag
cd ~/.copilot/skills/on-the-fly-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional editable install:
# pip install -e .
```

Then reload skills (`/skills reload`) and invoke with `/on-the-fly-rag`.

| Path | Purpose |
| --- | --- |
| `SKILL.md` | Copilot skill instructions |
| `on_the_fly_rag/` | Python package |
| `models/` | Bundled MiniLM + Granite ONNX (shards for large Granite) + EmbeddingGemma 2 safetensors (16 shards) |
| `scripts/` | Thin CLI wrappers (+ `eval_multimodal.py` retrieval eval) |
| `requirements.txt` / `pyproject.toml` | Deps |
| `requirements-gemma.txt` / `.[gemma]` | Optional PyTorch extras for `embeddinggemma-2` (multimodal) |

Run CLIs from this folder (or with `PYTHONPATH=.` / `pip install -e .`):

```bash
python -m on_the_fly_rag models
python -m on_the_fly_rag unshard --model granite
python -m on_the_fly_rag ingest /path/to/docs --model minilm
python scripts/search.py "query"

# Optional multimodal model (text + images + video + audio); ~2–5GB RAM, much slower than ONNX
pip install -r requirements-gemma.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m on_the_fly_rag unshard --model embeddinggemma-2        # SHA-256 verified
python -m on_the_fly_rag ingest /path/to/media_folder --model embeddinggemma-2
python -m on_the_fly_rag search "a red bicycle" --modality image
python -m on_the_fly_rag search --query-file photo.jpg --modality image
```

Development tests, fixtures, and docs stay at the **repository root**; this folder is the distributable skill.
