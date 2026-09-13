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
| `models/` | Bundled MiniLM + Granite ONNX (shards for large Granite) |
| `scripts/` | Thin CLI wrappers |
| `requirements.txt` / `pyproject.toml` | Deps |

Run CLIs from this folder (or with `PYTHONPATH=.` / `pip install -e .`):

```bash
python -m on_the_fly_rag models
python -m on_the_fly_rag unshard --input models/granite-embedding-english-r2/model.onnx.part
python -m on_the_fly_rag ingest /path/to/docs --model minilm
python scripts/search.py "query"
```

Development tests, fixtures, and docs stay at the **repository root**; this folder is the distributable skill.
