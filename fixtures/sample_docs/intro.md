# On-the-fly RAG

On-the-fly RAG indexes a local folder of documents or a codebase and answers
questions with retrieved chunks. It supports two modes:

1. Lightweight search with ripgrep / regex / path filters.
2. Embedding-based retrieval using a bundled MiniLM ONNX model.

Prefer grep when you know exact symbols, error strings, or file names.
Prefer embeddings when the question is conceptual or phrased differently from
the source text.

## Chunking

Chunks stay under MiniLM's 256-token context window. Typical settings use about
200 tokens with a 40-token overlap so neighboring sections remain coherent.
