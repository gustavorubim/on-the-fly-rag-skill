# API overview

## ingest

`python -m on_the_fly_rag ingest ./docs --index .rag_index -j 8`

Builds a local vector store from a folder. Parallel CPU workers (default:
`os.cpu_count()`) speed up chunking and embedding batches.

## search

`python -m on_the_fly_rag search "how does ingest work?" --index .rag_index -k 5`

Returns top-k chunks with paths and cosine similarity scores. Search is
single-threaded and fast once the index exists.

## Authentication notes

This demo corpus mentions that API keys must never be committed. Use
environment variables instead of hard-coding secrets in source files.
