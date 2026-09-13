# granite-embedding-small-english-r2

Vendored ONNX (fp16) export of [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)
via [onnx-community/granite-embedding-small-english-r2-ONNX](https://huggingface.co/onnx-community/granite-embedding-small-english-r2-ONNX).

- Embedding dim: 384
- Max context: 8192 tokens
- Pooling: CLS (`sentence_embedding` ONNX output)
- License: Apache-2.0 (IBM Granite)
- Format: single-file ONNX for onnxruntime CPU

If `model.onnx` is missing but `model.onnx.part*` shards exist, unshard first:
```bash
python -m on_the_fly_rag unshard --input models/granite-embedding-small-english-r2/model.onnx.part
```
