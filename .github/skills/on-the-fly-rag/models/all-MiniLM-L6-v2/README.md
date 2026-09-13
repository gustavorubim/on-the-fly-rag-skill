# all-MiniLM-L6-v2 (ONNX, vendored)

Bundled for **offline** embedding after clone. No Hugging Face download required for the default path.

| File | Notes |
| --- | --- |
| `model.onnx` | FP32 ONNX graph (~87MB, under GitHub’s 100MB limit) |
| `tokenizer.json` | Fast tokenizer |
| `1_Pooling/config.json` | Mean pooling (applied in code) |

Source: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0).

If you replace this with a larger encoder, use:

```bash
python -m on_the_fly_rag shard path/to/weights.onnx
python -m on_the_fly_rag unshard --input path/to/weights.onnx.part
```
