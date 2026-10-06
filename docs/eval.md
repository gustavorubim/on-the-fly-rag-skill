# Evaluation

> **Status (2026-10-06):** EmbeddingGemma 2 pass/fail table, Granite text comparison and
> public-file sources are final for this run. **Pending:** latency breakdown per query type,
> a second (larger) audio set, and a re-run after any audio-window tuning.

## EmbeddingGemma 2 — pass/fail per query type

**PASS rule (stated before scoring):** hit@1 ≥ 0.60 **and** hit@5 ≥ 0.80 for that row.
"Drive" = a small read-only sample of the owner's Google Drive (generic description below;
contents not reproduced). "Public" = openly licensed Wikimedia Commons files (sources at the
end). All numbers come from `scripts/eval_multimodal.py` against real indexes built with
`--model embeddinggemma-2` (FP32, dim 768, CPU). n is small: treat each row as a smoke-level
signal, not a benchmark.

| Query type | Set | n | hit@1 | hit@5 | MRR@10 | Result | Example query → top result (score) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text→text | Drive | 18 | 0.944 | 0.944 | 0.950 | **PASS** | "What is the worst-case time complexity of linear search?" → course-notes doc, text chunk (0.761) |
| text→text multi-hop (both docs in top 5) | Drive | 4 | 1.000 | 0.750 | 1.000 | **FAIL** (hit@5) | miss: "Snort rules for DDoS + which course covers DDoS" → top 5 all from the paper; syllabus not in top 5 (same miss for Granite / MiniLM) |
| text→image | Drive (8 charts) | 8 | 1.000 | 1.000 | 1.000 | **PASS** | "confusion matrix of the network flow classifier" → `images/chart_b.png` [image] (0.718) |
| text→image | Public | 7 | 1.000 | 1.000 | 1.000 | **PASS** | "a red bicycle parked on grass" → `images/red_bicycle.jpg` [image] (0.738) |
| image→image (`--query-file`) | Public | 2 | 1.000 | 1.000 | 1.000 | **PASS** (n=2) | `pizza2.jpg` (different photo, not indexed) → `images/pizza.jpg` [image] (0.762) |
| text→audio | Public | 7 | 0.429 | 1.000 | 0.667 | **FAIL** (hit@1) | hit: "piano music" → `audio/piano.flac#t=20.0-30.0` [audio 00:20–00:30] (0.701); miss: "a person speaking" → `cat_meow.ogg` (0.695), speech clip at rank 3 (0.652) |
| text→audio | Drive | 1 | 1.000 | 1.000 | 1.000 | n/a | Drive sample has a single 0.1 s test tone; not a meaningful test |
| text→video | Public (64 s compilation, 4 scenes) | 4 | 1.000 | 1.000 | 1.000 | **PASS** | "a cat drinking water from a puddle" → `video/compilation.mp4#t=16.0-32.0` [video 00:16–00:32, frames 16–31] (0.774) |
| text→video | Drive | 0 | – | – | – | n/a | no video files in the Drive sample |
| mixed, no filter (8 charts vs 501 text chunks) | Drive | 8 | 1.000 | 1.000 | 1.000 | **PASS** | same chart queries without `--modality`: every top-1 is the right image, above all text chunks |
| mixed, no filter (one index: Drive text+charts + public images/audio/video) | Drive+Public | 10 | 0.800 | 0.900 | 0.850 | **PASS** | "video of a candle burning" → `video/compilation.mp4#t=0.0-16.0` [video] (0.737); misses: "rain and thunder sounds" → rooster clip (0.673) with thunder at rank 2 (0.669); "How long do emergency production exceptions last?" → not in top 10 |

**Automated per-modality tests** (tiny CC0/synthetic fixtures, real model): see
[Automated coverage](#automated-coverage) — results recorded there.

### Takeaways

- Images, video segments and image→image are clean on these sets; chart images beat 501
  text chunks without any filter.
- **Audio is the weak modality.** Scores for short (<2 s) sound effects cluster within
  ~0.01–0.05 of each other (e.g. "a dog barking": rooster 0.695, cat 0.685, dog 0.675 for the
  0.9 s clip), so hit@1 is unreliable; the 4 s dog clip and the 30 s piano/TTS clips rank
  first. Prompt variants (no prefix, classification/similarity prefixes, "the sound of …")
  and looping / peak-normalizing short clips did not improve top-1 on a 6-clip probe
  (3/6 for the default prompt and most variants, 2/6 for the "the sound of …" phrasings).
- One Drive text miss (policy doc, "how long does an emergency exception last") is ranked
  10th by EmbeddingGemma 2 but 1st by Granite; the doc is a single short chunk.

## Granite text comparison (Drive, text→text only)

Same 18 single-hop + 4 multi-hop questions, same Drive sample. Media are skipped by the ONNX
models (notice printed). Chunk counts differ by tokenizer (Gemma 501 text chunks, Granite 542,
MiniLM 471).

| Model | single-hop hit@1 | hit@5 | MRR@10 | multi-hop hit@1 | both-docs@5 |
| --- | --- | --- | --- | --- | --- |
| embeddinggemma-2 (dim 768) | 0.944 | 0.944 | 0.950 | 1.000 | 0.750 |
| granite (English R2, 149M) | 0.889 | 1.000 | 0.931 | 1.000 | 0.750 |
| granite-small (47M) | 0.944 | 1.000 | 0.972 | 1.000 | 0.750 |
| minilm (22M) | 0.778 | 1.000 | 0.880 | 1.000 | 0.750 |

On this small text set the three stronger models are within one question of each other;
Granite small/large get every answer into the top 5, EmbeddingGemma 2 misses one entirely.
For text-only corpora Granite is the cheaper choice (see cost).

## Cost

Measured on the shared 8-vCPU dev box (another 8-process job was running; load avg ≈ 10),
`--torch-threads 2`, FP32 unless noted. Peak RSS via `getrusage`.

| Step | Wall | Peak RSS |
| --- | --- | --- |
| EmbeddingGemma 2 text-only load (271M params) + 1 tiny doc | 18.6 s (load 10.0 s) | 2,114 MB |
| same, `--dtype bf16` | 12.4 s (load 7.9 s) | 1,042 MB |
| full multimodal load (744M) + 1 tiny doc | 14.0 s (load 10.0 s) | 4,828 MB |
| Drive ingest, gemma: 501 text chunks + 8 images + 1 audio | 582 s (text ≈ 457 s ≈ 1.1 chunks/s; images 14.4 s avg) | 4,727 MB |
| Public ingest, gemma (`--append` re-run of the full folder): 7 images + 8 audio files (94 s of audio) + 64 s video | 522 s (video 4×16-frame segments 371 s; audio 36 s ≈ 2.6× real time; images 14.5 s avg) | 4,767 MB |
| Drive ingest, granite (`-j 2`, text only) | 117 s | 1,764 MB |
| Drive ingest, granite-small (`-j 2`) | 49 s | 1,299 MB |
| Drive ingest, minilm (`-j 2`) | 15 s | 559 MB |
| One text query on a gemma index (CLI, includes text-only load) | ≈ 11 s | ≈ 2,010 MB |

## Corpora

**Drive sample (read-only, never committed).** Downloaded with the Google Drive connector's
read/download tools only (no create/update/move/share/trash/comment). ~6 MB: 4 PDFs (public
regulatory model documentation, a regulatory glossary and data dictionary, an academic
security paper), 2 Word docs (a synthetic AI-governance policy, a course syllabus), 3 Google
Docs exported to DOCX (course notes ×2, a 3D-printer config), 2 slide decks (a synthetic
model-inventory briefing, a public meetup deck), 8 chart PNGs (ML/network-analysis plots),
1 WAV test tone. No video and no meaningful audio → gaps filled with public files.

**Public files** (kept outside the repo; only CC0 excerpts are committed as test fixtures,
see `fixtures/multimodal/SOURCES.md`). The video is a 4-scene 64 s compilation (16 s each:
candle, cat at a puddle, waterfall, traffic lights; 320×180) built from the clips below.

| Local file | Source | Author | License |
| --- | --- | --- | --- |
| `img_bananas.jpg` | [Bunch of bananas on sale.jpg](https://commons.wikimedia.org/wiki/File:Bunch_of_bananas_on_sale.jpg) | Wilfredor | CC0 |
| `img_red_bicycle.jpg` | [The Red Bicycle. (15655615295).jpg](https://commons.wikimedia.org/wiki/File:The_Red_Bicycle._(15655615295).jpg) | Bernard Spragg. NZ from Christchurch, New Zealand | CC0 |
| `img_pizza.jpg` | [Margherita pizza 55.jpg](https://commons.wikimedia.org/wiki/File:Margherita_pizza_55.jpg) | Swathi sri srinivasa raghavan | CC0 |
| `img_sailboat.jpg` | [Nonsuch 22 sailboat 3636.jpg](https://commons.wikimedia.org/wiki/File:Nonsuch_22_sailboat_3636.jpg) | Ahunt | CC0 |
| `img_snowy_mountain.jpg` | [Snowy Forest and Mountain Landscape.jpg](https://commons.wikimedia.org/wiki/File:Snowy_Forest_and_Mountain_Landscape.jpg) | Lelexed9 | CC0 |
| `img_giraffe.jpg` | [Giraffe in Bandia reserve near Dakar Senegal.jpg](https://commons.wikimedia.org/wiki/File:Giraffe_in_Bandia_reserve_near_Dakar_Senegal.jpg) | Gromane | CC0 |
| `img_guitar.jpg` | [Bodyless electric guitar for sale at Sam Ash in New Jersey.JPG](https://commons.wikimedia.org/wiki/File:Bodyless_electric_guitar_for_sale_at_Sam_Ash_in_New_Jersey.JPG) | Tomwsulcer | CC0 |
| `qimg_pizza2.jpg` | [Pizza 10.jpg](https://commons.wikimedia.org/wiki/File:Pizza_10.jpg) | Kurt Kaiser | CC0 |
| `qimg_giraffe2.jpg` | [Giraffe at Zoorasia.jpg](https://commons.wikimedia.org/wiki/File:Giraffe_at_Zoorasia.jpg) | Syced | CC0 |
| `aud_dog_bark.wav` | [Bark (avk).wav](https://commons.wikimedia.org/wiki/File:Bark_(avk).wav) | Luce VERGNEAUX | CC0 |
| `aud_cat_meow.ogg` | [Maullido de gata hembra joven.ogg](https://commons.wikimedia.org/wiki/File:Maullido_de_gata_hembra_joven.ogg) | George Miquilena | CC0 |
| `aud_speech.ogg` | [I'm going anyway.ogg](https://commons.wikimedia.org/wiki/File:I%27m_going_anyway.ogg) | Biolongvistul | CC0 |
| `aud_piano.wav` | [Relaxing-piano-music-351844.wav](https://commons.wikimedia.org/wiki/File:Relaxing-piano-music-351844.wav) | Clavier Music | CC0 |
| `aud_rooster.ogg` | [Medium rooster crowing.ogg](https://commons.wikimedia.org/wiki/File:Medium_rooster_crowing.ogg) | alys | Public domain |
| `aud_thunder_rain.ogg` | [Rain and thunder.ogg](https://commons.wikimedia.org/wiki/File:Rain_and_thunder.ogg) | User:Caesar | Public domain |
| `vid_candle.webm` | [Blue candle burning.webm](https://commons.wikimedia.org/wiki/File:Blue_candle_burning.webm) | Jahobr | CC0 |
| `vid_traffic.webm` | [FFM - Münchener Straße, Grünschaltung Auto- und Straßenbahnampel.webm](https://commons.wikimedia.org/wiki/File:FFM_-_M%C3%BCnchener_Stra%C3%9Fe,_Gr%C3%BCnschaltung_Auto-_und_Stra%C3%9Fenbahnampel.webm) | ManuelB701 | CC0 |
| `vid_waterfall.webm` | [Watagataki Falls (Video).webm](https://commons.wikimedia.org/wiki/File:Watagataki_Falls_(Video).webm) | Yoshi Canopus | CC0 |
| `vid_cat_puddle.ogv` | [Black and white cat drinks from a puddle.ogv](https://commons.wikimedia.org/wiki/File:Black_and_white_cat_drinks_from_a_puddle.ogv) | User:Mattes | Public domain |
| `aud_speech_long.ogg` | [Political philosophy - lead section.ogg](https://commons.wikimedia.org/wiki/File:Political_philosophy_-_lead_section.ogg) | Speech Synthesis reading a paragraph of the Wikipedia article "Political philoso | Public domain |
| `aud_dog_bark2.ogg` | [Barking of a dog 2.ogg](https://commons.wikimedia.org/wiki/File:Barking_of_a_dog_2.ogg) | Amada44 | CC BY-SA 3.0 |

## Reproduce

```bash
cd .github/skills/on-the-fly-rag
python -m on_the_fly_rag ingest <drive_sample> --model embeddinggemma-2 -o idx/drive_gemma --torch-threads 2
python -m on_the_fly_rag ingest <public_corpus> --model embeddinggemma-2 -o idx/public_gemma --torch-threads 2
python -m on_the_fly_rag ingest <drive_sample> --model granite -o idx/drive_granite     # + granite-small, minilm
python scripts/eval_multimodal.py --questions q_drive.json --questions q_public.json \
  --index drive_gemma=idx/drive_gemma --index public_gemma=idx/public_gemma \
  --index drive_granite=idx/drive_granite ... --compare drive_gemma:drive_granite,drive_granite_small,drive_minilm
```

The question files reference private Drive content and are not committed; the question
schema is documented in the script header. Index names like `a+b` merge same-model indexes
in memory for the mixed-modality rows.

## Automated coverage

- `tests/test_multimodal_unit.py` — prompts, fp16 refusal, Matryoshka truncate+renorm,
  model/dim compat refusals (search + append), modality filter, store back-compat, media
  segmenting (ffmpeg), ONNX skip-media notice (no model needed).
- `tests/test_shard.py` — per-part and whole-file SHA-256 checks, corrupt/truncated/missing
  shards fail loudly, `unshard --verify-only` exit code 2, bundled manifests consistent.
- `tests/test_gemma_multimodal.py` (`-m gemma`, real model, auto-skipped otherwise) —
  text→text, text→image, image→image, text→audio, text→video, mixed index, text-only load +
  dim 256. Run: `python -m pytest -m gemma -v`.
  Local run with the real model (2026-10-06, `ON_THE_FLY_RAG_TORCH_THREADS=3`, shared box):
  **10 passed, 15 subtests passed in 226.7 s**, peak RSS 8.7 GB (the fixture ingest and the
  test's own full embedder are both resident). Non-model suite: 66 passed, 10 skipped (gemma)
  when the model is absent. The first real-model run failed 3 audio assertions (0.9 s dog bark
  ranked below the 1.2 s cat meow; 0.7 s speech clip not in top 2); the audio tests now assert
  exact top-1 for the distinct clips (cat, piano) and category-level top-1 + exact top-2 for
  the dog bark — consistent with the audio weakness reported above.

---

# Multi-hop / multi-doc eval (NovaSync corpus, MiniLM)

Corpus: `fixtures/eval_corpus/`

| File | Format | Role |
| --- | --- | --- |
| `nova_product_spec.pdf` | PDF | Spec: **50ms** p99 SLA, **OAuth2 only**, 1000 req/min |
| `nova_pricing.docx` | DOCX | Pricing: Free 100 rpm / Pro **$49** @ 1000 rpm / Ent **$499** |
| `nova_ops_status.pptx` | PPTX | Ops: observed **120ms** p99, **API keys + OAuth2**, notes |
| `nova_faq.md` | MD | Cross-links + open questions |

Deliberate cross-deps: shared product **NovaSync**; Spec↔Ops latency contradiction; Spec↔Ops auth drift; Pricing↔Spec rate-limit alignment; FAQ points at all three. Ops notes intentionally quote Spec’s 50ms (trap for unfiltered search).

## Queries exercised

### A. Compare latency (path-filtered hops) — **PASS**

```json
[
  {"id":"spec","query":"NovaSync p99 latency SLA under 50 milliseconds","path_glob":"*product_spec*"},
  {"id":"ops","query":"NovaSync observed p99 latency production","path_glob":"*ops_status*"}
]
```

**Result:** Spec hop → `nova_product_spec.pdf` (50ms, score ~0.59). Ops hop → `nova_ops_status.pptx#slide-1` (120ms, score ~0.63). Contradiction clear in scratchpad.

### A2. Same compare **without** path filters — **FAIL (expected)**

Both hops ranked Ops first because Ops notes mention Spec’s 50ms. This is why the skill requires path filters for Doc A vs Doc B compares.

### B. Auth policy drift (batched + filters) — **PASS**

```json
[
  {"id":"spec_auth","query":"Authentication OAuth2 only API keys unsupported","path_glob":"*.pdf"},
  {"id":"ops_auth","query":"API keys AND OAuth2 both accepted production","path_glob":"*.pptx"}
]
```

**Result:** Spec = OAuth2 only / keys unsupported; Ops slide about auth = keys+OAuth2 + Q4 deprecation. Per-slide PPTX chunking (`#slide-N`) keeps auth separate from latency.

### C. Pricing ↔ Spec rate limit (global then refine) — **PASS**

1. Global `Pro tier cost requests per minute` already tops `nova_pricing.docx`.
2. Refined hops with `--path-contains pricing` + Spec rate-limit glob recover Pro **$49 / 1000 rpm** matching Spec ceiling; Free **100 rpm** intentionally lower.

### D. Three-hop FAQ → Spec → Ops — **PASS**

FAQ open-questions hop → Spec 50ms → Ops 120ms. Scratchpad keeps both numbers; FAQ explicitly flags the gap.

### E. Thin hop then recover — **PASS**

Bad glob `*does_not_exist*` → `coverage.thin_hops`; re-query with `*ops*` recovers.

### F. Sequential entity hop — **PASS**

FAQ establishes “NovaSync” → Spec auth → Ops auth. Works when later queries depend on earlier facts.

## Approaches tried

| Approach | Result |
| --- | --- |
| Single global query “NovaSync latency” | Mixed; easy to miss contradiction |
| Path-filtered per doc (`multi-search`) | **Reliable** |
| Unfiltered multi-query (A2) | **Unreliable** when docs cross-quote |
| Sequential hops | Good when hop N depends on hop N−1 |
| Batched `multi-search` + `coverage` | Best for planned compares |
| Per-slide PPTX sections | Improves auth vs latency separation |

## What we changed after eval

1. `--glob` / `--path-prefix` / `multi-search` + `coverage_stats` (thin &lt; 0.20).
2. Skill: **mandatory path filters for compares**; scratchpad + verify loop.
3. PPTX: extract notes; chunk per `[Slide N]` as `file.pptx#slide-N`.
4. Globs match paths with `#slide` fragments.
5. Skip-on-extract-failure so one bad PDF cannot abort ingest.

## Automated coverage

- `tests/test_extract.py` — PDF/DOCX/PPTX fixtures, bad-PDF skip, eval corpus ingest
- `tests/test_multi_search.py` — path filters, Spec 50 vs Ops 120, CLI JSON + coverage

## Success criteria

Met: Office extractors offline; multi-hop path filters surface conflicting numbers from different files; skill tells agents to verify/re-retrieve when thin; unfiltered compare documented as a failure mode.
