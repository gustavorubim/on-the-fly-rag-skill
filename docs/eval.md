# Multi-hop / multi-doc eval (NovaSync corpus)

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
