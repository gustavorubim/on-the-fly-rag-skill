#!/usr/bin/env python3
"""Stress-test multi-hop retrieval on fixtures/eval_corpus."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
# Fixtures live at the development repo root (skill_root/.github/skills -> repo)
REPO_ROOT = SKILL_ROOT
for cand in (SKILL_ROOT.parents[2], SKILL_ROOT.parents[1], SKILL_ROOT):
    if (cand / "fixtures" / "eval_corpus").is_dir():
        REPO_ROOT = cand
        break
import sys
sys.path.insert(0, str(SKILL_ROOT))

from on_the_fly_rag.ingest import ingest
from on_the_fly_rag.search import coverage_stats, multi_search, search

EVAL = REPO_ROOT / "fixtures" / "eval_corpus"


def show(title, results):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    cov = coverage_stats(results) if isinstance(results, list) and results and "hits" in results[0] else None
    if cov:
        for h in results:
            top = h["hits"][0] if h["hits"] else None
            print(f"\n[{h['id']}] q={h['query']!r} filters={h.get('filters')}")
            if not top:
                print("  NO HITS")
                continue
            print(f"  top={top['score']:.4f} path={top['path']}")
            snippet = top["text"].replace("\n", " ")[:180]
            print(f"  text: {snippet}")
            # key facts
            t = " ".join(x["text"] for x in h["hits"])
            facts = []
            for needle in ("50", "120", "OAuth2", "API key", "API keys", "$49", "$499", "1000", "100 "):
                if needle.lower() in t.lower() or needle in t:
                    facts.append(needle)
            print(f"  markers: {facts}")
        print(f"\nCOVERAGE: ok={cov['ok']} thin={cov['thin_hops']} paths={cov['unique_paths']}")
        return cov
    else:
        for i, hit in enumerate(results, 1):
            print(f"{i}. {hit['score']:.4f} {hit['path']}: {hit['text'][:120].replace(chr(10),' ')}")
        return None


def main():
    td = tempfile.mkdtemp(prefix="nova_eval_")
    index = Path(td) / "idx"
    print(f"Ingesting {EVAL} -> {index}")
    cfg = ingest(EVAL, index_dir=index, workers=1)
    print("ingest:", cfg)

    report = {"scenarios": []}

    # --- A: path-filtered compare latency ---
    q_a = [
        {"id": "spec", "query": "NovaSync p99 latency SLA under 50 milliseconds", "path_glob": "*product_spec*"},
        {"id": "ops", "query": "NovaSync observed p99 latency production", "path_glob": "*ops_status*"},
    ]
    ra = multi_search(q_a, index_dir=index, top_k=3)
    cov_a = show("A: path-filtered latency compare", ra)
    spec_txt = " ".join(h["text"] for h in ra[0]["hits"])
    ops_txt = " ".join(h["text"] for h in ra[1]["hits"])
    ok_a = ("50" in spec_txt) and ("120" in ops_txt) and cov_a["ok"]
    report["scenarios"].append({"name": "A_latency_compare", "ok": ok_a, "coverage": cov_a})

    # --- A2: same without filters (weaker?) ---
    q_a2 = [
        {"id": "spec", "query": "product spec p99 latency SLA 50 milliseconds"},
        {"id": "ops", "query": "ops status observed p99 latency 120 milliseconds"},
    ]
    ra2 = multi_search(q_a2, index_dir=index, top_k=3)
    cov_a2 = show("A2: latency compare WITHOUT path filters", ra2)
    # check if each hop's top path is the intended file
    path_ok = (
        "product_spec" in ra2[0]["hits"][0]["path"]
        and "ops_status" in ra2[1]["hits"][0]["path"]
    )
    # A2 is an expected failure mode when docs cross-quote each other.
    report["scenarios"].append({
        "name": "A2_no_filter_expected_weak",
        "ok": True,  # scenario ran; weakness documented
        "demonstrates_need_for_filters": not path_ok,
        "path_ok": path_ok,
        "coverage": cov_a2,
    })
    print("A2 demonstrates filter need:", not path_ok)

    # --- B: auth drift ---
    q_b = [
        {"id": "spec_auth", "query": "Authentication OAuth2 only API keys unsupported", "path_glob": "*.pdf"},
        {"id": "ops_auth", "query": "API keys AND OAuth2 both accepted production", "path_glob": "*.pptx"},
    ]
    rb = multi_search(q_b, index_dir=index, top_k=3)
    cov_b = show("B: auth policy drift", rb)
    st = " ".join(h["text"] for h in rb[0]["hits"]).lower()
    ot = " ".join(h["text"] for h in rb[1]["hits"]).lower()
    ok_b = ("oauth2" in st and "unsupported" in st) and ("api key" in ot and "oauth2" in ot)
    report["scenarios"].append({"name": "B_auth_drift", "ok": ok_b, "coverage": cov_b})

    # --- C: pricing + rate limit (global then refine) ---
    global_hits = search("Pro tier cost requests per minute", index_dir=index, top_k=5)
    show("C1: global Pro tier", global_hits)
    q_c = [
        {"id": "price", "query": "Pro tier $49 requests per minute", "path_contains": "pricing"},
        {"id": "spec_rate", "query": "default rate limit ceiling requests per minute", "path_glob": "*product_spec*"},
        {"id": "free", "query": "Free tier 100 requests per minute", "path_contains": "pricing"},
    ]
    rc = multi_search(q_c, index_dir=index, top_k=3)
    cov_c = show("C2: refined pricing vs Spec rate limit", rc)
    pt = " ".join(h["text"] for h in rc[0]["hits"])
    rt = " ".join(h["text"] for h in rc[1]["hits"])
    ok_c = ("49" in pt and "1000" in pt) and ("1000" in rt)
    report["scenarios"].append({"name": "C_pricing_rate", "ok": ok_c, "coverage": cov_c})

    # --- D: three-hop FAQ → Spec → Ops ---
    q_d = [
        {"id": "faq", "query": "known open questions latency Spec Ops", "path_glob": "*.md"},
        {"id": "spec", "query": "Latency SLA p99 milliseconds", "path_glob": "*product_spec*"},
        {"id": "ops", "query": "Observed p99 latency milliseconds last 7 days", "path_glob": "*ops_status*"},
    ]
    rd = multi_search(q_d, index_dir=index, top_k=3)
    cov_d = show("D: three-hop FAQ→Spec→Ops", rd)
    faq_t = " ".join(h["text"] for h in rd[0]["hits"])
    ok_d = ("50" in " ".join(h["text"] for h in rd[1]["hits"])) and ("120" in " ".join(h["text"] for h in rd[2]["hits"])) and ("50ms" in faq_t or "50" in faq_t)
    report["scenarios"].append({"name": "D_three_hop", "ok": ok_d, "coverage": cov_d})

    # --- E: thin hop then recover ---
    q_e_bad = [{"id": "bad", "query": "latency", "path_glob": "*does_not_exist*"}]
    re_bad = multi_search(q_e_bad, index_dir=index, top_k=3)
    cov_bad = coverage_stats(re_bad)
    q_e_fix = [{"id": "fixed", "query": "NovaSync observed p99 latency production", "path_glob": "*ops*"}]
    re_fix = multi_search(q_e_fix, index_dir=index, top_k=3)
    cov_fix = coverage_stats(re_fix)
    show("E1: intentional thin (bad glob)", re_bad)
    show("E2: recovered hop", re_fix)
    ok_e = (not cov_bad["ok"]) and cov_fix["ok"]
    report["scenarios"].append({"name": "E_thin_then_recover", "ok": ok_e})

    # --- F: sequential dependency simulation ---
    # hop1 find product name from FAQ, hop2 search Spec with that entity
    h1 = search("What is NovaSync product name", index_dir=index, top_k=2, path_glob="*.md")
    entity = "NovaSync"
    h2 = search(f"{entity} authentication policy OAuth2", index_dir=index, top_k=3, path_glob="*.pdf")
    h3 = search(f"{entity} API keys AND OAuth2 both accepted production", index_dir=index, top_k=3, path_glob="*.pptx")
    print("\nF: sequential entity hop")
    print("  h1 paths", [x["path"] for x in h1])
    print("  h2", h2[0]["score"], h2[0]["path"], "OAuth2" in h2[0]["text"])
    print("  h3", h3[0]["score"], h3[0]["path"], "API" in h3[0]["text"])
    ok_f = h1 and h2 and h3 and "OAuth2" in h2[0]["text"] and "slide-2" in h3[0]["path"] and "API keys" in h3[0]["text"]
    report["scenarios"].append({"name": "F_sequential", "ok": bool(ok_f)})

    print("\n" + "#" * 72)
    print("SUMMARY")
    all_ok = True
    for s in report["scenarios"]:
        print(f"  {s['name']}: {'PASS' if s['ok'] else 'FAIL'}")
        all_ok = all_ok and s["ok"]
    print("ALL_PASS" if all_ok else "SOME_FAIL")
    out = Path("/tmp/nova_eval_report.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("wrote", out)
    return 0 if all_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
