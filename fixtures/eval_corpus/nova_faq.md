# NovaSync FAQ (internal)

## What is NovaSync?
NovaSync is the real-time sync API described in the Product Spec v2.1.
Codename and product name are both **NovaSync**.

## Where do I look for numbers?
- Latency SLA and auth policy → `nova_product_spec.pdf`
- Tier prices and rate entitlements → `nova_pricing.docx`
- Live production metrics → `nova_ops_status.pptx`

## Known open questions
1. Spec says **50ms p99**; Ops observes **120ms p99**. Which is binding for customers?
2. Spec says **OAuth2 only**; Ops still accepts **API keys**. When is the cutoff?
3. Free tier is **100 req/min** but Spec default ceiling is **1000 req/min** — Free is intentionally lower.

## Contact
Platform team owns Spec; Growth owns Pricing; SRE owns Ops Status.
