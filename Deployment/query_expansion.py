"""
query_expansion.py

Fixes the "paraphrased question gets no answer" failure mode: hybrid
retrieval (BM25 + embeddings) can genuinely fail to connect "two supplier
numbers" to a document's "Multiple Supplier Numbers" -- BM25 needs literal
token overlap, and a small/fast embedding model may not bridge the gap
either. No amount of prompt tuning fixes an answer built from a chunk that
was never retrieved.

The fix: before retrieving, ask the LLM to generate a couple of alternate
phrasings of the question. Retrieve for the original question AND each
variant, then merge results (summing per-chunk relevance across variants,
i.e. a chunk that multiple phrasings agree on ranks higher). If the user's
wording and the document's wording diverge, at least one LLM-generated
variant is likely to land closer to the document's actual vocabulary.

This is entirely generic -- it doesn't know or care what the question is
about, so it helps any app's paraphrase-robustness equally, without any
app-specific keyword lists.

Trade-off: one extra LLM call per question (for generating variants). For
an internal support tool this is usually an acceptable latency/cost trade
for meaningfully better recall; if it ever becomes a bottleneck, this can
be made conditional (e.g. only expand when the first-pass retrieval score
is low) rather than running on every question.
"""

from chunking_main import _generate_content_with_retry

DEFAULT_NUM_VARIANTS = 2


def generate_query_variants(query, num_variants=DEFAULT_NUM_VARIANTS):
    prompt = f"""Generate {num_variants} alternative ways to phrase the following
question. Use different words and sentence structure while keeping exactly
the same meaning and intent -- don't add or remove any implied detail.
Return ONLY the alternative phrasings, one per line, with no numbering,
no quotation marks, and no other text.

Question: {query}"""

    try:
        response = _generate_content_with_retry(prompt)
        text = (response.text or "").strip()
        variants = [line.strip(' \n"\'') for line in text.split("\n") if line.strip()]
        return variants[:num_variants]
    except Exception:
        return []  # expansion is a best-effort enhancement, never a hard dependency


def multi_query_search(query, hybrid_index, search_fn, top_k=6, num_variants=DEFAULT_NUM_VARIANTS, **search_kwargs):
    """Runs search_fn (either hybrid_index.search or hybrid_index.search_diverse)
    for the original query plus its LLM-generated variants, then merges results
    by summing relevance score per chunk_id across all variants that surfaced
    it -- a chunk multiple phrasings agree on floats to the top."""
    variants = generate_query_variants(query, num_variants=num_variants)
    all_queries = [query] + variants

    combined = {}  # chunk_id -> result dict (text/source/etc kept from first sighting)
    scores = {}    # chunk_id -> summed score

    for q in all_queries:
        results = search_fn(q, top_k=top_k, **search_kwargs)
        for r in results:
            cid = r["chunk_id"]
            if cid not in combined:
                combined[cid] = r
            scores[cid] = scores.get(cid, 0.0) + r["score"]

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
    merged = []
    for rank, cid in enumerate(ranked_ids, start=1):
        result = dict(combined[cid])
        result["rank"] = rank
        result["score"] = scores[cid]
        merged.append(result)
    return merged