"""
retrieval_v2.py (generic rewrite)

Fully domain-agnostic chunking + retrieval. Nothing in this file references
column names, error tables, or any other content specific to the 450T docs --
onboarding a new application should never require editing this file.

Structural signals used (and why each generalizes):

  - Section = the document's own headings (docx) or sheet names (xlsx).
    Every reasonably-authored document already has this structure; we don't
    infer meaning, we just use what the author already organized.

  - is_table = whether a block is literally delimited tabular data (from
    loader.py). Reference/rule content in support docs is very often
    tabular regardless of the app; prose explanations are not. This is a
    structural fact about the block, not a guess about its topic.

  - Diagnostic vs. lookup question classification uses generic troubleshoot
    vocabulary ("why", "fail", "error", "not working", "issue", ...) that
    applies to asking about problems in *any* application, not just file
    validation.

  - Question widening isn't just "diagnostic vs. lookup" anymore. Two
    generic, structural signals independently trigger the wider MMR search:
    diagnostic vocabulary (as before), AND negation/polarity words ("not",
    "doesn't", "isn't", "without", ...). Negation questions are a known
    generic weak spot for hybrid retrieval -- a question like "does X load
    Y itself?" can score poorly against a document chunk phrased as "Y is
    NOT loaded by X", because the phrasing barely overlaps lexically and
    the negation flips the meaning entirely. This applies to any app's
    "does/is/can it NOT do such-and-such" questions, not just one doc's
    config section, so it's handled as a general linguistic signal rather
    than special-cased content.
"""

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from query_expansion import multi_query_search

DIAGNOSTIC_TERMS = (
    "why", "fail", "failing", "failed", "error", "invalid", "issue", "problem",
    "wrong", "incorrect", "not working", "doesn't work", "does not work",
    "broken", "reject", "rejected", "denied", "cause", "reason", "troubleshoot",
    "fix", "resolve", "missing",
)

NEGATION_TERMS = (
    "not", "n't", "never", "cannot", "without", "no longer", "unable",
    "doesn't", "does not", "isn't", "is not", "won't", "will not",
    "can't", "cant", "aren't", "wasn't", "weren't",
)

# A yes/no question ("does X do Y?", "is Z supported?") can be correctly
# answered by a document sentence phrased in the OPPOSITE polarity ("Y is
# not done by X"), with very little shared vocabulary between question and
# answer. The question itself need not contain any negation word for this
# mismatch to happen -- it's the question TYPE (polar/yes-no), not its
# wording, that predicts the risk. Detected generically by sentence-initial
# auxiliary/modal verbs, which works for any app's yes/no questions.
YES_NO_QUESTION_STARTERS = (
    "does", "do", "did", "is", "are", "was", "were",
    "can", "could", "will", "would", "should", "shall", "has", "have",
)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str = ""
    app: str = ""
    section: str = ""
    is_table: bool = False

    def as_result(self, rank=0, score=1.0):
        return {
            "rank": rank,
            "chunk_id": self.chunk_id,
            "score": float(score),
            "text": self.text,
            "source": self.source,
            "app": self.app,
            "section": self.section,
            "is_table": self.is_table,
        }


# ---------------------------------------------------------------------------
# Chunking: operates on structured Block objects (from loader.py), not a
# flattened string. Tables are kept whole up to a generous size cap, since
# splitting a reference table mid-way is more damaging than a slightly large
# chunk; narrative text is windowed by word count with overlap, as before.
# ---------------------------------------------------------------------------
def chunk_blocks(blocks, app="", chunk_size=200, overlap=50, max_table_words=600):
    chunks = []
    chunk_id = 0

    for block in blocks:
        words = block.text.split()

        if block.is_table:
            if len(words) <= max_table_words:
                chunks.append(Chunk(
                    chunk_id=chunk_id, text=_with_header(block), source=block.source,
                    app=app, section=block.section, is_table=True,
                ))
                chunk_id += 1
            else:
                # Oversized table: split by rows (lines) rather than mid-row,
                # keeping the header row attached to every piece so column
                # meaning isn't lost.
                lines = block.text.split("\n")
                header_line = lines[0] if lines else ""
                body_lines = lines[1:]
                window = []
                window_words = 0
                for line in body_lines:
                    window.append(line)
                    window_words += len(line.split())
                    if window_words >= chunk_size:
                        piece_text = "\n".join([header_line] + window) if header_line else "\n".join(window)
                        chunks.append(Chunk(
                            chunk_id=chunk_id, text=_with_header(block, override_text=piece_text),
                            source=block.source, app=app, section=block.section, is_table=True,
                        ))
                        chunk_id += 1
                        window, window_words = [], 0
                if window:
                    piece_text = "\n".join([header_line] + window) if header_line else "\n".join(window)
                    chunks.append(Chunk(
                        chunk_id=chunk_id, text=_with_header(block, override_text=piece_text),
                        source=block.source, app=app, section=block.section, is_table=True,
                    ))
                    chunk_id += 1
            continue

        # Narrative text: sliding window over words with overlap.
        if len(words) <= chunk_size:
            chunks.append(Chunk(
                chunk_id=chunk_id, text=_with_header(block), source=block.source,
                app=app, section=block.section, is_table=False,
            ))
            chunk_id += 1
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            piece_text = " ".join(words[start:end])
            chunks.append(Chunk(
                chunk_id=chunk_id, text=_with_header(block, override_text=piece_text),
                source=block.source, app=app, section=block.section, is_table=False,
            ))
            chunk_id += 1
            if end == len(words):
                break
            start = end - overlap

    return chunks


def _with_header(block, override_text=None):
    text = override_text if override_text is not None else block.text
    header = f"Source: {block.source} | Section: {block.section}"
    return f"{header}\n{text}"


# ---------------------------------------------------------------------------
# Hybrid retrieval: BM25 (lexical) + embeddings (semantic) via Reciprocal
# Rank Fusion. Domain-agnostic by construction -- it only ever looks at
# token overlap and vector similarity, never at what the tokens mean.
# ---------------------------------------------------------------------------
def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


class HybridIndex:
    def __init__(self, chunks, embedding_model, faiss_index):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def _embed_query(self, query):
        return self.embedding_model.encode(query, normalize_embeddings=True).astype("float32").reshape(1, -1)

    def _candidate_indices(self, app):
        indices = list(range(len(self.chunks)))
        if app:
            scoped = [i for i in indices if self.chunks[i].app == app]
            if scoped:
                return scoped
        return indices

    def search(self, query, top_k=5, app=None, rrf_k=60):
        if not self.chunks:
            return []

        candidate_indices = set(self._candidate_indices(app))

        query_vec = self._embed_query(query)
        sem_scores, sem_idx = self.faiss_index.search(query_vec, len(self.chunks))
        sem_rank = {}
        rank = 0
        for idx in sem_idx[0]:
            if idx < 0 or idx not in candidate_indices:
                continue
            rank += 1
            sem_rank[idx] = rank

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        lex_order = sorted(candidate_indices, key=lambda i: bm25_scores[i], reverse=True)
        lex_rank = {idx: rank + 1 for rank, idx in enumerate(lex_order)}

        fused = {}
        for idx in candidate_indices:
            score = 0.0
            if idx in sem_rank:
                score += 1.0 / (rrf_k + sem_rank[idx])
            if idx in lex_rank:
                score += 1.0 / (rrf_k + lex_rank[idx])
            fused[idx] = score

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self.chunks[idx].as_result(rank=r, score=score) for r, (idx, score) in enumerate(ranked, start=1)]

    def search_diverse(self, query, top_k=6, pool_size=20, app=None, lambda_relevance=0.7):
        """MMR re-ranking: widen the candidate pool, then greedily pick chunks
        that are relevant to the query but not redundant with chunks already
        picked (measured by embedding similarity + same-section penalty).
        This is the domain-agnostic replacement for retrieving 'by category'
        -- it doesn't need to know what a rule/table/error-type IS, it just
        avoids returning five near-duplicate chunks from the same section."""
        if not self.chunks:
            return []

        candidate_indices = self._candidate_indices(app)
        pool = self.search(query, top_k=min(pool_size, len(candidate_indices)), app=app)
        if not pool:
            return []

        pool_texts = [self.chunks[r["chunk_id"]].text for r in pool]
        pool_vectors = self.embedding_model.encode(pool_texts, normalize_embeddings=True).astype("float32")
        pool_scores = np.array([r["score"] for r in pool], dtype="float32")
        if pool_scores.max() > 0:
            pool_scores = pool_scores / pool_scores.max()

        selected = []
        selected_vec_idxs = []
        remaining = list(range(len(pool)))

        while remaining and len(selected) < top_k:
            best_idx, best_value = None, -1e9
            for i in remaining:
                relevance = pool_scores[i]
                if selected_vec_idxs:
                    redundancy = max(float(np.dot(pool_vectors[i], pool_vectors[j])) for j in selected_vec_idxs)
                    same_section_penalty = 0.15 if any(
                        pool[i]["section"] == pool[j]["section"] and pool[i]["source"] == pool[j]["source"]
                        for j in selected_vec_idxs
                    ) else 0.0
                else:
                    redundancy = 0.0
                    same_section_penalty = 0.0
                value = lambda_relevance * relevance - (1 - lambda_relevance) * redundancy - same_section_penalty
                if value > best_value:
                    best_value = value
                    best_idx = i
            selected.append(pool[best_idx])
            selected_vec_idxs.append(best_idx)
            remaining.remove(best_idx)

        for rank, result in enumerate(selected, start=1):
            result["rank"] = rank
        return selected


def build_hybrid_index(chunks, embedding_model, faiss_index):
    return HybridIndex(chunks, embedding_model, faiss_index)


def classify_question(query):
    lowered = query.lower()
    if any(term in lowered for term in DIAGNOSTIC_TERMS):
        return "diagnostic"
    return "lookup"


def _needs_wide_retrieval(query):
    """Structural signals that a question benefits from the wider MMR pool,
    independent of what app or topic it's about:
      - diagnostic/troubleshoot vocabulary
      - negation words IN the question
      - the question is a yes/no question at all (starts with an auxiliary
        verb) -- these are at risk of being answered by an oppositely-
        phrased document sentence even when the question itself has no
        negation word (e.g. "does X do Y?" answered by "Y is not done").
    """
    lowered = query.lower().strip()
    if any(term in lowered for term in DIAGNOSTIC_TERMS):
        return True
    if any(term in lowered for term in NEGATION_TERMS):
        return True
    first_word = re.match(r"[a-z']+", lowered)
    if first_word and first_word.group(0) in YES_NO_QUESTION_STARTERS:
        return True
    return False


def retrieve_for_question(query, hybrid_index, top_k=5, app=None, expand_query=True):
    """expand_query=True (default): also retrieve using LLM-generated
    rephrasings of the question and merge results (see query_expansion.py).
    Fixes cases where the user's wording diverges from the document's
    wording -- e.g. "two supplier numbers" vs. a doc's "Multiple Supplier
    Numbers" -- which plain hybrid search can miss entirely regardless of
    app or topic. Costs one extra LLM call per question; set to False to
    skip it if latency/cost matters more than paraphrase robustness."""
    effective_top_k = max(top_k, 6)
    wide = _needs_wide_retrieval(query)
    search_fn = hybrid_index.search_diverse if wide else hybrid_index.search
    search_kwargs = {"app": app, "pool_size": 20} if wide else {"app": app}

    if expand_query:
        return multi_query_search(query, hybrid_index, search_fn, top_k=effective_top_k, **search_kwargs)

    return search_fn(query, top_k=effective_top_k, **search_kwargs)