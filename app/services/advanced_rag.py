"""Advanced tenant-isolated Retrieval-Augmented Generation (RAG) pipeline.

Implements Section 9 & Fixes Defects #4 and #7:
1. Dense vector search (via VectorProvider)
2. Paginated lexical relevance & phrase matching
3. Query expansion (domain synonyms)
4. Bounded LLM query reformulation for follow-ups
5. Reciprocal Rank Fusion (RRF)
6. Cross-feature Reranking
7. Genuine Evidence Assessment (`SUPPORTED`, `PARTIAL`, `UNKNOWN`) that evaluates
   whether the retrieved evidence actually covers the important concepts in the question,
   rather than declaring SUPPORTED merely because two chunks pass a weak similarity threshold.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from app.domain.enums import SupportLevel
from app.providers.embeddings import BaseEmbeddingProvider, SYNONYM_MAP
from app.providers.llm import BaseLlmProvider, STOPWORDS
from app.providers.vector import BaseVectorProvider
from app.repositories.base import BaseRepository

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def extract_important_terms(query: str) -> List[str]:
    """Extract content-bearing query terms (excluding stopwords)."""
    raw_tokens = TOKEN_RE.findall((query or "").lower())
    terms: List[str] = []
    for t in raw_tokens:
        if t not in STOPWORDS and len(t) >= 2 and t not in terms:
            terms.append(t)
    return terms


def expand_query_terms(terms: List[str]) -> Set[str]:
    """Expand query terms with bidirectional domain synonyms and morphological stems."""
    expanded: Set[str] = set(terms)
    for t in terms:
        canonical = SYNONYM_MAP.get(t)
        if canonical:
            expanded.add(canonical)
            for k, v in SYNONYM_MAP.items():
                if v == canonical:
                    expanded.add(k)
        for k, v in SYNONYM_MAP.items():
            if v == t:
                expanded.add(k)
        if len(t) > 4 and t.endswith("s"):
            expanded.add(t[:-1])
        elif len(t) > 3:
            expanded.add(t + "s")
    return expanded


class AdvancedRagService:
    """Tenant-scoped hybrid retrieval, RRF fusion, reranking, and evidence evaluation."""

    def __init__(
        self,
        repository: BaseRepository,
        embedding_provider: BaseEmbeddingProvider,
        vector_provider: BaseVectorProvider,
        llm_provider: BaseLlmProvider,
        diagnostics_enabled: bool = False,
    ) -> None:
        self.repository = repository
        self.embeddings = embedding_provider
        self.vector = vector_provider
        self.llm = llm_provider
        self.diagnostics_enabled = diagnostics_enabled

    async def _lexical_search_paginated(
        self,
        tenant_id: str,
        query: str,
        expanded_terms: Set[str],
        top_k: int = 10,
        page_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """Paginate through active tenant chunks to compute lexical & phrase scores."""
        if not expanded_terms:
            return []

        lower_query = query.lower().strip()
        scored: List[Dict[str, Any]] = []
        offset = 0

        while True:
            batch = await self.repository.get_active_chunks_paginated(
                tenant_id=tenant_id,
                limit=page_size,
                offset=offset,
            )
            if not batch:
                break
            for item in batch:
                if item.get("tenant_id") != tenant_id or not item.get("is_active", True):
                    continue
                content = str(item.get("content") or "")
                lower_content = content.lower()
                chunk_tokens = set(TOKEN_RE.findall(lower_content))
                if not chunk_tokens:
                    continue

                matched_terms = expanded_terms & chunk_tokens
                if not matched_terms:
                    continue

                term_coverage = len(matched_terms) / max(1, len(expanded_terms))
                # Phrase match bonus
                phrase_bonus = 0.25 if (len(lower_query) >= 6 and lower_query in lower_content) else 0.0
                lexical_score = min(1.0, (term_coverage * 0.85) + phrase_bonus)
                if lexical_score > 0.05:
                    scored.append(
                        {
                            "id": item["id"],
                            "tenant_id": item["tenant_id"],
                            "source_id": item["source_id"],
                            "source_filename": item["source_filename"],
                            "source_version": item["source_version"],
                            "chunk_index": item["chunk_index"],
                            "content": content,
                            "lexical_score": round(lexical_score, 6),
                        }
                    )
            if len(batch) < page_size:
                break
            offset += page_size

        scored.sort(key=lambda x: x["lexical_score"], reverse=True)
        return scored[: max(1, top_k)]

    def _reciprocal_rank_fusion(
        self,
        dense_hits: List[Dict[str, Any]],
        lexical_hits: List[Dict[str, Any]],
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Combine dense vector hits and lexical hits via Reciprocal Rank Fusion (RRF)."""
        by_id: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["id"]
            by_id[cid] = {
                **hit,
                "similarity": float(hit.get("similarity", 0.0)),
                "lexical_score": float(hit.get("lexical_score", 0.0)),
            }
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        for rank, hit in enumerate(lexical_hits, start=1):
            cid = hit["id"]
            if cid not in by_id:
                by_id[cid] = {
                    **hit,
                    "similarity": float(hit.get("similarity", 0.0)),
                    "lexical_score": float(hit.get("lexical_score", 0.0)),
                }
            else:
                by_id[cid]["lexical_score"] = float(hit.get("lexical_score", 0.0))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        fused = []
        for cid, record in by_id.items():
            record["rrf_score"] = round(rrf_scores[cid], 6)
            fused.append(record)
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    def _rerank_and_assess_evidence(
        self,
        original_question: str,
        reformulated_query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
        min_similarity: float,
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """Rerank fused candidates and evaluate factual support coverage (Fixes Defect #7)."""
        if not candidates:
            return [], SupportLevel.UNKNOWN.value, 0.0

        important_terms = extract_important_terms(original_question)
        if not important_terms:
            important_terms = extract_important_terms(reformulated_query)

        # Group each important term with its valid synonyms
        concept_groups: List[Set[str]] = []
        for term in important_terms:
            grp = expand_query_terms([term])
            concept_groups.append(grp)

        reranked: List[Dict[str, Any]] = []
        for cand in candidates:
            content_lower = str(cand.get("content") or "").lower()
            chunk_tokens = set(TOKEN_RE.findall(content_lower))

            matched_concepts = 0
            for grp in concept_groups:
                if grp & chunk_tokens or any(g in content_lower for g in grp if len(g) >= 4):
                    matched_concepts += 1

            concept_coverage = matched_concepts / max(1, len(concept_groups))
            dense_sim = float(cand.get("similarity", 0.0))
            lex_score = float(cand.get("lexical_score", 0.0))

            # Rerank score weights concept coverage heavily to distinguish factual support from weak similarity
            combined_score = (concept_coverage * 0.55) + (dense_sim * 0.30) + (lex_score * 0.15)
            cand_copy = dict(cand)
            cand_copy["concept_coverage"] = round(concept_coverage, 4)
            cand_copy["combined_score"] = round(combined_score, 4)
            reranked.append(cand_copy)

        reranked.sort(key=lambda x: x["combined_score"], reverse=True)
        top_chunks = reranked[: max(1, top_k)]

        # Evaluate aggregate concept coverage across top retrieved chunks
        covered_across_top = 0
        combined_top_tokens: Set[str] = set()
        combined_top_text = " ".join(str(c.get("content") or "").lower() for c in top_chunks[:3])
        for c in top_chunks[:3]:
            combined_top_tokens.update(TOKEN_RE.findall(str(c.get("content") or "").lower()))

        for grp in concept_groups:
            if grp & combined_top_tokens or any(g in combined_top_text for g in grp if len(g) >= 4):
                covered_across_top += 1

        aggregate_coverage = covered_across_top / max(1, len(concept_groups))
        best_score = top_chunks[0]["combined_score"] if top_chunks else 0.0
        best_sim = top_chunks[0].get("similarity", 0.0) if top_chunks else 0.0

        # Strict Evidence Support Decision (Fixes Defect #7: weak similarity alone NEVER yields SUPPORTED)
        if aggregate_coverage >= 0.65 and (best_score >= 0.42 or best_sim >= max(min_similarity, 0.35)):
            support_level = SupportLevel.SUPPORTED.value
        elif aggregate_coverage >= 0.34 and best_score >= 0.25:
            support_level = SupportLevel.PARTIAL.value
        else:
            support_level = SupportLevel.UNKNOWN.value

        # Filter out noise chunks when support is UNKNOWN
        if support_level == SupportLevel.UNKNOWN.value:
            return [], SupportLevel.UNKNOWN.value, round(aggregate_coverage, 4)

        filtered = [c for c in top_chunks if c["combined_score"] >= 0.20]
        return filtered, support_level, round(aggregate_coverage, 4)

    async def retrieve_and_assess(
        self,
        tenant_id: str,
        question: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        min_similarity: float = 0.25,
    ) -> Dict[str, Any]:
        """Run full tenant-scoped hybrid RAG retrieval and evidence assessment."""
        history = conversation_history or []

        # 1. Bounded Query Reformulation
        reformulated = await self.llm.reformulate_query(question, history)
        terms = extract_important_terms(reformulated)
        expanded = expand_query_terms(terms)
        expanded_query_str = " ".join(sorted(expanded)) if expanded else reformulated

        # 2. Dense Vector Retrieval
        query_embedding = await self.embeddings.embed_query(
            f"{reformulated} {expanded_query_str}".strip()
        )
        dense_hits = await self.vector.search(
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            top_k=max(top_k * 2, 8),
            min_similarity=max(0.10, min_similarity * 0.6),
        )

        # 3. Paginated Lexical & Phrase Retrieval
        lexical_hits = await self._lexical_search_paginated(
            tenant_id=tenant_id,
            query=reformulated,
            expanded_terms=expanded,
            top_k=max(top_k * 2, 8),
        )

        # 4. Reciprocal Rank Fusion (RRF)
        fused = self._reciprocal_rank_fusion(dense_hits, lexical_hits)

        # 5. Reranking & Evidence Assessment
        selected_chunks, support_level, coverage = self._rerank_and_assess_evidence(
            original_question=question,
            reformulated_query=reformulated,
            candidates=fused,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        return {
            "question": question,
            "reformulated_query": reformulated,
            "support_level": support_level,
            "coverage_score": coverage,
            "chunks": selected_chunks,
        }
