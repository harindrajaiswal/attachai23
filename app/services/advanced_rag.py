"""Advanced tenant-isolated Multilingual & Multi-Query RAG pipeline.

Implements:
1. Multilingual Query Understanding (Hindi Devanagari, Hinglish, Urdu, English)
2. Context-Aware Reference Resolution (resolves 'it', 'its', 'iska', 'iske', 'second option', etc.)
3. Multi-Query Generation (exact, synonym, pricing/benefit/package variations)
4. Parallel Hybrid Retrieval (Dense Vector + Lexical + Semantic Synonym + Phrase Matching)
5. Reciprocal Rank Fusion (RRF) & Chunk Deduplication
6. Cross-Feature Reranking & Genuine Evidence Assessment (`SUPPORTED`, `PARTIAL`, `UNKNOWN`)
7. Secondary Expanded Retrieval Pass for weak initial matches (Scenario B)
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from app.domain.enums import SupportLevel
from app.providers.embeddings import BaseEmbeddingProvider, SYNONYM_MAP
from app.providers.llm import BaseLlmProvider, STOPWORDS
from app.providers.vector import BaseVectorProvider
from app.repositories.base import BaseRepository

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

# Comprehensive Hindi (Devanagari), Hinglish, and Urdu to English concept dictionary
MULTILINGUAL_CONCEPT_MAP: Dict[str, List[str]] = {
    # Hindi Devanagari terms
    "सदस्यता": ["membership", "plan", "subscription"],
    "मेम्बरशिप": ["membership", "plan", "tier"],
    "मेंबरशिप": ["membership", "plan", "tier"],
    "प्लान": ["plan", "membership", "package", "tier"],
    "प्लान्स": ["plans", "memberships", "packages", "tiers"],
    "योजना": ["plan", "package", "membership"],
    "पैकेज": ["package", "plan", "membership", "tier"],
    "कीमत": ["price", "pricing", "cost", "fee", "month"],
    "प्राइस": ["price", "pricing", "cost", "fee"],
    "शुल्क": ["fee", "price", "cost", "charges"],
    "चार्ज": ["charges", "price", "cost", "fee"],
    "चार्जेस": ["charges", "prices", "cost", "fees"],
    "फीस": ["fees", "price", "cost", "monthly"],
    "खर्च": ["cost", "price", "fee"],
    "किराया": ["price", "cost", "rental", "fee"],
    "समय": ["hours", "schedule", "open", "time"],
    "टाइमिंग": ["hours", "schedule", "open", "time"],
    "घंटे": ["hours", "schedule", "open"],
    "खुलने": ["opening", "open", "hours", "schedule"],
    "बंद": ["closing", "close", "hours"],
    "रद्द": ["cancel", "cancellation", "policy", "notice"],
    "कैंसिल": ["cancel", "cancellation", "policy"],
    "रिफंड": ["refund", "cancel", "policy"],
    "प्रीमियम": ["premium", "peak", "plus", "all-access", "membership"],
    "बेसिक": ["basic", "core", "standard", "membership"],
    "कोर": ["core", "basic", "membership"],
    "पीक": ["peak", "premium", "membership"],
    "वार्षिक": ["annual", "yearly", "year", "month", "membership", "freeze"],
    "सालाना": ["annual", "yearly", "year", "membership"],
    "मासिक": ["monthly", "month", "per", "price"],
    "महीने": ["month", "monthly", "per"],
    "सुविधाएं": ["benefits", "features", "includes", "access", "unlimited"],
    "सुविधा": ["benefit", "feature", "includes", "access"],
    "लाभ": ["benefits", "features", "includes", "access", "perks"],
    "फायदे": ["benefits", "features", "includes", "access"],
    "संपर्क": ["contact", "email", "phone", "support", "call"],
    "फोन": ["phone", "call", "contact", "hotline"],
    "ईमेल": ["email", "support", "contact"],
    "जिम": ["gym", "club", "fitness", "membership", "training"],
    "क्लब": ["club", "gym", "membership"],
    "सेवाएं": ["services", "plans", "offerings", "packages"],
    "सेवा": ["service", "plan", "offering"],
    "विकल्प": ["option", "options", "plan", "tier"],
    "ट्रेनिंग": ["training", "personal", "classes", "hiit"],
    "सौना": ["sauna", "recovery", "steam"],
    "डेंटल": ["dental", "clinic", "consultation", "teeth", "whitening"],
    "दांत": ["teeth", "dental", "whitening", "consultation"],
    "परामर्श": ["consultation", "initial", "appointment"],
    "बैकअप": ["backup", "cloud", "snapshots", "encrypted"],
    "क्लाउड": ["cloud", "backup", "enterprise", "uptime", "sla"],
    "सुरक्षा": ["security", "compliance", "soc", "iso", "encrypted"],
    "गारंटी": ["sla", "uptime", "guarantees"],
    # Romanized Hinglish / Urdu terms
    "sadasyata": ["membership", "plan", "subscription"],
    "charges": ["price", "pricing", "cost", "fee", "month"],
    "charge": ["price", "cost", "fee"],
    "keemat": ["price", "pricing", "cost", "fee"],
    "qimat": ["price", "pricing", "cost", "fee"],
    "daam": ["price", "cost", "fee"],
    "kharcha": ["cost", "price", "fee"],
    "kitna": ["how", "much", "price", "cost", "fee"],
    "kitne": ["how", "much", "price", "cost", "fee"],
    "kitni": ["how", "much", "price", "cost", "fee"],
    "paisa": ["price", "cost", "fee"],
    "paise": ["price", "cost", "fee"],
    "fayde": ["benefits", "features", "includes", "access"],
    "fayda": ["benefit", "feature", "includes", "access"],
    "laabh": ["benefits", "features", "includes", "access"],
    "suvidha": ["benefits", "features", "includes", "access", "facilities"],
    "suvidhayein": ["benefits", "features", "includes", "access"],
    "timing": ["hours", "schedule", "opening", "open"],
    "timings": ["hours", "schedule", "opening", "open"],
    "waqt": ["hours", "time", "schedule", "open"],
    "samay": ["hours", "time", "schedule", "open"],
    "khulne": ["opening", "open", "hours", "schedule"],
    "mahine": ["month", "monthly", "per"],
    "mahina": ["month", "monthly"],
    "saal": ["year", "annual", "yearly", "months"],
    "salana": ["annual", "yearly", "year", "months"],
    "wapas": ["refund", "cancel", "cancellation"],
    "sampark": ["contact", "email", "phone", "support"],
    "seva": ["service", "services", "plan"],
    "sevayein": ["services", "plans", "packages"],
    "maloomat": ["information", "details", "plans", "services"],
    "tafsil": ["details", "information", "included"],
    "doosra": ["second", "plus", "peak", "premium", "tier"],
    "pehla": ["first", "core", "basic", "tier"],
}

# Extended English domain synonym map for Scenario B query expansion
EXTENDED_SYNONYMS: Dict[str, List[str]] = {
    "long": ["annual", "yearly", "months", "year", "membership", "freeze", "plan"],
    "term": ["annual", "yearly", "months", "year", "membership", "plan", "contract"],
    "service": ["services", "plan", "plans", "membership", "package", "features", "support", "pricing", "integration"],
    "services": ["service", "plans", "plan", "memberships", "packages", "features", "support", "pricing", "integration"],
    "package": ["plan", "plans", "membership", "tier", "box", "service", "pricing"],
    "packages": ["plans", "memberships", "tiers", "boxes", "services", "pricing"],
    "plan": ["membership", "plans", "package", "tier", "pricing", "month"],
    "plans": ["membership", "memberships", "packages", "tiers", "pricing", "month"],
    "membership": ["plan", "plans", "package", "tier", "member", "monthly", "cost", "price"],
    "memberships": ["plans", "packages", "tiers", "membership", "monthly", "cost", "price"],
    "price": ["cost", "pricing", "fee", "fees", "charges", "month", "per", "rate"],
    "pricing": ["price", "cost", "fee", "fees", "charges", "month", "per", "plans"],
    "cost": ["price", "pricing", "fee", "fees", "charges", "month", "per"],
    "costs": ["price", "pricing", "fee", "fees", "charges", "month", "per"],
    "charges": ["price", "pricing", "cost", "fee", "fees", "month", "per"],
    "fee": ["price", "cost", "pricing", "charges", "month"],
    "fees": ["price", "cost", "pricing", "charges", "month"],
    "benefit": ["benefits", "includes", "included", "access", "features", "unlimited"],
    "benefits": ["includes", "included", "access", "features", "unlimited", "classes", "sauna", "guest", "passes"],
    "feature": ["features", "benefits", "includes", "included", "access"],
    "features": ["benefits", "includes", "included", "access", "unlimited"],
    "include": ["includes", "included", "benefits", "access", "features"],
    "included": ["includes", "benefits", "access", "features"],
    "hours": ["open", "opening", "schedule", "monday", "friday", "saturday", "sunday", "am", "pm", "24/7"],
    "opening": ["open", "hours", "schedule", "monday", "friday", "am", "pm"],
    "schedule": ["hours", "open", "opening", "monday", "friday", "classes"],
    "cancel": ["cancellation", "freeze", "notice", "billing", "days", "policy", "refund"],
    "cancellation": ["cancel", "freeze", "notice", "billing", "days", "policy"],
    "refund": ["cancel", "cancellation", "policy", "billing"],
    "contact": ["email", "phone", "call", "support", "hotline", "reach"],
    "annual": ["year", "yearly", "months", "month", "membership", "plan", "freeze"],
    "yearly": ["annual", "year", "months", "month", "membership", "plan"],
    "second": ["plus", "peak", "premium", "tier", "all-access", "whitening"],
    "first": ["core", "basic", "initial", "consultation"],
    "premium": ["peak", "plus", "all-access", "membership", "plan", "tier"],
    "basic": ["core", "standard", "membership", "plan"],
}

HINGLISH_STOPWORDS = {
    "mujhe", "mujhko", "aapke", "aapka", "aapki", "aap", "tumhare", "hamare",
    "bataiye", "batao", "bataye", "batana", "kya", "hai", "hain", "ho",
    "iska", "iske", "iski", "uska", "uske", "unke", "isme", "usme",
    "baare", "bare", "mein", "me", "ke", "ki", "ka", "ko", "se", "aur",
    "bhi", "toh", "ye", "yeh", "wo", "woh", "ek", "par", "pe", "liye",
}


def translate_multilingual_tokens_to_english(text: str) -> List[str]:
    """Translate Hindi (Devanagari), Hinglish, and Urdu tokens into English retrieval concepts."""
    if not text:
        return []
    mapped: List[str] = []
    # Match both Devanagari/Urdu words and Latin words
    raw_words = re.findall(r"[\w\u0900-\u097F\u0600-\u06FF]+", text.lower())
    for w in raw_words:
        if w in MULTILINGUAL_CONCEPT_MAP:
            for eng in MULTILINGUAL_CONCEPT_MAP[w]:
                if eng not in mapped:
                    mapped.append(eng)
        elif re.match(r"^[a-z0-9]{2,}$", w) and w not in STOPWORDS and w not in HINGLISH_STOPWORDS:
            if w not in mapped:
                mapped.append(w)
    return mapped


def extract_important_terms(query: str) -> List[str]:
    """Extract content-bearing English & translated query terms (excluding stopwords)."""
    translated = translate_multilingual_tokens_to_english(query)
    raw_tokens = TOKEN_RE.findall((query or "").lower())
    terms: List[str] = []
    for t in translated + raw_tokens:
        if t not in STOPWORDS and t not in HINGLISH_STOPWORDS and len(t) >= 2 and t not in terms:
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
        if t in EXTENDED_SYNONYMS:
            expanded.update(EXTENDED_SYNONYMS[t])
        if len(t) > 4 and t.endswith("s"):
            expanded.add(t[:-1])
        elif len(t) > 3:
            expanded.add(t + "s")
    return expanded


def resolve_context_and_normalize_query(
    question: str,
    conversation_history: List[Dict[str, Any]],
    memory_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve conversational references ('it', 'its', 'iska', 'iske', 'second option', etc.)
    and normalize multilingual Hindi/Hinglish/Urdu queries into English retrieval concepts.
    """
    meta = memory_metadata or {}
    eng_concepts = translate_multilingual_tokens_to_english(question)
    base_english = " ".join(eng_concepts) if eng_concepts else question

    lower_q = question.lower()
    raw_word_count = len(re.findall(r"[\w\u0900-\u097F\u0600-\u06FF]+", lower_q))
    has_reference = bool(
        re.search(
            r"\b(?:it|its|they|them|that|those|this|iska|iske|iski|uska|uske|unke|isme|usme|"
            r"second\s+option|first\s+option|annual\s+option|monthly\s+option|what\s+about|how\s+about)\b|"
            r"(?:इसका|इसके|इसकी|उसका|उसके|उनके)",
            lower_q,
        )
    ) or raw_word_count <= 4

    context_terms: List[str] = []
    if has_reference and (conversation_history or meta):
        # 1. Pull current_topic and entities from persistent conversation memory
        topic = str(meta.get("current_topic") or "").strip()
        if topic:
            context_terms.extend(extract_important_terms(topic))
        for ent in reversed(list(meta.get("entities") or [])[-3:]):
            for et in extract_important_terms(str(ent)):
                if et not in context_terms:
                    context_terms.append(et)

        # 2. Pull salient terms from the most recent user & assistant turns
        for turn in reversed(conversation_history[-4:]):
            turn_terms = extract_important_terms(str(turn.get("content") or ""))
            for tt in turn_terms[:6]:
                if tt not in context_terms:
                    context_terms.append(tt)
            if len(context_terms) >= 8:
                break

        # Handle ordinal references like "second option" / "doosra option"
        if "second" in lower_q or "doosra" in lower_q:
            for extra in ("plus", "peak", "premium", "tier", "whitening"):
                if extra not in context_terms:
                    context_terms.append(extra)
        elif "first" in lower_q or "pehla" in lower_q:
            for extra in ("core", "basic", "initial", "consultation"):
                if extra not in context_terms:
                    context_terms.append(extra)

    if context_terms:
        combined = f"{base_english} {' '.join(context_terms[:6])}".strip()
        return combined
    return base_english.strip() or question.strip()


def generate_multi_queries(
    original_question: str,
    normalized_query: str,
    important_terms: List[str],
    expanded_terms: Set[str],
    memory_metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate 3 to 5 diverse retrieval sub-queries for multi-query RAG."""
    queries: List[str] = []

    def add_q(q_str: str) -> None:
        cleaned = " ".join(q_str.split()).strip()
        if cleaned and cleaned.lower() not in {x.lower() for x in queries}:
            queries.append(cleaned)

    # Query 1: Normalized English query with resolved context
    add_q(normalized_query)

    # Query 2: Core important terms
    if important_terms:
        add_q(" ".join(important_terms[:8]))

    # Query 3: Domain aspect expansion (pricing / plans / benefits / hours)
    term_set = {t.lower() for t in important_terms}
    if term_set & {"price", "pricing", "cost", "charges", "fee", "fees", "month", "monthly", "annual"}:
        topic_words = [t for t in important_terms if t not in {"price", "pricing", "cost", "charges", "fee", "fees", "how", "much"}]
        prefix = " ".join(topic_words[:4]) or "membership service"
        add_q(f"{prefix} pricing cost per month monthly annual fees")
    if term_set & {"membership", "memberships", "plan", "plans", "package", "packages", "tier", "option"}:
        add_q("available membership plans packages tiers core basic peak plus premium pricing")
    if term_set & {"benefit", "benefits", "feature", "features", "includes", "included", "access"}:
        topic = (memory_metadata or {}).get("current_topic") or "membership plan"
        add_q(f"{topic} benefits features included unlimited access classes sauna")
    if term_set & {"long", "term", "annual", "yearly", "year"}:
        add_q("annual yearly membership plans monthly pricing freeze policy")

    # Query 4: Expanded synonym query
    if expanded_terms:
        add_q(" ".join(sorted(expanded_terms)[:12]))

    # Query 5: Original question if distinct
    add_q(original_question)

    return queries[:5]


class AdvancedRagService:
    """Tenant-scoped hybrid multi-query retrieval, RRF fusion, reranking, and evidence evaluation."""

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
        queries: List[str],
        expanded_terms: Set[str],
        top_k: int = 10,
        page_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """Paginate through active tenant chunks to compute multi-query lexical & phrase scores."""
        if not expanded_terms and not queries:
            return []

        lower_queries = [q.lower().strip() for q in queries if q and len(q.strip()) >= 4]
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

                term_coverage = len(matched_terms) / max(1, min(len(expanded_terms), 8))
                phrase_bonus = 0.0
                for lq in lower_queries:
                    if len(lq) >= 5 and lq in lower_content:
                        phrase_bonus = 0.25
                        break
                    # Check 2-word bigrams from query
                    q_words = TOKEN_RE.findall(lq)
                    for i in range(len(q_words) - 1):
                        bigram = f"{q_words[i]} {q_words[i + 1]}"
                        if len(bigram) >= 6 and bigram in lower_content:
                            phrase_bonus = max(phrase_bonus, 0.18)

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
        dense_hit_lists: List[List[Dict[str, Any]]],
        lexical_hits: List[Dict[str, Any]],
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Combine multiple dense vector result lists and lexical hits via Reciprocal Rank Fusion (RRF),
        deduplicating overlapping chunks by chunk ID.
        """
        by_id: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        for dense_hits in dense_hit_lists:
            for rank, hit in enumerate(dense_hits, start=1):
                cid = hit["id"]
                sim = float(hit.get("similarity", 0.0))
                if cid not in by_id:
                    by_id[cid] = {
                        **hit,
                        "similarity": sim,
                        "lexical_score": float(hit.get("lexical_score", 0.0)),
                    }
                else:
                    by_id[cid]["similarity"] = max(float(by_id[cid].get("similarity", 0.0)), sim)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        for rank, hit in enumerate(lexical_hits, start=1):
            cid = hit["id"]
            lex = float(hit.get("lexical_score", 0.0))
            if cid not in by_id:
                by_id[cid] = {
                    **hit,
                    "similarity": float(hit.get("similarity", 0.0)),
                    "lexical_score": lex,
                }
            else:
                by_id[cid]["lexical_score"] = max(float(by_id[cid].get("lexical_score", 0.0)), lex)
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
        """Rerank fused candidates and evaluate factual support coverage across English & Multilingual queries."""
        if not candidates:
            return [], SupportLevel.UNKNOWN.value, 0.0

        # Use translated/normalized terms from original question; if follow-up, combine with reformulated query
        important_terms = extract_important_terms(original_question)
        reformulated_terms = extract_important_terms(reformulated_query)
        if len(important_terms) <= 2 and reformulated_terms:
            for rt in reformulated_terms:
                if rt not in important_terms:
                    important_terms.append(rt)
        elif not important_terms:
            important_terms = reformulated_terms

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

            combined_score = (concept_coverage * 0.55) + (dense_sim * 0.30) + (lex_score * 0.15)
            cand_copy = dict(cand)
            cand_copy["concept_coverage"] = round(concept_coverage, 4)
            cand_copy["combined_score"] = round(combined_score, 4)
            reranked.append(cand_copy)

        reranked.sort(key=lambda x: x["combined_score"], reverse=True)
        top_chunks = reranked[: max(1, top_k)]

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

        # Strict Evidence Support Decision (weak similarity alone NEVER yields SUPPORTED)
        if aggregate_coverage >= 0.60 and (best_score >= 0.38 or best_sim >= max(min_similarity, 0.32)):
            support_level = SupportLevel.SUPPORTED.value
        elif aggregate_coverage >= 0.34 and best_score >= 0.24:
            support_level = SupportLevel.PARTIAL.value
        else:
            support_level = SupportLevel.UNKNOWN.value

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
        memory_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run full tenant-scoped multilingual, multi-query hybrid RAG retrieval and evidence assessment."""
        history = conversation_history or []
        meta = memory_metadata or {}

        # 1. Context-Aware & Multilingual Query Normalization
        context_normalized = resolve_context_and_normalize_query(question, history, meta)
        try:
            llm_reformulated = await self.llm.reformulate_query(question, history)
        except Exception:
            llm_reformulated = question

        reformulated = context_normalized if context_normalized != question.strip() else llm_reformulated
        terms = extract_important_terms(context_normalized)
        expanded = expand_query_terms(terms)

        # 2. Multi-Query Generation
        multi_queries = generate_multi_queries(
            original_question=question,
            normalized_query=context_normalized,
            important_terms=terms,
            expanded_terms=expanded,
            memory_metadata=meta,
        )

        # 3. Parallel Dense Vector Search across top generated queries + Paginated Lexical Search
        primary_vector_queries = multi_queries[:3]

        async def _dense_for_query(q_text: str) -> List[Dict[str, Any]]:
            q_emb = await self.embeddings.embed_query(q_text)
            return await self.vector.search(
                tenant_id=tenant_id,
                query_embedding=q_emb,
                top_k=max(top_k * 2, 8),
                min_similarity=max(0.08, min_similarity * 0.5),
            )

        dense_tasks = [_dense_for_query(q) for q in primary_vector_queries]
        lexical_task = self._lexical_search_paginated(
            tenant_id=tenant_id,
            queries=multi_queries,
            expanded_terms=expanded,
            top_k=max(top_k * 2, 8),
        )

        results = await asyncio.gather(*dense_tasks, lexical_task)
        dense_hit_lists = list(results[:-1])
        lexical_hits = results[-1]

        # 4. Reciprocal Rank Fusion (RRF) & Deduplication
        fused = self._reciprocal_rank_fusion(dense_hit_lists, lexical_hits)

        # 5. Reranking & Evidence Assessment
        selected_chunks, support_level, coverage = self._rerank_and_assess_evidence(
            original_question=context_normalized,
            reformulated_query=reformulated,
            candidates=fused,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        expanded_search_used = False
        best_initial_sim = max(
            (float(c.get("similarity", 0.0)) for c in selected_chunks),
            default=0.0,
        )
        # 6. Scenario B: If initial retrieval is weak/UNKNOWN or below strict similarity threshold, run expanded secondary search
        if (support_level == SupportLevel.UNKNOWN.value or best_initial_sim < min_similarity) and terms:
            expanded_search_used = True
            broad_terms = expand_query_terms(list(expanded))
            broad_lexical = await self._lexical_search_paginated(
                tenant_id=tenant_id,
                queries=multi_queries + [" ".join(sorted(broad_terms)[:15])],
                expanded_terms=broad_terms,
                top_k=max(top_k * 2, 10),
            )
            if broad_lexical:
                fused_retry = self._reciprocal_rank_fusion(dense_hit_lists, broad_lexical)
                selected_chunks, support_level, coverage = self._rerank_and_assess_evidence(
                    original_question=context_normalized,
                    reformulated_query=reformulated,
                    candidates=fused_retry,
                    top_k=top_k,
                    min_similarity=max(0.15, min_similarity * 0.5),
                )

        return {
            "question": question,
            "reformulated_query": context_normalized,
            "multi_queries": multi_queries,
            "expanded_search_used": expanded_search_used,
            "support_level": support_level,
            "coverage_score": coverage,
            "chunks": selected_chunks,
        }
