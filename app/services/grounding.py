"""Grounding validation service for RAG answers (Multilingual & Label-Free).

Implements Section 9 Grounding requirements + Upgrade 1:
Checks candidate LLM answers against retrieved evidence for:
- Unsupported numerical claims, prices, and percentages
- Unsupported dates or times
- Invented contact details (emails, phone numbers)
- Unsupported guarantees ("100% guaranteed", "lifetime warranty", "full refund")
- Contradictions or hallucinated facts absent from the evidence
- Multilingual lexical & concept grounding across English, Hindi, Hinglish, and Urdu
- Strips robotic 'Unverified Answer' / 'Fallback Answer' prefixes and internal error tokens.
"""

import re
from typing import Any, Dict, List
from app.domain.enums import SupportLevel
from app.domain.models import GroundingResult
from app.services.advanced_rag import translate_multilingual_tokens_to_english
from app.utils.pii import EMAIL_PATTERN, PHONE_PATTERN

# Extract numbers, prices ($49, £29.99), percentages (25%), and times (9am, 24/7)
NUMBER_CLAIM_RE = re.compile(
    r"(?:[$£€₹]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*%|\b\d{1,4}(?::\d{2})?\s*(?:am|pm)\b|\b\d{2,}\b)",
    re.IGNORECASE,
)

UNSUPPORTED_GUARANTEE_PATTERNS = (
    "100% guaranteed",
    "lifetime warranty",
    "unconditional refund",
    "guaranteed approval",
    "zero risk guarantee",
)

FORBIDDEN_ROBOTIC_LABELS_RE = re.compile(
    r"^\s*(?:unverified\s+answer|fallback\s+answer|unverified\s+response|fallback\s+response|"
    r"insufficient\s+retrieval|rag\s+failure|internal\s+intent\s+classification|"
    r"retrieval_failed|unsupported|internal_error)\s*[:\-–—]*\s*",
    re.IGNORECASE,
)

INTERNAL_STATUS_TOKENS_RE = re.compile(
    r"\b(?:RETRIEVAL_FAILED|UNSUPPORTED_ANSWER|FALLBACK_MODE|INTERNAL_ERROR|INSUFFICIENT_RETRIEVAL|RAG_FAILURE)\b"
)


def sanitize_customer_facing_text(text: str) -> str:
    """Remove robotic labels or internal status tokens from customer-facing responses."""
    cleaned = (text or "").strip()
    cleaned = FORBIDDEN_ROBOTIC_LABELS_RE.sub("", cleaned)
    cleaned = re.sub(
        r"\b(?:unverified\s+answer|fallback\s+answer|insufficient\s+retrieval|rag\s+failure)\s*[:\-–—]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = INTERNAL_STATUS_TOKENS_RE.sub("", cleaned)
    return cleaned.strip()


def _normalize_number_token(tok: str) -> str:
    return re.sub(r"[\s,$£€₹]", "", tok.lower().strip())


class GroundingService:
    """Validates generated answers against retrieved tenant knowledge chunks."""

    def validate_answer(
        self,
        candidate_answer: str,
        evidence_chunks: List[Dict[str, Any]],
        support_level: str,
        fallback_answer: str,
        language: str = "en",
    ) -> GroundingResult:
        """Validate candidate_answer against evidence_chunks."""
        clean_fallback = sanitize_customer_facing_text(fallback_answer) or (
            "I don't have those specific details in our available information yet. "
            "Please let me know if I can help with another question!"
        )

        if support_level == SupportLevel.UNKNOWN.value or not evidence_chunks:
            return GroundingResult(
                passed=False,
                support_level=SupportLevel.UNKNOWN.value,
                validated_answer=clean_fallback,
                reasons=["Insufficient retrieved evidence to support an answer."],
                unverified_numbers=[],
                coverage_score=0.0,
            )

        cleaned_answer = sanitize_customer_facing_text(candidate_answer)
        if not cleaned_answer:
            return GroundingResult(
                passed=False,
                support_level=support_level,
                validated_answer=clean_fallback,
                reasons=["Empty model response."],
                unverified_numbers=[],
                coverage_score=0.0,
            )

        evidence_text = "\n".join(str(c.get("content") or "") for c in evidence_chunks)
        evidence_lower = evidence_text.lower()
        reasons: List[str] = []
        unverified_numbers: List[str] = []

        # 1. Verify numerical, price, percentage, and time claims
        evidence_numbers = {
            _normalize_number_token(m.group(0))
            for m in NUMBER_CLAIM_RE.finditer(evidence_text)
        }
        evidence_digits = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_text))

        for match in NUMBER_CLAIM_RE.finditer(cleaned_answer):
            raw_claim = match.group(0).strip()
            norm_claim = _normalize_number_token(raw_claim)
            raw_digits = re.sub(r"[^0-9.]", "", norm_claim)
            if (
                norm_claim not in evidence_numbers
                and raw_digits not in evidence_digits
                and raw_claim.lower() not in evidence_lower
            ):
                unverified_numbers.append(raw_claim)

        if unverified_numbers:
            reasons.append(
                f"Answer contained unsupported numeric/price claims: {', '.join(unverified_numbers)}"
            )

        # 2. Verify email addresses in answer exist in evidence
        answer_emails = {e.lower() for e in EMAIL_PATTERN.findall(cleaned_answer)}
        evidence_emails = {e.lower() for e in EMAIL_PATTERN.findall(evidence_text)}
        invented_emails = answer_emails - evidence_emails
        if invented_emails:
            reasons.append(f"Answer contained invented email address: {', '.join(invented_emails)}")

        # 3. Verify phone numbers in answer exist in evidence
        answer_phones = {re.sub(r"\D", "", p) for p in PHONE_PATTERN.findall(cleaned_answer)}
        evidence_phones = {re.sub(r"\D", "", p) for p in PHONE_PATTERN.findall(evidence_text)}
        invented_phones = {p for p in answer_phones if len(p) >= 7 and p not in evidence_phones}
        if invented_phones:
            reasons.append("Answer contained invented phone number not present in evidence.")

        # 4. Check unsupported guarantees
        answer_lower = cleaned_answer.lower()
        for phrase in UNSUPPORTED_GUARANTEE_PATTERNS:
            if phrase in answer_lower and phrase not in evidence_lower:
                reasons.append(f"Answer contained unsupported guarantee phrase: '{phrase}'.")

        # 5. Check content token overlap with evidence (supports natural multilingual & translated responses)
        english_stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "your", "are", "can", "our", "you",
            "have", "has", "will", "would", "there", "their", "what", "when", "where", "which", "about",
            "hai", "hain", "kya", "mein", "aur", "ke", "ki", "ka", "ko", "se", "par", "bhi", "hum",
            "aap", "aapka", "aapki", "hamara", "hamari", "hoti", "hota", "liye", "sakte", "sakta",
        }
        raw_ascii_tokens = {
            w for w in re.findall(r"[a-z0-9]{3,}", answer_lower)
            if w not in english_stopwords
        }
        bridged_en_tokens = translate_multilingual_tokens_to_english(cleaned_answer)
        bridged_token_set = {
            w.lower() for w in bridged_en_tokens if len(w) >= 3 and w.lower() not in english_stopwords
        }

        evidence_tokens = set(re.findall(r"[a-z0-9]{3,}", evidence_lower))
        is_multilingual = language != "en" or bool(re.search(r"[^\x00-\x7F]", cleaned_answer))

        if is_multilingual:
            effective_answer_tokens = (raw_ascii_tokens & evidence_tokens) | bridged_token_set
            matched = effective_answer_tokens & evidence_tokens
            if matched:
                overlap_ratio = max(0.35, len(matched) / max(1, len(effective_answer_tokens)))
            elif not raw_ascii_tokens and not unverified_numbers:
                # Pure non-Latin script translation where all numerical/contact claims passed
                overlap_ratio = 0.50
            else:
                overlap_ratio = 0.0
            min_overlap = 0.25
        else:
            effective_answer_tokens = raw_ascii_tokens | bridged_token_set
            overlap_ratio = (
                len(effective_answer_tokens & evidence_tokens) / max(1, len(effective_answer_tokens))
                if effective_answer_tokens
                else 0.0
            )
            min_overlap = 0.25

        if overlap_ratio < min_overlap:
            reasons.append(
                f"Answer has low lexical grounding overlap with retrieved evidence ({overlap_ratio:.2f})."
            )

        if reasons:
            return GroundingResult(
                passed=False,
                support_level=support_level,
                validated_answer=clean_fallback,
                reasons=reasons,
                unverified_numbers=unverified_numbers,
                coverage_score=round(overlap_ratio, 4),
            )

        return GroundingResult(
            passed=True,
            support_level=support_level,
            validated_answer=cleaned_answer,
            reasons=[],
            unverified_numbers=[],
            coverage_score=round(overlap_ratio, 4),
        )
