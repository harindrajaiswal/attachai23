"""Grounding validation service for RAG answers.

Implements Section 9 Grounding requirements:
Checks candidate LLM answers against retrieved evidence for:
- Unsupported numerical claims, prices, and percentages
- Unsupported dates or times
- Invented contact details (emails, phone numbers)
- Unsupported guarantees ("100% guaranteed", "lifetime warranty", "full refund")
- Contradictions or hallucinated facts absent from the evidence.

When evidence is insufficient or grounding fails, returns the tenant's configured
fallback response instead of emitting unverified claims.
"""

import re
from typing import Any, Dict, List
from app.domain.enums import SupportLevel
from app.domain.models import GroundingResult
from app.utils.pii import EMAIL_PATTERN, PHONE_PATTERN

# Extract numbers, prices ($49, £29.99), percentages (25%), and times (9am, 24/7)
NUMBER_CLAIM_RE = re.compile(
    r"(?:[$£€]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*%|\b\d{1,4}(?::\d{2})?\s*(?:am|pm)\b|\b\d{2,}\b)",
    re.IGNORECASE,
)

UNSUPPORTED_GUARANTEE_PATTERNS = (
    "100% guaranteed",
    "lifetime warranty",
    "unconditional refund",
    "guaranteed approval",
    "zero risk guarantee",
)


def _normalize_number_token(tok: str) -> str:
    return re.sub(r"[\s,$£€]", "", tok.lower().strip())


class GroundingService:
    """Validates generated answers against retrieved tenant knowledge chunks."""

    def validate_answer(
        self,
        candidate_answer: str,
        evidence_chunks: List[Dict[str, Any]],
        support_level: str,
        fallback_answer: str,
    ) -> GroundingResult:
        """Validate candidate_answer against evidence_chunks."""
        if support_level == SupportLevel.UNKNOWN.value or not evidence_chunks:
            return GroundingResult(
                passed=False,
                support_level=SupportLevel.UNKNOWN.value,
                validated_answer=fallback_answer,
                reasons=["Insufficient retrieved evidence to support an answer."],
                unverified_numbers=[],
                coverage_score=0.0,
            )

        cleaned_answer = (candidate_answer or "").strip()
        if not cleaned_answer:
            return GroundingResult(
                passed=False,
                support_level=support_level,
                validated_answer=fallback_answer,
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
        # Also include raw digit sequences in evidence
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

        # 5. Check content token overlap with evidence
        answer_tokens = {
            w for w in re.findall(r"[a-z0-9]{3,}", answer_lower)
            if w not in ("the", "and", "for", "with", "that", "this", "from", "your", "are", "can", "our", "you")
        }
        evidence_tokens = set(re.findall(r"[a-z0-9]{3,}", evidence_lower))
        overlap_ratio = (
            len(answer_tokens & evidence_tokens) / max(1, len(answer_tokens))
            if answer_tokens
            else 0.0
        )
        if overlap_ratio < 0.35:
            reasons.append(
                f"Answer has low lexical grounding overlap with retrieved evidence ({overlap_ratio:.2f})."
            )

        if reasons:
            return GroundingResult(
                passed=False,
                support_level=support_level,
                validated_answer=fallback_answer,
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
