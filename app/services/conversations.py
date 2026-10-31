"""Conversation memory, intent understanding, language detection, and feedback service.

Implements:
- Persistent conversation memory (short-term bounded history + long-term summaries + entity/topic metadata)
- Multilingual language detection (English, Hindi Devanagari, Hinglish, Urdu)
- Lightweight intent classification (GREETING, GENERAL_CONVERSATION, SUMMARY_REQUEST,
  FOLLOW_UP_QUESTION, AMBIGUOUS, OUT_OF_SCOPE, KNOWLEDGE_QUESTION)
- Pronoun & follow-up reference resolution using conversation memory
- Strict tenant & conversation ownership validation
- Assistant message feedback persistence (Like / Dislike / Reason) with duplicate prevention
"""

import re
from typing import Any, Dict, List, Optional
from app.core.exceptions import AuthorizationError, NotFoundError, ValidationError
from app.providers.llm import BaseLlmProvider
from app.repositories.base import BaseRepository

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ARABIC_URDU_RE = re.compile(r"[\u0600-\u06FF]")

HINGLISH_MARKERS = {
    "mujhe", "mujhko", "aapke", "aapka", "aapki", "aap", "tumhare", "hamare",
    "bataiye", "batao", "bataye", "batana", "kya", "hai", "hain", "ho",
    "iska", "iske", "iski", "uska", "uske", "unke", "isme", "usme",
    "baare", "bare", "mein", "me", "ke", "ki", "ka", "ko", "se", "aur",
    "kitna", "kitne", "kitni", "kaise", "kab", "kahan", "kyun", "acha",
    "namaste", "namaskar", "dhanyawad", "shukriya", "chahiye", "milega",
    "milta", "lagta", "paisa", "paise", "kharcha", "mahine", "saal", "salana",
    "suvidha", "fayde", "fayda",
}

URDU_ROMAN_MARKERS = {
    "assalamualaikum", "walaikum", "shukriya", "meherbani", "khuwahish", "maloomat",
}

GREETING_PATTERNS = re.compile(
    r"^\s*(?:hello|hi|hey|hiya|good\s+(?:morning|afternoon|evening|day)|"
    r"namaste|namaskar|pranam|assalamu\s*alaikum|salam|hola|bonjour|"
    r"नमस्ते|नमस्कार|प्रणाम|हैलो|हाय|सलाम)"
    r"(?:\s*[!,.?]*)?(?:\s+(?:how\s+are\s+you(?:\s+doing)?|kaise\s+ho|kaise\s+hain|kya\s+haal\s+hai| कैसे\s+हैं|कैसे\s+हो))?\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

HOW_ARE_YOU_PATTERNS = re.compile(
    r"^\s*(?:how\s+are\s+you(?:\s+doing|\s+today)?|how's\s+it\s+going|"
    r"kya\s+haal\s+hai|(?:aap\s+)?kaise\s+(?:hain|ho)(?:\s+aap)?|आप\s+कैसे\s+हैं|कैसे\s+हो)\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

GRATITUDE_OR_CLOSING_PATTERNS = re.compile(
    r"^\s*(?:thank\s+you(?:\s+so\s+much|\s+very\s+much)?|thanks(?:\s+a\s+lot)?|"
    r"thx|ty|shukriya|dhanyawad|dhanyavaad|bahut\s+dhanyawad|bahut\s+shukriya|"
    r"धन्यवाद|शुक्रिया|बहुत\s+धन्यवाद|"
    r"that(?:'s|\s+is)\s+(?:very\s+)?helpful|got\s+it|understood|makes\s+sense|"
    r"okay\s+thank\s+you|ok\s+thanks|great\s+thanks|awesome\s+thanks|perfect\s+thanks|"
    r"bye|goodbye|see\s+you|have\s+a\s+good\s+day)\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

REPEAT_OR_META_PATTERNS = re.compile(
    r"^\s*(?:can\s+you\s+explain\s+that\s+again|please\s+explain\s+that\s+again|"
    r"explain\s+that\s+again|repeat\s+that|can\s+you\s+repeat\s+that|"
    r"i\s+have\s+another\s+question|one\s+more\s+question|"
    r"what\s+can\s+you\s+help\s+(?:me\s+)?with|how\s+can\s+you\s+help(?:\s+me)?|"
    r"who\s+are\s+you|what\s+do\s+you\s+do|"
    r"aap\s+kya\s+kar\s+sakte\s+(?:ho|hain)|phir\s+se\s+batao|dobara\s+bataiye)\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

SUMMARY_INTENT_RE = re.compile(
    r"\b(?:summarize|summary|recap|sum\s+up)\b.*?(?:\bconversation\b|\bchat\b|\bdiscussion\b|\bmessages\b|\bso\s+far\b)?|"
    r"\bwhat\s+have\s+we\s+(?:discussed|talked\s+about)\b|"
    r"(?:बातचीत\s+का\s+सारांश|हमारी\s+बातचीत)",
    re.IGNORECASE,
)

FOLLOW_UP_PRONOUN_RE = re.compile(
    r"\b(?:it|its|they|them|their|that|those|this|these|"
    r"iska|iske|iski|uska|uske|uski|unke|unki|isme|usme|is|us|"
    r"second\s+option|first\s+option|third\s+option|other\s+option|previous|"
    r"annual\s+option|monthly\s+option|what\s+about|how\s+about)\b|"
    r"(?:इसका|इसके|इसकी|उसका|उसके|उसकी|उनके|इसमें)",
    re.IGNORECASE,
)

OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b(?:weather\s+forecast|weather\s+today|weather\s+in|stock\s+market|stock\s+price|bitcoin\s+price|crypto\s+trading|"
    r"who\s+won\s+the\s+(?:cricket\s+)?world\s+cup|write\s+a\s+python\s+script|solve\s+this\s+math|"
    r"capital\s+of\s+france|president\s+of\s+the\s+united\s+states|movie\s+recommendations)\b",
    re.IGNORECASE,
)

TOPIC_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "are", "is", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "the", "a", "an", "and", "or", "but", "if", "then",
    "for", "of", "on", "in", "to", "from", "with", "about", "into", "through",
    "your", "you", "our", "we", "they", "their", "it", "its", "this", "that",
    "these", "those", "tell", "me", "please", "know", "interested", "want",
    "mujhe", "aapke", "aapka", "aapki", "baare", "bare", "mein", "bataiye",
    "batao", "kya", "hai", "hain", "iska", "iske", "aur", "kitna", "price",
    "cost", "charges", "fee", "fees", "benefits", "details", "option", "options",
}


def detect_language(text: str) -> str:
    """Detect whether the user message is Hindi ('hi'), Urdu ('ur'), Hinglish ('hinglish'), or English ('en')."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "en"
    if DEVANAGARI_RE.search(cleaned):
        return "hi"
    if ARABIC_URDU_RE.search(cleaned):
        return "ur"

    tokens = re.findall(r"[a-zA-Z]+", cleaned.lower())
    if not tokens:
        return "en"

    hinglish_hits = sum(1 for t in tokens if t in HINGLISH_MARKERS)
    urdu_hits = sum(1 for t in tokens if t in URDU_ROMAN_MARKERS)
    if urdu_hits >= 1 and hinglish_hits >= 1:
        return "hinglish"
    if hinglish_hits >= 2 or (len(tokens) <= 4 and hinglish_hits >= 1 and tokens[0] in ("namaste", "namaskar", "mujhe", "kya", "iska", "shukriya", "dhanyawad", "kaise")):
        return "hinglish"
    return "en"


def is_summary_request(message: str) -> bool:
    """Detect whether the customer is asking for a summary of the current conversation."""
    cleaned = (message or "").strip().lower()
    if any(
        phrase in cleaned
        for phrase in (
            "summarize our conversation",
            "summarize this conversation",
            "summarize the chat",
            "summary of our conversation",
            "summary of the chat",
            "give me a summary",
            "can you summarize",
            "please summarize",
            "what have we discussed",
            "recap our chat",
        )
    ):
        return True
    return bool(SUMMARY_INTENT_RE.search(cleaned) and len(cleaned.split()) <= 12)


def classify_message_intent(
    message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    memory_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Classify user input into GREETING, GENERAL_CONVERSATION, SUMMARY_REQUEST,
    AMBIGUOUS, OUT_OF_SCOPE, FOLLOW_UP_QUESTION, or KNOWLEDGE_QUESTION.
    """
    cleaned = (message or "").strip()
    if not cleaned:
        return "AMBIGUOUS"

    if is_summary_request(cleaned):
        return "SUMMARY_REQUEST"

    if GREETING_PATTERNS.match(cleaned) or HOW_ARE_YOU_PATTERNS.match(cleaned):
        return "GREETING"

    if GRATITUDE_OR_CLOSING_PATTERNS.match(cleaned) or REPEAT_OR_META_PATTERNS.match(cleaned):
        return "GENERAL_CONVERSATION"

    if OUT_OF_SCOPE_PATTERNS.search(cleaned):
        return "OUT_OF_SCOPE"

    history = conversation_history or []
    meta = memory_metadata or {}
    has_prior_context = bool(history) or bool(meta.get("current_topic")) or bool(meta.get("entities"))

    words = re.findall(r"[\w\u0900-\u097F\u0600-\u06FF]+", cleaned)
    lower_words = [w.lower() for w in words]

    # Check if input is extremely short/vague without prior context
    if len(words) <= 1 and lower_words and lower_words[0] in ("?", "hmm", "ok", "okay", "yes", "no", "details", "info", "more", "options", "price"):
        if has_prior_context and lower_words[0] in ("details", "info", "more", "options", "price"):
            return "FOLLOW_UP_QUESTION"
        if lower_words[0] in ("ok", "okay", "yes", "no"):
            return "GENERAL_CONVERSATION"
        return "AMBIGUOUS"

    if has_prior_context:
        if FOLLOW_UP_PRONOUN_RE.search(cleaned):
            return "FOLLOW_UP_QUESTION"
        if len(words) <= 6 and any(
            w in lower_words
            for w in ("price", "pricing", "cost", "charges", "benefits", "features", "hours", "annual", "monthly", "isme", "iska", "iske", "कीमत", "लाभ", "चार्ज")
        ):
            return "FOLLOW_UP_QUESTION"

    return "KNOWLEDGE_QUESTION"


def extract_entities_and_topic(
    message: str,
    prior_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract salient business entities and current topic from the user message to update conversation memory."""
    meta = dict(prior_metadata or {})
    existing_entities: List[str] = list(meta.get("entities") or [])

    # Look for multi-word capitalized or domain-important phrases (e.g., "Premium membership", "Core Membership", "Cloud Backup")
    cleaned = (message or "").strip()
    candidates: List[str] = []

    # 1. Explicit plan/service patterns
    plan_matches = re.findall(
        r"\b([A-Za-z0-9]+(?:\s+[A-Za-z0-9&-]+){0,2}\s+(?:membership|plan|package|tier|service|club|box|consultation|whitening|backup|sla|training))\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    for pm in plan_matches:
        tokens = [t for t in pm.split() if t.lower() not in ("your", "our", "the", "a", "an", "in", "about", "for", "of", "ki", "ke", "ka", "aapke")]
        if tokens:
            phrase = " ".join(tokens).strip()
            if len(phrase) >= 3:
                candidates.append(phrase)

    # 2. Content words if no explicit phrase matched
    if not candidates:
        content_words = [
            w for w in re.findall(r"[A-Za-z]{3,}", cleaned)
            if w.lower() not in TOPIC_STOPWORDS
        ]
        if content_words:
            candidates.append(" ".join(content_words[:3]))

    current_topic = meta.get("current_topic") or ""
    if candidates:
        current_topic = candidates[0]
        for c in candidates:
            if c.lower() not in {e.lower() for e in existing_entities}:
                existing_entities.append(c)

    # Keep bounded list of recent entities
    existing_entities = existing_entities[-8:]
    return {
        **meta,
        "current_topic": current_topic,
        "entities": existing_entities,
    }


class ConversationService:
    """Manages tenant-scoped conversations, persistent memory metadata, messages, feedback, and LLM summaries."""

    def __init__(
        self,
        repository: BaseRepository,
        llm_provider: BaseLlmProvider,
        history_limit: int = 12,
    ) -> None:
        self.repository = repository
        self.llm = llm_provider
        self.history_limit = max(6, history_limit)

    async def resolve_conversation(
        self,
        tenant_id: str,
        session_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve or create a conversation while strictly preventing cross-tenant or deleted access."""
        if conversation_id and conversation_id.strip():
            conv = await self.repository.get_conversation(tenant_id, conversation_id.strip())
            if not conv:
                raise AuthorizationError(
                    "Conversation not found or belongs to a different tenant."
                )
            return conv

        return await self.repository.get_or_create_conversation(
            tenant_id=tenant_id,
            session_id=session_id,
            conversation_id=None,
        )

    async def get_bounded_history(self, tenant_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        """Return bounded prior messages for the conversation."""
        messages = await self.repository.list_messages(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            limit=self.history_limit * 2,
        )
        return messages[-self.history_limit :]

    async def update_memory_state(
        self,
        tenant_id: str,
        conversation_id: str,
        user_message: str,
        assistant_answer: str,
        detected_language: str,
        intent: str,
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update persistent conversation memory metadata and trigger periodic long-term summary."""
        updated_meta = extract_entities_and_topic(user_message, existing_metadata)
        # Also extract key entities from assistant answer if topic is still empty
        if not updated_meta.get("current_topic") and assistant_answer:
            updated_meta = extract_entities_and_topic(assistant_answer, updated_meta)

        turn_count = int(updated_meta.get("turn_count", 0)) + 1
        updated_meta["turn_count"] = turn_count
        updated_meta["preferred_language"] = detected_language
        updated_meta["last_intent"] = intent

        await self.repository.update_conversation_metadata(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            metadata=updated_meta,
        )

        # Automatically generate/refresh long-term conversation summary every 3 turns (6 messages)
        if turn_count >= 2 and turn_count % 3 == 0:
            try:
                await self.generate_and_store_summary(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    language=detected_language,
                )
            except Exception:
                pass

        return updated_meta

    async def generate_and_store_summary(
        self,
        tenant_id: str,
        conversation_id: str,
        assistant_name: str = "AI Assistant",
        language: str = "en",
    ) -> str:
        """Generate an actual summary from prior messages using the LLM and store it."""
        history = await self.get_bounded_history(tenant_id, conversation_id)
        try:
            summary = await self.llm.summarize_conversation(
                conversation_history=history,
                assistant_name=assistant_name,
                language=language,
            )
        except Exception:
            if getattr(self.llm, "_http_handler", None) is not None:
                raise
            from app.providers.llm import LocalLlmProvider

            summary = await LocalLlmProvider().summarize_conversation(
                conversation_history=history,
                assistant_name=assistant_name,
                language=language,
            )
        await self.repository.update_conversation_summary(tenant_id, conversation_id, summary)
        return summary

    async def get_conversation_detail(self, tenant_id: str, conversation_id: str) -> Dict[str, Any]:
        conv = await self.repository.get_conversation(tenant_id, conversation_id)
        if not conv:
            raise NotFoundError("Conversation not found for this tenant.")
        messages = await self.repository.list_messages(tenant_id, conversation_id, limit=200)
        feedback_list = await self.repository.list_conversation_feedback(tenant_id, conversation_id)
        feedback_by_msg = {fb["message_id"]: fb for fb in feedback_list}
        for m in messages:
            if m["id"] in feedback_by_msg:
                m["feedback"] = feedback_by_msg[m["id"]]["feedback_type"]
                m["feedback_reason"] = feedback_by_msg[m["id"]].get("reason")
        return {**conv, "messages": messages, "feedback": feedback_list}

    async def delete_conversation(self, tenant_id: str, conversation_id: str) -> bool:
        conv = await self.repository.get_conversation(tenant_id, conversation_id)
        if not conv:
            raise NotFoundError("Conversation not found for this tenant.")
        return await self.repository.delete_conversation(tenant_id, conversation_id)

    async def submit_feedback(
        self,
        tenant_id: str,
        conversation_id: str,
        message_id: str,
        feedback_type: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate tenant, conversation, and assistant message ownership before persisting feedback."""
        norm_type = (feedback_type or "").strip().lower()
        if norm_type not in ("like", "dislike"):
            raise ValidationError("feedback_type must be 'like' or 'dislike'.")

        conv = await self.repository.get_conversation(tenant_id, conversation_id)
        if not conv:
            raise AuthorizationError("Conversation not found or does not belong to this tenant.")

        msg = await self.repository.get_message(tenant_id, message_id)
        if not msg or msg.get("conversation_id") != conversation_id:
            raise AuthorizationError("Message not found in the specified tenant conversation.")
        if msg.get("role") != "assistant":
            raise ValidationError("Feedback can only be submitted for assistant messages.")

        clean_reason = (reason or "").strip()[:500] or None
        return await self.repository.upsert_message_feedback(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_id=message_id,
            feedback_type=norm_type,
            reason=clean_reason,
        )

    async def list_conversations(
        self, tenant_id: Optional[str] = None, search: str = "", limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        return await self.repository.list_conversations(
            tenant_id=tenant_id, search=search, limit=limit, offset=offset
        )
