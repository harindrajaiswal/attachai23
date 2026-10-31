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
NON_LATIN_SCRIPT_RE = re.compile(r"[\u0400-\u052F\u0980-\u0DFF\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]")

HINGLISH_MARKERS = {
    "mujhe", "mujhko", "mera", "meri", "mere", "hum", "humne", "hamara", "hamari", "hamare",
    "aap", "aapke", "aapka", "aapki", "aapse", "tum", "tumhare", "tumhara", "tumhari", "tumse",
    "bataiye", "batao", "bataye", "batana", "samjhao", "samjhaiye", "likho", "bolo", "karo", "kariye",
    "kya", "kaun", "kaise", "kaisa", "kaisi", "kab", "kahan", "kyun", "kitna", "kitne", "kitni",
    "hai", "hain", "ho", "hoon", "tha", "thi", "the", "hoga", "hogi",
    "iska", "iske", "iski", "uska", "uske", "uski", "unke", "isme", "usme",
    "baare", "bare", "mein", "ke", "ki", "ka", "ko", "se", "aur", "ya", "par", "bhi", "toh", "ab",
    "pehle", "pahle", "poochha", "poocha", "pucha", "puchha", "baat", "karte", "karta", "karti",
    "nahi", "nahin", "mat", "sakte", "sakta", "sakti", "chahiye", "milega", "milta", "lagta",
    "namaste", "namaskar", "dhanyawad", "dhanyavaad", "shukriya", "alvida",
    "paisa", "paise", "kharcha", "mahine", "saal", "salana", "suvidha", "fayde", "fayda", "doosra", "pehla",
}

STRONG_HINGLISH_MARKERS = {
    "mujhe", "mujhko", "humne", "tumse", "aapke", "aapka", "aapki", "bataiye", "batao",
    "kya", "kaun", "kaise", "hai", "hain", "iska", "iske", "iski", "baare", "mein",
    "pehle", "poochha", "poocha", "pucha", "baat", "karo", "likho", "karte", "sakte",
    "namaste", "namaskar", "shukriya", "dhanyawad",
}

URDU_ROMAN_MARKERS = {
    "assalamualaikum", "walaikum", "meherbani", "khuwahish", "maloomat", "sahulat", "qimat",
}

LANGUAGE_SWITCH_DIRECTIVE_RE = re.compile(
    r"(?:\b(?:speak|talk|reply|respond|write|explain|translate|chat|converse|continue|switch)\b.*?\b(?:in|to|into)\s+([a-zA-Z]+)\b)|"
    r"(?:\b([a-zA-Z]+)\s+mein\s+(?:baat|likho|likhiye|batao|bataiye|samjhao|samjhaiye|boliye|bolo|jawaab|uttar))|"
    r"(?:हिंदी\s+में\s+(?:बात|लिखो|लिखिए|बताओ|बताइए|समझाओ|समझाइए|जवाब))|"
    r"(?:अंग्रेजी\s+में\s+(?:बात|लिखो|बताओ|समझाओ))",
    re.IGNORECASE,
)

TRANSLATE_OR_EXPLAIN_PREVIOUS_RE = re.compile(
    r"\b(?:explain\s+(?:it|that|this|your\s+previous\s+answer|the\s+previous\s+answer)\s+in\s+[a-zA-Z]+|"
    r"translate\s+(?:it|that|this|your\s+previous\s+answer|the\s+previous\s+answer|your\s+last\s+message)\s+(?:to|into|in)\s+[a-zA-Z]+|"
    r"now\s+explain\s+(?:that|it|this)\s+in\s+[a-zA-Z]+|"
    r"isko\s+[a-zA-Z]+\s+mein\s+(?:batao|bataiye|samjhao|translate\s+karo)|"
    r"इसे\s+हिंदी\s+में\s+(?:समझाओ|बताओ|अनुवाद\s+करो))\b",
    re.IGNORECASE,
)

GREETING_PATTERNS = re.compile(
    r"^\s*(?:hello|hi|hey|hiya|good\s+(?:morning|afternoon|evening|day)|"
    r"namaste|namaskar|pranam|assalamu\s*alaikum|salam|hola|bonjour|ciao|hallo|ola|"
    r"नमस्ते|नमस्कार|प्रणाम|हैलो|हाय|सलाम)"
    r"(?:\s*[!,.?]*)?(?:\s+(?:there|friend|assistant|ji|जी|how\s+are\s+you(?:\s+doing|\s+today)?|kaise\s+ho|kaise\s+hain|kya\s+haal\s+hai|कैसे\s+हैं|कैसे\s+हो|आप\s+कैसे\s+हैं))?\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

HOW_ARE_YOU_PATTERNS = re.compile(
    r"^\s*(?:how\s+are\s+you(?:\s+doing|\s+today)?|how's\s+it\s+going|how\s+do\s+you\s+do|"
    r"kya\s+haal\s+hai|(?:aap\s+|tum\s+)?kaise\s+(?:hain|ho)(?:\s+aap)?|आप\s+कैसे\s+हैं|आप\s+कैसे\s+हो|तुम\s+कैसे\s+हो|कैसे\s+हो)\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

IDENTITY_PATTERNS = re.compile(
    r"^\s*(?:who\s+are\s+you|what\s+is\s+your\s+name|tell\s+me\s+about\s+yourself|introduce\s+yourself|"
    r"(?:aap|tum)\s+kaun\s+(?:ho|hain)|aapka\s+naam\s+kya\s+hai|tumhara\s+naam\s+kya\s+hai|"
    r"(?:आप|तुम)\s+कौन\s+(?:हो|हैं)|आपका\s+नाम\s+क्या\s+है|तुम्हारा\s+नाम\s+क्या\s+है)\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

CAPABILITY_PATTERNS = re.compile(
    r"^\s*(?:what\s+can\s+you\s+do|what\s+do\s+you\s+do|how\s+can\s+you\s+help(?:\s+me)?|"
    r"what\s+can\s+you\s+help(?:\s+me)?\s+with|what\s+information\s+can\s+you\s+provide|"
    r"can\s+i\s+ask\s+you\s+questions|what\s+are\s+your\s+capabilities|what\s+is\s+your\s+(?:job|role|work)|"
    r"(?:aap|tum)\s+kya\s+(?:karte|karti|kar\s+sakte|kar\s+sakti)\s+(?:ho|hain)|"
    r"(?:tumhara|aapka)\s+kaam\s+kya\s+hai|kya\s+kaam\s+hai\s+(?:tumhara|aapka)|"
    r"(?:aap|tum)\s+meri\s+kya\s+(?:madad|help)\s+kar\s+sakte\s+(?:ho|hain)|"
    r"(?:आप|तुम)\s+क्या\s+(?:करते|करती|कर\s+सकते|कर\s+सकती)\s+(?:हो|हैं)|"
    r"(?:तुम्हारा|आपका)\s+काम\s+क्या\s+है|"
    r"आप\s+मेरी\s+कैसे\s+मदद\s+कर\s+सकते\s+हैं)\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

LANGUAGE_ABILITY_PATTERNS = re.compile(
    r"^\s*(?:can\s+you\s+(?:speak|talk|understand|write|chat)(?:\s+in)?\s+[a-zA-Z]+|"
    r"do\s+you\s+(?:speak|know|understand)\s+[a-zA-Z]+|"
    r"(?:kya\s+)?(?:aap|tum)\s+[a-zA-Z]+(?:\s+mein)?\s+(?:baat|bol|likh|samajh)(?:\s+nahi)?\s*(?:kar\s+)?(?:sakte|sakta|sakti)?\s*(?:ho|hain)?|"
    r"(?:क्या\s+)?(?:आप|तुम)\s+हिंदी(?:\s+में)?\s+(?:बात|बोल|लिख|समझ)(?:\s+नहीं)?\s*(?:कर\s+)?(?:सकते|सकता|सकती)?\s*(?:हो|हैं)?)\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

MEMORY_RECALL_PATTERNS = re.compile(
    r"^\s*(?:what\s+did\s+i\s+ask(?:\s+you)?\s+(?:earlier|before|previously|first|last\s+time)|"
    r"what\s+was\s+my\s+(?:previous|earlier|first|last)\s+question|"
    r"do\s+you\s+remember\s+what\s+i\s+asked|"
    r"(?:humne|maine)\s+(?:tumse|aapse)?\s*(?:pehle|pahle|abhi)\s+kya\s+(?:poochha|poocha|pucha|puchha|kaha)\s*(?:tha)?|"
    r"(?:मैंने|हमने)\s+(?:तुमसे|आपसे)?\s*(?:पहले|अभी)\s+क्या\s+(?:पूछा|कहा)\s*(?:था)?|"
    r"(?:can\s+you\s+|please\s+)?explain\s+that\s+again|repeat\s+that|can\s+you\s+repeat\s+that|"
    r"what\s+did\s+you\s+mean(?:\s+by\s+that)?|continue\s+from\s+where\s+we\s+stopped|"
    r"phir\s+se\s+(?:batao|bataiye|samjhao)|dobara\s+(?:batao|bataiye|samjhaiye))\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

GENERAL_CHITCHAT_PATTERNS = re.compile(
    r"^\s*(?:tell\s+me\s+something\s+interesting|tell\s+me\s+a\s+fun\s+fact|tell\s+me\s+a\s+joke|"
    r"kuch\s+interesting\s+(?:batao|bataiye)|कुछ\s+रोचक\s+बताओ|"
    r"nice\s+to\s+meet\s+you|pleasure\s+talking\s+to\s+you|you\s+are\s+(?:great|helpful|smart|awesome)|"
    r"i\s+have\s+a\s+question|i\s+have\s+another\s+question|one\s+more\s+question)\s*[!,.?]*\s*$",
    re.IGNORECASE,
)

GRATITUDE_OR_CLOSING_PATTERNS = re.compile(
    r"^\s*(?:thank\s+you(?:\s+so\s+much|\s+very\s+much)?|thanks(?:\s+a\s+lot)?|"
    r"thx|ty|shukriya|dhanyawad|dhanyavaad|bahut\s+dhanyawad|bahut\s+shukriya|"
    r"धन्यवाद|शुक्रिया|बहुत\s+धन्यवाद|"
    r"that(?:'s|\s+is)\s+(?:very\s+)?helpful|got\s+it|understood|makes\s+sense|"
    r"okay\s+thank\s+you|ok\s+thanks|great\s+thanks|awesome\s+thanks|perfect\s+thanks|"
    r"bye|goodbye|see\s+you|have\s+a\s+good\s+day|alvida)\s*[!,.?]*\s*$",
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
    r"iska|iske|iski|uska|uske|uski|unke|unki|isme|usme|"
    r"second\s+(?:option|plan|tier|package)|first\s+(?:option|plan|tier|package)|third\s+(?:option|plan|tier|package)|"
    r"other\s+(?:option|plan)|previous\s+(?:one|plan|option)|"
    r"better\s+than|compare\s+it|annual\s+option|monthly\s+option|what\s+about|how\s+about)\b|"
    r"(?:इसका|इसके|इसकी|उसका|उसके|उसकी|उनके|इसमें|दूसरा\s+प्लान|पहला\s+प्लान)",
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

LANGUAGE_NAME_NORMALIZATION = {
    "hindi": "hi",
    "हिंदी": "hi",
    "english": "en",
    "अंग्रेजी": "en",
    "angrezi": "en",
    "hinglish": "hinglish",
    "urdu": "ur",
    "اردو": "ur",
}


def extract_requested_language(text: str) -> Optional[str]:
    """Check if the user message explicitly asks to switch to or respond in a specific language."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if LANGUAGE_ABILITY_PATTERNS.match(cleaned):
        return None
    if re.search(r"हिंदी\s+में", cleaned):
        return "hi"
    if re.search(r"अंग्रेजी\s+में", cleaned):
        return "en"

    m = LANGUAGE_SWITCH_DIRECTIVE_RE.search(cleaned)
    if m:
        raw_lang = (m.group(1) or m.group(2) or "").strip().lower()
        if raw_lang and raw_lang not in ("detail", "details", "short", "brief", "depth", "full", "simple", "plain", "person", "advance", "general"):
            return LANGUAGE_NAME_NORMALIZATION.get(raw_lang, raw_lang)

    # Also catch short patterns like "Hindi mein likho", "Hindi mein baat karo", "Explain it in Hindi"
    m2 = re.search(
        r"\b(?:in|into|to)\s+(hindi|english|hinglish|urdu|spanish|french|german|tamil|bengali|marathi|gujarati|punjabi|telugu|arabic|japanese|korean|chinese)\b",
        cleaned,
        re.IGNORECASE,
    )
    if m2:
        raw_lang = m2.group(1).strip().lower()
        return LANGUAGE_NAME_NORMALIZATION.get(raw_lang, raw_lang)

    m3 = re.search(
        r"\b(hindi|english|hinglish|urdu|spanish|french|german|tamil|bengali|marathi|gujarati|punjabi|telugu|arabic)\s+mein\b",
        cleaned,
        re.IGNORECASE,
    )
    if m3:
        raw_lang = m3.group(1).strip().lower()
        return LANGUAGE_NAME_NORMALIZATION.get(raw_lang, raw_lang)

    return None


def detect_language(text: str, prior_language: Optional[str] = None) -> str:
    """Dynamically detect the user's language or requested language switch without a fixed allowlist."""
    cleaned = (text or "").strip()
    if not cleaned:
        return prior_language or "en"

    # 1. Explicit language switch or translation target takes highest priority
    requested = extract_requested_language(cleaned)
    if requested:
        return requested

    # 2. Script-based detection
    if DEVANAGARI_RE.search(cleaned):
        return "hi"
    if ARABIC_URDU_RE.search(cleaned):
        return "ur"
    if NON_LATIN_SCRIPT_RE.search(cleaned):
        return "auto"

    tokens = re.findall(r"[a-zA-Z]+", cleaned.lower())
    if not tokens:
        return prior_language or "en"

    hinglish_hits = sum(1 for t in tokens if t in HINGLISH_MARKERS)
    strong_hinglish_hits = sum(1 for t in tokens if t in STRONG_HINGLISH_MARKERS)
    urdu_hits = sum(1 for t in tokens if t in URDU_ROMAN_MARKERS)

    if urdu_hits >= 1 and hinglish_hits >= 1:
        return "hinglish"
    if hinglish_hits >= 2 or strong_hinglish_hits >= 1:
        return "hinglish"

    # If the user previously explicitly switched to Hindi ('hi') and sends a very short 1-2 word prompt like "Ok"
    if prior_language and len(tokens) <= 1 and tokens[0] in ("ok", "okay", "yes", "no", "acha", "haan"):
        return prior_language

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


def classify_conversational_sub_intent(
    message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Identify specific conversational sub-modes (IDENTITY, CAPABILITIES, LANGUAGE_ABILITY,
    MEMORY_RECALL, LANGUAGE_SWITCH, TRANSLATE_PREVIOUS, CHITCHAT) so responses are never forced into RAG fallback.
    """
    cleaned = (message or "").strip()
    if not cleaned:
        return None

    if TRANSLATE_OR_EXPLAIN_PREVIOUS_RE.search(cleaned):
        return "TRANSLATE_PREVIOUS"

    # Pure language switch command (e.g., "Hindi mein likho", "Ab Hindi mein baat karo", "Speak in Hindi")
    req_lang = extract_requested_language(cleaned)
    if req_lang and len(cleaned.split()) <= 8 and not any(
        bw in cleaned.lower()
        for bw in ("price", "pricing", "plan", "plans", "refund", "cancel", "policy", "membership", "hours", "address")
    ):
        if any(
            w in cleaned.lower()
            for w in ("explain", "translate", "that", "it", "previous", "isko", "इसे")
        ) and conversation_history:
            return "TRANSLATE_PREVIOUS"
        if LANGUAGE_ABILITY_PATTERNS.match(cleaned):
            return "LANGUAGE_ABILITY"
        return "LANGUAGE_SWITCH"

    if MEMORY_RECALL_PATTERNS.match(cleaned):
        return "MEMORY_RECALL"

    if IDENTITY_PATTERNS.match(cleaned):
        return "IDENTITY"

    if CAPABILITY_PATTERNS.match(cleaned):
        return "CAPABILITIES"

    if LANGUAGE_ABILITY_PATTERNS.match(cleaned):
        return "LANGUAGE_ABILITY"

    if GENERAL_CHITCHAT_PATTERNS.match(cleaned):
        return "CHITCHAT"

    return None


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

    sub_intent = classify_conversational_sub_intent(cleaned, conversation_history)
    if sub_intent in ("IDENTITY", "CAPABILITIES", "LANGUAGE_ABILITY", "MEMORY_RECALL", "LANGUAGE_SWITCH", "TRANSLATE_PREVIOUS", "CHITCHAT"):
        return "GENERAL_CONVERSATION"

    if GRATITUDE_OR_CLOSING_PATTERNS.match(cleaned):
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
