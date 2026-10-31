"""LLM provider abstraction and implementations (LocalLlmProvider & OpenAICompatibleLlmProvider).

Implements:
- Bounded prior conversation history + persistent conversation memory metadata in LLM calls
- Genuine conversation summarization
- Context-aware query reformulation
- Multilingual answer synthesis & conversational replies (English, Hindi Devanagari, Hinglish, Urdu)
- Strict remote LLM error handling (401, 403, 429, 5xx, timeouts, malformed JSON, empty choices)
"""

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.core.exceptions import ConfigurationError, ProviderError
from app.utils.pii import redact_text

STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "are", "is", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "the", "a", "an", "and", "or", "but", "if", "then",
    "for", "of", "on", "in", "to", "from", "with", "about", "into", "through",
    "your", "you", "our", "we", "they", "their", "it", "its", "this", "that",
    "these", "those", "tell", "me", "please", "know",
    "kya", "hai", "hain", "mein", "aur", "ke", "ki", "ka", "ko", "se", "batao",
}


class BaseLlmProvider(ABC):
    """Contract for chat synthesis, conversational replies, summarization, and query reformulation."""

    @abstractmethod
    async def generate_answer(
        self,
        system_prompt: str,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        language: str = "en",
        temperature: float = 0.2,
        model_override: Optional[str] = None,
        memory_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate an evidence-grounded response using system prompt, chunks, and history."""

    @abstractmethod
    async def generate_conversational_reply(
        self,
        intent: str,
        message: str,
        language: str,
        tenant: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        memory_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a natural, helpful conversational response for greetings, small talk, ambiguity, or out-of-scope."""

    @abstractmethod
    async def summarize_conversation(
        self,
        conversation_history: List[Dict[str, Any]],
        assistant_name: str = "AI Assistant",
        language: str = "en",
    ) -> str:
        """Generate a real summary of the conversation history."""

    @abstractmethod
    async def reformulate_query(
        self,
        question: str,
        conversation_history: List[Dict[str, Any]],
    ) -> str:
        """Reformulate a follow-up question using bounded prior conversation history."""

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check LLM provider configuration and connectivity."""


class LocalLlmProvider(BaseLlmProvider):
    """Deterministic, evidence-grounded local LLM synthesizer with multilingual support."""

    def __init__(self, model_name: str = "local-rag-synthesizer") -> None:
        self.model_name = model_name

    async def reformulate_query(
        self,
        question: str,
        conversation_history: List[Dict[str, Any]],
    ) -> str:
        q = (question or "").strip()
        if not q or not conversation_history:
            return q

        lower_q = q.lower()
        needs_context = (
            len(q.split()) <= 7
            or any(
                pron in re.findall(r"\b[a-z]+\b", lower_q)
                for pron in (
                    "it", "its", "they", "them", "that", "those", "there", "also",
                    "more", "iska", "iske", "iski", "usme", "plan", "option", "second", "first",
                )
            )
        )
        if not needs_context:
            return q

        recent_terms: List[str] = []
        for turn in reversed(conversation_history[-4:]):
            content = str(turn.get("content") or "")
            words = [
                w for w in re.findall(r"[A-Za-z0-9]{3,}", content.lower())
                if w not in STOPWORDS
            ]
            for w in words[:6]:
                if w not in recent_terms:
                    recent_terms.append(w)
            if len(recent_terms) >= 6:
                break

        if not recent_terms:
            return q
        return f"{q} ({' '.join(recent_terms[:5])})"

    async def generate_conversational_reply(
        self,
        intent: str,
        message: str,
        language: str,
        tenant: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        memory_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        assistant_name = str(tenant.get("assistant_name") or "AI Assistant").strip()
        company_name = str(tenant.get("company_name") or tenant.get("business_name") or "our team").strip()
        meta = memory_metadata or {}
        current_topic = str(meta.get("current_topic") or "").strip()
        lower_msg = (message or "").strip().lower()

        if intent == "GREETING":
            if language == "hi":
                return (
                    f"नमस्ते! मैं {assistant_name} हूँ ({company_name})। "
                    f"मैं आपके सवालों—जैसे हमारे प्लान्स, कीमत (pricing), रिफंड पॉलिसी, या सपोर्ट—में आपकी मदद कर सकता हूँ। "
                    f"बताइए, आज मैं आपकी क्या सहायता कर सकता हूँ?"
                )
            if language == "hinglish":
                return (
                    f"Namaste! Main {assistant_name} hoon from {company_name}. "
                    f"Main aapki pricing, plans, features, refund policy, ya support se jude sawalon mein madad kar sakta hoon. "
                    f"Bataiye, aaj main aapki kaise help kar sakta hoon?"
                )
            if language == "ur":
                return (
                    f"السلام علیکم! میں {company_name} کی جانب سے {assistant_name} ہوں۔ "
                    f"میں آپ کے سوالات، پلانز، قیمتوں اور سپورٹ کے بارے میں مدد کر سکتا ہوں۔ آج میں آپ کی کیا مدد کر سکتا ہوں؟"
                )
            return (
                f"Hello! I'm {assistant_name} with {company_name}. "
                f"I can help answer questions about our plans, pricing, features, policies, and support options. "
                f"How can I help you today?"
            )

        if intent == "GENERAL_CONVERSATION":
            is_thanks = any(
                w in lower_msg for w in ("thank", "thanks", "shukriya", "dhanyavad", "धन्यवाद", "शुक्रिया")
            )
            is_bye = any(w in lower_msg for w in ("bye", "goodbye", "alvida", "see you"))
            if is_thanks:
                if language == "hi":
                    return f"आपका स्वागत है! अगर आपको {company_name} के बारे में और कोई जानकारी चाहिए, तो बेझिझक पूछें।"
                if language == "hinglish":
                    return f"You're most welcome! Agar aapko {company_name} ke baare mein kuch aur poochna ho, toh zaroor bataiye."
                if language == "ur":
                    return f"آپ کا شکریہ! اگر آپ کو مزید کسی معلومات کی ضرورت ہو تو بلا جھجھک پوچھیں۔"
                return f"You're very welcome! Let me know if there is anything else I can help you with regarding {company_name}."
            if is_bye:
                if language == "hi":
                    return f"धन्यवाद! आपका दिन शुभ हो। जब भी आवश्यकता हो, हमसे फिर जुड़ें।"
                if language == "hinglish":
                    return f"Take care! Jab bhi aapko {company_name} se judi koi help chahiye ho, hum yahin hain."
                return f"Goodbye and have a great day! Feel free to reach out anytime you need help from {company_name}."

            if language == "hi":
                return (
                    f"मैं {assistant_name} हूँ, {company_name} का AI सहायक। "
                    f"मैं हमारे उपलब्ध दस्तावेज़ों के आधार पर आपके सवालों के सटीक उत्तर देने के लिए यहाँ हूँ। आप क्या जानना चाहेंगे?"
                )
            if language == "hinglish":
                return (
                    f"Main {assistant_name} hoon, {company_name} ka AI support assistant. "
                    f"Aap mujhse hamare plans, pricing, policies, ya services ke baare mein pooch sakte hain!"
                )
            if language == "ur":
                return (
                    f"میں {assistant_name} ہوں اور {company_name} کے متعلق آپ کے سوالات کے جوابات دینے کے لیے حاضر ہوں۔"
                )
            return (
                f"I'm doing well, thank you! I'm {assistant_name}, the virtual assistant for {company_name}. "
                f"I can help you explore our services, pricing plans, policies, and support details. What would you like to know?"
            )

        if intent == "AMBIGUOUS":
            topic_hint = f" about {current_topic}" if current_topic else ""
            if language == "hi":
                return (
                    f"मैं आपकी पूरी मदद करना चाहता हूँ, लेकिन आपका प्रश्न थोड़ा संक्षिप्त है। "
                    f"क्या आप स्पष्ट कर सकते हैं कि आप हमारे किस प्लान, कीमत (pricing), या सेवा के बारे में जानना चाहते हैं?"
                )
            if language == "hinglish":
                return (
                    f"Main aapki help zaroor karna chahunga! Kya aap thoda detail mein bata sakte hain ki aap "
                    f"kis plan, pricing, ya policy ke baare mein poochna chahte hain?"
                )
            if language == "ur":
                return (
                    f"میں آپ کی مدد کرنے کے لیے تیار ہوں۔ براہ کرم تھوڑی مزید تفصیل بتائیں کہ آپ کس پلان، قیمت یا سروس کے بارے میں جاننا چاہتے ہیں؟"
                )
            return (
                f"I'd love to help{topic_hint}, could you share a little more detail about what you're looking for? "
                f"For example, are you asking about our pricing plans, refund policy, integrations, or support hours?"
            )

        # OUT_OF_SCOPE or unsupported knowledge in non-English
        if language == "hi":
            return (
                f"मेरे पास {company_name} के सत्यापित ज्ञानकोश (knowledge base) में इस विषय पर विशिष्ट जानकारी उपलब्ध नहीं है। "
                f"मैं हमारे प्लान्स, कीमत, नीतियों और सपोर्ट के बारे में आपके सवालों का जवाब दे सकता हूँ, या आप अपने संपर्क विवरण (contact details) छोड़ सकते हैं ताकि हमारी टीम आपकी सहायता कर सके।"
            )
        if language == "hinglish":
            return (
                f"Mere paas {company_name} ke verified knowledge base mein is topic par specific details abhi available nahi hain. "
                f"Aap mujhse hamare plans, pricing, refund policy, ya support ke baare mein pooch sakte hain, ya apni contact details share kar sakte hain taaki hamari team aapse connect kar sake."
            )
        if language == "ur":
            return (
                f"ہمارے تصدیق شدہ ریکارڈ میں اس موضوع پر مخصوص معلومات موجود نہیں ہیں۔ "
                f"آپ ہمارے پلانز، قیمتوں یا پالیسیوں کے بارے میں پوچھ سکتے ہیں، یا اپنی رابطہ معلومات چھوڑ سکتے ہیں تاکہ ہماری ٹیم آپ کی رہنمائی کر سکے۔"
            )
        return (
            f"I specialize in helping with {company_name}'s products, pricing, policies, and support, so I don't have verified information on that topic. "
            f"If you have a question about our services—or would like our team to follow up directly—please let me know!"
        )

    async def summarize_conversation(
        self,
        conversation_history: List[Dict[str, Any]],
        assistant_name: str = "AI Assistant",
        language: str = "en",
    ) -> str:
        if not conversation_history:
            return "We have just started our conversation and there are no prior messages to summarize yet."

        user_points: List[str] = []
        assistant_points: List[str] = []
        for msg in conversation_history:
            role = msg.get("role", "")
            content = " ".join(str(msg.get("content") or "").split()).strip()
            if not content:
                continue
            if role == "user":
                user_points.append(content)
            elif role == "assistant":
                first_sent = re.split(r"(?<=[.!?।])\s+", content)[0].strip()
                assistant_points.append(first_sent)

        summary_parts: List[str] = ["Conversation Summary:"]
        if user_points:
            asked = "; ".join(user_points[-5:])
            summary_parts.append(f"• Topics asked by customer: {asked}")
        if assistant_points:
            answered = " ".join(assistant_points[-4:])
            summary_parts.append(f"• Key information provided by {assistant_name}: {answered}")
        return "\n".join(summary_parts)

    async def generate_answer(
        self,
        system_prompt: str,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        language: str = "en",
        temperature: float = 0.2,
        model_override: Optional[str] = None,
        memory_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not evidence_chunks:
            return ""

        from app.services.advanced_rag import (
            expand_query_terms,
            translate_multilingual_tokens_to_english,
        )

        bridged_en_tokens = translate_multilingual_tokens_to_english(question)
        base_q_tokens = [
            w for w in re.findall(r"[a-z0-9]{2,}", question.lower()) if w not in STOPWORDS
        ] + [w.lower() for w in bridged_en_tokens if w.lower() not in STOPWORDS]
        primary_q_tokens = expand_query_terms(base_q_tokens) | set(base_q_tokens)

        context_tokens: set[str] = set()
        meta = memory_metadata or {}
        lower_q = question.lower()
        is_follow_up_or_short = (
            len(question.split()) <= 7
            or any(
                p in lower_q
                for p in (
                    "it", "its", "that", "this", "iska", "iske", "iski", "usme",
                    "second", "first", "third", "annual", "cancel", "included",
                )
            )
        )
        if is_follow_up_or_short:
            for ent in meta.get("entities") or []:
                for tok in re.findall(r"[a-z0-9]{2,}", str(ent).lower()):
                    if tok not in STOPWORDS:
                        context_tokens.add(tok)
            if meta.get("current_topic"):
                for tok in re.findall(r"[a-z0-9]{2,}", str(meta["current_topic"]).lower()):
                    if tok not in STOPWORDS:
                        context_tokens.add(tok)
            for msg in reversed(conversation_history[-4:]):
                if msg.get("role") == "user":
                    en_hist = translate_multilingual_tokens_to_english(str(msg.get("content", "")))
                    context_tokens.update(
                        w.lower() for w in en_hist if len(w) >= 3 and w.lower() not in STOPWORDS
                    )
                    break

        scored_sentences: List[Tuple[float, str]] = []
        seen_normalized: set[str] = set()

        for rank_idx, chunk in enumerate(evidence_chunks[:4]):
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            raw_parts = re.split(r"(?<=[.!?])\s+|\n+", content)
            for part in raw_parts:
                s = part.strip()
                if len(s) < 8:
                    continue
                if s.endswith(":") and len(s.split()) <= 6:
                    continue
                norm_s = s.lower()
                if norm_s in seen_normalized:
                    continue
                seen_normalized.add(norm_s)
                s_tokens = set(re.findall(r"[a-z0-9]{2,}", norm_s))
                primary_overlap = len(primary_q_tokens & s_tokens)
                context_overlap = len(context_tokens & s_tokens)
                entity_boost = 0.0
                if is_follow_up_or_short and meta.get("entities"):
                    last_ent = str(meta["entities"][-1]).lower()
                    if last_ent and last_ent in norm_s:
                        entity_boost = 1.5
                price_boost = 0.0
                if (
                    any(p in primary_q_tokens for p in ("price", "pricing", "cost", "costs", "charges", "fee", "fees"))
                    and ("$" in s or "per month" in norm_s or "/month" in norm_s)
                ):
                    price_boost = 2.5
                rank_bonus = max(0.1, 1.0 - (rank_idx * 0.2))
                score = (primary_overlap * 3.0) + (context_overlap * 1.0) + entity_boost + price_boost + rank_bonus
                scored_sentences.append((score, s))

        if not scored_sentences:
            base_answer = str(evidence_chunks[0].get("content") or "").strip()
        else:
            scored_sentences.sort(key=lambda item: item[0], reverse=True)
            selected = [s for _, s in scored_sentences[:3]]
            base_answer = " ".join(selected).strip()
            if not base_answer.endswith((".", "!", "?")):
                base_answer += "."

        # Adapt output naturally when user asked in Hindi, Hinglish, or Urdu while keeping verified facts intact
        if language == "hi":
            return f"हमारे सत्यापित रिकॉर्ड के अनुसार: {base_answer}"
        if language == "hinglish":
            return f"Hamare verified details ke mutabik: {base_answer}"
        if language == "ur":
            return f"ہمارے تصدیق شدہ ریکارڈ کے مطابق: {base_answer}"
        return base_answer

    async def check_health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "provider_type": "local",
            "provider_name": "local",
            "model": self.model_name,
        }


class OpenAICompatibleLlmProvider(BaseLlmProvider):
    """Production OpenAI-compatible HTTP LLM provider with retries and strict error handling."""

    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_ms: int = 15000,
        max_retries: int = 2,
        temperature: float = 0.2,
        max_tokens: int = 800,
        chat_path: str = "/chat/completions",
        health_path: str = "/models",
        auth_header_name: str = "Authorization",
        auth_scheme: str = "Bearer",
        extra_headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        http_handler: Optional[Callable[[str, str, Dict[str, str], Optional[bytes]], Tuple[int, Any]]] = None,
    ) -> None:
        if not base_url or not api_key or not model:
            raise ConfigurationError(
                "LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL are required for OpenAICompatibleLlmProvider."
            )
        self.provider_name = provider_name or "openai-compatible"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_sec = max(1.0, timeout_ms / 1000.0)
        self.max_retries = max(0, int(max_retries))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_path = "/" + chat_path.lstrip("/")
        self.health_path = "/" + health_path.lstrip("/")
        self.auth_header_name = auth_header_name.strip() or "Authorization"
        self.auth_scheme = auth_scheme.strip()
        self.extra_headers = dict(extra_headers or {})
        self.query_params = dict(query_params or {})
        self._http_handler = http_handler

    def _build_url(self, path: str) -> str:
        url = f"{self.base_url}{path}"
        if self.query_params:
            qs = urllib.parse.urlencode(self.query_params)
            url = f"{url}?{qs}"
        return url

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.extra_headers)
        auth_val = f"{self.auth_scheme} {self.api_key}".strip() if self.auth_scheme else self.api_key
        headers[self.auth_header_name] = auth_val
        return headers

    async def _post_completions(self, messages: List[Dict[str, str]], temperature: float, model: str) -> str:
        url = self._build_url(self.chat_path)
        headers = self._build_headers()
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
        ).encode("utf-8")

        attempts = 0
        last_exc: Optional[Exception] = None
        effective_retries = self.max_retries if self._http_handler is not None else min(1, self.max_retries)
        effective_timeout = self.timeout_sec if self._http_handler is not None else min(self.timeout_sec, 6.0)

        while attempts <= effective_retries:
            attempts += 1
            try:
                if self._http_handler is not None:
                    status, raw_body = self._http_handler("POST", url, headers, payload)
                else:
                    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                    try:
                        with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                            status = resp.status
                            raw_body = resp.read().decode("utf-8")
                    except urllib.error.HTTPError as http_err:
                        status = http_err.code
                        raw_body = ""
                    except TimeoutError as te:
                        raise ProviderError(
                            "LLM provider request timed out.",
                            status_code=504,
                            provider=self.provider_name,
                        ) from te

                if status == 401:
                    raise ProviderError(
                        "LLM provider authentication failed (HTTP 401).",
                        status_code=502,
                        provider=self.provider_name,
                    )
                if status == 403:
                    raise ProviderError(
                        "LLM provider authorization denied (HTTP 403).",
                        status_code=502,
                        provider=self.provider_name,
                    )
                if status == 429 or status >= 500:
                    raise ProviderError(
                        f"LLM provider transient error (HTTP {status}).",
                        status_code=503 if status == 429 else 502,
                        provider=self.provider_name,
                    )
                if status >= 400:
                    raise ProviderError(
                        f"LLM provider rejected request (HTTP {status}).",
                        status_code=502,
                        provider=self.provider_name,
                    )

                if isinstance(raw_body, str):
                    try:
                        parsed = json.loads(raw_body)
                    except json.JSONDecodeError as je:
                        raise ProviderError(
                            "LLM provider returned malformed JSON.",
                            status_code=502,
                            provider=self.provider_name,
                        ) from je
                elif isinstance(raw_body, dict):
                    parsed = raw_body
                else:
                    raise ProviderError(
                        "LLM provider returned unexpected payload type.",
                        status_code=502,
                        provider=self.provider_name,
                    )

                choices = parsed.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ProviderError(
                        "LLM provider returned no completion choices.",
                        status_code=502,
                        provider=self.provider_name,
                    )
                msg_obj = choices[0].get("message") or {}
                content = str(msg_obj.get("content") or "").strip()
                if not content:
                    raise ProviderError(
                        "LLM provider returned an empty response.",
                        status_code=502,
                        provider=self.provider_name,
                    )
                return content

            except ProviderError as pe:
                last_exc = pe
                if "HTTP 401" in pe.message or "HTTP 403" in pe.message or "malformed JSON" in pe.message:
                    raise
                if attempts > effective_retries:
                    raise
                await asyncio.sleep(0.05 * attempts)
            except Exception as exc:
                safe_err = redact_text(str(exc), extra_secrets=[self.api_key])
                last_exc = ProviderError(
                    f"LLM provider request failed: {safe_err}",
                    status_code=502,
                    provider=self.provider_name,
                )
                if attempts > effective_retries:
                    raise last_exc from exc
                await asyncio.sleep(0.05 * attempts)

        raise last_exc or ProviderError("LLM provider failed.", provider=self.provider_name)

    async def reformulate_query(
        self,
        question: str,
        conversation_history: List[Dict[str, Any]],
    ) -> str:
        if not conversation_history or len(question.split()) > 10:
            return question
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conversation_history[-4:]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the follow-up customer question into a standalone search query using the recent conversation context. "
                    "Output only the standalone search query."
                ),
            },
            {
                "role": "user",
                "content": f"Conversation:\n{history_text}\n\nFollow-up question: {question}",
            },
        ]
        try:
            rewritten = await self._post_completions(messages, temperature=0.0, model=self.model)
            return rewritten.strip() or question
        except Exception:
            return question

    async def generate_conversational_reply(
        self,
        intent: str,
        message: str,
        language: str,
        tenant: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        memory_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return await LocalLlmProvider().generate_conversational_reply(
            intent=intent,
            message=message,
            language=language,
            tenant=tenant,
            conversation_history=conversation_history,
            memory_metadata=memory_metadata,
        )

    async def summarize_conversation(
        self,
        conversation_history: List[Dict[str, Any]],
        assistant_name: str = "AI Assistant",
        language: str = "en",
    ) -> str:
        if not conversation_history:
            return "No messages in this conversation yet."
        transcript = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in conversation_history[-12:]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {assistant_name}. Summarize the customer support conversation clearly and concisely in {language}. "
                    "Do not invent any facts outside the transcript."
                ),
            },
            {"role": "user", "content": f"Summarize this conversation:\n\n{transcript}"},
        ]
        return await self._post_completions(messages, temperature=0.2, model=self.model)

    async def generate_answer(
        self,
        system_prompt: str,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        language: str = "en",
        temperature: float = 0.2,
        model_override: Optional[str] = None,
        memory_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        evidence_block = "\n\n---\n\n".join(
            f"[Source: {c.get('source_filename', 'doc')}#{c.get('chunk_index', 0)}]\n{c.get('content', '')}"
            for c in evidence_chunks
        )
        lang_instruction = {
            "hi": "Respond naturally in polite Hindi (Devanagari script), keeping exact numbers, prices, and plan names from the evidence.",
            "hinglish": "Respond naturally in polite Hinglish (conversational Hindi in Latin script mixed with English), keeping exact numbers, prices, and plan names from the evidence.",
            "ur": "Respond naturally in polite Urdu, keeping exact numbers, prices, and plan names from the evidence.",
            "en": "Respond naturally in clear, helpful English.",
        }.get(language, f"Respond in {language}.")

        meta = memory_metadata or {}
        memory_context = ""
        if meta.get("current_topic") or meta.get("entities"):
            memory_context = (
                f"\nActive Conversation Topic: {meta.get('current_topic', '')}\n"
                f"Active Entities Discussed: {', '.join(meta.get('entities') or [])}\n"
            )

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    f"Language Instruction: {lang_instruction}\n"
                    f"{memory_context}"
                    "CRITICAL GROUNDING RULE: Answer strictly using the verified knowledge evidence below. "
                    "Never output labels like 'Unverified Answer' or 'Fallback Answer'. "
                    "Never invent numbers, prices, dates, guarantees, or contact details not present in the evidence.\n\n"
                    f"VERIFIED EVIDENCE:\n{evidence_block}"
                ),
            }
        ]
        for turn in conversation_history[-8:]:
            role = turn.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": str(turn.get("content", ""))})
        messages.append({"role": "user", "content": question})
        return await self._post_completions(
            messages=messages,
            temperature=temperature,
            model=model_override or self.model,
        )

    async def check_health(self) -> Dict[str, Any]:
        url = self._build_url(self.health_path)
        headers = self._build_headers()
        try:
            if self._http_handler is not None:
                status, _ = self._http_handler("GET", url, headers, None)
            else:
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    status = resp.status
            return {
                "ok": 200 <= status < 400,
                "provider_type": "openai-compatible",
                "provider_name": self.provider_name,
                "model": self.model,
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider_type": "openai-compatible",
                "provider_name": self.provider_name,
                "model": self.model,
                "error": redact_text(str(exc), extra_secrets=[self.api_key]),
            }
