"""LLM provider abstraction and implementations (LocalLlmProvider & OpenAICompatibleLlmProvider).

Fixes Defects #2, #3, #4:
- Includes bounded prior conversation history in relevant LLM calls
- Summary requests genuinely use the LLM and conversation history
- Query reformulation is connected to the retrieval pipeline
- Remote LLM provider handles 401, 403, 429, 5xx, timeouts, invalid JSON, and empty responses
  with transient-only retries and never silently falls back to fake answers on remote errors.
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
}


class BaseLlmProvider(ABC):
    """Contract for chat synthesis, summarization, and query reformulation."""

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
    ) -> str:
        """Generate an evidence-grounded response using system prompt, chunks, and history."""

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
    """Deterministic, evidence-grounded local LLM synthesizer for offline development & tests."""

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
            len(q.split()) <= 6
            or any(
                pron in re.findall(r"\b[a-z]+\b", lower_q)
                for pron in ("it", "they", "them", "that", "those", "there", "also", "more", "what about")
            )
        )
        if not needs_context:
            return q

        # Extract topic terms from the most recent user/assistant turns
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
                first_sent = re.split(r"(?<=[.!?])\s+", content)[0].strip()
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
    ) -> str:
        if not evidence_chunks:
            return ""

        q_tokens = {
            w for w in re.findall(r"[a-z0-9]{2,}", question.lower()) if w not in STOPWORDS
        }
        # Also include context tokens from the last user message for follow-up questions
        for msg in reversed(conversation_history[-4:]):
            if msg.get("role") == "user":
                q_tokens.update(
                    w for w in re.findall(r"[a-z0-9]{3,}", str(msg.get("content", "")).lower())
                    if w not in STOPWORDS
                )
                break

        scored_sentences: List[Tuple[float, str]] = []
        seen_Normalized: set[str] = set()

        for rank_idx, chunk in enumerate(evidence_chunks[:4]):
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            raw_parts = re.split(r"(?<=[.!?])\s+|\n+", content)
            for part in raw_parts:
                s = part.strip()
                if len(s) < 8:
                    continue
                norm_s = s.lower()
                if norm_s in seen_Normalized:
                    continue
                seen_Normalized.add(norm_s)
                s_tokens = set(re.findall(r"[a-z0-9]{2,}", norm_s))
                overlap = len(q_tokens & s_tokens)
                rank_bonus = max(0.1, 1.0 - (rank_idx * 0.2))
                score = (overlap * 2.0) + rank_bonus
                scored_sentences.append((score, s))

        if not scored_sentences:
            return str(evidence_chunks[0].get("content") or "").strip()

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [s for _, s in scored_sentences[:3]]
        answer = " ".join(selected).strip()
        if not answer.endswith((".", "!", "?")):
            answer += "."
        return answer

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

        while attempts <= self.max_retries:
            attempts += 1
            try:
                if self._http_handler is not None:
                    status, raw_body = self._http_handler("POST", url, headers, payload)
                else:
                    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                    try:
                        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
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
                # Do NOT retry permanent auth/validation errors (401, 403, malformed JSON)
                if "HTTP 401" in pe.message or "HTTP 403" in pe.message or "malformed JSON" in pe.message:
                    raise
                if attempts > self.max_retries:
                    raise
                await asyncio.sleep(0.05 * attempts)
            except Exception as exc:
                safe_err = redact_text(str(exc), extra_secrets=[self.api_key])
                last_exc = ProviderError(
                    f"LLM provider request failed: {safe_err}",
                    status_code=502,
                    provider=self.provider_name,
                )
                if attempts > self.max_retries:
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
    ) -> str:
        evidence_block = "\n\n---\n\n".join(
            f"[Source: {c.get('source_filename', 'doc')}#{c.get('chunk_index', 0)}]\n{c.get('content', '')}"
            for c in evidence_chunks
        )
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    f"Language: {language}.\n"
                    "CRITICAL GROUNDING RULE: Answer strictly using the verified knowledge evidence below. "
                    "Never invent numbers, prices, dates, guarantees, or contact details not present in the evidence.\n\n"
                    f"VERIFIED EVIDENCE:\n{evidence_block}"
                ),
            }
        ]
        for turn in conversation_history[-6:]:
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
