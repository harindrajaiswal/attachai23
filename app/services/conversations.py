"""Conversation memory, session management, and summarization service.

Implements Section 9 Conversation Memory & Fixes Defects #2 and #3:
- Enforces strict tenant ownership of conversations and sessions
- Rejects cross-tenant conversation IDs with AuthorizationError
- Retrieves bounded prior conversation history for LLM context
- Generates and persists real LLM-powered conversation summaries
"""

import re
from typing import Any, Dict, List, Optional
from app.core.exceptions import AuthorizationError, NotFoundError
from app.providers.llm import BaseLlmProvider
from app.repositories.base import BaseRepository

SUMMARY_INTENT_RE = re.compile(
    r"\b(?:summarize|summary|recap|sum\s+up)\b.*?(?:\bconversation\b|\bchat\b|\bdiscussion\b|\bmessages\b|\bso\s+far\b)?|"
    r"\bwhat\s+have\s+we\s+(?:discussed|talked\s+about)\b",
    re.IGNORECASE,
)


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


class ConversationService:
    """Manages tenant-scoped conversations, messages, and LLM summaries."""

    def __init__(
        self,
        repository: BaseRepository,
        llm_provider: BaseLlmProvider,
        history_limit: int = 10,
    ) -> None:
        self.repository = repository
        self.llm = llm_provider
        self.history_limit = history_limit

    async def resolve_conversation(
        self,
        tenant_id: str,
        session_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve or create a conversation while strictly preventing cross-tenant access."""
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

    async def generate_and_store_summary(
        self,
        tenant_id: str,
        conversation_id: str,
        assistant_name: str = "AI Assistant",
        language: str = "en",
    ) -> str:
        """Generate an actual summary from prior messages using the LLM and store it."""
        history = await self.get_bounded_history(tenant_id, conversation_id)
        summary = await self.llm.summarize_conversation(
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
        return {**conv, "messages": messages}

    async def list_conversations(
        self, tenant_id: Optional[str] = None, search: str = "", limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        return await self.repository.list_conversations(
            tenant_id=tenant_id, search=search, limit=limit, offset=offset
        )
