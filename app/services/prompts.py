"""Centralized conversational system instructions, prompt versioning, restoration, and live assistant preview service."""

from typing import Any, Dict, List, Optional
from app.core.exceptions import NotFoundError, ValidationError
from app.repositories.base import BaseRepository
from app.services.advanced_rag import AdvancedRagService
from app.services.grounding import GroundingService
from app.providers.llm import BaseLlmProvider


def format_language_directive(language: str) -> str:
    """Return a dynamic language instruction for the LLM without restricting to a fixed allowlist."""
    lang = (language or "en").strip()
    low = lang.lower()
    if low == "hi":
        return (
            "Respond naturally in fluent, conversational Hindi (Devanagari script: हिंदी). "
            "Keep exact numbers, currency values, and official plan/product names accurate."
        )
    if low == "hinglish":
        return (
            "Respond naturally in conversational Hinglish (Hindi written in Latin script mixed naturally with English). "
            "Keep exact numbers, currency values, and official plan/product names accurate."
        )
    if low == "ur":
        return (
            "Respond naturally in fluent, polite Urdu (اردو). "
            "Keep exact numbers, currency values, and official plan/product names accurate."
        )
    if low == "en":
        return "Respond naturally in clear, warm, conversational English."
    if low == "auto":
        return (
            "Detect the exact language of the user's message and respond naturally in that same language."
        )
    return (
        f"Respond naturally and fluently in {lang}, matching the user's chosen language while keeping "
        "exact numbers, currency values, and official plan/product names accurate."
    )


def build_conversational_system_prompt(
    tenant: Dict[str, Any],
    language: str = "en",
    intent: str = "GENERAL_CONVERSATION",
    memory_metadata: Optional[Dict[str, Any]] = None,
    conversation_summary: str = "",
) -> str:
    """Build centralized system instructions for natural, context-aware, multilingual conversation."""
    assistant_name = str(tenant.get("assistant_name") or "AI Assistant").strip()
    business_name = str(
        tenant.get("name") or tenant.get("company_name") or tenant.get("business_name") or ""
    ).strip()
    tenant_prompt = str(tenant.get("system_prompt") or "").strip()
    tone = str(tenant.get("tone") or "helpful, warm, and natural").strip()
    meta = memory_metadata or {}

    identity_line = (
        f"You are {assistant_name}, the AI assistant for {business_name}."
        if business_name
        else f"You are {assistant_name}, a helpful AI assistant."
    )

    memory_lines: List[str] = []
    if conversation_summary:
        memory_lines.append(f"Earlier Conversation Summary:\n{conversation_summary}")
    if meta.get("current_topic"):
        memory_lines.append(f"Current Active Topic: {meta['current_topic']}")
    if meta.get("entities"):
        memory_lines.append(f"Entities Discussed So Far: {', '.join(str(e) for e in meta['entities'])}")
    memory_block = "\n".join(memory_lines)
    tenant_prompt_line = f"Tenant Instructions: {tenant_prompt}\n" if tenant_prompt else ""
    memory_section = f"{memory_block}\n" if memory_block else ""

    return (
        f"{identity_line}\n"
        f"Personality & Tone: {tone}.\n"
        f"{tenant_prompt_line}\n"
        f"Language Directive: {format_language_directive(language)}\n"
        f"Current Conversational Context Mode: {intent}\n"
        f"{memory_section}\n"
        "CORE CONVERSATIONAL GUIDELINES:\n"
        "1. Communicate like an intelligent, natural human conversational partner (comparable to ChatGPT or Gemini).\n"
        "2. Never use robotic templates, never repeat the same canned introduction on every turn, and never mention 'verified knowledge base', 'Unverified Answer', 'Fallback Answer', or internal system states.\n"
        "3. Understand and reply in the user's chosen language—including Hindi (Devanagari), Hinglish, English, or any other language—and switch languages immediately whenever the user asks.\n"
        "4. When the user asks who you are or what you can do, introduce yourself naturally using your configured identity and explain how you can help.\n"
        "5. When the user asks what they asked earlier or refers to prior messages, use the actual conversation history to answer accurately. Never invent prior turns that did not happen.\n"
        "6. Never fabricate company-specific prices, policies, addresses, guarantees, or contact details."
    ).strip()


def build_rag_system_prompt(
    tenant: Dict[str, Any],
    language: str,
    evidence_chunks: List[Dict[str, Any]],
    memory_metadata: Optional[Dict[str, Any]] = None,
    conversation_summary: str = "",
) -> str:
    """Build centralized system instructions for grounded business RAG responses."""
    base_prompt = build_conversational_system_prompt(
        tenant=tenant,
        language=language,
        intent="KNOWLEDGE_QUESTION",
        memory_metadata=memory_metadata,
        conversation_summary=conversation_summary,
    )
    if evidence_chunks:
        evidence_block = "\n\n---\n\n".join(
            f"[Source: {c.get('source_filename', 'doc')}#{c.get('chunk_index', 0)}]\n{c.get('content', '')}"
            for c in evidence_chunks
        )
        return (
            f"{base_prompt}\n\n"
            "FACTUAL GROUNDING RULES:\n"
            "- Answer the user's question naturally and directly in the target language using the retrieved business evidence below.\n"
            "- Do not start your response with robotic phrases like 'According to our verified records' or 'Hamare verified details ke mutabik'. Just answer naturally.\n"
            "- Preserve exact numbers, prices, durations, and policy terms from the evidence.\n"
            "- If the user asks to translate, compare, or explain earlier facts, use both the evidence below and the conversation history.\n"
            "- Never invent prices, policies, addresses, guarantees, or contact details not supported by the evidence.\n\n"
            f"RETRIEVED BUSINESS EVIDENCE:\n{evidence_block}"
        )
    return (
        f"{base_prompt}\n\n"
        "FACTUAL GROUNDING RULES (NO SPECIFIC DOCUMENT MATCHED):\n"
        "- The user is asking for a specific business detail that is not present in the uploaded company documents.\n"
        "- Politely and naturally let the user know in their language that you don't have that specific detail (such as an unlisted office address or unconfigured policy) in the available company information.\n"
        "- Do not invent any company facts, prices, addresses, or contact details.\n"
        "- Do not use robotic phrases like 'Unverified Answer' or 'Fallback Answer'."
    )


class PromptService:
    """Manages versioned tenant system prompts and preview execution."""

    def __init__(
        self,
        repository: BaseRepository,
        rag_service: AdvancedRagService,
        grounding_service: GroundingService,
        llm_provider: BaseLlmProvider,
    ) -> None:
        self.repository = repository
        self.rag = rag_service
        self.grounding = grounding_service
        self.llm = llm_provider

    async def list_versions(self, tenant_id: str) -> List[Dict[str, Any]]:
        client = await self.repository.get_client(tenant_id)
        if not client:
            raise NotFoundError("Tenant not found.")
        return await self.repository.list_prompt_versions(tenant_id)

    async def save_new_version(
        self,
        tenant_id: str,
        system_prompt: str,
        fallback_answer: Optional[str] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        client = await self.repository.get_client(tenant_id)
        if not client:
            raise NotFoundError("Tenant not found.")
        cleaned_prompt = (system_prompt or "").strip()
        if not cleaned_prompt:
            raise ValidationError("System prompt cannot be empty.")
        resolved_fallback = (
            fallback_answer.strip()
            if fallback_answer and fallback_answer.strip()
            else client.get("fallback_answer", "I do not have enough verified information to answer that.")
        )

        await self.repository.update_client(
            tenant_id,
            {"system_prompt": cleaned_prompt, "fallback_answer": resolved_fallback},
        )
        version = await self.repository.create_prompt_version(
            tenant_id=tenant_id,
            system_prompt=cleaned_prompt,
            fallback_answer=resolved_fallback,
            note=note or "Saved from prompt manager",
        )
        await self.repository.record_audit(
            actor="admin",
            action="prompt.version_saved",
            tenant_id=tenant_id,
            details={"version_number": version["version_number"]},
        )
        return version

    async def restore_version(self, tenant_id: str, version_number: int) -> Dict[str, Any]:
        target = await self.repository.get_prompt_version(tenant_id, version_number)
        if not target:
            raise NotFoundError(f"Prompt version {version_number} not found for this tenant.")
        return await self.save_new_version(
            tenant_id=tenant_id,
            system_prompt=target["system_prompt"],
            fallback_answer=target["fallback_answer"],
            note=f"Restored from version v{version_number}",
        )

    async def preview_prompt(
        self,
        tenant_id: str,
        test_message: str,
        draft_system_prompt: Optional[str] = None,
        draft_fallback_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Preview how the tenant's assistant responds to a sample question with current knowledge."""
        client = await self.repository.get_client(tenant_id)
        if not client:
            raise NotFoundError("Tenant not found.")
        sys_prompt = (draft_system_prompt or client["system_prompt"]).strip()
        fallback = (draft_fallback_answer or client["fallback_answer"]).strip()

        rag_res = await self.rag.retrieve_and_assess(
            tenant_id=tenant_id,
            question=test_message,
            conversation_history=[],
            top_k=int(client.get("top_k", 5)),
            min_similarity=float(client.get("min_similarity", 0.25)),
        )
        chunks = rag_res["chunks"]
        support_level = rag_res["support_level"]
        if not chunks or support_level == "UNKNOWN":
            return {
                "answer": fallback,
                "support_level": "UNKNOWN",
                "grounding_passed": False,
                "sources": [],
            }

        candidate = await self.llm.generate_answer(
            system_prompt=sys_prompt,
            question=test_message,
            evidence_chunks=chunks,
            conversation_history=[],
            language=client.get("language", "en"),
            temperature=float(client.get("temperature", 0.2)),
            model_override=client.get("model_override"),
        )
        grounding_res = self.grounding.validate_answer(
            candidate_answer=candidate,
            evidence_chunks=chunks,
            support_level=support_level,
            fallback_answer=fallback,
        )
        return {
            "answer": grounding_res.validated_answer,
            "support_level": grounding_res.support_level,
            "grounding_passed": grounding_res.passed,
            "reasons": grounding_res.reasons,
            "sources": [
                {
                    "source": c["source_filename"],
                    "chunkIndex": c["chunk_index"],
                    "score": c.get("combined_score", c.get("similarity", 0.0)),
                }
                for c in chunks
            ],
        }
