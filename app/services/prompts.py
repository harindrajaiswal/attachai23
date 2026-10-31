"""Prompt versioning, restoration, and live assistant preview service."""

from typing import Any, Dict, List, Optional
from app.core.exceptions import NotFoundError, ValidationError
from app.repositories.base import BaseRepository
from app.services.advanced_rag import AdvancedRagService
from app.services.grounding import GroundingService
from app.providers.llm import BaseLlmProvider


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
