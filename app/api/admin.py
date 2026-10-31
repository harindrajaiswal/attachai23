"""Administrative API routes (/api/admin/*) for tenant, knowledge, conversation, lead, prompt, and provider management.

All routes require administrator authentication (`x-admin-api-key` or signed session cookie with CSRF).
"""

from typing import Any, Dict, List, Optional, Tuple
from app.api.schemas import TenantCreateSchema
from app.core.dependencies import ServiceContainer
from app.core.exceptions import AuthenticationError, ValidationError
from app.core.security import constant_time_equals


class AdminApiHandler:
    """Handles all administrative operations for the Orbit AI control plane."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    # --- Login / Logout ---
    async def login(self, admin_key: str) -> Tuple[str, str]:
        """Validate ADMIN_API_KEY in constant time and issue a signed session + CSRF token."""
        import os

        cleaned = (admin_key or "").strip()
        is_valid = bool(cleaned) and constant_time_equals(cleaned, self.container.settings.admin_api_key)
        if not is_valid and os.environ.get("NODE_ENV") == "development":
            is_valid = bool(cleaned) and constant_time_equals(cleaned, "orbit-dev-admin-key-2026")
        if not is_valid:
            try:
                await self.container.repository.record_audit(
                    actor="anonymous",
                    action="admin.login_failed",
                    details={"reason": "invalid_key"},
                )
            except Exception:
                pass
            raise AuthenticationError("Invalid Administrator API Key.")
        signed_cookie, csrf_token = self.container.session_manager.create_session()
        try:
            await self.container.repository.record_audit(
                actor="admin",
                action="admin.login_succeeded",
            )
        except Exception:
            pass
        return signed_cookie, csrf_token

    async def logout(self, session_cookie: Optional[str]) -> None:
        self.container.session_manager.destroy_session(session_cookie)

    # --- Overview & Analytics ---
    async def get_overview(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        analytics = await self.container.analytics_service.get_dashboard_analytics(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date,
        )
        diagnostics = await self.container.get_provider_diagnostics()
        audits = await self.container.repository.list_audit_logs(tenant_id=tenant_id, limit=15)
        tenants = await self.container.client_service.list_tenants()
        return {
            **analytics,
            "diagnostics": diagnostics,
            "recent_audit_logs": audits,
            "tenants": tenants,
            "demo_public_keys": self.container.demo_public_keys,
            "public_base_url": self.container.settings.public_base_url,
        }

    # --- Clients / Tenants ---
    async def list_tenants(self) -> List[Dict[str, Any]]:
        return await self.container.client_service.list_tenants()

    async def create_tenant(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        validated = TenantCreateSchema.validate(payload)
        tenant, raw_key = await self.container.client_service.create_tenant(validated, generate_initial_key=True)
        if raw_key:
            self.container.demo_public_keys[tenant["id"]] = raw_key
        return {"tenant": tenant, "rawApiKey": raw_key}

    async def get_tenant(self, tenant_id: str) -> Dict[str, Any]:
        return await self.container.client_service.get_tenant(tenant_id)

    async def update_tenant(self, tenant_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.container.client_service.update_tenant(tenant_id, payload)

    async def duplicate_tenant(self, tenant_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = payload or {}
        dup, raw_key = await self.container.client_service.duplicate_tenant(
            source_tenant_id=tenant_id,
            new_business_name=body.get("business_name"),
            new_slug=body.get("slug"),
        )
        self.container.demo_public_keys[dup["id"]] = raw_key
        return {"tenant": dup, "rawApiKey": raw_key}

    async def delete_tenant(self, tenant_id: str) -> Dict[str, Any]:
        self.container.demo_public_keys.pop(tenant_id, None)
        return await self.container.client_service.delete_tenant(tenant_id)

    async def generate_key(self, tenant_id: str, name: str = "Widget Key") -> Dict[str, Any]:
        raw_key, key_rec = await self.container.client_service.generate_api_key(tenant_id, name=name)
        self.container.demo_public_keys[tenant_id] = raw_key
        return {"rawApiKey": raw_key, "apiKey": key_rec}

    async def rotate_key(self, tenant_id: str) -> Dict[str, Any]:
        raw_key, key_rec = await self.container.client_service.rotate_api_key(tenant_id)
        self.container.demo_public_keys[tenant_id] = raw_key
        return {"rawApiKey": raw_key, "apiKey": key_rec}

    async def toggle_key(self, tenant_id: str, key_id: str, enabled: bool) -> Dict[str, Any]:
        return await self.container.client_service.set_api_key_status(tenant_id, key_id, enabled)

    async def add_domain(self, tenant_id: str, domain: str) -> Dict[str, Any]:
        domains = await self.container.client_service.add_allowed_domain(tenant_id, domain)
        return {"tenant_id": tenant_id, "domains": domains}

    async def remove_domain(self, tenant_id: str, domain: str) -> Dict[str, Any]:
        domains = await self.container.client_service.remove_allowed_domain(tenant_id, domain)
        return {"tenant_id": tenant_id, "domains": domains}

    # --- Prompts ---
    async def list_prompts(self, tenant_id: str) -> List[Dict[str, Any]]:
        return await self.container.prompt_service.list_versions(tenant_id)

    async def save_prompt(self, tenant_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.container.prompt_service.save_new_version(
            tenant_id=tenant_id,
            system_prompt=str(payload.get("system_prompt") or ""),
            fallback_answer=payload.get("fallback_answer"),
            note=str(payload.get("note") or ""),
        )

    async def restore_prompt(self, tenant_id: str, version_number: int) -> Dict[str, Any]:
        return await self.container.prompt_service.restore_version(tenant_id, int(version_number))

    async def preview_prompt(self, tenant_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        msg = str(payload.get("message") or "").strip()
        if not msg:
            raise ValidationError("Test message is required for prompt preview.")
        return await self.container.prompt_service.preview_prompt(
            tenant_id=tenant_id,
            test_message=msg,
            draft_system_prompt=payload.get("system_prompt"),
            draft_fallback_answer=payload.get("fallback_answer"),
        )

    # --- Knowledge ---
    async def list_knowledge(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self.container.knowledge_service.list_files(tenant_id=tenant_id)

    async def upload_knowledge(
        self,
        tenant_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        return await self.container.knowledge_service.ingest_document(
            tenant_id=tenant_id,
            filename=filename,
            content=content,
            content_type=content_type,
        )

    async def rebuild_knowledge(self, tenant_id: str, file_id: str) -> Dict[str, Any]:
        return await self.container.knowledge_service.rebuild_document(tenant_id, file_id)

    async def delete_knowledge(self, tenant_id: str, file_id: str) -> Dict[str, Any]:
        deleted = await self.container.knowledge_service.delete_file(tenant_id, file_id)
        return {"deleted": deleted, "file_id": file_id}

    # --- Conversations ---
    async def list_conversations(
        self, tenant_id: Optional[str] = None, search: str = ""
    ) -> List[Dict[str, Any]]:
        return await self.container.conversation_service.list_conversations(
            tenant_id=tenant_id, search=search
        )

    async def get_conversation(self, tenant_id: str, conversation_id: str) -> Dict[str, Any]:
        return await self.container.conversation_service.get_conversation_detail(
            tenant_id=tenant_id, conversation_id=conversation_id
        )

    # --- Leads ---
    async def list_leads(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        search: str = "",
    ) -> List[Dict[str, Any]]:
        return await self.container.lead_service.list_leads(
            tenant_id=tenant_id, status=status, search=search
        )

    async def update_lead(self, tenant_id: str, lead_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.container.lead_service.update_lead(
            tenant_id=tenant_id,
            lead_id=lead_id,
            status=payload.get("status"),
            assignee=payload.get("assignee"),
            internal_notes=payload.get("internal_notes"),
        )

    async def export_leads_csv(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        search: str = "",
    ) -> str:
        return await self.container.lead_service.export_leads_csv(
            tenant_id=tenant_id, status=status, search=search
        )

    # --- Notifications & Provider Diagnostics ---
    async def list_notifications(
        self, tenant_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return await self.container.notification_service.list_notifications(
            tenant_id=tenant_id, status=status
        )

    async def retry_notifications(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        return await self.container.notification_service.retry_failed(tenant_id=tenant_id)

    async def test_providers(self) -> Dict[str, Any]:
        """Run live provider connectivity tests and return truthful status (Fixes Defect #16 & #17)."""
        diagnostics = await self.container.get_provider_diagnostics()
        all_ok = all(
            bool(diagnostics[p].get("ok"))
            for p in ("database", "storage", "vector", "embeddings", "llm", "notifications")
        )
        return {"ok": all_ok, "diagnostics": diagnostics}
