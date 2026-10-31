"""Client / Tenant management service.

Implements Section 5 Multi-Tenant Management:
1. Create a tenant
2. List all tenants
3. View tenant details
4. Edit tenant settings
5. Enable or disable a tenant
6. Delete a tenant (cleaning up conversations, leads, knowledge metadata, vectors, storage objects, notifications)
7. Duplicate a tenant
8. Generate a new API key
9. Rotate a tenant API key
10. Enable or disable an existing key
11. Add allowed website domains
12. Remove allowed website domains
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
)
from app.core.security import constant_time_equals, generate_tenant_api_key, hash_api_key
from app.domain.models import WidgetConfig
from app.providers.storage import BaseStorageProvider
from app.repositories.base import BaseRepository
from app.utils.validators import (
    domain_matches,
    normalize_domain,
    normalize_slug,
    validate_hex_color,
)


class ClientService:
    """Service for multi-tenant lifecycle, API keys, domains, and public widget config."""

    def __init__(self, repository: BaseRepository, storage: BaseStorageProvider) -> None:
        self.repository = repository
        self.storage = storage

    async def create_tenant(
        self,
        data: Dict[str, Any],
        generate_initial_key: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Create a new tenant and optionally generate its first API key (returned raw once)."""
        business_name = str(data.get("business_name") or "").strip()
        if not business_name:
            raise ValidationError("Business name is required.")

        raw_slug = str(data.get("slug") or business_name).strip()
        slug = normalize_slug(raw_slug)
        existing_slug = await self.repository.get_client_by_slug(slug)
        if existing_slug:
            raise ValidationError(f"Tenant slug '{slug}' is already in use.")

        raw_domains = data.get("domains") or []
        if isinstance(raw_domains, str):
            raw_domains = [d.strip() for d in raw_domains.split(",") if d.strip()]
        normalized_domains = [normalize_domain(d) for d in raw_domains if str(d).strip()]

        widget_raw = dict(data.get("widget_config") or {})
        if "primary_color" in widget_raw and widget_raw["primary_color"]:
            widget_raw["primary_color"] = validate_hex_color(str(widget_raw["primary_color"]))
        widget_cfg = WidgetConfig.from_dict(widget_raw).to_dict()

        created = await self.repository.create_client(
            {
                **data,
                "business_name": business_name,
                "slug": slug,
                "company_name": str(data.get("company_name") or business_name).strip(),
                "assistant_name": str(data.get("assistant_name") or "AI Assistant").strip(),
                "domains": normalized_domains,
                "widget_config": widget_cfg,
            }
        )

        raw_key: Optional[str] = None
        if generate_initial_key:
            raw_key, _ = await self.generate_api_key(created["id"], name="Initial Widget Key")

        await self.repository.record_audit(
            actor="admin",
            action="tenant.created",
            tenant_id=created["id"],
            details={"slug": slug, "business_name": business_name},
        )
        return created, raw_key

    async def list_tenants(self) -> List[Dict[str, Any]]:
        tenants = await self.repository.list_clients()
        for t in tenants:
            t["api_keys"] = await self.repository.list_api_keys(t["id"])
        return tenants

    async def get_tenant(self, tenant_id: str) -> Dict[str, Any]:
        tenant = await self.repository.get_client(tenant_id)
        if not tenant:
            raise NotFoundError(f"Tenant '{tenant_id}' not found.")
        tenant["api_keys"] = await self.repository.list_api_keys(tenant_id)
        return tenant

    async def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        existing = await self.get_tenant(tenant_id)
        payload = dict(updates)
        if "slug" in payload and payload["slug"]:
            new_slug = normalize_slug(str(payload["slug"]))
            if new_slug != existing["slug"]:
                conflict = await self.repository.get_client_by_slug(new_slug)
                if conflict and conflict["id"] != tenant_id:
                    raise ValidationError(f"Slug '{new_slug}' is already in use.")
            payload["slug"] = new_slug

        if "domains" in payload and payload["domains"] is not None:
            doms = payload["domains"]
            if isinstance(doms, str):
                doms = [d.strip() for d in doms.split(",") if d.strip()]
            payload["domains"] = [normalize_domain(d) for d in doms if str(d).strip()]

        if "widget_config" in payload and isinstance(payload["widget_config"], dict):
            w_cfg = dict(existing.get("widget_config") or {})
            w_cfg.update(payload["widget_config"])
            if "primary_color" in w_cfg and w_cfg["primary_color"]:
                w_cfg["primary_color"] = validate_hex_color(str(w_cfg["primary_color"]))
            payload["widget_config"] = WidgetConfig.from_dict(w_cfg).to_dict()

        updated = await self.repository.update_client(tenant_id, payload)
        if not updated:
            raise NotFoundError(f"Tenant '{tenant_id}' not found.")
        await self.repository.record_audit(
            actor="admin",
            action="tenant.updated",
            tenant_id=tenant_id,
            details={"updated_fields": list(payload.keys())},
        )
        return await self.get_tenant(tenant_id)

    async def set_tenant_enabled(self, tenant_id: str, enabled: bool) -> Dict[str, Any]:
        return await self.update_tenant(tenant_id, {"enabled": enabled})

    async def duplicate_tenant(
        self, source_tenant_id: str, new_business_name: Optional[str] = None, new_slug: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        source = await self.get_tenant(source_tenant_id)
        name = (new_business_name or f"{source['business_name']} (Copy)").strip()
        base_slug = normalize_slug(new_slug or f"{source['slug']}-copy")
        candidate_slug = base_slug
        suffix = 2
        while await self.repository.get_client_by_slug(candidate_slug):
            candidate_slug = f"{base_slug}-{suffix}"
            suffix += 1

        dup_data = {
            "business_name": name,
            "slug": candidate_slug,
            "company_name": source.get("company_name", name),
            "assistant_name": source.get("assistant_name", "AI Assistant"),
            "enabled": True,
            "system_prompt": source.get("system_prompt"),
            "fallback_answer": source.get("fallback_answer"),
            "welcome_message": source.get("welcome_message"),
            "language": source.get("language", "en"),
            "model_override": source.get("model_override"),
            "temperature": source.get("temperature", 0.2),
            "top_k": source.get("top_k", 5),
            "min_similarity": source.get("min_similarity", 0.25),
            "lead_capture_enabled": source.get("lead_capture_enabled", True),
            "lead_settings": source.get("lead_settings"),
            "notification_settings": source.get("notification_settings"),
            "widget_config": source.get("widget_config"),
            "analytics_settings": source.get("analytics_settings"),
            "domains": source.get("domains", []),
        }
        duplicated, raw_key = await self.create_tenant(dup_data, generate_initial_key=True)
        assert raw_key is not None
        return duplicated, raw_key

    async def delete_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """Safely delete a tenant and clean up all private storage objects and database records."""
        existing = await self.get_tenant(tenant_id)
        deleted_objects = await self.storage.delete_tenant_objects(tenant_id)
        deleted_db = await self.repository.delete_client(tenant_id)
        await self.repository.record_audit(
            actor="admin",
            action="tenant.deleted",
            tenant_id=None,
            details={
                "deleted_tenant_id": tenant_id,
                "slug": existing["slug"],
                "deleted_storage_objects": deleted_objects,
            },
        )
        return {
            "deleted": deleted_db,
            "tenant_id": tenant_id,
            "deleted_storage_objects": deleted_objects,
        }

    # --- API Key Management ---
    async def generate_api_key(self, tenant_id: str, name: str = "Widget Key") -> Tuple[str, Dict[str, Any]]:
        await self.get_tenant(tenant_id)
        raw_key, key_prefix, key_hash = generate_tenant_api_key()
        key_rec = await self.repository.create_api_key(
            tenant_id=tenant_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
            name=name,
        )
        await self.repository.record_audit(
            actor="admin",
            action="api_key.generated",
            tenant_id=tenant_id,
            details={"key_id": key_rec["id"], "key_prefix": key_prefix},
        )
        return raw_key, key_rec

    async def rotate_api_key(self, tenant_id: str) -> Tuple[str, Dict[str, Any]]:
        """Revoke all existing keys for tenant_id and generate a fresh API key."""
        await self.get_tenant(tenant_id)
        revoked_count = await self.repository.revoke_tenant_api_keys(tenant_id)
        raw_key, key_rec = await self.generate_api_key(tenant_id, name="Rotated Widget Key")
        await self.repository.record_audit(
            actor="admin",
            action="api_key.rotated",
            tenant_id=tenant_id,
            details={"revoked_keys": revoked_count, "new_key_id": key_rec["id"]},
        )
        return raw_key, key_rec

    async def set_api_key_status(self, tenant_id: str, key_id: str, enabled: bool) -> Dict[str, Any]:
        updated = await self.repository.set_api_key_enabled(tenant_id, key_id, enabled)
        if not updated:
            raise NotFoundError("API key not found for this tenant.")
        return updated

    # --- Domain Management ---
    async def add_allowed_domain(self, tenant_id: str, domain: str) -> List[str]:
        await self.get_tenant(tenant_id)
        normalized = normalize_domain(domain)
        await self.repository.add_domain(tenant_id, normalized)
        return await self.repository.list_domains(tenant_id)

    async def remove_allowed_domain(self, tenant_id: str, domain: str) -> List[str]:
        await self.get_tenant(tenant_id)
        normalized = normalize_domain(domain)
        await self.repository.remove_domain(tenant_id, normalized)
        return await self.repository.list_domains(tenant_id)

    # --- Public Widget Authentication & Safe Config ---
    async def authenticate_public_request(
        self,
        raw_api_key: Optional[str],
        origin_header: Optional[str] = None,
        expected_client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Authenticate a public widget request using x-api-key, constant-time hash check, and domain allowlist."""
        if not raw_api_key or not raw_api_key.strip():
            raise AuthenticationError("Missing x-api-key header.")

        computed_hash = hash_api_key(raw_api_key.strip())
        key_rec = await self.repository.get_api_key_by_hash(computed_hash)
        if not key_rec or not constant_time_equals(key_rec.get("key_hash", ""), computed_hash):
            raise AuthenticationError("Invalid tenant API key.")

        if not key_rec.get("enabled", False):
            raise AuthenticationError("Tenant API key has been disabled or revoked.")

        tenant_id = key_rec["tenant_id"]
        if expected_client_id and expected_client_id.strip() and expected_client_id.strip() != tenant_id:
            raise AuthorizationError("Cross-tenant access denied: API key does not belong to requested client ID.")

        tenant = await self.repository.get_client(tenant_id)
        if not tenant or not tenant.get("enabled", False):
            raise AuthorizationError("Tenant account is disabled or unavailable.")

        # Enforce domain allowlist when tenant has configured domains and caller sends Origin/Referer
        allowed_domains: List[str] = tenant.get("domains") or []
        if allowed_domains and origin_header and origin_header.strip():
            try:
                caller_host = normalize_domain(origin_header)
            except ValidationError as exc:
                raise AuthorizationError("Invalid request Origin header.") from exc

            # Always permit localhost/127.0.0.1/testserver when in local dev or if explicitly listed
            if not any(domain_matches(caller_host, d) for d in allowed_domains):
                raise AuthorizationError(
                    f"Origin domain '{caller_host}' is not authorized for this tenant."
                )

        return tenant

    def build_safe_public_config(self, tenant: Dict[str, Any]) -> Dict[str, Any]:
        """Return strictly safe tenant widget configuration (never secrets, prompts, or key hashes)."""
        w_cfg = WidgetConfig.from_dict(tenant.get("widget_config")).to_dict()
        lead_cfg = tenant.get("lead_settings") or {}
        return {
            "clientId": tenant["id"],
            "businessName": tenant["business_name"],
            "companyName": tenant.get("company_name") or tenant["business_name"],
            "assistantName": tenant.get("assistant_name") or "AI Assistant",
            "welcomeMessage": tenant.get("welcome_message") or "Hello! How can I help you today?",
            "language": tenant.get("language") or "en",
            "leadCaptureEnabled": bool(tenant.get("lead_capture_enabled", True)),
            "leadSettings": {
                "requireEmail": bool(lead_cfg.get("require_email", True)),
                "requirePhone": bool(lead_cfg.get("require_phone", False)),
                "customPrompt": str(
                    lead_cfg.get("custom_prompt") or "Leave your contact details so our team can follow up."
                ),
            },
            "theme": {
                "primaryColor": w_cfg["primary_color"],
                "themeMode": w_cfg["theme_mode"],
                "position": w_cfg["position"],
                "width": w_cfg["width"],
                "height": w_cfg["height"],
                "borderRadius": w_cfg["border_radius"],
                "logoUrl": w_cfg["logo_url"],
                "avatarUrl": w_cfg["avatar_url"],
                "launcherIcon": w_cfg["launcher_icon"],
                "inputPlaceholder": w_cfg["input_placeholder"],
                "showBranding": w_cfg["show_branding"],
                "suggestedQuestions": w_cfg["suggested_questions"],
            },
        }
