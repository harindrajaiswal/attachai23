"""Lead capture, idempotency, CRM workflow, and CSV export service.

Implements Section 14 & Fixes Defects #11 and #12:
- Validates lead contact fields and associates lead with authenticated tenant
- Validates conversation ownership if conversationId is supplied
- Computes or uses a deterministic idempotency key so retries never create duplicate leads
- Persists lead record BEFORE attempting notification delivery via the outbox provider
"""

import hashlib
from typing import Any, Dict, List, Optional
from app.core.exceptions import AuthorizationError, NotFoundError, ValidationError
from app.domain.enums import LeadStatus
from app.providers.notifications import NotificationProvider
from app.repositories.base import BaseRepository
from app.utils.csv_parser import generate_csv_string
from app.utils.validators import validate_email_address

VALID_LEAD_STATES = {s.value for s in LeadStatus}


class LeadService:
    """Manages tenant-isolated lead capture, workflow state transitions, and CSV export."""

    def __init__(self, repository: BaseRepository, notification_provider: NotificationProvider) -> None:
        self.repository = repository
        self.notifications = notification_provider

    def _compute_idempotency_key(
        self,
        tenant_id: str,
        email: str,
        name: str,
        conversation_id: Optional[str],
        explicit_key: Optional[str],
    ) -> str:
        if explicit_key and explicit_key.strip():
            return explicit_key.strip()[:120]
        raw = f"{tenant_id}:{email.lower().strip()}:{name.lower().strip()}:{conversation_id or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]

    async def submit_lead(
        self,
        tenant: Dict[str, Any],
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a customer lead idempotently and enqueue an outbox notification."""
        tenant_id = tenant["id"]
        name = str(payload.get("name") or "").strip()
        if not name or len(name) < 2:
            raise ValidationError("Customer name is required (minimum 2 characters).")

        email = validate_email_address(str(payload.get("email") or ""))
        phone = str(payload.get("phone") or "").strip()
        lead_settings = tenant.get("lead_settings") or {}
        if lead_settings.get("require_phone") and not phone:
            raise ValidationError("Phone number is required by this tenant.")

        requirement = str(payload.get("requirement") or "").strip()[:500]
        message = str(payload.get("message") or "").strip()[:2000]
        conversation_id = str(payload.get("conversationId") or payload.get("conversation_id") or "").strip() or None

        if conversation_id:
            conv = await self.repository.get_conversation(tenant_id, conversation_id)
            if not conv:
                raise AuthorizationError("Cannot associate lead with a conversation from another tenant.")

        idem_key = self._compute_idempotency_key(
            tenant_id=tenant_id,
            email=email,
            name=name,
            conversation_id=conversation_id,
            explicit_key=idempotency_key or payload.get("idempotencyKey"),
        )

        lead_record, created_new = await self.repository.upsert_lead_idempotent(
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "idempotency_key": idem_key,
                "name": name,
                "email": email,
                "phone": phone,
                "requirement": requirement,
                "message": message,
            }
        )

        if created_new:
            notif_cfg = tenant.get("notification_settings") or {}
            recipients = notif_cfg.get("notify_emails") or []
            recipient_str = ", ".join(recipients) if isinstance(recipients, list) else str(recipients)
            notif = await self.notifications.enqueue_and_dispatch(
                tenant_id=tenant_id,
                lead_id=lead_record["id"],
                recipient=recipient_str or "outbox@orbit-ai.local",
                subject=f"New Lead for {tenant.get('business_name', 'Tenant')}: {name}",
                payload={
                    "lead_id": lead_record["id"],
                    "tenant_id": tenant_id,
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "requirement": requirement,
                    "message": message,
                    "conversation_id": conversation_id,
                },
            )
            lead_record["notification_status"] = notif.get("status", "pending")
            await self.repository.record_usage(
                {
                    "tenant_id": tenant_id,
                    "operation": "lead_capture",
                    "request_count": 1,
                    "usage_units": 1,
                    "latency_ms": 5,
                    "support_level": "SUPPORTED",
                }
            )

        lead_record["createdNew"] = created_new
        return lead_record

    async def list_leads(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        search: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if status and status not in VALID_LEAD_STATES:
            raise ValidationError(f"Invalid lead status '{status}'. Allowed: {', '.join(sorted(VALID_LEAD_STATES))}")
        return await self.repository.list_leads(
            tenant_id=tenant_id,
            status=status,
            search=search,
            limit=limit,
            offset=offset,
        )

    async def get_lead_detail(self, tenant_id: str, lead_id: str) -> Dict[str, Any]:
        lead = await self.repository.get_lead(tenant_id, lead_id)
        if not lead:
            raise NotFoundError("Lead not found for this tenant.")
        return lead

    async def update_lead(
        self,
        tenant_id: str,
        lead_id: str,
        status: Optional[str] = None,
        assignee: Optional[str] = None,
        internal_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        if status is not None and status not in VALID_LEAD_STATES:
            raise ValidationError(
                f"Invalid lead status '{status}'. Allowed: new, contacted, qualified, won, lost."
            )
        updated = await self.repository.update_lead_workflow(
            tenant_id=tenant_id,
            lead_id=lead_id,
            status=status,
            assignee=assignee,
            internal_notes=internal_notes,
        )
        if not updated:
            raise NotFoundError("Lead not found for this tenant.")
        await self.repository.record_audit(
            actor="admin",
            action="lead.updated",
            tenant_id=tenant_id,
            details={"lead_id": lead_id, "status": updated.get("status"), "assignee": updated.get("assignee")},
        )
        return updated

    async def export_leads_csv(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        search: str = "",
    ) -> str:
        leads = await self.list_leads(tenant_id=tenant_id, status=status, search=search, limit=5000, offset=0)
        headers = [
            "id",
            "tenant_id",
            "name",
            "email",
            "phone",
            "requirement",
            "message",
            "status",
            "assignee",
            "internal_notes",
            "notification_status",
            "conversation_id",
            "created_at",
        ]
        return generate_csv_string(headers, leads)
