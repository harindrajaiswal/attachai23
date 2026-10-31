"""Public Widget API routes (/v1/config, /v1/chat, /v1/chat/stream, /v1/leads).

Implements Section 8 Public API Contract:
- GET  /v1/config
- POST /v1/chat
- POST /v1/chat/stream (Server-Sent Events, grounded before emission)
- POST /v1/leads (Idempotent lead submission)
"""

from typing import Any, AsyncGenerator, Dict, Optional, Tuple
from app.api.schemas import ChatRequestSchema, LeadSubmitSchema
from app.core.dependencies import ServiceContainer
from app.middleware.authentication import verify_public_tenant_request


class PublicApiHandler:
    """Framework-agnostic handler for /v1/* routes used by FastAPI and built-in ASGI server."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    async def get_config(
        self,
        api_key: Optional[str],
        origin: Optional[str] = None,
        client_id: Optional[str] = None,
        client_ip: str = "127.0.0.1",
    ) -> Dict[str, Any]:
        tenant = await verify_public_tenant_request(
            container=self.container,
            api_key_header=api_key,
            origin_header=origin,
            client_id_header=client_id,
            client_ip=client_ip,
        )
        return self.container.client_service.build_safe_public_config(tenant)

    async def post_chat(
        self,
        body: Dict[str, Any],
        api_key: Optional[str],
        origin: Optional[str] = None,
        client_id: Optional[str] = None,
        client_ip: str = "127.0.0.1",
    ) -> Dict[str, Any]:
        tenant = await verify_public_tenant_request(
            container=self.container,
            api_key_header=api_key,
            origin_header=origin,
            client_id_header=client_id,
            client_ip=client_ip,
        )
        schema = ChatRequestSchema.from_dict(body)
        return await self.container.chat_service.process_chat(
            tenant=tenant,
            message=schema.message,
            session_id=schema.sessionId,
            conversation_id=schema.conversationId,
        )

    async def post_chat_stream(
        self,
        body: Dict[str, Any],
        api_key: Optional[str],
        origin: Optional[str] = None,
        client_id: Optional[str] = None,
        client_ip: str = "127.0.0.1",
    ) -> AsyncGenerator[str, None]:
        tenant = await verify_public_tenant_request(
            container=self.container,
            api_key_header=api_key,
            origin_header=origin,
            client_id_header=client_id,
            client_ip=client_ip,
        )
        schema = ChatRequestSchema.from_dict(body)
        async for chunk in self.container.chat_service.stream_validated_chat(
            tenant=tenant,
            message=schema.message,
            session_id=schema.sessionId,
            conversation_id=schema.conversationId,
        ):
            yield chunk

    async def post_leads(
        self,
        body: Dict[str, Any],
        api_key: Optional[str],
        origin: Optional[str] = None,
        client_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        client_ip: str = "127.0.0.1",
    ) -> Tuple[int, Dict[str, Any]]:
        tenant = await verify_public_tenant_request(
            container=self.container,
            api_key_header=api_key,
            origin_header=origin,
            client_id_header=client_id,
            client_ip=client_ip,
        )
        schema = LeadSubmitSchema.from_dict(body)
        lead = await self.container.lead_service.submit_lead(
            tenant=tenant,
            payload=schema.to_dict(),
            idempotency_key=idempotency_key or schema.idempotencyKey,
        )
        status_code = 201 if lead.get("createdNew") else 200
        return status_code, lead
