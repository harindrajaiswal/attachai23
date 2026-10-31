"""Notification outbox service for inspecting and retrying notifications."""

from typing import Any, Dict, List, Optional
from app.providers.notifications import NotificationProvider
from app.repositories.base import BaseRepository


class NotificationService:
    """Service wrapper around persistent notification records and retry operations."""

    def __init__(self, repository: BaseRepository, provider: NotificationProvider) -> None:
        self.repository = repository
        self.provider = provider

    async def list_notifications(
        self, tenant_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        return await self.repository.list_notifications(tenant_id=tenant_id, status=status, limit=limit)

    async def retry_failed(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        count = await self.provider.retry_failed_notifications(tenant_id=tenant_id)
        return {"retried": count}
