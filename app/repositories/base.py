"""Abstract repository interface enforcing tenant-scoped persistence across all 14 tables."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseRepository(ABC):
    """Contract for SupabaseRepository and SQLiteRepository."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database tables or verify schema readiness."""

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Verify database connectivity and schema compatibility."""

    # --- Clients / Tenants ---
    @abstractmethod
    async def create_client(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def list_clients(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_client(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_client_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def update_client(self, tenant_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_client(self, tenant_id: str) -> bool:
        pass

    # --- API Keys ---
    @abstractmethod
    async def create_api_key(
        self, tenant_id: str, key_prefix: str, key_hash: str, name: str = "Default Widget Key"
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def list_api_keys(self, tenant_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def set_api_key_enabled(self, tenant_id: str, key_id: str, enabled: bool) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def revoke_tenant_api_keys(self, tenant_id: str) -> int:
        pass

    # --- Domains ---
    @abstractmethod
    async def add_domain(self, tenant_id: str, domain: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def remove_domain(self, tenant_id: str, domain: str) -> bool:
        pass

    @abstractmethod
    async def list_domains(self, tenant_id: str) -> List[str]:
        pass

    # --- Prompt Versions ---
    @abstractmethod
    async def create_prompt_version(
        self, tenant_id: str, system_prompt: str, fallback_answer: str, note: str = ""
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def list_prompt_versions(self, tenant_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_prompt_version(self, tenant_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        pass

    # --- Knowledge Files & Vectors ---
    @abstractmethod
    async def create_knowledge_file(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def get_knowledge_file(self, tenant_id: str, file_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def find_knowledge_file_by_name(self, tenant_id: str, original_filename: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def list_knowledge_files(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def update_knowledge_file(
        self, tenant_id: str, file_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_knowledge_file(self, tenant_id: str, file_id: str) -> bool:
        pass

    @abstractmethod
    async def insert_staging_vectors(
        self, tenant_id: str, source_id: str, version: int, chunks: List[Dict[str, Any]]
    ) -> int:
        pass

    @abstractmethod
    async def count_vectors_for_version(self, tenant_id: str, source_id: str, version: int) -> int:
        pass

    @abstractmethod
    async def activate_vector_version(self, tenant_id: str, source_id: str, active_version: int) -> None:
        pass

    @abstractmethod
    async def delete_vectors_for_version(self, tenant_id: str, source_id: str, version: int) -> None:
        pass

    @abstractmethod
    async def get_active_chunks_paginated(
        self, tenant_id: str, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        pass

    # --- Conversations & Messages ---
    @abstractmethod
    async def get_or_create_conversation(
        self, tenant_id: str, session_id: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def get_conversation(self, tenant_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def list_conversations(
        self, tenant_id: Optional[str] = None, search: str = "", limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def update_conversation_summary(self, tenant_id: str, conversation_id: str, summary: str) -> None:
        pass

    async def update_conversation_metadata(
        self, tenant_id: str, conversation_id: str, metadata: Dict[str, Any]
    ) -> None:
        pass

    async def delete_conversation(self, tenant_id: str, conversation_id: str) -> bool:
        return False

    @abstractmethod
    async def add_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    async def get_message(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        return None

    @abstractmethod
    async def list_messages(
        self, tenant_id: str, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        pass

    # --- Message Feedback ---
    async def upsert_message_feedback(
        self,
        tenant_id: str,
        conversation_id: str,
        message_id: str,
        feedback_type: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def get_message_feedback(
        self, tenant_id: str, message_id: str
    ) -> Optional[Dict[str, Any]]:
        return None

    async def list_conversation_feedback(
        self, tenant_id: str, conversation_id: str
    ) -> List[Dict[str, Any]]:
        return []

    # --- Leads & Lead Workflow ---
    @abstractmethod
    async def upsert_lead_idempotent(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Insert lead or return existing lead if (tenant_id, idempotency_key) matches.
        Returns (lead_record, created_new).
        """

    @abstractmethod
    async def get_lead(self, tenant_id: str, lead_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def list_leads(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        search: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def update_lead_workflow(
        self,
        tenant_id: str,
        lead_id: str,
        status: Optional[str] = None,
        assignee: Optional[str] = None,
        internal_notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass

    # --- Notifications / Outbox ---
    @abstractmethod
    async def create_notification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def update_notification(self, notification_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def list_notifications(
        self, tenant_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        pass

    # --- Usage Logs, Audit Logs & Settings ---
    @abstractmethod
    async def record_usage(self, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def list_usage_logs(
        self, tenant_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def record_audit(self, actor: str, action: str, tenant_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    async def list_audit_logs(self, tenant_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_setting(self, key: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def set_setting(self, key: str, value: Dict[str, Any]) -> Dict[str, Any]:
        pass
