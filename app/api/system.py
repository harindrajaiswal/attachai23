"""System health, readiness, and operational metrics routes (/health, /ready, /metrics).

Implements Section 19:
- GET /health: Safe application health summary
- GET /ready: Verifies database connectivity, required schema tables, vector RPC, and storage bucket
  (Temporary LLM outages do NOT take the web service out of readiness)
- GET /metrics: Operational metrics restricted to authenticated administrator calls
"""

from typing import Any, Dict, Optional, Tuple
from app.core.dependencies import ServiceContainer
from app.middleware.authentication import verify_admin_credentials


class SystemApiHandler:
    """Handler for /health, /ready, and /metrics."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    async def get_health(self) -> Tuple[int, Dict[str, Any]]:
        return 200, {
            "status": "ok",
            "service": "orbit-ai",
            "environment": self.container.settings.app_env,
            "version": "1.0.0",
        }

    async def get_ready(self) -> Tuple[int, Dict[str, Any]]:
        """Check operational dependencies required to accept traffic (DB, Vector Schema, Storage)."""
        db_health = await self.container.repository.check_health()
        storage_health = await self.container.storage_provider.check_health()
        vector_health = await self.container.vector_provider.check_health()
        emb_health = await self.container.embedding_provider.check_health()
        llm_health = await self.container.llm_provider.check_health()

        # Critical dependencies for readiness: Database, Vector schema/RPC, Storage, and Embeddings
        # Note per Section 19: Temporary LLM outages do not take the entire web service out of readiness
        critical_ready = bool(
            db_health.get("ok")
            and storage_health.get("ok")
            and vector_health.get("ok")
            and emb_health.get("ok")
        )
        status_code = 200 if critical_ready else 503
        return status_code, {
            "ready": critical_ready,
            "status": "ready" if critical_ready else "not_ready",
            "checks": {
                "database": {
                    "ok": bool(db_health.get("ok")),
                    "provider": db_health.get("provider"),
                    "error": db_health.get("error"),
                },
                "storage": {
                    "ok": bool(storage_health.get("ok")),
                    "provider": storage_health.get("provider"),
                    "error": storage_health.get("error"),
                },
                "vector": {
                    "ok": bool(vector_health.get("ok")),
                    "provider": vector_health.get("provider"),
                    "dimensions": vector_health.get("dimensions"),
                    "error": vector_health.get("error"),
                },
                "embeddings": {
                    "ok": bool(emb_health.get("ok")),
                    "dimensions": emb_health.get("dimensions"),
                },
                "llm": {
                    "ok": bool(llm_health.get("ok")),
                    "provider_type": llm_health.get("provider_type"),
                    "degraded_non_blocking": not bool(llm_health.get("ok")),
                },
            },
        }

    async def get_metrics(
        self,
        admin_header_key: Optional[str] = None,
        session_cookie: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Return operational metrics (requires admin authentication to protect infrastructure telemetry)."""
        verify_admin_credentials(
            container=self.container,
            admin_header_key=admin_header_key,
            session_cookie=session_cookie,
            method="GET",
        )
        analytics = await self.container.analytics_service.get_dashboard_analytics()
        summary = analytics["summary"]
        return 200, {
            "service": "orbit-ai",
            "metrics": {
                "orbit_tenants_total": summary["total_tenants"],
                "orbit_tenants_active": summary["active_tenants"],
                "orbit_conversations_total": summary["total_conversations"],
                "orbit_messages_total": summary["total_messages"],
                "orbit_leads_total": summary["total_leads"],
                "orbit_knowledge_files_ready": summary["ready_knowledge_files"],
                "orbit_knowledge_chunks_total": summary["total_knowledge_chunks"],
                "orbit_requests_total": summary["total_requests"],
                "orbit_usage_units_total": summary["total_usage_units"],
                "orbit_latency_avg_ms": summary["avg_latency_ms"],
                "orbit_latency_p95_ms": summary["p95_latency_ms"],
            },
        }
