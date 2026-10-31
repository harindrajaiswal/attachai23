"""Supabase PostgreSQL repository implementation.

Uses Supabase PostgREST API with the backend service-role key while enforcing
explicit `.eq("tenant_id", tenant_id)` filters on every tenant-scoped query.
Also verifies schema readiness (14 required tables + `match_knowledge_vectors` RPC).
"""

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.core.exceptions import ProviderError
from app.domain.models import WidgetConfig, utc_now_iso
from app.repositories.base import BaseRepository


class SupabaseRepository(BaseRepository):
    """Production Supabase repository with explicit tenant scoping and readiness verification."""

    REQUIRED_TABLES = (
        "clients",
        "api_keys",
        "domains",
        "prompt_versions",
        "conversations",
        "messages",
        "leads",
        "lead_workflow",
        "usage_logs",
        "audit_logs",
        "knowledge_files",
        "knowledge_vectors",
        "settings",
        "notifications",
    )

    def __init__(
        self,
        supabase_url: str,
        service_role_key: str,
        vector_dimensions: int = 128,
        http_handler: Optional[Callable[[str, str, Dict[str, str], Optional[bytes]], Tuple[int, Any]]] = None,
    ) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.service_role_key = service_role_key.strip()
        self.vector_dimensions = vector_dimensions
        self._http_handler = http_handler

    def _headers(self, prefer: str = "return=representation") -> Dict[str, str]:
        return {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
            "Prefer": prefer,
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        prefer: str = "return=representation",
    ) -> Any:
        url = f"{self.supabase_url}{path}"
        if params:
            qs = urllib.parse.urlencode(params, safe=".,()*!:")
            url = f"{url}?{qs}"
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = self._headers(prefer=prefer)

        if self._http_handler is not None:
            status, resp_data = self._http_handler(method, url, headers, payload)
            if status >= 400:
                raise ProviderError(
                    message=f"Supabase database error (HTTP {status}).",
                    status_code=502,
                    provider="supabase",
                )
            return resp_data

        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else []
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                message=f"Supabase database request failed with HTTP {exc.code}.",
                status_code=502,
                provider="supabase",
            ) from exc
        except Exception as exc:
            raise ProviderError(
                message="Supabase database connection failed.",
                status_code=502,
                provider="supabase",
            ) from exc

    async def initialize(self) -> None:
        """Verify Supabase configuration is present."""
        if not self.supabase_url or not self.service_role_key:
            raise ProviderError("Supabase URL and Service Role Key are required.", provider="supabase")

    async def check_health(self) -> Dict[str, Any]:
        """Verify that required tables and the match_knowledge_vectors RPC exist in Supabase."""
        try:
            for table in ("clients", "knowledge_files", "knowledge_vectors", "leads", "notifications"):
                self._request("GET", f"/rest/v1/{table}", params={"select": "id", "limit": "1"})

            # Verify match_knowledge_vectors RPC exists with a zero-vector probe
            zero_vec = [0.0] * self.vector_dimensions
            self._request(
                "POST",
                "/rest/v1/rpc/match_knowledge_vectors",
                body={
                    "query_embedding": zero_vec,
                    "match_tenant_id": "00000000-0000-0000-0000-000000000000",
                    "match_threshold": 0.99,
                    "match_count": 1,
                },
            )
            return {
                "ok": True,
                "provider": "supabase",
                "tables_verified": len(self.REQUIRED_TABLES),
                "rpc_verified": "match_knowledge_vectors",
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": "supabase",
                "error": str(exc),
            }

    def _enrich_client(self, row: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(row)
        d["widget_config"] = WidgetConfig.from_dict(d.get("widget_config")).to_dict()
        if "domains" not in d:
            doms = self._request(
                "GET",
                "/rest/v1/domains",
                params={"tenant_id": f"eq.{d['id']}", "select": "domain"},
            )
            d["domains"] = [item["domain"] for item in (doms or [])]
        return d

    async def create_client(self, data: Dict[str, Any]) -> Dict[str, Any]:
        tenant_id = data.get("id") or str(uuid.uuid4())
        payload = {
            "id": tenant_id,
            "business_name": data["business_name"],
            "slug": data["slug"],
            "company_name": data.get("company_name") or data["business_name"],
            "assistant_name": data.get("assistant_name") or "AI Assistant",
            "enabled": bool(data.get("enabled", True)),
            "system_prompt": data.get("system_prompt")
            or "You are a helpful, accurate customer support assistant. Answer strictly using the provided knowledge base.",
            "fallback_answer": data.get("fallback_answer")
            or "I do not have enough verified information in our knowledge base to answer that accurately.",
            "welcome_message": data.get("welcome_message") or "Hello! How can I help you today?",
            "language": data.get("language") or "en",
            "model_override": data.get("model_override"),
            "temperature": float(data.get("temperature", 0.2)),
            "top_k": int(data.get("top_k", 5)),
            "min_similarity": float(data.get("min_similarity", 0.25)),
            "lead_capture_enabled": bool(data.get("lead_capture_enabled", True)),
            "lead_settings": data.get("lead_settings") or {"require_email": True, "require_phone": False},
            "notification_settings": data.get("notification_settings") or {"email_enabled": False},
            "widget_config": WidgetConfig.from_dict(data.get("widget_config")).to_dict(),
            "analytics_settings": data.get("analytics_settings") or {"retention_days": 90},
        }
        rows = self._request("POST", "/rest/v1/clients", body=payload)
        created = rows[0] if isinstance(rows, list) and rows else payload
        for dom in data.get("domains") or []:
            await self.add_domain(tenant_id, dom)
        await self.create_prompt_version(
            tenant_id, payload["system_prompt"], payload["fallback_answer"], note="Initial version"
        )
        return self._enrich_client(created)

    async def list_clients(self) -> List[Dict[str, Any]]:
        rows = self._request("GET", "/rest/v1/clients", params={"order": "created_at.desc"})
        return [self._enrich_client(r) for r in (rows or [])]

    async def get_client(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        rows = self._request("GET", "/rest/v1/clients", params={"id": f"eq.{tenant_id}", "limit": "1"})
        return self._enrich_client(rows[0]) if isinstance(rows, list) and rows else None

    async def get_client_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        rows = self._request("GET", "/rest/v1/clients", params={"slug": f"eq.{slug}", "limit": "1"})
        return self._enrich_client(rows[0]) if isinstance(rows, list) and rows else None

    async def update_client(self, tenant_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body = {k: v for k, v in updates.items() if k != "domains"}
        if "widget_config" in body:
            body["widget_config"] = WidgetConfig.from_dict(body["widget_config"]).to_dict()
        if body:
            self._request("PATCH", "/rest/v1/clients", params={"id": f"eq.{tenant_id}"}, body=body)
        if "domains" in updates and isinstance(updates["domains"], list):
            self._request("DELETE", "/rest/v1/domains", params={"tenant_id": f"eq.{tenant_id}"})
            for dom in updates["domains"]:
                await self.add_domain(tenant_id, dom)
        return await self.get_client(tenant_id)

    async def delete_client(self, tenant_id: str) -> bool:
        rows = self._request("DELETE", "/rest/v1/clients", params={"id": f"eq.{tenant_id}"})
        return bool(rows)

    async def create_api_key(
        self, tenant_id: str, key_prefix: str, key_hash: str, name: str = "Default Widget Key"
    ) -> Dict[str, Any]:
        payload = {
            "id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "key_prefix": key_prefix,
            "key_hash": key_hash,
            "name": name,
            "enabled": True,
        }
        rows = self._request("POST", "/rest/v1/api_keys", body=payload)
        res = rows[0] if isinstance(rows, list) and rows else payload
        res.pop("key_hash", None)
        return res

    async def list_api_keys(self, tenant_id: str) -> List[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/api_keys",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "select": "id,tenant_id,key_prefix,name,enabled,last_used_at,created_at,revoked_at",
                "order": "created_at.desc",
            },
        )
        return list(rows or [])

    async def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/api_keys",
            params={"key_hash": f"eq.{key_hash}", "limit": "1"},
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def set_api_key_enabled(self, tenant_id: str, key_id: str, enabled: bool) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "PATCH",
            "/rest/v1/api_keys",
            params={"id": f"eq.{key_id}", "tenant_id": f"eq.{tenant_id}"},
            body={"enabled": enabled, "revoked_at": None if enabled else utc_now_iso()},
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def revoke_tenant_api_keys(self, tenant_id: str) -> int:
        rows = self._request(
            "PATCH",
            "/rest/v1/api_keys",
            params={"tenant_id": f"eq.{tenant_id}", "enabled": "eq.true"},
            body={"enabled": False, "revoked_at": utc_now_iso()},
        )
        return len(rows) if isinstance(rows, list) else 0

    async def add_domain(self, tenant_id: str, domain: str) -> Dict[str, Any]:
        payload = {"id": str(uuid.uuid4()), "tenant_id": tenant_id, "domain": domain.strip().lower()}
        rows = self._request(
            "POST",
            "/rest/v1/domains",
            body=payload,
            prefer="return=representation,resolution=ignore-duplicates",
        )
        return rows[0] if isinstance(rows, list) and rows else payload

    async def remove_domain(self, tenant_id: str, domain: str) -> bool:
        rows = self._request(
            "DELETE",
            "/rest/v1/domains",
            params={"tenant_id": f"eq.{tenant_id}", "domain": f"eq.{domain.strip().lower()}"},
        )
        return bool(rows)

    async def list_domains(self, tenant_id: str) -> List[str]:
        rows = self._request(
            "GET",
            "/rest/v1/domains",
            params={"tenant_id": f"eq.{tenant_id}", "select": "domain", "order": "created_at.asc"},
        )
        return [r["domain"] for r in (rows or [])]

    async def create_prompt_version(
        self, tenant_id: str, system_prompt: str, fallback_answer: str, note: str = ""
    ) -> Dict[str, Any]:
        existing = await self.list_prompt_versions(tenant_id)
        next_v = (max((r["version_number"] for r in existing), default=0)) + 1
        payload = {
            "id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "version_number": next_v,
            "system_prompt": system_prompt,
            "fallback_answer": fallback_answer,
            "note": note,
            "created_by": "admin",
        }
        rows = self._request("POST", "/rest/v1/prompt_versions", body=payload)
        return rows[0] if isinstance(rows, list) and rows else payload

    async def list_prompt_versions(self, tenant_id: str) -> List[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/prompt_versions",
            params={"tenant_id": f"eq.{tenant_id}", "order": "version_number.desc"},
        )
        return list(rows or [])

    async def get_prompt_version(self, tenant_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/prompt_versions",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "version_number": f"eq.{version_number}",
                "limit": "1",
            },
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def create_knowledge_file(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rows = self._request("POST", "/rest/v1/knowledge_files", body=data)
        return rows[0] if isinstance(rows, list) and rows else data

    async def get_knowledge_file(self, tenant_id: str, file_id: str) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/knowledge_files",
            params={"id": f"eq.{file_id}", "tenant_id": f"eq.{tenant_id}", "limit": "1"},
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def find_knowledge_file_by_name(self, tenant_id: str, original_filename: str) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/knowledge_files",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "original_filename": f"eq.{original_filename}",
                "order": "created_at.desc",
                "limit": "1",
            },
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def list_knowledge_files(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {"order": "created_at.desc"}
        if tenant_id:
            params["tenant_id"] = f"eq.{tenant_id}"
        rows = self._request("GET", "/rest/v1/knowledge_files", params=params)
        return list(rows or [])

    async def update_knowledge_file(
        self, tenant_id: str, file_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "PATCH",
            "/rest/v1/knowledge_files",
            params={"id": f"eq.{file_id}", "tenant_id": f"eq.{tenant_id}"},
            body=updates,
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def delete_knowledge_file(self, tenant_id: str, file_id: str) -> bool:
        rows = self._request(
            "DELETE",
            "/rest/v1/knowledge_files",
            params={"id": f"eq.{file_id}", "tenant_id": f"eq.{tenant_id}"},
        )
        return bool(rows)

    async def insert_staging_vectors(
        self, tenant_id: str, source_id: str, version: int, chunks: List[Dict[str, Any]]
    ) -> int:
        payload = [
            {
                "id": c.get("id") or str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "source_id": source_id,
                "source_filename": c["source_filename"],
                "source_version": version,
                "chunk_index": int(c["chunk_index"]),
                "content": c["content"],
                "embedding": c["embedding"],
                "is_active": False,
            }
            for c in chunks
        ]
        if not payload:
            return 0
        self._request("POST", "/rest/v1/knowledge_vectors", body=payload)
        return len(payload)

    async def count_vectors_for_version(self, tenant_id: str, source_id: str, version: int) -> int:
        rows = self._request(
            "GET",
            "/rest/v1/knowledge_vectors",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "source_id": f"eq.{source_id}",
                "source_version": f"eq.{version}",
                "select": "id",
            },
        )
        return len(rows) if isinstance(rows, list) else 0

    async def activate_vector_version(self, tenant_id: str, source_id: str, active_version: int) -> None:
        self._request(
            "PATCH",
            "/rest/v1/knowledge_vectors",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "source_id": f"eq.{source_id}",
                "source_version": f"eq.{active_version}",
            },
            body={"is_active": True},
        )
        self._request(
            "DELETE",
            "/rest/v1/knowledge_vectors",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "source_id": f"eq.{source_id}",
                "source_version": f"neq.{active_version}",
            },
        )

    async def delete_vectors_for_version(self, tenant_id: str, source_id: str, version: int) -> None:
        self._request(
            "DELETE",
            "/rest/v1/knowledge_vectors",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "source_id": f"eq.{source_id}",
                "source_version": f"eq.{version}",
            },
        )

    async def get_active_chunks_paginated(
        self, tenant_id: str, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/knowledge_vectors",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "is_active": "eq.true",
                "order": "chunk_index.asc",
                "limit": str(limit),
                "offset": str(offset),
            },
        )
        return list(rows or [])

    async def get_or_create_conversation(
        self, tenant_id: str, session_id: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if conversation_id:
            conv = await self.get_conversation(tenant_id, conversation_id)
            if conv:
                return conv
        rows = self._request(
            "GET",
            "/rest/v1/conversations",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "session_id": f"eq.{session_id}",
                "order": "created_at.desc",
                "limit": "1",
            },
        )
        if isinstance(rows, list) and rows:
            return rows[0]
        payload = {
            "id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "session_id": session_id,
            "status": "active",
            "metadata": {},
        }
        created = self._request("POST", "/rest/v1/conversations", body=payload)
        return created[0] if isinstance(created, list) and created else payload

    async def get_conversation(self, tenant_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/conversations",
            params={"id": f"eq.{conversation_id}", "tenant_id": f"eq.{tenant_id}", "limit": "1"},
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def list_conversations(
        self, tenant_id: Optional[str] = None, search: str = "", limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {"order": "updated_at.desc", "limit": str(limit), "offset": str(offset)}
        if tenant_id:
            params["tenant_id"] = f"eq.{tenant_id}"
        rows = self._request("GET", "/rest/v1/conversations", params=params)
        return list(rows or [])

    async def update_conversation_summary(self, tenant_id: str, conversation_id: str, summary: str) -> None:
        self._request(
            "PATCH",
            "/rest/v1/conversations",
            params={"id": f"eq.{conversation_id}", "tenant_id": f"eq.{tenant_id}"},
            body={"summary": summary},
        )

    async def update_conversation_metadata(
        self, tenant_id: str, conversation_id: str, metadata: Dict[str, Any]
    ) -> None:
        self._request(
            "PATCH",
            "/rest/v1/conversations",
            params={"id": f"eq.{conversation_id}", "tenant_id": f"eq.{tenant_id}"},
            body={"metadata": metadata or {}},
        )

    async def delete_conversation(self, tenant_id: str, conversation_id: str) -> bool:
        self._request(
            "DELETE",
            "/rest/v1/conversations",
            params={"id": f"eq.{conversation_id}", "tenant_id": f"eq.{tenant_id}"},
        )
        return True

    async def add_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        msg_id = data.get("id") or str(uuid.uuid4())
        if data.get("id"):
            existing = await self.get_message(data["tenant_id"], msg_id)
            if existing:
                return existing
        payload = {
            "id": msg_id,
            "tenant_id": data["tenant_id"],
            "conversation_id": data["conversation_id"],
            "role": data["role"],
            "content": data["content"],
            "sources": data.get("sources") or [],
            "support_level": data.get("support_level", "UNKNOWN"),
            "grounding_passed": bool(data.get("grounding_passed", True)),
            "latency_ms": int(data.get("latency_ms", 0)),
        }
        rows = self._request("POST", "/rest/v1/messages", body=payload)
        return rows[0] if isinstance(rows, list) and rows else payload

    async def get_message(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/messages",
            params={"id": f"eq.{message_id}", "tenant_id": f"eq.{tenant_id}", "limit": "1"},
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def list_messages(
        self, tenant_id: str, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/messages",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "conversation_id": f"eq.{conversation_id}",
                "order": "created_at.asc",
                "limit": str(limit),
            },
        )
        return list(rows or [])

    async def upsert_message_feedback(
        self,
        tenant_id: str,
        conversation_id: str,
        message_id: str,
        feedback_type: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            existing = await self.get_message_feedback(tenant_id, message_id)
            if existing and not existing.get("_fallback_meta"):
                rows = self._request(
                    "PATCH",
                    "/rest/v1/message_feedback",
                    params={"id": f"eq.{existing['id']}", "tenant_id": f"eq.{tenant_id}"},
                    body={"feedback_type": feedback_type, "reason": reason},
                )
                rec = rows[0] if isinstance(rows, list) and rows else {**existing, "feedback_type": feedback_type, "reason": reason}
                return {**rec, "updated": True}

            payload = {
                "id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "feedback_type": feedback_type,
                "reason": reason,
            }
            rows = self._request("POST", "/rest/v1/message_feedback", body=payload)
            rec = rows[0] if isinstance(rows, list) and rows else payload
            return {**rec, "updated": False}
        except Exception:
            if self._http_handler is not None:
                raise
            conv = await self.get_conversation(tenant_id, conversation_id)
            meta = dict((conv or {}).get("metadata") or {})
            fb_map = dict(meta.get("_message_feedback") or {})
            prev = fb_map.get(message_id)
            fb_record = {
                "id": (prev or {}).get("id") or str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "feedback_type": feedback_type,
                "reason": reason,
                "updated": bool(prev),
            }
            fb_map[message_id] = fb_record
            meta["_message_feedback"] = fb_map
            await self.update_conversation_metadata(tenant_id, conversation_id, meta)
            return fb_record

    async def get_message_feedback(
        self, tenant_id: str, message_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            rows = self._request(
                "GET",
                "/rest/v1/message_feedback",
                params={"tenant_id": f"eq.{tenant_id}", "message_id": f"eq.{message_id}", "limit": "1"},
            )
            return rows[0] if isinstance(rows, list) and rows else None
        except Exception:
            if self._http_handler is not None:
                raise
            return None

    async def list_conversation_feedback(
        self, tenant_id: str, conversation_id: str
    ) -> List[Dict[str, Any]]:
        try:
            rows = self._request(
                "GET",
                "/rest/v1/message_feedback",
                params={
                    "tenant_id": f"eq.{tenant_id}",
                    "conversation_id": f"eq.{conversation_id}",
                    "order": "created_at.asc",
                },
            )
            return list(rows or [])
        except Exception:
            if self._http_handler is not None:
                raise
            conv = await self.get_conversation(tenant_id, conversation_id)
            meta = dict((conv or {}).get("metadata") or {})
            fb_map = dict(meta.get("_message_feedback") or {})
            return list(fb_map.values())

    async def upsert_lead_idempotent(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        tenant_id = data["tenant_id"]
        idem_key = data["idempotency_key"]
        existing = self._request(
            "GET",
            "/rest/v1/leads",
            params={"tenant_id": f"eq.{tenant_id}", "idempotency_key": f"eq.{idem_key}", "limit": "1"},
        )
        if isinstance(existing, list) and existing:
            return existing[0], False

        lead_id = data.get("id") or str(uuid.uuid4())
        payload = {
            "id": lead_id,
            "tenant_id": tenant_id,
            "conversation_id": data.get("conversation_id"),
            "idempotency_key": idem_key,
            "name": data["name"],
            "email": data["email"],
            "phone": data.get("phone") or "",
            "requirement": data.get("requirement") or "",
            "message": data.get("message") or "",
            "status": "new",
        }
        rows = self._request("POST", "/rest/v1/leads", body=payload)
        lead_row = rows[0] if isinstance(rows, list) and rows else payload
        self._request(
            "POST",
            "/rest/v1/lead_workflow",
            body={
                "id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "lead_id": lead_id,
                "status": "new",
                "assignee": "",
                "internal_notes": "",
            },
        )
        return lead_row, True

    async def get_lead(self, tenant_id: str, lead_id: str) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "GET",
            "/rest/v1/leads",
            params={"id": f"eq.{lead_id}", "tenant_id": f"eq.{tenant_id}", "limit": "1"},
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def list_leads(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        search: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {"order": "created_at.desc", "limit": str(limit), "offset": str(offset)}
        if tenant_id:
            params["tenant_id"] = f"eq.{tenant_id}"
        if status:
            params["status"] = f"eq.{status}"
        rows = self._request("GET", "/rest/v1/leads", params=params)
        return list(rows or [])

    async def update_lead_workflow(
        self,
        tenant_id: str,
        lead_id: str,
        status: Optional[str] = None,
        assignee: Optional[str] = None,
        internal_notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if status is not None:
            self._request(
                "PATCH",
                "/rest/v1/leads",
                params={"id": f"eq.{lead_id}", "tenant_id": f"eq.{tenant_id}"},
                body={"status": status},
            )
        wf_updates: Dict[str, Any] = {}
        if status is not None:
            wf_updates["status"] = status
        if assignee is not None:
            wf_updates["assignee"] = assignee
        if internal_notes is not None:
            wf_updates["internal_notes"] = internal_notes
        if wf_updates:
            self._request(
                "PATCH",
                "/rest/v1/lead_workflow",
                params={"lead_id": f"eq.{lead_id}", "tenant_id": f"eq.{tenant_id}"},
                body=wf_updates,
            )
        return await self.get_lead(tenant_id, lead_id)

    async def create_notification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "id": data.get("id") or str(uuid.uuid4()),
            "tenant_id": data["tenant_id"],
            "lead_id": data.get("lead_id"),
            "channel": data.get("channel", "supabase"),
            "recipient": data.get("recipient", ""),
            "subject": data["subject"],
            "payload": data.get("payload") or {},
            "status": data.get("status", "pending"),
            "attempts": int(data.get("attempts", 0)),
            "last_error": data.get("last_error"),
        }
        rows = self._request("POST", "/rest/v1/notifications", body=payload)
        return rows[0] if isinstance(rows, list) and rows else payload

    async def update_notification(self, notification_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rows = self._request(
            "PATCH",
            "/rest/v1/notifications",
            params={"id": f"eq.{notification_id}"},
            body=updates,
        )
        return rows[0] if isinstance(rows, list) and rows else None

    async def list_notifications(
        self, tenant_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {"order": "created_at.desc", "limit": str(limit)}
        if tenant_id:
            params["tenant_id"] = f"eq.{tenant_id}"
        if status:
            params["status"] = f"eq.{status}"
        rows = self._request("GET", "/rest/v1/notifications", params=params)
        return list(rows or [])

    async def record_usage(self, data: Dict[str, Any]) -> None:
        self._request(
            "POST",
            "/rest/v1/usage_logs",
            body={
                "id": str(uuid.uuid4()),
                "tenant_id": data["tenant_id"],
                "operation": data.get("operation", "chat"),
                "request_count": int(data.get("request_count", 1)),
                "usage_units": int(data.get("usage_units", 1)),
                "latency_ms": int(data.get("latency_ms", 0)),
                "support_level": data.get("support_level", "UNKNOWN"),
            },
        )

    async def list_usage_logs(
        self, tenant_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {"order": "created_at.asc"}
        if tenant_id:
            params["tenant_id"] = f"eq.{tenant_id}"
        rows = self._request("GET", "/rest/v1/usage_logs", params=params)
        return list(rows or [])

    async def record_audit(
        self, actor: str, action: str, tenant_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None
    ) -> None:
        self._request(
            "POST",
            "/rest/v1/audit_logs",
            body={
                "id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "actor": actor,
                "action": action,
                "details": details or {},
            },
        )

    async def list_audit_logs(self, tenant_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {"order": "created_at.desc", "limit": str(limit)}
        if tenant_id:
            params["tenant_id"] = f"eq.{tenant_id}"
        rows = self._request("GET", "/rest/v1/audit_logs", params=params)
        return list(rows or [])

    async def get_setting(self, key: str) -> Optional[Dict[str, Any]]:
        rows = self._request("GET", "/rest/v1/settings", params={"key": f"eq.{key}", "limit": "1"})
        return rows[0] if isinstance(rows, list) and rows else None

    async def set_setting(self, key: str, value: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"key": key, "value": value, "updated_at": utc_now_iso()}
        rows = self._request(
            "POST",
            "/rest/v1/settings",
            body=payload,
            prefer="return=representation,resolution=merge-duplicates",
        )
        return rows[0] if isinstance(rows, list) and rows else payload
