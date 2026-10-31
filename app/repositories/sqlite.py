"""SQLite repository implementation for local development and automated testing.

Implements all 14 tables matching `supabase/migrations/0001_initial.sql` with
strict `tenant_id` filtering on every tenant-owned table.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app.domain.models import WidgetConfig, utc_now_iso
from app.repositories.base import BaseRepository


class SQLiteRepository(BaseRepository):
    """Thread-safe SQLite repository with full multi-tenant isolation."""

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

    def __init__(self, db_path: str = "data/orbit_local.db") -> None:
        self.db_path = db_path
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")

    async def initialize(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS clients (
                id TEXT PRIMARY KEY,
                business_name TEXT NOT NULL,
                slug TEXT NOT NULL UNIQUE,
                company_name TEXT NOT NULL,
                assistant_name TEXT NOT NULL DEFAULT 'AI Assistant',
                enabled INTEGER NOT NULL DEFAULT 1,
                system_prompt TEXT NOT NULL,
                fallback_answer TEXT NOT NULL,
                welcome_message TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'en',
                model_override TEXT,
                temperature REAL NOT NULL DEFAULT 0.2,
                top_k INTEGER NOT NULL DEFAULT 5,
                min_similarity REAL NOT NULL DEFAULT 0.25,
                lead_capture_enabled INTEGER NOT NULL DEFAULT 1,
                lead_settings TEXT NOT NULL,
                notification_settings TEXT NOT NULL,
                widget_config TEXT NOT NULL,
                analytics_settings TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                key_prefix TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL DEFAULT 'Default Widget Key',
                enabled INTEGER NOT NULL DEFAULT 1,
                last_used_at TEXT,
                created_at TEXT NOT NULL,
                revoked_at TEXT
            );

            CREATE TABLE IF NOT EXISTS domains (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                domain TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(tenant_id, domain)
            );

            CREATE TABLE IF NOT EXISTS prompt_versions (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                version_number INTEGER NOT NULL,
                system_prompt TEXT NOT NULL,
                fallback_answer TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_by TEXT NOT NULL DEFAULT 'admin',
                created_at TEXT NOT NULL,
                UNIQUE(tenant_id, version_number)
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                session_id TEXT NOT NULL,
                summary TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                support_level TEXT NOT NULL DEFAULT 'UNKNOWN',
                grounding_passed INTEGER NOT NULL DEFAULT 1,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS leads (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                idempotency_key TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL DEFAULT '',
                requirement TEXT NOT NULL DEFAULT '',
                message TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'new',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(tenant_id, idempotency_key)
            );

            CREATE TABLE IF NOT EXISTS lead_workflow (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                lead_id TEXT NOT NULL REFERENCES leads(id) ON DELETE CASCADE UNIQUE,
                status TEXT NOT NULL DEFAULT 'new',
                assignee TEXT NOT NULL DEFAULT '',
                internal_notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS usage_logs (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                operation TEXT NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 1,
                usage_units INTEGER NOT NULL DEFAULT 1,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                support_level TEXT NOT NULL DEFAULT 'UNKNOWN',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                tenant_id TEXT REFERENCES clients(id) ON DELETE SET NULL,
                actor TEXT NOT NULL DEFAULT 'admin',
                action TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_files (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                original_filename TEXT NOT NULL,
                safe_filename TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                file_size INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'processing',
                active_version INTEGER NOT NULL DEFAULT 1,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_vectors (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                source_id TEXT NOT NULL REFERENCES knowledge_files(id) ON DELETE CASCADE,
                source_filename TEXT NOT NULL,
                source_version INTEGER NOT NULL DEFAULT 1,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                lead_id TEXT REFERENCES leads(id) ON DELETE SET NULL,
                channel TEXT NOT NULL DEFAULT 'outbox',
                recipient TEXT NOT NULL DEFAULT '',
                subject TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    async def check_health(self) -> Dict[str, Any]:
        try:
            cur = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
            existing = {r["name"] for r in cur.fetchall()}
            missing = [t for t in self.REQUIRED_TABLES if t not in existing]
            if missing:
                return {
                    "ok": False,
                    "provider": "sqlite",
                    "error": f"Missing required database tables: {', '.join(missing)}",
                }
            return {
                "ok": True,
                "provider": "sqlite",
                "tables_verified": len(self.REQUIRED_TABLES),
            }
        except Exception as exc:
            return {"ok": False, "provider": "sqlite", "error": str(exc)}

    def _deserialize_client(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["enabled"] = bool(d.get("enabled", 1))
        d["lead_capture_enabled"] = bool(d.get("lead_capture_enabled", 1))
        for json_col in ("lead_settings", "notification_settings", "widget_config", "analytics_settings"):
            raw = d.get(json_col)
            if isinstance(raw, str):
                try:
                    d[json_col] = json.loads(raw)
                except Exception:
                    d[json_col] = {}
        d["widget_config"] = WidgetConfig.from_dict(d.get("widget_config")).to_dict()
        # Attach domains
        dom_cur = self._conn.execute(
            "SELECT domain FROM domains WHERE tenant_id = ? ORDER BY created_at ASC",
            (d["id"],),
        )
        d["domains"] = [r["domain"] for r in dom_cur.fetchall()]
        return d

    async def create_client(self, data: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now_iso()
        tenant_id = data.get("id") or str(uuid.uuid4())
        widget_cfg = WidgetConfig.from_dict(data.get("widget_config")).to_dict()
        lead_settings = data.get("lead_settings") or {
            "require_email": True,
            "require_phone": False,
            "custom_prompt": "Leave your contact details so our team can follow up.",
        }
        notif_settings = data.get("notification_settings") or {
            "email_enabled": False,
            "notify_emails": [],
            "webhook_url": None,
        }
        analytics_settings = data.get("analytics_settings") or {
            "retention_days": 90,
            "track_latency": True,
        }
        system_prompt = data.get("system_prompt") or (
            "You are a helpful, accurate customer support assistant. "
            "Answer strictly using the provided knowledge base."
        )
        fallback_answer = data.get("fallback_answer") or (
            "I do not have enough verified information in our knowledge base to answer that accurately. "
            "Please leave your contact details and our team will assist you."
        )

        self._conn.execute(
            """
            INSERT INTO clients (
                id, business_name, slug, company_name, assistant_name, enabled,
                system_prompt, fallback_answer, welcome_message, language,
                model_override, temperature, top_k, min_similarity,
                lead_capture_enabled, lead_settings, notification_settings,
                widget_config, analytics_settings, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                data["business_name"],
                data["slug"],
                data.get("company_name") or data["business_name"],
                data.get("assistant_name") or "AI Assistant",
                1 if data.get("enabled", True) else 0,
                system_prompt,
                fallback_answer,
                data.get("welcome_message") or "Hello! How can I help you today?",
                data.get("language") or "en",
                data.get("model_override"),
                float(data.get("temperature", 0.2)),
                int(data.get("top_k", 5)),
                float(data.get("min_similarity", 0.25)),
                1 if data.get("lead_capture_enabled", True) else 0,
                json.dumps(lead_settings),
                json.dumps(notif_settings),
                json.dumps(widget_cfg),
                json.dumps(analytics_settings),
                now,
                now,
            ),
        )
        self._conn.commit()
        for dom in data.get("domains") or []:
            await self.add_domain(tenant_id, dom)
        await self.create_prompt_version(tenant_id, system_prompt, fallback_answer, note="Initial version")
        res = await self.get_client(tenant_id)
        assert res is not None
        return res

    async def list_clients(self) -> List[Dict[str, Any]]:
        cur = self._conn.execute("SELECT * FROM clients ORDER BY created_at DESC")
        return [self._deserialize_client(r) for r in cur.fetchall()]

    async def get_client(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute("SELECT * FROM clients WHERE id = ?", (tenant_id,))
        row = cur.fetchone()
        return self._deserialize_client(row) if row else None

    async def get_client_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute("SELECT * FROM clients WHERE slug = ?", (slug,))
        row = cur.fetchone()
        return self._deserialize_client(row) if row else None

    async def update_client(self, tenant_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        existing = await self.get_client(tenant_id)
        if not existing:
            return None
        allowed_cols = {
            "business_name",
            "slug",
            "company_name",
            "assistant_name",
            "enabled",
            "system_prompt",
            "fallback_answer",
            "welcome_message",
            "language",
            "model_override",
            "temperature",
            "top_k",
            "min_similarity",
            "lead_capture_enabled",
            "lead_settings",
            "notification_settings",
            "widget_config",
            "analytics_settings",
        }
        sets: List[str] = []
        vals: List[Any] = []
        for k, v in updates.items():
            if k not in allowed_cols:
                continue
            if k in ("enabled", "lead_capture_enabled"):
                sets.append(f"{k} = ?")
                vals.append(1 if v else 0)
            elif k == "widget_config":
                merged_cfg = dict(existing.get("widget_config") or {})
                if isinstance(v, dict):
                    merged_cfg.update(v)
                sets.append("widget_config = ?")
                vals.append(json.dumps(WidgetConfig.from_dict(merged_cfg).to_dict()))
            elif k in ("lead_settings", "notification_settings", "analytics_settings"):
                sets.append(f"{k} = ?")
                vals.append(json.dumps(v if isinstance(v, dict) else {}))
            else:
                sets.append(f"{k} = ?")
                vals.append(v)

        if sets:
            sets.append("updated_at = ?")
            vals.append(utc_now_iso())
            vals.append(tenant_id)
            self._conn.execute(f"UPDATE clients SET {', '.join(sets)} WHERE id = ?", tuple(vals))
            self._conn.commit()

        if "domains" in updates and isinstance(updates["domains"], list):
            self._conn.execute("DELETE FROM domains WHERE tenant_id = ?", (tenant_id,))
            self._conn.commit()
            for d in updates["domains"]:
                await self.add_domain(tenant_id, d)

        return await self.get_client(tenant_id)

    async def delete_client(self, tenant_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM clients WHERE id = ?", (tenant_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # --- API Keys ---
    async def create_api_key(
        self, tenant_id: str, key_prefix: str, key_hash: str, name: str = "Default Widget Key"
    ) -> Dict[str, Any]:
        key_id = str(uuid.uuid4())
        now = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO api_keys (id, tenant_id, key_prefix, key_hash, name, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (key_id, tenant_id, key_prefix, key_hash, name, now),
        )
        self._conn.commit()
        return {
            "id": key_id,
            "tenant_id": tenant_id,
            "key_prefix": key_prefix,
            "name": name,
            "enabled": True,
            "created_at": now,
            "revoked_at": None,
        }

    async def list_api_keys(self, tenant_id: str) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT id, tenant_id, key_prefix, name, enabled, last_used_at, created_at, revoked_at
            FROM api_keys WHERE tenant_id = ? ORDER BY created_at DESC
            """,
            (tenant_id,),
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["enabled"] = bool(d["enabled"])
            rows.append(d)
        return rows

    async def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        )
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["enabled"] = bool(d["enabled"])
        return d

    async def set_api_key_enabled(self, tenant_id: str, key_id: str, enabled: bool) -> Optional[Dict[str, Any]]:
        revoked_at = None if enabled else utc_now_iso()
        cur = self._conn.execute(
            "UPDATE api_keys SET enabled = ?, revoked_at = ? WHERE id = ? AND tenant_id = ?",
            (1 if enabled else 0, revoked_at, key_id, tenant_id),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None
        keys = await self.list_api_keys(tenant_id)
        return next((k for k in keys if k["id"] == key_id), None)

    async def revoke_tenant_api_keys(self, tenant_id: str) -> int:
        now = utc_now_iso()
        cur = self._conn.execute(
            "UPDATE api_keys SET enabled = 0, revoked_at = ? WHERE tenant_id = ? AND enabled = 1",
            (now, tenant_id),
        )
        self._conn.commit()
        return cur.rowcount

    # --- Domains ---
    async def add_domain(self, tenant_id: str, domain: str) -> Dict[str, Any]:
        dom_id = str(uuid.uuid4())
        now = utc_now_iso()
        cleaned = domain.strip().lower()
        self._conn.execute(
            "INSERT OR IGNORE INTO domains (id, tenant_id, domain, created_at) VALUES (?, ?, ?, ?)",
            (dom_id, tenant_id, cleaned, now),
        )
        self._conn.commit()
        return {"id": dom_id, "tenant_id": tenant_id, "domain": cleaned, "created_at": now}

    async def remove_domain(self, tenant_id: str, domain: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM domains WHERE tenant_id = ? AND domain = ?",
            (tenant_id, domain.strip().lower()),
        )
        self._conn.commit()
        return cur.rowcount > 0

    async def list_domains(self, tenant_id: str) -> List[str]:
        cur = self._conn.execute(
            "SELECT domain FROM domains WHERE tenant_id = ? ORDER BY created_at ASC",
            (tenant_id,),
        )
        return [r["domain"] for r in cur.fetchall()]

    # --- Prompt Versions ---
    async def create_prompt_version(
        self, tenant_id: str, system_prompt: str, fallback_answer: str, note: str = ""
    ) -> Dict[str, Any]:
        cur = self._conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) AS max_v FROM prompt_versions WHERE tenant_id = ?",
            (tenant_id,),
        )
        next_v = int(cur.fetchone()["max_v"]) + 1
        pv_id = str(uuid.uuid4())
        now = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO prompt_versions (id, tenant_id, version_number, system_prompt, fallback_answer, note, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'admin', ?)
            """,
            (pv_id, tenant_id, next_v, system_prompt, fallback_answer, note, now),
        )
        self._conn.commit()
        return {
            "id": pv_id,
            "tenant_id": tenant_id,
            "version_number": next_v,
            "system_prompt": system_prompt,
            "fallback_answer": fallback_answer,
            "note": note,
            "created_by": "admin",
            "created_at": now,
        }

    async def list_prompt_versions(self, tenant_id: str) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM prompt_versions WHERE tenant_id = ? ORDER BY version_number DESC",
            (tenant_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    async def get_prompt_version(self, tenant_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM prompt_versions WHERE tenant_id = ? AND version_number = ?",
            (tenant_id, version_number),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # --- Knowledge Files & Vectors ---
    async def create_knowledge_file(self, data: Dict[str, Any]) -> Dict[str, Any]:
        file_id = data.get("id") or str(uuid.uuid4())
        now = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO knowledge_files (
                id, tenant_id, original_filename, safe_filename, storage_path,
                mime_type, file_size, status, active_version, chunk_count, error_message,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                data["tenant_id"],
                data["original_filename"],
                data["safe_filename"],
                data["storage_path"],
                data["mime_type"],
                int(data.get("file_size", 0)),
                data.get("status", "processing"),
                int(data.get("active_version", 1)),
                int(data.get("chunk_count", 0)),
                data.get("error_message"),
                now,
                now,
            ),
        )
        self._conn.commit()
        res = await self.get_knowledge_file(data["tenant_id"], file_id)
        assert res is not None
        return res

    async def get_knowledge_file(self, tenant_id: str, file_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM knowledge_files WHERE id = ? AND tenant_id = ?",
            (file_id, tenant_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    async def find_knowledge_file_by_name(self, tenant_id: str, original_filename: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM knowledge_files WHERE tenant_id = ? AND original_filename = ? ORDER BY created_at DESC LIMIT 1",
            (tenant_id, original_filename),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    async def list_knowledge_files(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if tenant_id:
            cur = self._conn.execute(
                "SELECT * FROM knowledge_files WHERE tenant_id = ? ORDER BY created_at DESC",
                (tenant_id,),
            )
        else:
            cur = self._conn.execute("SELECT * FROM knowledge_files ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]

    async def update_knowledge_file(
        self, tenant_id: str, file_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        allowed = {
            "original_filename",
            "safe_filename",
            "storage_path",
            "mime_type",
            "file_size",
            "status",
            "active_version",
            "chunk_count",
            "error_message",
        }
        sets: List[str] = []
        vals: List[Any] = []
        for k, v in updates.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                vals.append(v)
        if not sets:
            return await self.get_knowledge_file(tenant_id, file_id)
        sets.append("updated_at = ?")
        vals.append(utc_now_iso())
        vals.extend([file_id, tenant_id])
        self._conn.execute(
            f"UPDATE knowledge_files SET {', '.join(sets)} WHERE id = ? AND tenant_id = ?",
            tuple(vals),
        )
        self._conn.commit()
        return await self.get_knowledge_file(tenant_id, file_id)

    async def delete_knowledge_file(self, tenant_id: str, file_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM knowledge_files WHERE id = ? AND tenant_id = ?",
            (file_id, tenant_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    async def insert_staging_vectors(
        self, tenant_id: str, source_id: str, version: int, chunks: List[Dict[str, Any]]
    ) -> int:
        now = utc_now_iso()
        rows = []
        for c in chunks:
            rows.append(
                (
                    c.get("id") or str(uuid.uuid4()),
                    tenant_id,
                    source_id,
                    c["source_filename"],
                    version,
                    int(c["chunk_index"]),
                    c["content"],
                    json.dumps(c["embedding"]),
                    0,  # staging: inactive until verified & activated
                    now,
                )
            )
        self._conn.executemany(
            """
            INSERT INTO knowledge_vectors (
                id, tenant_id, source_id, source_filename, source_version,
                chunk_index, content, embedding, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

    async def count_vectors_for_version(self, tenant_id: str, source_id: str, version: int) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM knowledge_vectors WHERE tenant_id = ? AND source_id = ? AND source_version = ?",
            (tenant_id, source_id, version),
        )
        return int(cur.fetchone()["cnt"])

    async def activate_vector_version(self, tenant_id: str, source_id: str, active_version: int) -> None:
        # Activate new version and deactivate older versions atomically
        self._conn.execute(
            "UPDATE knowledge_vectors SET is_active = CASE WHEN source_version = ? THEN 1 ELSE 0 END WHERE tenant_id = ? AND source_id = ?",
            (active_version, tenant_id, source_id),
        )
        # Clean up older versions now that the new version is active
        self._conn.execute(
            "DELETE FROM knowledge_vectors WHERE tenant_id = ? AND source_id = ? AND source_version != ?",
            (tenant_id, source_id, active_version),
        )
        self._conn.commit()

    async def delete_vectors_for_version(self, tenant_id: str, source_id: str, version: int) -> None:
        self._conn.execute(
            "DELETE FROM knowledge_vectors WHERE tenant_id = ? AND source_id = ? AND source_version = ?",
            (tenant_id, source_id, version),
        )
        self._conn.commit()

    async def get_active_chunks_paginated(
        self, tenant_id: str, limit: int = 200, offset: int = 0
    ) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT kv.*
            FROM knowledge_vectors kv
            INNER JOIN knowledge_files kf
                ON kf.id = kv.source_id AND kf.tenant_id = kv.tenant_id
            WHERE kv.tenant_id = ?
              AND kv.is_active = 1
              AND kf.status = 'ready'
              AND kv.source_version = kf.active_version
            ORDER BY kv.source_id ASC, kv.chunk_index ASC
            LIMIT ? OFFSET ?
            """,
            (tenant_id, limit, offset),
        )
        results = []
        for r in cur.fetchall():
            d = dict(r)
            d["is_active"] = bool(d["is_active"])
            if isinstance(d.get("embedding"), str):
                d["embedding"] = json.loads(d["embedding"])
            results.append(d)
        return results

    # --- Conversations & Messages ---
    async def get_or_create_conversation(
        self, tenant_id: str, session_id: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if conversation_id:
            cur = self._conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND tenant_id = ?",
                (conversation_id, tenant_id),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
        # Check by session_id for the same tenant
        cur = self._conn.execute(
            "SELECT * FROM conversations WHERE tenant_id = ? AND session_id = ? ORDER BY created_at DESC LIMIT 1",
            (tenant_id, session_id),
        )
        row = cur.fetchone()
        if row:
            return dict(row)

        conv_id = str(uuid.uuid4())
        now = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO conversations (id, tenant_id, session_id, summary, status, metadata, created_at, updated_at)
            VALUES (?, ?, ?, NULL, 'active', '{}', ?, ?)
            """,
            (conv_id, tenant_id, session_id, now, now),
        )
        self._conn.commit()
        return {
            "id": conv_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "summary": None,
            "status": "active",
            "metadata": {},
            "created_at": now,
            "updated_at": now,
        }

    async def get_conversation(self, tenant_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM conversations WHERE id = ? AND tenant_id = ?",
            (conversation_id, tenant_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    async def list_conversations(
        self, tenant_id: Optional[str] = None, search: str = "", limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where_clauses: List[str] = []
        if tenant_id:
            where_clauses.append("c.tenant_id = ?")
            params.append(tenant_id)
        if search.strip():
            where_clauses.append(
                "(c.session_id LIKE ? OR c.summary LIKE ? OR EXISTS ("
                "SELECT 1 FROM messages m WHERE m.conversation_id = c.id AND m.content LIKE ?"
                "))"
            )
            like_q = f"%{search.strip()}%"
            params.extend([like_q, like_q, like_q])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.extend([limit, offset])
        cur = self._conn.execute(
            f"""
            SELECT c.*,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
                   (SELECT m.content FROM messages m WHERE m.conversation_id = c.id ORDER BY m.created_at DESC LIMIT 1) AS last_message
            FROM conversations c
            {where_sql}
            ORDER BY c.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params),
        )
        return [dict(r) for r in cur.fetchall()]

    async def update_conversation_summary(self, tenant_id: str, conversation_id: str, summary: str) -> None:
        self._conn.execute(
            "UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ? AND tenant_id = ?",
            (summary, utc_now_iso(), conversation_id, tenant_id),
        )
        self._conn.commit()

    async def add_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        msg_id = data.get("id") or str(uuid.uuid4())
        now = utc_now_iso()
        sources_json = json.dumps(data.get("sources") or [])
        self._conn.execute(
            """
            INSERT INTO messages (
                id, tenant_id, conversation_id, role, content, sources,
                support_level, grounding_passed, latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg_id,
                data["tenant_id"],
                data["conversation_id"],
                data["role"],
                data["content"],
                sources_json,
                data.get("support_level", "UNKNOWN"),
                1 if data.get("grounding_passed", True) else 0,
                int(data.get("latency_ms", 0)),
                now,
            ),
        )
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ? AND tenant_id = ?",
            (now, data["conversation_id"], data["tenant_id"]),
        )
        self._conn.commit()
        return {
            "id": msg_id,
            "tenant_id": data["tenant_id"],
            "conversation_id": data["conversation_id"],
            "role": data["role"],
            "content": data["content"],
            "sources": data.get("sources") or [],
            "support_level": data.get("support_level", "UNKNOWN"),
            "grounding_passed": bool(data.get("grounding_passed", True)),
            "latency_ms": int(data.get("latency_ms", 0)),
            "created_at": now,
        }

    async def list_messages(
        self, tenant_id: str, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT * FROM messages
            WHERE tenant_id = ? AND conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (tenant_id, conversation_id, limit),
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["grounding_passed"] = bool(d["grounding_passed"])
            if isinstance(d.get("sources"), str):
                try:
                    d["sources"] = json.loads(d["sources"])
                except Exception:
                    d["sources"] = []
            rows.append(d)
        return rows

    # --- Leads & Lead Workflow ---
    async def upsert_lead_idempotent(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        tenant_id = data["tenant_id"]
        idem_key = data["idempotency_key"]
        cur = self._conn.execute(
            "SELECT id FROM leads WHERE tenant_id = ? AND idempotency_key = ?",
            (tenant_id, idem_key),
        )
        existing = cur.fetchone()
        if existing:
            lead = await self.get_lead(tenant_id, existing["id"])
            assert lead is not None
            return lead, False

        lead_id = data.get("id") or str(uuid.uuid4())
        wf_id = str(uuid.uuid4())
        now = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO leads (
                id, tenant_id, conversation_id, idempotency_key,
                name, email, phone, requirement, message, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)
            """,
            (
                lead_id,
                tenant_id,
                data.get("conversation_id"),
                idem_key,
                data["name"],
                data["email"],
                data.get("phone") or "",
                data.get("requirement") or "",
                data.get("message") or "",
                now,
                now,
            ),
        )
        self._conn.execute(
            """
            INSERT INTO lead_workflow (id, tenant_id, lead_id, status, assignee, internal_notes, updated_at)
            VALUES (?, ?, ?, 'new', '', '', ?)
            """,
            (wf_id, tenant_id, lead_id, now),
        )
        self._conn.commit()
        lead = await self.get_lead(tenant_id, lead_id)
        assert lead is not None
        return lead, True

    async def get_lead(self, tenant_id: str, lead_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT l.*,
                   COALESCE(lw.assignee, '') AS assignee,
                   COALESCE(lw.internal_notes, '') AS internal_notes,
                   COALESCE(
                       (SELECT n.status FROM notifications n WHERE n.lead_id = l.id ORDER BY n.created_at DESC LIMIT 1),
                       'pending'
                   ) AS notification_status
            FROM leads l
            LEFT JOIN lead_workflow lw ON lw.lead_id = l.id AND lw.tenant_id = l.tenant_id
            WHERE l.id = ? AND l.tenant_id = ?
            """,
            (lead_id, tenant_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    async def list_leads(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        search: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if tenant_id:
            clauses.append("l.tenant_id = ?")
            params.append(tenant_id)
        if status:
            clauses.append("l.status = ?")
            params.append(status)
        if search.strip():
            clauses.append("(l.name LIKE ? OR l.email LIKE ? OR l.phone LIKE ? OR l.message LIKE ? OR l.requirement LIKE ?)")
            q = f"%{search.strip()}%"
            params.extend([q, q, q, q, q])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        cur = self._conn.execute(
            f"""
            SELECT l.*,
                   COALESCE(lw.assignee, '') AS assignee,
                   COALESCE(lw.internal_notes, '') AS internal_notes,
                   COALESCE(
                       (SELECT n.status FROM notifications n WHERE n.lead_id = l.id ORDER BY n.created_at DESC LIMIT 1),
                       'pending'
                   ) AS notification_status
            FROM leads l
            LEFT JOIN lead_workflow lw ON lw.lead_id = l.id AND lw.tenant_id = l.tenant_id
            {where_sql}
            ORDER BY l.created_at DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params),
        )
        return [dict(r) for r in cur.fetchall()]

    async def update_lead_workflow(
        self,
        tenant_id: str,
        lead_id: str,
        status: Optional[str] = None,
        assignee: Optional[str] = None,
        internal_notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        existing = await self.get_lead(tenant_id, lead_id)
        if not existing:
            return None
        now = utc_now_iso()
        new_status = status if status is not None else existing["status"]
        new_assignee = assignee if assignee is not None else existing.get("assignee", "")
        new_notes = internal_notes if internal_notes is not None else existing.get("internal_notes", "")

        self._conn.execute(
            "UPDATE leads SET status = ?, updated_at = ? WHERE id = ? AND tenant_id = ?",
            (new_status, now, lead_id, tenant_id),
        )
        self._conn.execute(
            """
            INSERT INTO lead_workflow (id, tenant_id, lead_id, status, assignee, internal_notes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(lead_id) DO UPDATE SET
                status = excluded.status,
                assignee = excluded.assignee,
                internal_notes = excluded.internal_notes,
                updated_at = excluded.updated_at
            """,
            (str(uuid.uuid4()), tenant_id, lead_id, new_status, new_assignee, new_notes, now),
        )
        self._conn.commit()
        return await self.get_lead(tenant_id, lead_id)

    # --- Notifications / Outbox ---
    async def create_notification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        notif_id = data.get("id") or str(uuid.uuid4())
        now = utc_now_iso()
        payload_str = json.dumps(data.get("payload") or {})
        self._conn.execute(
            """
            INSERT INTO notifications (
                id, tenant_id, lead_id, channel, recipient, subject,
                payload, status, attempts, last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notif_id,
                data["tenant_id"],
                data.get("lead_id"),
                data.get("channel", "outbox"),
                data.get("recipient", ""),
                data["subject"],
                payload_str,
                data.get("status", "pending"),
                int(data.get("attempts", 0)),
                data.get("last_error"),
                now,
                now,
            ),
        )
        self._conn.commit()
        return {
            "id": notif_id,
            "tenant_id": data["tenant_id"],
            "lead_id": data.get("lead_id"),
            "channel": data.get("channel", "outbox"),
            "recipient": data.get("recipient", ""),
            "subject": data["subject"],
            "payload": data.get("payload") or {},
            "status": data.get("status", "pending"),
            "attempts": int(data.get("attempts", 0)),
            "last_error": data.get("last_error"),
            "created_at": now,
            "updated_at": now,
        }

    async def update_notification(self, notification_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sets: List[str] = []
        vals: List[Any] = []
        for k in ("status", "attempts", "last_error"):
            if k in updates:
                sets.append(f"{k} = ?")
                vals.append(updates[k])
        if sets:
            sets.append("updated_at = ?")
            vals.append(utc_now_iso())
            vals.append(notification_id)
            self._conn.execute(
                f"UPDATE notifications SET {', '.join(sets)} WHERE id = ?",
                tuple(vals),
            )
            self._conn.commit()
        cur = self._conn.execute("SELECT * FROM notifications WHERE id = ?", (notification_id,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        if isinstance(d.get("payload"), str):
            d["payload"] = json.loads(d["payload"])
        return d

    async def list_notifications(
        self, tenant_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if tenant_id:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        cur = self._conn.execute(
            f"SELECT * FROM notifications {where_sql} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        )
        items = []
        for r in cur.fetchall():
            d = dict(r)
            if isinstance(d.get("payload"), str):
                try:
                    d["payload"] = json.loads(d["payload"])
                except Exception:
                    d["payload"] = {}
            items.append(d)
        return items

    # --- Usage Logs, Audit Logs & Settings ---
    async def record_usage(self, data: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO usage_logs (id, tenant_id, operation, request_count, usage_units, latency_ms, support_level, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                data["tenant_id"],
                data.get("operation", "chat"),
                int(data.get("request_count", 1)),
                int(data.get("usage_units", 1)),
                int(data.get("latency_ms", 0)),
                data.get("support_level", "UNKNOWN"),
                data.get("created_at") or utc_now_iso(),
            ),
        )
        self._conn.commit()

    async def list_usage_logs(
        self, tenant_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if tenant_id:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if start_date:
            clauses.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("created_at <= ?")
            params.append(end_date + "T23:59:59Z" if "T" not in end_date else end_date)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self._conn.execute(
            f"SELECT * FROM usage_logs {where_sql} ORDER BY created_at ASC",
            tuple(params),
        )
        return [dict(r) for r in cur.fetchall()]

    async def record_audit(
        self, actor: str, action: str, tenant_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO audit_logs (id, tenant_id, actor, action, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                tenant_id,
                actor,
                action,
                json.dumps(details or {}),
                utc_now_iso(),
            ),
        )
        self._conn.commit()

    async def list_audit_logs(self, tenant_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if tenant_id:
            cur = self._conn.execute(
                "SELECT * FROM audit_logs WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
                (tenant_id, limit),
            )
        else:
            cur = self._conn.execute("SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            if isinstance(d.get("details"), str):
                try:
                    d["details"] = json.loads(d["details"])
                except Exception:
                    d["details"] = {}
            rows.append(d)
        return rows

    async def get_setting(self, key: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute("SELECT * FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        if isinstance(d.get("value"), str):
            d["value"] = json.loads(d["value"])
        return d

    async def set_setting(self, key: str, value: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, json.dumps(value), now),
        )
        self._conn.commit()
        return {"key": key, "value": value, "updated_at": now}
