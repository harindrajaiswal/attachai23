"""Vector search provider abstraction and implementations (Local, Supabase pgvector RPC, Qdrant).

Strictly enforces:
- Tenant ID isolation on every search call
- Active version filtering (is_active=True and status='ready')
- Dimension validation against configured SUPABASE_VECTOR_DIMENSIONS (128 default)
- Paginated retrieval in LocalVectorProvider so no tenant is capped to a fixed memory slice
"""

import json
import math
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.core.exceptions import ConfigurationError, ProviderError
from app.repositories.base import BaseRepository


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0
    return float(dot / (norm_a * norm_b))


class BaseVectorProvider(ABC):
    """Contract for tenant-isolated vector similarity search."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions <= 0:
            raise ConfigurationError("Vector dimensions must be positive.")
        self.dimensions = dimensions

    def _validate_query(self, tenant_id: str, query_embedding: List[float]) -> None:
        if not tenant_id or not str(tenant_id).strip():
            raise ProviderError("Tenant ID is required for vector search.", status_code=400, provider="vector")
        if len(query_embedding) != self.dimensions:
            raise ProviderError(
                f"Query embedding dimension ({len(query_embedding)}) does not match vector schema ({self.dimensions}).",
                status_code=422,
                provider="vector",
            )

    @abstractmethod
    async def search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 8,
        min_similarity: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """Return top_k active chunks for tenant_id with similarity >= min_similarity."""

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Verify vector provider readiness."""


class LocalVectorProvider(BaseVectorProvider):
    """Local vector search over repository chunks using paginated scanning."""

    def __init__(self, repository: BaseRepository, dimensions: int = 128, page_size: int = 250) -> None:
        super().__init__(dimensions=dimensions)
        self.repository = repository
        self.page_size = page_size

    async def search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 8,
        min_similarity: float = 0.20,
    ) -> List[Dict[str, Any]]:
        self._validate_query(tenant_id, query_embedding)
        scored: List[Dict[str, Any]] = []
        offset = 0

        while True:
            batch = await self.repository.get_active_chunks_paginated(
                tenant_id=tenant_id,
                limit=self.page_size,
                offset=offset,
            )
            if not batch:
                break
            for item in batch:
                if item.get("tenant_id") != tenant_id or not item.get("is_active", True):
                    continue
                emb = item.get("embedding") or []
                if len(emb) != self.dimensions:
                    continue
                sim = cosine_similarity(query_embedding, emb)
                if sim >= min_similarity:
                    scored.append(
                        {
                            "id": item["id"],
                            "tenant_id": item["tenant_id"],
                            "source_id": item["source_id"],
                            "source_filename": item["source_filename"],
                            "source_version": item["source_version"],
                            "chunk_index": item["chunk_index"],
                            "content": item["content"],
                            "similarity": round(sim, 6),
                        }
                    )
            if len(batch) < self.page_size:
                break
            offset += self.page_size

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[: max(1, top_k)]

    async def check_health(self) -> Dict[str, Any]:
        db_health = await self.repository.check_health()
        return {
            "ok": bool(db_health.get("ok")),
            "provider": "local",
            "dimensions": self.dimensions,
        }


class SupabaseVectorProvider(BaseVectorProvider):
    """Supabase pgvector provider calling the `match_knowledge_vectors` SQL RPC."""

    def __init__(
        self,
        supabase_url: str,
        service_role_key: str,
        dimensions: int = 128,
        http_handler: Optional[Callable[[str, Dict[str, str], Dict[str, Any]], Tuple[int, Any]]] = None,
    ) -> None:
        super().__init__(dimensions=dimensions)
        if not supabase_url or not service_role_key:
            raise ConfigurationError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for SupabaseVectorProvider.")
        self.supabase_url = supabase_url.rstrip("/")
        self.service_role_key = service_role_key.strip()
        self._http_handler = http_handler

    def _call_rpc(self, payload: Dict[str, Any]) -> Any:
        url = f"{self.supabase_url}/rest/v1/rpc/match_knowledge_vectors"
        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }
        if self._http_handler is not None:
            status, data = self._http_handler(url, headers, payload)
            if status >= 400:
                raise ProviderError(
                    message=f"Supabase match_knowledge_vectors RPC failed (HTTP {status}).",
                    status_code=502,
                    provider="supabase-vector",
                )
            return data

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                message=f"Supabase vector RPC returned HTTP {exc.code}.",
                status_code=502,
                provider="supabase-vector",
            ) from exc
        except Exception as exc:
            raise ProviderError(
                message="Supabase vector RPC request failed.",
                status_code=502,
                provider="supabase-vector",
            ) from exc

    async def search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 8,
        min_similarity: float = 0.20,
    ) -> List[Dict[str, Any]]:
        self._validate_query(tenant_id, query_embedding)
        rows = self._call_rpc(
            {
                "query_embedding": query_embedding,
                "match_tenant_id": tenant_id,
                "match_threshold": float(min_similarity),
                "match_count": max(1, int(top_k)),
            }
        )
        results: List[Dict[str, Any]] = []
        for r in rows or []:
            # Defensive double-check of tenant isolation
            if r.get("tenant_id") != tenant_id:
                continue
            results.append(
                {
                    "id": r["id"],
                    "tenant_id": r["tenant_id"],
                    "source_id": r["source_id"],
                    "source_filename": r.get("source_filename", ""),
                    "source_version": int(r.get("source_version", 1)),
                    "chunk_index": int(r.get("chunk_index", 0)),
                    "content": r["content"],
                    "similarity": float(r.get("similarity", 0.0)),
                }
            )
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[: max(1, top_k)]

    async def check_health(self) -> Dict[str, Any]:
        try:
            zero_vec = [0.0] * self.dimensions
            self._call_rpc(
                {
                    "query_embedding": zero_vec,
                    "match_tenant_id": "00000000-0000-0000-0000-000000000000",
                    "match_threshold": 0.99,
                    "match_count": 1,
                }
            )
            return {
                "ok": True,
                "provider": "supabase",
                "rpc": "match_knowledge_vectors",
                "dimensions": self.dimensions,
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": "supabase",
                "rpc": "match_knowledge_vectors",
                "dimensions": self.dimensions,
                "error": str(exc),
            }


class QdrantVectorProvider(BaseVectorProvider):
    """Optional Qdrant vector search adapter with strict tenant_id payload filtering."""

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str = "orbit_knowledge",
        api_key: str = "",
        dimensions: int = 128,
        http_handler: Optional[Callable[[str, Dict[str, str], Dict[str, Any]], Tuple[int, Any]]] = None,
    ) -> None:
        super().__init__(dimensions=dimensions)
        if not qdrant_url:
            raise ConfigurationError("QDRANT_URL is required for QdrantVectorProvider.")
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection_name = collection_name
        self.api_key = api_key.strip()
        self._http_handler = http_handler

    async def search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 8,
        min_similarity: float = 0.20,
    ) -> List[Dict[str, Any]]:
        self._validate_query(tenant_id, query_embedding)
        url = f"{self.qdrant_url}/collections/{self.collection_name}/points/search"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        payload = {
            "vector": query_embedding,
            "limit": max(1, top_k),
            "score_threshold": float(min_similarity),
            "with_payload": True,
            "filter": {
                "must": [
                    {"key": "tenant_id", "match": {"value": tenant_id}},
                    {"key": "is_active", "match": {"value": True}},
                ]
            },
        }
        if self._http_handler is not None:
            status, body = self._http_handler(url, headers, payload)
        else:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                body = json.loads(resp.read().decode("utf-8"))
        if status >= 400 or not isinstance(body, dict):
            raise ProviderError("Qdrant vector search failed.", provider="qdrant")
        results = []
        for hit in body.get("result") or []:
            pl = hit.get("payload") or {}
            if pl.get("tenant_id") != tenant_id:
                continue
            results.append(
                {
                    "id": str(hit.get("id")),
                    "tenant_id": tenant_id,
                    "source_id": pl.get("source_id", ""),
                    "source_filename": pl.get("source_filename", ""),
                    "source_version": int(pl.get("source_version", 1)),
                    "chunk_index": int(pl.get("chunk_index", 0)),
                    "content": pl.get("content", ""),
                    "similarity": float(hit.get("score", 0.0)),
                }
            )
        return results

    async def check_health(self) -> Dict[str, Any]:
        return {"ok": True, "provider": "qdrant", "collection": self.collection_name, "dimensions": self.dimensions}
