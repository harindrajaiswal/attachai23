"""Private file storage providers (LocalStorage and SupabaseStorage).

Fixes Defect #9:
Implements proper pagination (`limit` and `offset`) when listing and deleting
all objects for a tenant so tenants with >1,000 files are fully cleaned up.

Fixes Defect #13:
Uses `<tenant-id>/<source-id>/<safe-filename>` object paths to prevent collisions.
"""

import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.core.exceptions import ConfigurationError, NotFoundError, ProviderError


class BaseStorageProvider(ABC):
    """Contract for private tenant knowledge document storage."""

    @abstractmethod
    async def upload(self, storage_path: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        """Store raw bytes at storage_path (`<tenant-id>/<source-id>/<safe-filename>`)."""

    @abstractmethod
    async def download(self, storage_path: str) -> bytes:
        """Download raw bytes for storage_path."""

    @abstractmethod
    async def delete(self, storage_path: str) -> bool:
        """Delete a single object at storage_path."""

    @abstractmethod
    async def list_tenant_objects(self, tenant_id: str, page_size: int = 100) -> List[str]:
        """Paginate through all objects belonging to tenant_id."""

    @abstractmethod
    async def delete_tenant_objects(self, tenant_id: str, page_size: int = 100) -> int:
        """Delete all objects belonging to tenant_id using paginated listing."""

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Verify storage bucket readiness."""


class LocalStorageProvider(BaseStorageProvider):
    """Filesystem-backed private storage for local development and tests."""

    def __init__(self, base_dir: str = "local_storage") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, storage_path: str) -> Path:
        cleaned = storage_path.strip().lstrip("/\\")
        target = (self.base_dir / cleaned).resolve()
        if not str(target).startswith(str(self.base_dir)):
            raise ProviderError("Invalid storage path traversal attempt.", status_code=400, provider="local-storage")
        return target

    async def upload(self, storage_path: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        target = self._resolve_safe_path(storage_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return storage_path

    async def download(self, storage_path: str) -> bytes:
        target = self._resolve_safe_path(storage_path)
        if not target.exists() or not target.is_file():
            raise NotFoundError(f"Stored object '{storage_path}' not found.")
        return target.read_bytes()

    async def delete(self, storage_path: str) -> bool:
        target = self._resolve_safe_path(storage_path)
        if target.exists() and target.is_file():
            target.unlink()
            return True
        return False

    async def list_tenant_objects(self, tenant_id: str, page_size: int = 100) -> List[str]:
        tenant_dir = self._resolve_safe_path(tenant_id)
        if not tenant_dir.exists():
            return []
        all_files = sorted(
            str(p.relative_to(self.base_dir)).replace("\\", "/")
            for p in tenant_dir.rglob("*")
            if p.is_file()
        )
        # Paginate in slices of page_size
        collected: List[str] = []
        offset = 0
        while offset < len(all_files):
            page = all_files[offset : offset + max(1, page_size)]
            collected.extend(page)
            offset += max(1, page_size)
        return collected

    async def delete_tenant_objects(self, tenant_id: str, page_size: int = 100) -> int:
        objects = await self.list_tenant_objects(tenant_id, page_size=page_size)
        count = 0
        for obj_path in objects:
            if await self.delete(obj_path):
                count += 1
        tenant_dir = self._resolve_safe_path(tenant_id)
        if tenant_dir.exists():
            shutil.rmtree(tenant_dir, ignore_errors=True)
        return count

    async def check_health(self) -> Dict[str, Any]:
        return {
            "ok": self.base_dir.exists() and self.base_dir.is_dir(),
            "provider": "local",
            "bucket": str(self.base_dir.name),
            "private": True,
        }


class SupabaseStorageProvider(BaseStorageProvider):
    """Supabase private bucket storage with full pagination support."""

    def __init__(
        self,
        supabase_url: str,
        service_role_key: str,
        bucket: str = "knowledge",
        http_handler: Optional[Callable[[str, str, Dict[str, str], Optional[bytes]], Tuple[int, Any]]] = None,
    ) -> None:
        if not supabase_url or not service_role_key or not bucket:
            raise ConfigurationError(
                "SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and SUPABASE_STORAGE_BUCKET are required."
            )
        self.supabase_url = supabase_url.rstrip("/")
        self.service_role_key = service_role_key.strip()
        self.bucket = bucket.strip()
        self._http_handler = http_handler

    def _headers(self, content_type: str = "application/json") -> Dict[str, str]:
        return {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": content_type,
        }

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        content_type: str = "application/json",
    ) -> Any:
        url = f"{self.supabase_url}{path}"
        headers = self._headers(content_type=content_type)
        if self._http_handler is not None:
            status, data = self._http_handler(method, url, headers, body)
            if status >= 400:
                raise ProviderError(
                    f"Supabase Storage operation failed with HTTP {status}.",
                    status_code=502,
                    provider="supabase-storage",
                )
            return data

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                if content_type == "application/json" and raw:
                    return json.loads(raw.decode("utf-8"))
                return raw
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"Supabase Storage returned HTTP {exc.code}.",
                status_code=502,
                provider="supabase-storage",
            ) from exc
        except Exception as exc:
            raise ProviderError(
                "Supabase Storage request failed.",
                status_code=502,
                provider="supabase-storage",
            ) from exc

    def _ensure_bucket(self) -> None:
        if self._http_handler is not None:
            return
        try:
            payload = json.dumps(
                {"id": self.bucket, "name": self.bucket, "public": False, "file_size_limit": 10485760}
            ).encode("utf-8")
            self._request("POST", "/storage/v1/bucket", body=payload)
        except Exception:
            pass

    async def upload(self, storage_path: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        encoded = urllib.parse.quote(storage_path.lstrip("/"), safe="/")
        try:
            self._request(
                "POST",
                f"/storage/v1/object/{self.bucket}/{encoded}",
                body=content,
                content_type=content_type,
            )
        except ProviderError:
            if self._http_handler is None:
                self._ensure_bucket()
                self._request(
                    "POST",
                    f"/storage/v1/object/{self.bucket}/{encoded}",
                    body=content,
                    content_type=content_type,
                )
            else:
                raise
        return storage_path

    async def download(self, storage_path: str) -> bytes:
        encoded = urllib.parse.quote(storage_path.lstrip("/"), safe="/")
        data = self._request(
            "GET",
            f"/storage/v1/object/{self.bucket}/{encoded}",
            content_type="application/octet-stream",
        )
        return data if isinstance(data, bytes) else bytes(data)

    async def delete(self, storage_path: str) -> bool:
        body = json.dumps({"prefixes": [storage_path.lstrip("/")]}).encode("utf-8")
        self._request("DELETE", f"/storage/v1/object/{self.bucket}", body=body)
        return True

    async def list_tenant_objects(self, tenant_id: str, page_size: int = 100) -> List[str]:
        """Paginate through all objects under `<tenant_id>/` using limit & offset."""
        collected: List[str] = []
        offset = 0
        prefix = tenant_id.strip().strip("/")
        while True:
            payload = json.dumps(
                {
                    "prefix": prefix,
                    "limit": max(1, page_size),
                    "offset": offset,
                    "sortBy": {"column": "name", "order": "asc"},
                }
            ).encode("utf-8")
            items = self._request("POST", f"/storage/v1/object/list/{self.bucket}", body=payload)
            if not isinstance(items, list) or not items:
                break
            for item in items:
                name = item.get("name")
                if name:
                    full_path = name if name.startswith(prefix + "/") else f"{prefix}/{name}"
                    collected.append(full_path)
            if len(items) < page_size:
                break
            offset += page_size
        return collected

    async def delete_tenant_objects(self, tenant_id: str, page_size: int = 100) -> int:
        all_paths = await self.list_tenant_objects(tenant_id, page_size=page_size)
        if not all_paths:
            return 0
        deleted = 0
        for idx in range(0, len(all_paths), max(1, page_size)):
            batch = all_paths[idx : idx + max(1, page_size)]
            body = json.dumps({"prefixes": batch}).encode("utf-8")
            self._request("DELETE", f"/storage/v1/object/{self.bucket}", body=body)
            deleted += len(batch)
        return deleted

    async def check_health(self) -> Dict[str, Any]:
        try:
            try:
                info = self._request("GET", f"/storage/v1/bucket/{self.bucket}")
            except ProviderError:
                if self._http_handler is None:
                    self._ensure_bucket()
                    info = self._request("GET", f"/storage/v1/bucket/{self.bucket}")
                else:
                    raise
            is_public = bool(info.get("public", False)) if isinstance(info, dict) else False
            if is_public:
                return {
                    "ok": False,
                    "provider": "supabase",
                    "bucket": self.bucket,
                    "error": f"Storage bucket '{self.bucket}' must be private, not public.",
                }
            return {
                "ok": True,
                "provider": "supabase",
                "bucket": self.bucket,
                "private": True,
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": "supabase",
                "bucket": self.bucket,
                "error": str(exc),
            }
