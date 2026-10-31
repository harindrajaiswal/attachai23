"""Shared test fixtures and factory helpers for Orbit AI tests.

Compatible with both `pytest` and `python3 -m unittest discover -s tests`.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from app.core.config import Settings
from app.main import OrbitApplication, OrbitTestClient


class IsolatedOrbitTestEnv:
    """Creates a clean temporary SQLite database and storage directory per test."""

    def __init__(self, env_overrides: Optional[Dict[str, Any]] = None) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="orbit_test_"))
        base_env: Dict[str, Any] = {
            "APP_ENV": "test",
            "ADMIN_API_KEY": "test-super-secret-admin-key-2026",
            "DATABASE_PROVIDER": "sqlite",
            "STORAGE_PROVIDER": "local",
            "VECTOR_PROVIDER": "local",
            "NOTIFICATION_PROVIDER": "outbox",
            "LLM_PROVIDER_TYPE": "local",
            "EMBEDDING_PROVIDER": "local",
            "EMBEDDING_DIMENSIONS": "128",
            "SUPABASE_VECTOR_DIMENSIONS": "128",
            "SQLITE_DB_PATH": str(self.temp_dir / "test_orbit.db"),
            "LOCAL_STORAGE_DIR": str(self.temp_dir / "storage"),
            "ALLOWED_ORIGINS": "http://localhost:8000,http://localhost:3000,https://example.com",
        }
        if env_overrides:
            base_env.update(env_overrides)
        self.settings = Settings(env_overrides=base_env)
        self.app = OrbitApplication(settings=self.settings)
        self.client = OrbitTestClient(self.app)
        self.admin_headers = {"x-admin-api-key": self.settings.admin_api_key}

    def create_test_tenant(
        self,
        business_name: str = "Test Company",
        slug: str = "test-company",
        domains: Optional[list[str]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        resp = self.client.post(
            "/api/admin/clients",
            json_body={
                "business_name": business_name,
                "slug": slug,
                "company_name": f"{business_name} Inc.",
                "assistant_name": "Test Assistant",
                "domains": domains if domains is not None else ["example.com", "localhost"],
            },
            headers=self.admin_headers,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        return body["tenant"], body["rawApiKey"]

    def cleanup(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
