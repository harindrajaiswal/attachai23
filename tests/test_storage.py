"""Storage provider upload, download, collision prevention, and paginated deletion tests."""

import asyncio
import unittest
from typing import Any, Dict, List, Optional, Tuple
from app.providers.storage import LocalStorageProvider, SupabaseStorageProvider
from app.utils.validators import build_storage_object_path
from tests.conftest import IsolatedOrbitTestEnv


class TestStorage(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_local_storage_collision_prevention_and_pagination(self) -> None:
        storage = LocalStorageProvider(base_dir=str(self.env.temp_dir / "store_test"))
        p1 = build_storage_object_path("tenant-1", "src-aaa", "My File (1).txt")
        p2 = build_storage_object_path("tenant-1", "src-bbb", "My File [1].txt")
        self.assertNotEqual(p1, p2)

        asyncio.run(storage.upload(p1, b"alpha"))
        asyncio.run(storage.upload(p2, b"beta"))
        self.assertEqual(asyncio.run(storage.download(p1)), b"alpha")
        self.assertEqual(asyncio.run(storage.download(p2)), b"beta")

        # Paginated listing with page_size=1
        listed = asyncio.run(storage.list_tenant_objects("tenant-1", page_size=1))
        self.assertEqual(len(listed), 2)
        deleted = asyncio.run(storage.delete_tenant_objects("tenant-1", page_size=1))
        self.assertEqual(deleted, 2)
        self.assertEqual(asyncio.run(storage.list_tenant_objects("tenant-1")), [])

    def test_supabase_storage_paginated_listing_and_deletion(self) -> None:
        calls: List[Tuple[str, str]] = []

        def mock_http(
            method: str, url: str, headers: Dict[str, str], body: Optional[bytes]
        ) -> Tuple[int, Any]:
            calls.append((method, url))
            if "/object/list/" in url and body:
                import json
                req = json.loads(body.decode("utf-8"))
                offset = req.get("offset", 0)
                if offset == 0:
                    return 200, [{"name": "doc1.txt"}, {"name": "doc2.txt"}]
                if offset == 2:
                    return 200, [{"name": "doc3.txt"}]
                return 200, []
            return 200, {"public": False}

        sb_store = SupabaseStorageProvider(
            supabase_url="https://example.supabase.co",
            service_role_key="sbp_test_secret_key_123456",
            bucket="knowledge",
            http_handler=mock_http,
        )
        objs = asyncio.run(sb_store.list_tenant_objects("tenant-xyz", page_size=2))
        self.assertEqual(len(objs), 3)
        del_count = asyncio.run(sb_store.delete_tenant_objects("tenant-xyz", page_size=2))
        self.assertEqual(del_count, 3)


if __name__ == "__main__":
    unittest.main()
