"""Supabase SQL migration verification, vector RPC contract, and LLM retry/error handling tests."""

import asyncio
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from app.core.exceptions import ConfigurationError, ProviderError
from app.providers.embeddings import OpenAICompatibleEmbeddingProvider
from app.providers.llm import OpenAICompatibleLlmProvider
from app.providers.vector import SupabaseVectorProvider
from app.repositories.supabase import SupabaseRepository


class TestSupabaseAndProviders(unittest.TestCase):
    def test_sql_migration_contains_all_14_tables_and_128d_vector_rpc(self) -> None:
        sql_path = Path(__file__).resolve().parent.parent / "supabase" / "migrations" / "0001_initial.sql"
        self.assertTrue(sql_path.exists())
        sql_text = sql_path.read_text(encoding="utf-8")

        for table in SupabaseRepository.REQUIRED_TABLES:
            self.assertIn(f"CREATE TABLE IF NOT EXISTS public.{table}", sql_text)

        self.assertIn("vector(128)", sql_text)
        self.assertIn("match_knowledge_vectors", sql_text)
        self.assertIn("ENABLE ROW LEVEL SECURITY", sql_text)
        self.assertIn("'knowledge'", sql_text)

    def test_supabase_vector_rpc_filters_by_tenant_and_validates_dimension(self) -> None:
        def mock_rpc(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, Any]:
            self.assertEqual(payload["match_tenant_id"], "tenant-123")
            self.assertEqual(len(payload["query_embedding"]), 128)
            return 200, [
                {
                    "id": "vec-1",
                    "tenant_id": "tenant-123",
                    "source_id": "src-1",
                    "source_filename": "guide.txt",
                    "source_version": 1,
                    "chunk_index": 0,
                    "content": "Verified chunk content",
                    "similarity": 0.88,
                },
                # Simulated rogue cross-tenant row must be filtered out defensively
                {
                    "id": "vec-rogue",
                    "tenant_id": "other-tenant",
                    "source_id": "src-2",
                    "source_filename": "secret.txt",
                    "source_version": 1,
                    "chunk_index": 0,
                    "content": "Should be excluded",
                    "similarity": 0.95,
                },
            ]

        vec_prov = SupabaseVectorProvider(
            supabase_url="https://example.supabase.co",
            service_role_key="service-role-secret",
            dimensions=128,
            http_handler=mock_rpc,
        )
        hits = asyncio.run(vec_prov.search("tenant-123", [0.1] * 128, top_k=5))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["tenant_id"], "tenant-123")

        # Dimension mismatch must raise ProviderError
        with self.assertRaises(ProviderError):
            asyncio.run(vec_prov.search("tenant-123", [0.1] * 64, top_k=5))

    def test_embeddings_dimension_mismatch_rejected(self) -> None:
        def bad_emb_http(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, Any]:
            return 200, {"data": [{"embedding": [0.1] * 64}]}

        emb = OpenAICompatibleEmbeddingProvider(
            base_url="https://api.example.com/v1",
            api_key="sk-test1234567890",
            model="text-embedding-3-small",
            dimensions=128,
            http_handler=bad_emb_http,
        )
        with self.assertRaises(ProviderError):
            asyncio.run(emb.embed_query("hello"))

    def test_llm_retries_transient_429_and_fails_fast_on_401(self) -> None:
        call_count = 0

        def transient_then_ok(
            method: str, url: str, headers: Dict[str, str], body: Optional[bytes]
        ) -> Tuple[int, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 429, {"error": "rate limited"}
            return 200, {"choices": [{"message": {"content": "Recovered answer."}}]}

        llm = OpenAICompatibleLlmProvider(
            provider_name="openai",
            base_url="https://api.example.com/v1",
            api_key="sk-secret-llm-key",
            model="gpt-4o-mini",
            max_retries=2,
            http_handler=transient_then_ok,
        )
        ans = asyncio.run(
            llm.generate_answer(
                system_prompt="You are helpful.",
                question="Test?",
                evidence_chunks=[{"source_filename": "a.txt", "chunk_index": 0, "content": "Fact"}],
                conversation_history=[],
            )
        )
        self.assertEqual(ans, "Recovered answer.")
        self.assertEqual(call_count, 2)

        # 401 unauthorized must fail immediately without retrying
        auth_calls = 0

        def unauthorized_http(
            method: str, url: str, headers: Dict[str, str], body: Optional[bytes]
        ) -> Tuple[int, Any]:
            nonlocal auth_calls
            auth_calls += 1
            return 401, {"error": "invalid api key"}

        llm_unauth = OpenAICompatibleLlmProvider(
            provider_name="openai",
            base_url="https://api.example.com/v1",
            api_key="sk-bad-key",
            model="gpt-4o-mini",
            max_retries=2,
            http_handler=unauthorized_http,
        )
        with self.assertRaises(ProviderError):
            asyncio.run(
                llm_unauth.generate_answer("Prompt", "Q?", [{"content": "Fact"}], [])
            )
        self.assertEqual(auth_calls, 1)


if __name__ == "__main__":
    unittest.main()
