"""Security, PII redaction, rate limiting, and startup configuration validation tests."""

import unittest
from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.utils.pii import redact_data, redact_text
from tests.conftest import IsolatedOrbitTestEnv


class TestSecurity(unittest.TestCase):
    def test_pii_and_secret_redaction(self) -> None:
        raw = (
            "Customer email is jane.doe@example.org and phone is +1 (555) 234-5678. "
            "Token is Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.aaa.bbb "
            "and key is orbit_live_1234567890abcdef and sk-proj1234567890abcdef."
        )
        redacted = redact_text(raw, extra_secrets=["custom-super-secret-val"])
        self.assertNotIn("jane.doe@example.org", redacted)
        self.assertNotIn("234-5678", redacted)
        self.assertNotIn("orbit_live_1234567890abcdef", redacted)
        self.assertNotIn("sk-proj1234567890abcdef", redacted)
        self.assertIn("[REDACTED_EMAIL]", redacted)
        self.assertIn("[REDACTED_PHONE]", redacted)
        self.assertIn("[REDACTED_TOKEN]", redacted)

        nested = {
            "admin_api_key": "my-secret",
            "customer_note": "Reach me at bob@company.io",
        }
        clean_dict = redact_data(nested)
        self.assertEqual(clean_dict["admin_api_key"], "[REDACTED_SECRET]")
        self.assertIn("[REDACTED_EMAIL]", clean_dict["customer_note"])

    def test_startup_validation_rejects_invalid_production_and_dimension_mismatch(self) -> None:
        # Missing strong ADMIN_API_KEY in production
        with self.assertRaises(ConfigurationError):
            Settings(env_overrides={"APP_ENV": "production", "ADMIN_API_KEY": "short"})

        # Vector dimension mismatch
        with self.assertRaises(ConfigurationError):
            Settings(
                env_overrides={
                    "APP_ENV": "development",
                    "EMBEDDING_DIMENSIONS": "256",
                    "SUPABASE_VECTOR_DIMENSIONS": "128",
                }
            )

        # Supabase selected without SUPABASE_URL
        with self.assertRaises(ConfigurationError):
            Settings(
                env_overrides={
                    "APP_ENV": "development",
                    "DATABASE_PROVIDER": "supabase",
                    "SUPABASE_URL": "",
                    "SUPABASE_SERVICE_ROLE_KEY": "",
                }
            )

        # Optional integrations (Qdrant, SMTP, external embeddings, TTS) gracefully fall back when not configured
        graceful = Settings(
            env_overrides={
                "APP_ENV": "development",
                "DATABASE_PROVIDER": "sqlite",
                "STORAGE_PROVIDER": "local",
                "VECTOR_PROVIDER": "qdrant",
                "QDRANT_URL": "",
                "NOTIFICATION_PROVIDER": "smtp",
                "SMTP_HOST": "",
                "EMBEDDING_PROVIDER": "openai-compatible",
                "EMBEDDING_API_KEY": "",
                "TTS_API_KEY": "",
            }
        )
        self.assertEqual(graceful.vector_provider, "local")
        self.assertEqual(graceful.notification_provider, "outbox")
        self.assertEqual(graceful.embedding_provider, "local")
        self.assertFalse(graceful.tts_enabled)

    def test_rate_limiting_and_protected_metrics(self) -> None:
        env = IsolatedOrbitTestEnv(env_overrides={"PUBLIC_RATE_LIMIT_PER_MINUTE": "3"})
        try:
            _, api_key = env.create_test_tenant()
            # Unauthenticated /metrics must be rejected
            self.assertEqual(env.client.get("/metrics").status_code, 401)
            # Admin authenticated /metrics succeeds
            self.assertEqual(env.client.get("/metrics", headers=env.admin_headers).status_code, 200)

            # Exceed public rate limit (3 req/min)
            for _ in range(3):
                r = env.client.get("/v1/config", headers={"x-api-key": api_key})
                self.assertEqual(r.status_code, 200)
            r_limited = env.client.get("/v1/config", headers={"x-api-key": api_key})
            self.assertEqual(r_limited.status_code, 429)
        finally:
            env.cleanup()


if __name__ == "__main__":
    unittest.main()
