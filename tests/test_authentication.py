"""Authentication and authorization tests (Admin & Tenant keys, domain checks, cross-tenant isolation)."""

import unittest
from tests.conftest import IsolatedOrbitTestEnv


class TestAuthentication(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_missing_admin_key_rejected(self) -> None:
        resp = self.env.client.get("/api/admin/clients")
        self.assertEqual(resp.status_code, 401)
        self.assertIn("AUTHENTICATION_FAILED", resp.json()["code"])

    def test_invalid_admin_key_rejected(self) -> None:
        resp = self.env.client.get(
            "/api/admin/clients",
            headers={"x-admin-api-key": "wrong-admin-key"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_valid_admin_key_accepted(self) -> None:
        resp = self.env.client.get("/api/admin/clients", headers=self.env.admin_headers)
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_admin_session_login_and_csrf_protection(self) -> None:
        bad = self.env.client.post("/api/admin/login", json_body={"admin_api_key": "bad-key"})
        self.assertEqual(bad.status_code, 401)

        good = self.env.client.post(
            "/api/admin/login",
            json_body={"admin_api_key": self.env.settings.admin_api_key},
        )
        self.assertEqual(good.status_code, 200)
        csrf_token = good.json()["csrfToken"]
        self.assertTrue(csrf_token)

        # State-changing request with session cookie but missing CSRF token must fail (403)
        no_csrf = self.env.client.post(
            "/api/admin/clients",
            json_body={"business_name": "Blocked Co", "slug": "blocked-co"},
        )
        self.assertEqual(no_csrf.status_code, 403)

        # With valid CSRF header it succeeds
        with_csrf = self.env.client.post(
            "/api/admin/clients",
            json_body={"business_name": "Allowed Co", "slug": "allowed-co"},
            headers={"x-csrf-token": csrf_token},
        )
        self.assertEqual(with_csrf.status_code, 201)

    def test_missing_and_invalid_tenant_api_key(self) -> None:
        resp_missing = self.env.client.get("/v1/config")
        self.assertEqual(resp_missing.status_code, 401)

        resp_invalid = self.env.client.get("/v1/config", headers={"x-api-key": "orbit_live_invalid12345"})
        self.assertEqual(resp_invalid.status_code, 401)

    def test_disabled_tenant_and_disabled_api_key_rejected(self) -> None:
        tenant, raw_key = self.env.create_test_tenant()
        keys = self.env.client.get(f"/api/admin/clients/{tenant['id']}", headers=self.env.admin_headers).json()["api_keys"]
        key_id = keys[0]["id"]

        # Disable key
        self.env.client.patch(
            f"/api/admin/clients/{tenant['id']}/keys/{key_id}",
            json_body={"enabled": False},
            headers=self.env.admin_headers,
        )
        resp = self.env.client.get("/v1/config", headers={"x-api-key": raw_key})
        self.assertEqual(resp.status_code, 401)

        # Re-enable key, then disable tenant
        self.env.client.patch(
            f"/api/admin/clients/{tenant['id']}/keys/{key_id}",
            json_body={"enabled": True},
            headers=self.env.admin_headers,
        )
        self.env.client.patch(
            f"/api/admin/clients/{tenant['id']}",
            json_body={"enabled": False},
            headers=self.env.admin_headers,
        )
        resp_disabled_tenant = self.env.client.get("/v1/config", headers={"x-api-key": raw_key})
        self.assertEqual(resp_disabled_tenant.status_code, 403)

    def test_domain_allowlist_and_cross_tenant_rejection(self) -> None:
        t1, key1 = self.env.create_test_tenant("Tenant One", "tenant-one", domains=["allowed.example.com"])
        t2, _ = self.env.create_test_tenant("Tenant Two", "tenant-two", domains=["two.example.com"])

        # Allowed origin succeeds
        ok_resp = self.env.client.get(
            "/v1/config",
            headers={"x-api-key": key1, "Origin": "https://allowed.example.com"},
        )
        self.assertEqual(ok_resp.status_code, 200)

        # Unauthorized origin fails with 403
        bad_origin = self.env.client.get(
            "/v1/config",
            headers={"x-api-key": key1, "Origin": "https://evil-domain.org"},
        )
        self.assertEqual(bad_origin.status_code, 403)

        # Cross-tenant client ID header mismatch fails with 403
        cross_resp = self.env.client.get(
            "/v1/config",
            headers={"x-api-key": key1, "x-client-id": t2["id"]},
        )
        self.assertEqual(cross_resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
