"""Tenant lifecycle, duplication, key rotation, domain management, and deletion cleanup tests."""

import unittest
from tests.conftest import IsolatedOrbitTestEnv


class TestTenants(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_create_retrieve_update_duplicate_rotate_and_delete_tenant(self) -> None:
        tenant, raw_key = self.env.create_test_tenant("Orbit Gym", "orbit-gym", domains=["orbitgym.com"])
        tid = tenant["id"]
        self.assertEqual(tenant["business_name"], "Orbit Gym")
        self.assertTrue(raw_key.startswith("orbit_live_"))

        # Update tenant
        updated = self.env.client.patch(
            f"/api/admin/clients/{tid}",
            json_body={"assistant_name": "Coach AI", "temperature": 0.1},
            headers=self.env.admin_headers,
        ).json()
        self.assertEqual(updated["assistant_name"], "Coach AI")

        # Add & remove domain
        dom_add = self.env.client.post(
            f"/api/admin/clients/{tid}/domains",
            json_body={"domain": "app.orbitgym.com"},
            headers=self.env.admin_headers,
        ).json()
        self.assertIn("app.orbitgym.com", dom_add["domains"])

        dom_rem = self.env.client.delete(
            f"/api/admin/clients/{tid}/domains?domain=app.orbitgym.com",
            headers=self.env.admin_headers,
        ).json()
        self.assertNotIn("app.orbitgym.com", dom_rem["domains"])

        # Duplicate tenant
        dup_resp = self.env.client.post(
            f"/api/admin/clients/{tid}/duplicate",
            json_body={"business_name": "Orbit Gym North"},
            headers=self.env.admin_headers,
        )
        self.assertEqual(dup_resp.status_code, 201)
        dup_data = dup_resp.json()
        self.assertEqual(dup_data["tenant"]["business_name"], "Orbit Gym North")
        self.assertNotEqual(dup_data["rawApiKey"], raw_key)

        # Rotate API key -> old key must stop working, new key must work
        rot = self.env.client.post(
            f"/api/admin/clients/{tid}/keys/rotate",
            headers=self.env.admin_headers,
        ).json()
        new_key = rot["rawApiKey"]
        self.assertNotEqual(new_key, raw_key)

        old_check = self.env.client.get("/v1/config", headers={"x-api-key": raw_key})
        self.assertEqual(old_check.status_code, 401)
        new_check = self.env.client.get("/v1/config", headers={"x-api-key": new_key})
        self.assertEqual(new_check.status_code, 200)

        # Upload a file, then delete tenant and verify cleanup
        self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": tid,
                "filename": "gym_rules.txt",
                "content": "Members must bring a towel and rack all weights after workout.",
            },
            headers=self.env.admin_headers,
        )
        del_resp = self.env.client.delete(f"/api/admin/clients/{tid}", headers=self.env.admin_headers)
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(del_resp.json()["deleted_storage_objects"], 1)

        not_found = self.env.client.get(f"/api/admin/clients/{tid}", headers=self.env.admin_headers)
        self.assertEqual(not_found.status_code, 404)


if __name__ == "__main__":
    unittest.main()
