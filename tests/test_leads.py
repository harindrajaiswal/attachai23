"""Lead capture, idempotency, notification failure resilience, workflow status updates, and CSV export tests."""

import unittest
from tests.conftest import IsolatedOrbitTestEnv


class TestLeads(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()
        self.tenant, self.api_key = self.env.create_test_tenant()

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_lead_idempotency_workflow_and_csv_export(self) -> None:
        payload = {
            "name": "Marcus Vance",
            "email": "marcus@example.com",
            "phone": "+1-555-0199",
            "requirement": "Enterprise Plan Quote",
            "message": "Need quote for 50 seats.",
            "idempotencyKey": "idem-marcus-001",
        }
        r1 = self.env.client.post("/v1/leads", json_body=payload, headers={"x-api-key": self.api_key})
        self.assertEqual(r1.status_code, 201)
        lead_id = r1.json()["id"]
        self.assertTrue(r1.json()["createdNew"])

        # Duplicate submission with same idempotencyKey returns existing lead (200) without duplicating (Fixes Defect #11)
        r2 = self.env.client.post("/v1/leads", json_body=payload, headers={"x-api-key": self.api_key})
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["id"], lead_id)
        self.assertFalse(r2.json()["createdNew"])

        leads_list = self.env.client.get(
            f"/api/admin/leads?tenant_id={self.tenant['id']}",
            headers=self.env.admin_headers,
        ).json()
        self.assertEqual(len(leads_list), 1)

        # Update lead status and assignee
        updated = self.env.client.patch(
            f"/api/admin/leads/{self.tenant['id']}/{lead_id}",
            json_body={"status": "qualified", "assignee": "Alex Rivera", "internal_notes": "Scheduled demo call"},
            headers=self.env.admin_headers,
        ).json()
        self.assertEqual(updated["status"], "qualified")
        self.assertEqual(updated["assignee"], "Alex Rivera")

        # Export CSV
        csv_resp = self.env.client.get(
            f"/api/admin/leads/export?tenant_id={self.tenant['id']}",
            headers=self.env.admin_headers,
        )
        self.assertEqual(csv_resp.status_code, 200)
        self.assertIn("marcus@example.com", csv_resp.text)
        self.assertIn("qualified", csv_resp.text)

    def test_notification_failure_does_not_lose_or_duplicate_lead(self) -> None:
        # Configure notification hook to throw an error (Fixes Defect #12)
        def failing_hook(_notif: dict) -> None:
            raise RuntimeError("Simulated SMTP outage")

        self.env.app.container.notification_provider._send_hook = failing_hook

        resp = self.env.client.post(
            "/v1/leads",
            json_body={
                "name": "Resilient Customer",
                "email": "resilient@example.com",
                "message": "Testing outbox resilience",
            },
            headers={"x-api-key": self.api_key},
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["notification_status"], "failed")

        # Lead is safely stored in DB
        leads = self.env.client.get(
            f"/api/admin/leads?tenant_id={self.tenant['id']}",
            headers=self.env.admin_headers,
        ).json()
        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["email"], "resilient@example.com")


if __name__ == "__main__":
    unittest.main()
