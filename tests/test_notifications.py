"""Notification outbox persistence and retry mechanism tests."""

import unittest
from tests.conftest import IsolatedOrbitTestEnv


class TestNotifications(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()
        self.tenant, self.api_key = self.env.create_test_tenant()

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_outbox_retry_recovers_failed_notifications(self) -> None:
        should_fail = True

        def toggle_hook(_notif: dict) -> None:
            if should_fail:
                raise ConnectionError("Temporary webhook timeout")

        self.env.app.container.notification_provider._send_hook = toggle_hook

        self.env.client.post(
            "/v1/leads",
            json_body={"name": "Retry User", "email": "retry@example.com"},
            headers={"x-api-key": self.api_key},
        )
        notifs = self.env.client.get("/api/admin/notifications", headers=self.env.admin_headers).json()
        self.assertEqual(notifs[0]["status"], "failed")

        # Restore connectivity and trigger retry
        should_fail = False
        retry_res = self.env.client.post("/api/admin/notifications/retry", headers=self.env.admin_headers).json()
        self.assertEqual(retry_res["retried"], 1)

        notifs_after = self.env.client.get("/api/admin/notifications", headers=self.env.admin_headers).json()
        self.assertEqual(notifs_after[0]["status"], "sent")


if __name__ == "__main__":
    unittest.main()
