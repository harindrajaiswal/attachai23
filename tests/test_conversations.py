"""Conversation persistence, session reuse, and cross-tenant session isolation tests."""

import unittest
from tests.conftest import IsolatedOrbitTestEnv


class TestConversations(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()
        self.t1, self.key1 = self.env.create_test_tenant("Tenant Alpha", "tenant-alpha")
        self.t2, self.key2 = self.env.create_test_tenant("Tenant Beta", "tenant-beta")

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_cross_tenant_conversation_access_blocked(self) -> None:
        c1 = self.env.client.post(
            "/v1/chat",
            json_body={"message": "Hello Alpha", "sessionId": "alpha-sess"},
            headers={"x-api-key": self.key1},
        ).json()
        conv_id = c1["conversationId"]

        # Tenant 2 trying to use Tenant 1's conversationId must be rejected with 403
        cross = self.env.client.post(
            "/v1/chat",
            json_body={"message": "Hijack attempt", "conversationId": conv_id},
            headers={"x-api-key": self.key2},
        )
        self.assertEqual(cross.status_code, 403)

        # Admin inspecting conversation under wrong tenant ID returns 404
        admin_wrong = self.env.client.get(
            f"/api/admin/conversations/{self.t2['id']}/{conv_id}",
            headers=self.env.admin_headers,
        )
        self.assertEqual(admin_wrong.status_code, 404)


if __name__ == "__main__":
    unittest.main()
