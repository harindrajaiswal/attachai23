"""Chat, SSE streaming, conversation memory, and LLM summary tests."""

import unittest
from tests.conftest import IsolatedOrbitTestEnv


class TestChatAndConversations(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()
        self.tenant, self.api_key = self.env.create_test_tenant()
        self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.tenant["id"],
                "filename": "hours_and_pricing.txt",
                "content": (
                    "Our clinic is open Monday to Friday from 8:00am to 6:00pm. "
                    "Initial dental consultations cost $95 and teeth whitening costs $250."
                ),
            },
            headers=self.env.admin_headers,
        )

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_chat_follow_up_and_actual_summary_generation(self) -> None:
        # Turn 1
        r1 = self.env.client.post(
            "/v1/chat",
            json_body={"message": "What are the clinic opening hours?", "sessionId": "sess-100"},
            headers={"x-api-key": self.api_key},
        ).json()
        conv_id = r1["conversationId"]
        self.assertIn("8:00am", r1["answer"])

        # Turn 2 (reusing sessionId/conversationId)
        r2 = self.env.client.post(
            "/v1/chat",
            json_body={
                "message": "How much do initial dental consultations cost?",
                "sessionId": "sess-100",
                "conversationId": conv_id,
            },
            headers={"x-api-key": self.api_key},
        ).json()
        self.assertEqual(r2["conversationId"], conv_id)
        self.assertIn("$95", r2["answer"])

        # Turn 3: Request a conversation summary (Fixes Defect #2)
        r3 = self.env.client.post(
            "/v1/chat",
            json_body={
                "message": "Can you summarize our conversation so far?",
                "sessionId": "sess-100",
                "conversationId": conv_id,
            },
            headers={"x-api-key": self.api_key},
        ).json()
        self.assertFalse(r3["isFallback"])
        self.assertIn("Conversation Summary", r3["answer"])
        self.assertIn("opening hours", r3["answer"].lower())

    def test_sse_streaming_chat_returns_validated_events(self) -> None:
        stream_resp = self.env.client.post(
            "/v1/chat/stream",
            json_body={"message": "How much does teeth whitening cost?"},
            headers={"x-api-key": self.api_key},
        )
        self.assertEqual(stream_resp.status_code, 200)
        self.assertIn("text/event-stream", stream_resp.headers.get("Content-Type", ""))
        body_text = stream_resp.text
        self.assertIn("event: token", body_text)
        self.assertIn("event: done", body_text)
        self.assertIn("$250", body_text)


if __name__ == "__main__":
    unittest.main()
