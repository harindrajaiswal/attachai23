"""Comprehensive Test Suite for Attach-AI Natural Conversational Intelligence Overhaul.

Covers Scenarios A through H from Section 14, Section 1 regression cases,
streaming parity (/v1/chat/stream), and OpenAICompatibleLlmProvider conversational LLM routing.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any, Dict, List

from app.domain.enums import SupportLevel
from app.providers.llm import OpenAICompatibleLlmProvider
from tests.conftest import IsolatedOrbitTestEnv


class ConversationalIntelligenceTestSuite(unittest.TestCase):
    """End-to-end validation of Scenarios A–H and Section 1 failure regressions."""

    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv(env_overrides={"APP_ENV": "development"})
        self.app = self.env.app
        self.container = self.app.container
        asyncio.run(self.app.ensure_initialized())
        self.client = self.env.client

        # Create Attach-Ai tenant with business knowledge base (plans, pricing, refund policy, features)
        self.tenant, raw_key = asyncio.run(
            self.container.client_service.create_tenant(
                {
                    "business_name": "Attach-Ai",
                    "slug": "attach-ai-test",
                    "assistant_name": "Test Assistant",
                    "company_name": "Attach-Ai",
                    "system_prompt": "You are Test Assistant for Attach-Ai. Help users naturally and accurately.",
                    "fallback_answer": (
                        "I don't have confirmed details on that right now. "
                        "Please let me know if I can help with our plans, pricing, or policies."
                    ),
                    "top_k": 5,
                    "min_similarity": 0.20,
                }
            )
        )
        self.api_key = str(raw_key)
        self.headers = {"X-API-Key": self.api_key}

        # Seed business knowledge base: Pricing plans, refund policy, features, booking channels
        kb_text = (
            "Attach-Ai Pricing Plans:\n"
            "1. Starter Plan (First Plan): $29 per month. Includes 1 AI assistant, 1,000 messages/month, "
            "and standard email support. Best for solo founders and small websites.\n"
            "2. Growth Plan (Second Plan): $79 per month. Includes 5 AI assistants, 10,000 messages/month, "
            "WhatsApp & website widget integration, priority support, and custom branding. "
            "The Growth Plan is better than the Starter Plan for growing teams that need higher volume and multi-channel support.\n\n"
            "Refund and Cancellation Policy:\n"
            "We offer a 14-day full refund policy for all new subscriptions if you are not satisfied. "
            "Subscriptions can be cancelled anytime from the billing dashboard with no cancellation fee.\n\n"
            "Booking Channels:\n"
            "Customers can book demos via our website calendar or WhatsApp automation channel."
        )
        asyncio.run(
            self.container.knowledge_service.ingest_document(
                tenant_id=self.tenant["id"],
                filename="attach_ai_guide.txt",
                content=kb_text.encode("utf-8"),
                content_type="text/plain",
            )
        )

    def tearDown(self) -> None:
        self.env.cleanup()

    def _chat(self, message: str, session_id: str, conversation_id: str | None = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"message": message, "sessionId": session_id}
        if conversation_id:
            body["conversationId"] = conversation_id
        resp = self.client.post("/v1/chat", headers=self.headers, json_body=body)
        self.assertEqual(resp.status_code, 200, f"Failed on message {message!r}: {resp.text}")
        return resp.json()

    # =========================================================================
    # SECTION 1 REGRESSION TESTS (Exact failure examples from problem statement)
    # =========================================================================

    def test_section1_exact_failure_regressions(self) -> None:
        """Verify every failure from Section 1 now receives a natural, intelligent response."""
        sess = "sess-sec1-regression"

        # 1. User: Hello
        r1 = self._chat("Hello", sess)
        conv_id = r1["conversationId"]
        self.assertEqual(r1["intent"], "GREETING")
        self.assertFalse(r1["isFallback"])
        self.assertNotIn("plans, pricing, features, policies, and support options", r1["answer"])

        # 2. User: Aap kaun ho? (Must NOT dump random KB chunk like 'Hamare verified details ke mutabik: Booking Channels...')
        r2 = self._chat("Aap kaun ho?", sess, conv_id)
        self.assertEqual(r2["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r2["isFallback"])
        self.assertNotIn("Hamare verified details ke mutabik", r2["answer"])
        self.assertNotIn("Booking Channels", r2["answer"])
        self.assertIn("Test Assistant", r2["answer"])
        self.assertIn("Attach-Ai", r2["answer"])

        # 3. User: Hindi mein likho. (Must NOT trigger KB fallback)
        r3 = self._chat("Hindi mein likho.", sess, conv_id)
        self.assertEqual(r3["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r3["isFallback"])
        self.assertNotIn("I do not have enough verified information", r3["answer"])

        # 4. User: Kya aap Hindi mein baat nahi kar sakte? (Must confirm Hindi capability naturally)
        r4 = self._chat("Kya aap Hindi mein baat nahi kar sakte?", sess, conv_id)
        self.assertEqual(r4["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r4["isFallback"])
        self.assertNotIn("verified knowledge base", r4["answer"].lower())
        self.assertTrue("hindi" in r4["answer"].lower() or "हिंदी" in r4["answer"])

        # 5. User: Tum kya karte ho? (Must explain assistant capabilities, not fallback)
        r5 = self._chat("Tum kya karte ho?", sess, conv_id)
        self.assertEqual(r5["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r5["isFallback"])
        self.assertNotIn("verified knowledge base", r5["answer"].lower())

        # 6. User: Humne tumse pehle kya poochha tha? (Must recall earlier user questions from conversation history)
        r6 = self._chat("Humne tumse pehle kya poochha tha?", sess, conv_id)
        self.assertEqual(r6["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r6["isFallback"])
        self.assertTrue(
            "Aap kaun ho" in r6["answer"]
            or "Tum kya karte ho" in r6["answer"]
            or "Hindi" in r6["answer"]
        )

    # =========================================================================
    # SCENARIO A — BASIC CONVERSATION
    # =========================================================================

    def test_scenario_a_basic_conversation_english(self) -> None:
        sess = "sess-scenario-a"
        r1 = self._chat("Hello", sess)
        conv_id = r1["conversationId"]
        self.assertEqual(r1["intent"], "GREETING")
        self.assertFalse(r1["isFallback"])

        r2 = self._chat("Who are you?", sess, conv_id)
        self.assertEqual(r2["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r2["isFallback"])
        self.assertIn("Test Assistant", r2["answer"])
        self.assertIn("Attach-Ai", r2["answer"])

        r3 = self._chat("What can you do?", sess, conv_id)
        self.assertEqual(r3["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r3["isFallback"])
        self.assertTrue("help" in r3["answer"].lower() or "answer" in r3["answer"].lower())

        r4 = self._chat("Can you speak Hindi?", sess, conv_id)
        self.assertEqual(r4["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r4["isFallback"])
        self.assertIn("Hindi", r4["answer"])

    # =========================================================================
    # SCENARIO B — HINDI CONVERSATION
    # =========================================================================

    def test_scenario_b_hindi_conversation(self) -> None:
        sess = "sess-scenario-b"
        r1 = self._chat("नमस्ते", sess)
        conv_id = r1["conversationId"]
        self.assertEqual(r1["intent"], "GREETING")
        self.assertEqual(r1["language"], "hi")
        self.assertFalse(r1["isFallback"])
        self.assertIn("नमस्ते", r1["answer"])

        r2 = self._chat("आप कौन हो?", sess, conv_id)
        self.assertEqual(r2["intent"], "GENERAL_CONVERSATION")
        self.assertEqual(r2["language"], "hi")
        self.assertFalse(r2["isFallback"])
        self.assertIn("Test Assistant", r2["answer"])
        self.assertIn("Attach-Ai", r2["answer"])

        r3 = self._chat("तुम क्या कर सकते हो?", sess, conv_id)
        self.assertEqual(r3["intent"], "GENERAL_CONVERSATION")
        self.assertEqual(r3["language"], "hi")
        self.assertFalse(r3["isFallback"])
        self.assertIn("मदद", r3["answer"])

        r4 = self._chat("मैंने तुमसे पहले क्या पूछा था?", sess, conv_id)
        self.assertEqual(r4["intent"], "GENERAL_CONVERSATION")
        self.assertEqual(r4["language"], "hi")
        self.assertFalse(r4["isFallback"])
        self.assertTrue("आप कौन हो" in r4["answer"] or "तुम क्या कर सकते हो" in r4["answer"])

    # =========================================================================
    # SCENARIO C — HINGLISH CONVERSATION
    # =========================================================================

    def test_scenario_c_hinglish_conversation(self) -> None:
        sess = "sess-scenario-c"
        r1 = self._chat("Hello, kaise ho?", sess)
        conv_id = r1["conversationId"]
        self.assertEqual(r1["intent"], "GREETING")
        self.assertEqual(r1["language"], "hinglish")
        self.assertFalse(r1["isFallback"])

        r2 = self._chat("Aap kaun ho?", sess, conv_id)
        self.assertEqual(r2["intent"], "GENERAL_CONVERSATION")
        self.assertEqual(r2["language"], "hinglish")
        self.assertFalse(r2["isFallback"])
        self.assertIn("Test Assistant", r2["answer"])

        r3 = self._chat("Hindi mein baat karo", sess, conv_id)
        self.assertEqual(r3["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r3["isFallback"])
        self.assertNotIn("knowledge base", r3["answer"].lower())

        r4 = self._chat("Aap kya karte ho?", sess, conv_id)
        self.assertEqual(r4["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r4["isFallback"])
        self.assertNotIn("Hamare verified details ke mutabik", r4["answer"])

        r5 = self._chat("Humne pehle kya poochha tha?", sess, conv_id)
        self.assertEqual(r5["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r5["isFallback"])
        self.assertTrue("Aap kaun ho" in r5["answer"] or "Aap kya karte ho" in r5["answer"])

    # =========================================================================
    # SCENARIO D — LANGUAGE SWITCHING
    # =========================================================================

    def test_scenario_d_language_switching(self) -> None:
        sess = "sess-scenario-d"
        r1 = self._chat("What do you do?", sess)
        conv_id = r1["conversationId"]
        self.assertEqual(r1["intent"], "GENERAL_CONVERSATION")
        self.assertEqual(r1["language"], "en")
        self.assertFalse(r1["isFallback"])

        r2 = self._chat("Ab Hindi mein baat karo", sess, conv_id)
        self.assertEqual(r2["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r2["isFallback"])

        r3 = self._chat("Tumhara kaam kya hai?", sess, conv_id)
        self.assertEqual(r3["intent"], "GENERAL_CONVERSATION")
        self.assertIn(r3["language"], ("hi", "hinglish"))
        self.assertFalse(r3["isFallback"])

        r4 = self._chat("Now explain that in English", sess, conv_id)
        self.assertEqual(r4["language"], "en")
        self.assertFalse(r4["isFallback"])
        self.assertIn("Test Assistant", r4["answer"])

    # =========================================================================
    # SCENARIO E — BUSINESS KNOWLEDGE RETRIEVAL
    # =========================================================================

    def test_scenario_e_business_knowledge_retrieval(self) -> None:
        sess = "sess-scenario-e"
        r1 = self._chat("What are your pricing plans?", sess)
        conv_id = r1["conversationId"]
        self.assertFalse(r1["isFallback"])
        self.assertTrue(r1["groundingPassed"])
        self.assertIn("$29", r1["answer"])
        self.assertIn("$79", r1["answer"])

        r2 = self._chat("Is there a refund policy?", sess, conv_id)
        self.assertFalse(r2["isFallback"])
        self.assertTrue(r2["groundingPassed"])
        self.assertIn("14-day", r2["answer"])

        r3 = self._chat("Explain it in Hindi", sess, conv_id)
        self.assertFalse(r3["isFallback"])
        self.assertTrue(r3["groundingPassed"])
        self.assertNotIn("हमारे सत्यापित रिकॉर्ड के अनुसार", r3["answer"])
        self.assertTrue("14" in r3["answer"] or "refund" in r3["answer"].lower())

    # =========================================================================
    # SCENARIO F — FOLLOW-UP MEMORY
    # =========================================================================

    def test_scenario_f_follow_up_memory(self) -> None:
        sess = "sess-scenario-f"
        r1 = self._chat("Tell me about your first plan", sess)
        conv_id = r1["conversationId"]
        self.assertFalse(r1["isFallback"])
        self.assertIn("Starter Plan", r1["answer"])
        self.assertIn("$29", r1["answer"])

        r2 = self._chat("What is its price?", sess, conv_id)
        self.assertFalse(r2["isFallback"])
        self.assertIn("$29", r2["answer"])

        r3 = self._chat("Is it better than the second plan?", sess, conv_id)
        self.assertFalse(r3["isFallback"])
        self.assertTrue("Growth Plan" in r3["answer"] or "Starter Plan" in r3["answer"])
        self.assertTrue("$79" in r3["answer"] or "$29" in r3["answer"])

    # =========================================================================
    # SCENARIO G — UNKNOWN BUSINESS INFORMATION
    # =========================================================================

    def test_scenario_g_unknown_business_information(self) -> None:
        sess = "sess-scenario-g"
        r1 = self._chat("What is your office address?", sess)
        self.assertTrue(r1["isFallback"])
        self.assertFalse(r1["groundingPassed"])
        # Must NOT invent a street address or city
        self.assertNotIn("Street", r1["answer"])
        self.assertNotIn("Avenue", r1["answer"])
        self.assertNotIn("Unverified Answer", r1["answer"])

    # =========================================================================
    # SCENARIO H — MIXED CONVERSATION + KNOWLEDGE
    # =========================================================================

    def test_scenario_h_mixed_conversation_and_knowledge(self) -> None:
        sess = "sess-scenario-h"
        r1 = self._chat("Hi", sess)
        conv_id = r1["conversationId"]
        self.assertEqual(r1["intent"], "GREETING")
        self.assertFalse(r1["isFallback"])

        r2 = self._chat("Tell me something interesting", sess, conv_id)
        self.assertEqual(r2["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(r2["isFallback"])
        self.assertNotIn("knowledge base", r2["answer"].lower())

        r3 = self._chat("Now tell me about your services", sess, conv_id)
        self.assertFalse(r3["isFallback"])
        self.assertTrue(r3["groundingPassed"])
        self.assertTrue("$29" in r3["answer"] or "$79" in r3["answer"] or "Starter" in r3["answer"])

        r4 = self._chat("Summarize our chat", sess, conv_id)
        self.assertEqual(r4["intent"], "SUMMARY_REQUEST")
        self.assertFalse(r4["isFallback"])
        self.assertTrue(len(r4["answer"]) > 20)

    # =========================================================================
    # STREAMING PARITY & CONFIGURED LLM ROUTING TESTS
    # =========================================================================

    def test_streaming_endpoint_handles_multilingual_conversation_and_memory(self) -> None:
        """Verify /v1/chat/stream emits natural conversational and memory responses."""
        sess = "sess-stream-conv"
        s1 = self.client.post(
            "/v1/chat/stream",
            headers=self.headers,
            json_body={"message": "Aap kaun ho?", "sessionId": sess},
        )
        self.assertEqual(s1.status_code, 200)
        events1 = [
            json.loads(line[len("data: ") :])
            for line in s1.text.splitlines()
            if line.startswith("data: ")
        ]
        done1 = [e for e in events1 if e.get("type") == "done"][0]
        self.assertEqual(done1["intent"], "GENERAL_CONVERSATION")
        self.assertFalse(done1["isFallback"])
        self.assertIn("Test Assistant", done1["answer"])

    def test_openai_compatible_provider_routes_conversational_reply_to_llm(self) -> None:
        """Verify OpenAICompatibleLlmProvider calls the LLM endpoint with centralized prompts."""
        captured_payloads: List[Dict[str, Any]] = []

        def fake_http_handler(
            method: str, url: str, headers: Dict[str, str], body: bytes | None
        ) -> tuple[int, Dict[str, Any]]:
            if body:
                captured_payloads.append(json.loads(body.decode("utf-8")))
            return (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Main Test Assistant hoon, Attach-Ai ka AI assistant. Main aapki Hindi aur English dono mein madad kar sakta hoon."
                            }
                        }
                    ]
                },
            )

        provider = OpenAICompatibleLlmProvider(
            provider_name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            http_handler=fake_http_handler,
        )

        reply = asyncio.run(
            provider.generate_conversational_reply(
                intent="GENERAL_CONVERSATION",
                message="Aap kaun ho?",
                language="hinglish",
                tenant={"name": "Attach-Ai", "assistant_name": "Test Assistant"},
                conversation_history=[{"role": "user", "content": "Hello"}],
                memory_metadata={"preferred_language": "hinglish"},
            )
        )
        self.assertIn("Test Assistant", reply)
        self.assertEqual(len(captured_payloads), 1)
        messages_sent = captured_payloads[0]["messages"]
        self.assertEqual(messages_sent[0]["role"], "system")
        self.assertIn("Attach-Ai", messages_sent[0]["content"])
        self.assertEqual(messages_sent[-1]["content"], "Aap kaun ho?")


if __name__ == "__main__":
    unittest.main()
