"""Comprehensive test suite for Upgrades 1, 2, and 3:
- Category A: Intelligent Response Engine & Multilingual Multi-Query RAG (18 tests)
- Category B: Persistent Conversation Memory & Context Resolution (10 tests)
- Category C: Premium Dynamic Widget UI & Message Feedback Persistence (9+ tests)
"""

import asyncio
import unittest
from pathlib import Path
from app.domain.enums import SupportLevel
from tests.conftest import IsolatedOrbitTestEnv


class TestUpgradesSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv(env_overrides={"APP_ENV": "development"})
        self.app = self.env.app
        self.container = self.app.container
        asyncio.run(self.app.ensure_initialized())
        self.client = self.env.client
        tenants = asyncio.run(self.container.client_service.list_tenants())
        by_slug = {t["slug"]: t for t in tenants}
        self.tenant_a = by_slug["apex-fitness"]
        self.tenant_b = by_slug["cloudscale-b2b"]
        self.key_a = self.container.demo_public_keys[self.tenant_a["id"]]
        self.key_b = self.container.demo_public_keys[self.tenant_b["id"]]

    def tearDown(self) -> None:
        self.env.cleanup()

    # =========================================================================
    # CATEGORY A — INTELLIGENT RESPONSE & MULTILINGUAL RAG TESTS (18 TESTS)
    # =========================================================================

    def test_a01_greeting_hi_natural_response(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Hi", "sessionId": "sess-a01"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "GREETING")
        self.assertFalse(data["isFallback"])
        self.assertNotIn("unverified answer", data["answer"].lower())
        self.assertNotIn("fallback answer", data["answer"].lower())
        self.assertIn(self.tenant_a["assistant_name"], data["answer"])

    def test_a02_greeting_namaste_natural_response(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Namaste", "sessionId": "sess-a02"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "GREETING")
        self.assertEqual(data["language"], "hinglish")
        self.assertFalse(data["isFallback"])
        self.assertIn("Namaste", data["answer"])
        self.assertNotIn("unverified answer", data["answer"].lower())

    def test_a03_greeting_kaise_ho_natural_response(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Kaise ho?", "sessionId": "sess-a03"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["intent"], ("GREETING", "GENERAL_CONVERSATION"))
        self.assertFalse(data["isFallback"])
        self.assertNotIn("unverified answer", data["answer"].lower())

    def test_a04_english_knowledge_question_grounded(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is the price of the Core Membership?", "sessionId": "sess-a04"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["supportLevel"], SupportLevel.SUPPORTED.value)
        self.assertTrue(data["groundingPassed"])
        self.assertFalse(data["isFallback"])
        self.assertIn("$59", data["answer"])

    def test_a05_paraphrased_english_knowledge_question_grounded(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={
                "message": "How much do you charge each month for your Core and Peak membership tiers?",
                "sessionId": "sess-a05",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["supportLevel"], (SupportLevel.SUPPORTED.value, SupportLevel.PARTIAL.value))
        self.assertTrue(data["groundingPassed"])
        self.assertTrue("$59" in data["answer"] or "$99" in data["answer"])

    def test_a06_hindi_devanagari_question_against_english_docs(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "आपकी सदस्यता की कीमत और प्लान्स क्या हैं?", "sessionId": "sess-a06"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["language"], "hi")
        self.assertFalse(data["isFallback"])
        self.assertTrue(data["groundingPassed"])
        self.assertIn("हमारे सत्यापित रिकॉर्ड के अनुसार", data["answer"])
        self.assertTrue("$59" in data["answer"] or "$99" in data["answer"])

    def test_a07_hinglish_question_against_english_docs(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Aapke membership plans aur charges kya hain?", "sessionId": "sess-a07"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["language"], "hinglish")
        self.assertFalse(data["isFallback"])
        self.assertTrue(data["groundingPassed"])
        self.assertIn("Hamare verified details ke mutabik", data["answer"])
        self.assertTrue("$59" in data["answer"] or "$99" in data["answer"])

    def test_a08_urdu_question_against_english_docs(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "آپ کے پلان کی قیمت کیا ہے؟ membership pricing", "sessionId": "sess-a08"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["language"], "ur")
        self.assertFalse(data["isFallback"])
        self.assertTrue(data["groundingPassed"])
        self.assertIn("ہمارے تصدیق شدہ ریکارڈ کے مطابق", data["answer"])

    def test_a09_synonym_based_query_matches_relevant_documents(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is your membership cancellation and freeze policy?", "sessionId": "sess-a09"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["isFallback"])
        self.assertTrue("14 days" in data["answer"] or "3 months" in data["answer"])

    def test_a10_multi_query_retrieval_merges_and_deduplicates_chunks(self) -> None:
        rag_res = asyncio.run(
            self.container.rag_service.retrieve_and_assess(
                tenant_id=self.tenant_a["id"],
                question="Aapke membership plans aur charges batao",
                conversation_history=[],
                top_k=5,
            )
        )
        self.assertGreaterEqual(len(rag_res["multi_queries"]), 2)
        chunk_ids = [c["id"] for c in rag_res["chunks"]]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertGreater(len(chunk_ids), 0)

    def test_a11_weak_initial_retrieval_triggers_expanded_secondary_search(self) -> None:
        rag_res = asyncio.run(
            self.container.rag_service.retrieve_and_assess(
                tenant_id=self.tenant_a["id"],
                question="membership plans pricing",
                conversation_history=[],
                top_k=5,
                min_similarity=0.95,  # High threshold triggers Scenario B expanded pass
            )
        )
        self.assertTrue(rag_res["expanded_search_used"])
        self.assertGreater(len(rag_res["chunks"]), 0)

    def test_a12_partial_evidence_answers_supported_part_without_fabricating(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={
                "message": "What is the Core Membership price and do you have a Mars colony gym?",
                "sessionId": "sess-a12",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("$59", data["answer"])
        self.assertNotIn("Mars", data["answer"])

    def test_a13_unsupported_question_gets_polite_natural_response_without_unverified_label(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={
                "message": "Kya aap cryptocurrency mining hardware rent par dete hain?",
                "sessionId": "sess-a13",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["isFallback"])
        self.assertNotIn("unverified answer", data["answer"].lower())
        self.assertNotIn("fallback answer", data["answer"].lower())
        self.assertIn("verified", data["answer"].lower())

    def test_a14_ambiguous_query_asks_clarifying_question(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Details?", "sessionId": "sess-a14"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "AMBIGUOUS")
        self.assertFalse(data["isFallback"])
        self.assertIn("?", data["answer"])

    def test_a15_out_of_scope_query_redirects_politely(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Who won the cricket world cup and what is the weather today?", "sessionId": "sess-a15"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "OUT_OF_SCOPE")
        self.assertTrue(data["isFallback"])
        self.assertNotIn("unverified answer", data["answer"].lower())

    def test_a16_response_never_invents_prices_hours_policies_or_contacts(self) -> None:
        chunks = [
            {
                "id": "c1",
                "content": "Core Membership is $59 per month. Support email is support@apexperformance.example.com.",
                "source_filename": "pricing.md",
                "chunk_index": 0,
            }
        ]
        bad_res = self.container.grounding_service.validate_answer(
            candidate_answer="Core Membership is $999 per month with 100% guaranteed lifetime warranty at fake@evil.com.",
            evidence_chunks=chunks,
            support_level=SupportLevel.SUPPORTED.value,
            fallback_answer="Polite fallback response.",
        )
        self.assertFalse(bad_res.passed)
        self.assertIn("$999", bad_res.unverified_numbers)

    def test_a17_response_never_exposes_raw_internal_status_labels(self) -> None:
        chunks = [
            {
                "id": "c1",
                "content": "Core Membership is $59 per month and includes unlimited access.",
                "source_filename": "pricing.md",
                "chunk_index": 0,
            }
        ]
        res = self.container.grounding_service.validate_answer(
            candidate_answer="Unverified Answer: Core Membership is $59 per month and includes unlimited access.",
            evidence_chunks=chunks,
            support_level=SupportLevel.SUPPORTED.value,
            fallback_answer="Fallback Answer: Please contact us.",
        )
        self.assertTrue(res.passed)
        self.assertNotIn("Unverified Answer", res.validated_answer)
        self.assertNotIn("Fallback Answer", res.validated_answer)

    def test_a18_cross_tenant_knowledge_isolation_enforced(self) -> None:
        # Ask Tenant A (Apex Fitness) about Tenant B's (CloudScale) SOC2 and 99.99% SLA
        resp = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is your SOC 2 Type II and ISO 27001 enterprise uptime SLA?", "sessionId": "sess-a18"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotIn("99.99%", data["answer"])
        self.assertNotIn("CloudScale", data["answer"])

    # =========================================================================
    # CATEGORY B — CONVERSATION MEMORY TESTS (10 TESTS)
    # =========================================================================

    def test_b01_multi_turn_follow_up_resolves_prior_topic_and_entity(self) -> None:
        r1 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is the price of the Peak All-Access Membership?", "sessionId": "sess-b01"},
        ).json()
        conv_id = r1["conversationId"]
        self.assertIn("$99", r1["answer"])

        r2 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={
                "message": "What is included in it?",
                "sessionId": "sess-b01",
                "conversationId": conv_id,
            },
        ).json()
        self.assertEqual(r2["conversationId"], conv_id)
        self.assertFalse(r2["isFallback"])
        self.assertTrue("HIIT" in r2["answer"] or "sauna" in r2["answer"] or "$99" in r2["answer"])

    def test_b02_pronoun_reference_resolves_using_conversation_history(self) -> None:
        r1 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Tell me about the Core Membership.", "sessionId": "sess-b02"},
        ).json()
        conv_id = r1["conversationId"]

        r2 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={
                "message": "How do I cancel it?",
                "sessionId": "sess-b02",
                "conversationId": conv_id,
            },
        ).json()
        self.assertFalse(r2["isFallback"])
        self.assertIn("14 days", r2["answer"])

    def test_b03_hinglish_follow_up_resolves_prior_context(self) -> None:
        r1 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Peak membership ke baare mein batao", "sessionId": "sess-b03"},
        ).json()
        conv_id = r1["conversationId"]

        r2 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={
                "message": "Iska price kya hai?",
                "sessionId": "sess-b03",
                "conversationId": conv_id,
            },
        ).json()
        self.assertEqual(r2["language"], "hinglish")
        self.assertFalse(r2["isFallback"])
        self.assertTrue("$99" in r2["answer"] or "$59" in r2["answer"])

    def test_b04_conversation_history_persists_across_turns(self) -> None:
        r1 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Hi there", "sessionId": "sess-b04"},
        ).json()
        conv_id = r1["conversationId"]
        self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={
                "message": "What is the Core Membership price?",
                "sessionId": "sess-b04",
                "conversationId": conv_id,
            },
        )
        hist_resp = self.client.get(
            f"/v1/conversations/{conv_id}/messages",
            headers={"X-API-Key": self.key_a},
        )
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.json()
        self.assertEqual(len(hist_data["messages"]), 4)
        self.assertEqual(hist_data["messages"][0]["role"], "user")
        self.assertEqual(hist_data["messages"][1]["role"], "assistant")

    def test_b05_conversation_summary_updates_for_long_conversations(self) -> None:
        conv_id = None
        for i in range(4):
            res = self.client.post(
                "/v1/chat",
                headers={"X-API-Key": self.key_a},
                json_body={
                    "message": f"Question {i}: What are the membership plans and opening hours?",
                    "sessionId": "sess-b05",
                    "conversationId": conv_id,
                },
            ).json()
            conv_id = res["conversationId"]

        detail = self.client.get(
            f"/v1/conversations/{conv_id}/messages",
            headers={"X-API-Key": self.key_a},
        ).json()
        self.assertTrue(len(detail["summary"]) > 0)
        self.assertIn("turn_count", detail["metadata"])

    def test_b06_new_conversation_starts_with_clean_context(self) -> None:
        r1 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is the Core Membership price?", "sessionId": "sess-b06-1"},
        ).json()
        r2 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Hello!", "sessionId": "sess-b06-2"},
        ).json()
        self.assertNotEqual(r1["conversationId"], r2["conversationId"])
        hist2 = self.client.get(
            f"/v1/conversations/{r2['conversationId']}/messages",
            headers={"X-API-Key": self.key_a},
        ).json()
        self.assertEqual(len(hist2["messages"]), 2)

    def test_b07_clearing_conversation_resets_context(self) -> None:
        r1 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is the Core Membership price?", "sessionId": "sess-b07"},
        ).json()
        conv_id = r1["conversationId"]

        del_resp = self.client.delete(
            f"/v1/conversations/{conv_id}",
            headers={"X-API-Key": self.key_a},
        )
        self.assertEqual(del_resp.status_code, 200)
        self.assertTrue(del_resp.json()["deleted"])

        get_after = self.client.get(
            f"/v1/conversations/{conv_id}/messages",
            headers={"X-API-Key": self.key_a},
        )
        self.assertEqual(get_after.status_code, 404)

    def test_b08_tenant_a_cannot_read_or_delete_tenant_b_conversation_memory(self) -> None:
        rb = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_b},
            json_body={"message": "What is the enterprise uptime SLA?", "sessionId": "sess-b08"},
        ).json()
        conv_b_id = rb["conversationId"]

        # Tenant A tries to read Tenant B's conversation messages
        leak_resp = self.client.get(
            f"/v1/conversations/{conv_b_id}/messages",
            headers={"X-API-Key": self.key_a},
        )
        self.assertEqual(leak_resp.status_code, 404)

        # Tenant A tries to delete Tenant B's conversation
        del_leak = self.client.delete(
            f"/v1/conversations/{conv_b_id}",
            headers={"X-API-Key": self.key_a},
        )
        self.assertEqual(del_leak.status_code, 404)

    def test_b09_conversation_a_cannot_leak_into_conversation_b(self) -> None:
        ra = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Tell me about Peak All-Access Membership.", "sessionId": "sess-b09-a"},
        ).json()
        rb = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Summarize our conversation so far.", "sessionId": "sess-b09-b"},
        ).json()
        self.assertNotEqual(ra["conversationId"], rb["conversationId"])
        self.assertNotIn("Peak", rb["answer"])

    def test_b10_retried_message_does_not_create_duplicate_stored_messages(self) -> None:
        fixed_msg_id = "retry-msg-uuid-12345"
        r1 = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={
                "message": "What is the Core Membership price?",
                "sessionId": "sess-b10",
                "messageId": fixed_msg_id,
            },
        ).json()
        conv_id = r1["conversationId"]

        # Simulate retry with identical user messageId
        self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={
                "message": "What is the Core Membership price?",
                "sessionId": "sess-b10",
                "conversationId": conv_id,
                "messageId": fixed_msg_id,
            },
        )
        hist = self.client.get(
            f"/v1/conversations/{conv_id}/messages",
            headers={"X-API-Key": self.key_a},
        ).json()
        user_msgs = [m for m in hist["messages"] if m["role"] == "user"]
        self.assertEqual(len(user_msgs), 1)
        self.assertEqual(user_msgs[0]["id"], fixed_msg_id)

    # =========================================================================
    # CATEGORY C — PREMIUM UI & FEEDBACK PERSISTENCE TESTS (9 TESTS)
    # =========================================================================

    def test_c01_like_feedback_saves_correctly_for_valid_assistant_message(self) -> None:
        chat_res = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is the Core Membership price?", "sessionId": "sess-c01"},
        ).json()
        fb_resp = self.client.post(
            "/v1/feedback",
            headers={"X-API-Key": self.key_a},
            json_body={
                "conversationId": chat_res["conversationId"],
                "messageId": chat_res["messageId"],
                "feedbackType": "like",
            },
        )
        self.assertEqual(fb_resp.status_code, 200)
        fb_data = fb_resp.json()
        self.assertEqual(fb_data["feedbackType"], "like")
        self.assertEqual(fb_data["messageId"], chat_res["messageId"])

    def test_c02_dislike_feedback_saves_with_and_without_reason(self) -> None:
        chat_res = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is the Core Membership price?", "sessionId": "sess-c02"},
        ).json()
        # Without reason
        fb1 = self.client.post(
            "/v1/feedback",
            headers={"X-API-Key": self.key_a},
            json_body={
                "conversationId": chat_res["conversationId"],
                "messageId": chat_res["messageId"],
                "feedbackType": "dislike",
            },
        )
        self.assertEqual(fb1.status_code, 200)
        self.assertIsNone(fb1.json()["reason"])

        # With reason
        fb2 = self.client.post(
            "/v1/feedback",
            headers={"X-API-Key": self.key_a},
            json_body={
                "conversationId": chat_res["conversationId"],
                "messageId": chat_res["messageId"],
                "feedbackType": "dislike",
                "reason": "Wanted annual billing breakdown as well",
            },
        )
        self.assertEqual(fb2.status_code, 200)
        self.assertEqual(fb2.json()["reason"], "Wanted annual billing breakdown as well")

    def test_c03_changing_like_to_dislike_updates_existing_record_without_duplicate(self) -> None:
        chat_res = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is the Core Membership price?", "sessionId": "sess-c03"},
        ).json()
        conv_id = chat_res["conversationId"]
        msg_id = chat_res["messageId"]

        r_like = self.client.post(
            "/v1/feedback",
            headers={"X-API-Key": self.key_a},
            json_body={"conversationId": conv_id, "messageId": msg_id, "feedbackType": "like"},
        ).json()

        r_dislike = self.client.post(
            "/v1/feedback",
            headers={"X-API-Key": self.key_a},
            json_body={
                "conversationId": conv_id,
                "messageId": msg_id,
                "feedbackType": "dislike",
                "reason": "Changed my mind",
            },
        ).json()

        self.assertEqual(r_like["id"], r_dislike["id"])
        self.assertEqual(r_dislike["feedbackType"], "dislike")

    def test_c04_feedback_persists_when_reloading_conversation_messages(self) -> None:
        chat_res = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "What is the Core Membership price?", "sessionId": "sess-c04"},
        ).json()
        conv_id = chat_res["conversationId"]
        msg_id = chat_res["messageId"]

        self.client.post(
            "/v1/feedback",
            headers={"X-API-Key": self.key_a},
            json_body={"conversationId": conv_id, "messageId": msg_id, "feedbackType": "like"},
        )

        hist = self.client.get(
            f"/v1/conversations/{conv_id}/messages",
            headers={"X-API-Key": self.key_a},
        ).json()
        assistant_msgs = [m for m in hist["messages"] if m["id"] == msg_id]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0]["feedbackType"], "like")

    def test_c05_feedback_rejected_for_invalid_or_missing_api_key(self) -> None:
        resp = self.client.post(
            "/v1/feedback",
            headers={"X-API-Key": "invalid-key"},
            json_body={"conversationId": "c1", "messageId": "m1", "feedbackType": "like"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_c06_feedback_rejected_for_another_tenants_message(self) -> None:
        chat_b = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_b},
            json_body={"message": "What is the enterprise uptime SLA?", "sessionId": "sess-c06"},
        ).json()

        cross_fb = self.client.post(
            "/v1/feedback",
            headers={"X-API-Key": self.key_a},
            json_body={
                "conversationId": chat_b["conversationId"],
                "messageId": chat_b["messageId"],
                "feedbackType": "like",
            },
        )
        self.assertIn(cross_fb.status_code, (403, 404))

    def test_c07_feedback_rejected_for_user_role_messages(self) -> None:
        chat_res = self.client.post(
            "/v1/chat",
            headers={"X-API-Key": self.key_a},
            json_body={"message": "Hello!", "sessionId": "sess-c07"},
        ).json()

        user_fb = self.client.post(
            "/v1/feedback",
            headers={"X-API-Key": self.key_a},
            json_body={
                "conversationId": chat_res["conversationId"],
                "messageId": chat_res["userMessageId"],
                "feedbackType": "like",
            },
        )
        self.assertEqual(user_fb.status_code, 422)

    def test_c08_widget_js_contains_action_toolbar_markdown_and_streaming_controls(self) -> None:
        widget_js = (Path(__file__).resolve().parent.parent / "static" / "js" / "widget.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('data-action", "like"', widget_js)
        self.assertIn('data-action", "dislike"', widget_js)
        self.assertIn('data-action", "copy"', widget_js)
        self.assertIn('data-action", "share"', widget_js)
        self.assertIn("Understanding your question...", widget_js)
        self.assertIn("Searching relevant information...", widget_js)
        self.assertIn("Preparing your answer...", widget_js)
        self.assertIn("renderSafeMarkdownToHtml", widget_js)
        self.assertIn("/v1/feedback", widget_js)
        self.assertIn("ev.shiftKey", widget_js)

    def test_c09_widget_css_supports_dark_mode_mobile_and_reduced_motion(self) -> None:
        widget_css = (Path(__file__).resolve().parent.parent / "static" / "css" / "widget.css").read_text(
            encoding="utf-8"
        )
        self.assertIn(".orbit-theme-dark", widget_css)
        self.assertIn("prefers-reduced-motion: reduce", widget_css)
        self.assertIn(".orbit-msg-toolbar", widget_css)
        self.assertIn(".orbit-action-btn", widget_css)


if __name__ == "__main__":
    unittest.main()
