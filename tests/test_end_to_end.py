"""End-to-End Acceptance Scenario test verifying all 10 mandatory steps from Section 27."""

import unittest
from tests.conftest import IsolatedOrbitTestEnv


class TestEndToEndScenario(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_complete_10_step_acceptance_scenario(self) -> None:
        client = self.env.client

        # Step 1 — Application startup, /health, and /ready
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        ready = client.get("/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertTrue(ready.json()["ready"])

        # Step 2 — Administrator login
        bad_login = client.post("/api/admin/login", json_body={"admin_api_key": "invalid-secret"})
        self.assertEqual(bad_login.status_code, 401)

        good_login = client.post(
            "/api/admin/login",
            json_body={"admin_api_key": self.env.settings.admin_api_key},
        )
        self.assertEqual(good_login.status_code, 200)
        csrf_token = good_login.json()["csrfToken"]
        admin_session_headers = {"x-csrf-token": csrf_token}

        dash_html = client.get("/admin")
        self.assertEqual(dash_html.status_code, 200)
        self.assertIn("Multi-Tenant Operations Overview", dash_html.text)

        # Step 3 — Create a tenant (Test Company / test-company / example.com / Test Assistant)
        create_res = client.post(
            "/api/admin/clients",
            json_body={
                "business_name": "Test Company",
                "slug": "test-company",
                "domains": ["example.com", "localhost"],
                "assistant_name": "Test Assistant",
            },
            headers=admin_session_headers,
        )
        self.assertEqual(create_res.status_code, 201)
        t1 = create_res.json()["tenant"]
        t1_key = create_res.json()["rawApiKey"]
        self.assertTrue(t1_key.startswith("orbit_live_"))

        # Step 4 — Upload knowledge document
        doc_text = (
            "Test Company Enterprise Support Guide:\n"
            "Our Cloud Backup service costs $120 per month and includes daily encrypted snapshots.\n"
            "Technical support hours are Monday through Friday from 9:00am to 7:00pm EST.\n"
            "Emergency hotline is support@testcompany.example.com."
        )
        up_res = client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": t1["id"],
                "filename": "support_guide.txt",
                "content": doc_text,
            },
            headers=admin_session_headers,
        )
        self.assertEqual(up_res.status_code, 201)
        self.assertEqual(up_res.json()["status"], "ready")
        self.assertGreaterEqual(up_res.json()["chunk_count"], 1)

        # Step 5 — Test RAG (supported question vs unrelated question)
        rag_ok = client.post(
            "/v1/chat",
            json_body={"message": "How much does the Cloud Backup service cost per month?", "sessionId": "e2e-session"},
            headers={"x-api-key": t1_key, "Origin": "https://example.com"},
        ).json()
        self.assertFalse(rag_ok["isFallback"])
        self.assertIn("$120", rag_ok["answer"])
        conv_id = rag_ok["conversationId"]

        rag_unrel = client.post(
            "/v1/chat",
            json_body={
                "message": "Do you offer submarine rental services in Antarctica?",
                "sessionId": "e2e-session",
                "conversationId": conv_id,
            },
            headers={"x-api-key": t1_key, "Origin": "https://example.com"},
        ).json()
        self.assertTrue(rag_unrel["isFallback"])
        self.assertEqual(rag_unrel["answer"], t1["fallback_answer"])

        # Step 6 — Conversation history & summary
        followup = client.post(
            "/v1/chat",
            json_body={
                "message": "What are the technical support hours?",
                "sessionId": "e2e-session",
                "conversationId": conv_id,
            },
            headers={"x-api-key": t1_key, "Origin": "https://example.com"},
        ).json()
        self.assertIn("9:00am", followup["answer"])

        summary_res = client.post(
            "/v1/chat",
            json_body={
                "message": "Please summarize our conversation",
                "sessionId": "e2e-session",
                "conversationId": conv_id,
            },
            headers={"x-api-key": t1_key, "Origin": "https://example.com"},
        ).json()
        self.assertFalse(summary_res["isFallback"])
        self.assertIn("Conversation Summary", summary_res["answer"])

        # Step 7 — Website widget & /demo.html
        demo_page = client.get("/demo.html")
        self.assertEqual(demo_page.status_code, 200)
        w_cfg = client.get("/v1/config", headers={"x-api-key": t1_key, "Origin": "https://example.com"}).json()
        self.assertEqual(w_cfg["assistantName"], "Test Assistant")

        # Step 8 — Lead capture & duplicate protection
        lead_payload = {
            "name": "Jordan Taylor",
            "email": "jordan@example.com",
            "phone": "+1-555-333-4444",
            "requirement": "Cloud Backup Service",
            "message": "Ready to subscribe for 3 servers.",
            "conversationId": conv_id,
            "idempotencyKey": "e2e-lead-key-1",
        }
        lead1 = client.post(
            "/v1/leads",
            json_body=lead_payload,
            headers={"x-api-key": t1_key, "Origin": "https://example.com"},
        )
        self.assertEqual(lead1.status_code, 201)
        lead_dup = client.post(
            "/v1/leads",
            json_body=lead_payload,
            headers={"x-api-key": t1_key, "Origin": "https://example.com"},
        )
        self.assertEqual(lead_dup.status_code, 200)
        self.assertEqual(lead1.json()["id"], lead_dup.json()["id"])

        # Step 9 — Dashboard verification
        overview = client.get("/api/admin/overview", headers=admin_session_headers).json()
        self.assertEqual(overview["summary"]["total_tenants"], 1)
        self.assertEqual(overview["summary"]["total_conversations"], 1)
        self.assertEqual(overview["summary"]["total_leads"], 1)
        self.assertEqual(overview["summary"]["ready_knowledge_files"], 1)

        # Step 10 — Multi-tenant isolation
        t2_res = client.post(
            "/api/admin/clients",
            json_body={
                "business_name": "Second Tenant Corp",
                "slug": "second-tenant",
                "domains": ["second.example.com"],
                "assistant_name": "Second Bot",
            },
            headers=admin_session_headers,
        ).json()
        t2 = t2_res["tenant"]
        t2_key = t2_res["rawApiKey"]

        # First tenant's key against second tenant's client ID is rejected
        cross_cfg = client.get(
            "/v1/config",
            headers={"x-api-key": t1_key, "x-client-id": t2["id"], "Origin": "https://example.com"},
        )
        self.assertEqual(cross_cfg.status_code, 403)

        # Second tenant querying with first tenant's conversationId is rejected
        cross_chat = client.post(
            "/v1/chat",
            json_body={"message": "Hello", "conversationId": conv_id},
            headers={"x-api-key": t2_key, "Origin": "https://second.example.com"},
        )
        self.assertEqual(cross_chat.status_code, 403)


if __name__ == "__main__":
    unittest.main()
