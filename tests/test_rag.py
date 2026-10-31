"""Advanced RAG retrieval, tenant isolation, query reformulation, and evidence support tests."""

import unittest
from tests.conftest import IsolatedOrbitTestEnv


class TestRag(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()
        self.t1, self.key1 = self.env.create_test_tenant("PureGym Club", "puregym-club")
        self.t2, self.key2 = self.env.create_test_tenant("Gousto Kitchen", "gousto-kitchen")

        self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.t1["id"],
                "filename": "gym_info.txt",
                "content": (
                    "PureGym Club memberships cost $45 per month with 24/7 access. "
                    "Personal training sessions are $60 per hour. "
                    "Sauna and steam room access is included in the Plus tier for $65 per month."
                ),
            },
            headers=self.env.admin_headers,
        )
        self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.t2["id"],
                "filename": "recipe_boxes.txt",
                "content": (
                    "Gousto Kitchen delivers fresh recipe meal boxes every Tuesday and Friday. "
                    "A 4-recipe box for two people costs $52 including insulated chilled delivery."
                ),
            },
            headers=self.env.admin_headers,
        )

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_relevant_evidence_supported_and_fallback_on_unsupported(self) -> None:
        # Supported question for Tenant 1
        res = self.env.client.post(
            "/v1/chat",
            json_body={"message": "How much do PureGym Club memberships cost per month?"},
            headers={"x-api-key": self.key1},
        ).json()
        self.assertEqual(res["supportLevel"], "SUPPORTED")
        self.assertFalse(res["isFallback"])
        self.assertIn("$45", res["answer"])

        # Unrelated question must NOT be marked SUPPORTED (Fixes Defect #7)
        unrel = self.env.client.post(
            "/v1/chat",
            json_body={"message": "Do you sell organic dog food and veterinary insurance?"},
            headers={"x-api-key": self.key1},
        ).json()
        self.assertEqual(unrel["supportLevel"], "UNKNOWN")
        self.assertTrue(unrel["isFallback"])
        self.assertEqual(unrel["answer"], self.t1["fallback_answer"])

    def test_strict_tenant_filtered_retrieval(self) -> None:
        # Asking Tenant 1 about Tenant 2's meal boxes must return Tenant 1 fallback
        cross = self.env.client.post(
            "/v1/chat",
            json_body={"message": "How much is a Gousto Kitchen 4-recipe meal box on Tuesday?"},
            headers={"x-api-key": self.key1},
        ).json()
        self.assertTrue(cross["isFallback"])
        self.assertNotIn("$52", cross["answer"])


if __name__ == "__main__":
    unittest.main()
