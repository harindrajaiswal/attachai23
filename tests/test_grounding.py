"""Grounding validation tests (unsupported numeric claims, invented contacts, unsupported guarantees)."""

import unittest
from app.services.grounding import GroundingService


class TestGrounding(unittest.TestCase):
    def setUp(self) -> None:
        self.grounding = GroundingService()
        self.evidence = [
            {
                "id": "c1",
                "source_filename": "pricing.txt",
                "chunk_index": 0,
                "content": "Our Pro Plan is $79 per month and support is available at help@company.com.",
            }
        ]
        self.fallback = "Fallback: unable to verify answer."

    def test_valid_grounded_answer_passes(self) -> None:
        res = self.grounding.validate_answer(
            candidate_answer="The Pro Plan is $79 per month and you can email help@company.com for support.",
            evidence_chunks=self.evidence,
            support_level="SUPPORTED",
            fallback_answer=self.fallback,
        )
        self.assertTrue(res.passed)
        self.assertIn("$79", res.validated_answer)

    def test_unsupported_numeric_claim_rejected(self) -> None:
        res = self.grounding.validate_answer(
            candidate_answer="The Pro Plan is $29 per month with a 50% discount.",
            evidence_chunks=self.evidence,
            support_level="SUPPORTED",
            fallback_answer=self.fallback,
        )
        self.assertFalse(res.passed)
        self.assertEqual(res.validated_answer, self.fallback)
        self.assertTrue(res.unverified_numbers)

    def test_invented_contact_email_and_guarantee_rejected(self) -> None:
        res_email = self.grounding.validate_answer(
            candidate_answer="Contact our billing team at fake-billing@company.com for Pro Plan help.",
            evidence_chunks=self.evidence,
            support_level="SUPPORTED",
            fallback_answer=self.fallback,
        )
        self.assertFalse(res_email.passed)
        self.assertEqual(res_email.validated_answer, self.fallback)

        res_guar = self.grounding.validate_answer(
            candidate_answer="Our Pro Plan is $79 per month with a lifetime warranty.",
            evidence_chunks=self.evidence,
            support_level="SUPPORTED",
            fallback_answer=self.fallback,
        )
        self.assertFalse(res_guar.passed)


if __name__ == "__main__":
    unittest.main()
