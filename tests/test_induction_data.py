from __future__ import annotations

import unittest

from privacy_runtime.induction_data import (
    build_induction_record,
    build_scoring_target,
    evaluate_prediction,
    extract_json_object,
    is_p1_sample,
    split_records_by_domain,
    validate_scoring_target,
)


class InductionDataTests(unittest.TestCase):
    def sample(self, *, sample_id: str = "medical_1", domain: str = "medical") -> dict:
        return {
            "sample_id": sample_id,
            "domain": domain,
            "metadata": {
                "domain": domain,
                "privacy_level": 1,
                "privacy_type": "explicit_field_constraints",
            },
            "generated_texts": {
                "source_document_text": "Name: Noah Baumann. Symptom: lower back pain.",
                "privacy_policy_text": "Share symptoms. Do not disclose my name.",
                "task_instruction_text": "Classify urgency.",
            },
            "scoring_targets": {
                "allowed_values": ["lower back pain"],
                "do_not_disclose_values": ["Noah Baumann"],
            },
        }

    def test_p1_filter(self) -> None:
        sample = self.sample()
        self.assertTrue(is_p1_sample(sample))

        sample["metadata"]["privacy_level"] = 2
        self.assertFalse(is_p1_sample(sample))

    def test_build_scoring_target_keeps_wrapper(self) -> None:
        target = build_scoring_target(self.sample())

        self.assertEqual(
            target,
            {
                "scoring_targets": {
                    "allowed_values": ["lower back pain"],
                    "do_not_disclose_values": ["Noah Baumann"],
                }
            },
        )
        valid, error = validate_scoring_target(target)
        self.assertTrue(valid, error)

    def test_build_induction_record_contains_prompt_and_target_text(self) -> None:
        record = build_induction_record(self.sample())

        self.assertIn("Source document:", record["input"])
        self.assertIn("Privacy policy:", record["input"])
        self.assertIn("Task instruction:", record["input"])
        self.assertIn('"scoring_targets"', record["target_text"])
        self.assertEqual(record["messages"][-1]["role"], "assistant")

    def test_extract_json_object_from_fenced_output(self) -> None:
        parsed, error = extract_json_object(
            '```json\n{"scoring_targets":{"allowed_values":[],"do_not_disclose_values":[]}}\n```'
        )

        self.assertEqual(error, "")
        self.assertIsNotNone(parsed)
        self.assertIn("scoring_targets", parsed or {})

    def test_evaluate_prediction(self) -> None:
        gold = build_scoring_target(self.sample())
        prediction = {
            "scoring_targets": {
                "allowed_values": ["lower back pain", "extra"],
                "do_not_disclose_values": ["Noah Baumann"],
            }
        }

        metrics = evaluate_prediction(gold, prediction)

        self.assertTrue(metrics["schema_valid"])
        self.assertEqual(metrics["do_not_disclose_values"]["recall"], 1.0)
        self.assertEqual(metrics["allowed_values"]["recall"], 1.0)
        self.assertLess(metrics["allowed_values"]["precision"], 1.0)
        self.assertFalse(metrics["exact_set_match"])

    def test_split_records_by_domain_is_stratified(self) -> None:
        records = [
            build_induction_record(self.sample(sample_id=f"medical_{i}", domain="medical"))
            for i in range(10)
        ] + [
            build_induction_record(self.sample(sample_id=f"finance_{i}", domain="finance"))
            for i in range(10)
        ]

        splits = split_records_by_domain(records, seed=7)

        self.assertEqual(len(splits["train"]), 14)
        self.assertEqual(len(splits["val"]), 2)
        self.assertEqual(len(splits["test"]), 4)
        self.assertEqual({record["domain"] for record in splits["test"]}, {"medical", "finance"})


if __name__ == "__main__":
    unittest.main()

