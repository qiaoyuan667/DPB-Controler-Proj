from __future__ import annotations

import unittest

from privacy_runtime.induction_data import (
    build_induction_record,
    build_key_value_scoring_target,
    build_protected_key_value_target,
    build_scoring_target,
    detect_target_schema,
    evaluate_prediction,
    extract_json_object,
    is_p1_sample,
    split_records_by_domain,
    validate_key_value_scoring_target,
    validate_protected_key_value_target,
    validate_scoring_target,
)
from scripts.train_inducer_qlora import tokenize_record


class TinyTokenizer:
    eos_token = "<eos>"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        ids = list(range(1, len(text.split()) + 1))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


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
            "hidden_target": {
                "allowed_fields": ["symptom"],
                "withheld_fields": ["name"],
                "gold_sensitive_values": {"name": "Noah Baumann"},
            },
            "source_document_inputs": {
                "task_relevant_fields": {"symptom": "lower back pain"},
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

    def test_build_key_value_scoring_target_uses_polar_fields(self) -> None:
        target = build_key_value_scoring_target(self.sample())

        self.assertEqual(
            target,
            {
                "scoring_targets": {
                    "allowed_values": [
                        {"key": "symptom", "value": "lower back pain"},
                    ],
                    "do_not_disclose_values": [
                        {"key": "name", "value": "Noah Baumann"},
                    ],
                }
            },
        )
        valid, error = validate_key_value_scoring_target(target)
        self.assertTrue(valid, error)

    def test_build_protected_key_value_target_omits_allowed_values(self) -> None:
        target = build_protected_key_value_target(self.sample())

        self.assertEqual(
            target,
            {
                "policy_targets": {
                    "protected_values": [
                        {"key": "name", "value": "Noah Baumann"},
                    ],
                }
            },
        )
        self.assertEqual(detect_target_schema(target), "protected_key_value")
        valid, error = validate_protected_key_value_target(target)
        self.assertTrue(valid, error)
        self.assertNotIn("allowed_values", str(target))

    def test_build_induction_record_contains_prompt_and_target_text(self) -> None:
        record = build_induction_record(self.sample())

        self.assertIn("Source document:", record["input"])
        self.assertIn("Privacy policy:", record["input"])
        self.assertIn("Task instruction:", record["input"])
        self.assertIn('"scoring_targets"', record["target_text"])
        self.assertEqual(record["messages"][-1]["role"], "assistant")

    def test_build_key_value_induction_record_contains_key_value_prompt(self) -> None:
        record = build_induction_record(self.sample(), target_schema="key_value")

        self.assertEqual(record["target_schema"], "key_value")
        self.assertIn('"key":"symptom"', record["target_text"])
        self.assertIn('"value":"Noah Baumann"', record["target_text"])
        self.assertIn('"key":"","value":""', record["messages"][0]["content"])

    def test_build_protected_key_value_induction_record_contains_runtime_prompt(self) -> None:
        record = build_induction_record(self.sample(), target_schema="protected_key_value")

        self.assertEqual(record["target_schema"], "protected_key_value")
        self.assertIn('"policy_targets"', record["target_text"])
        self.assertIn('"protected_values"', record["target_text"])
        self.assertNotIn("allowed_values", record["target_text"])
        self.assertIn("Do not output allowed_values", record["messages"][0]["content"])

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

    def test_evaluate_key_value_prediction_scores_key_errors(self) -> None:
        gold = build_key_value_scoring_target(self.sample())
        prediction = {
            "scoring_targets": {
                "allowed_values": [
                    {"key": "condition", "value": "lower back pain"},
                ],
                "do_not_disclose_values": [
                    {"key": "name", "value": "Noah Baumann"},
                ],
            }
        }

        metrics = evaluate_prediction(gold, prediction)

        self.assertTrue(metrics["schema_valid"])
        self.assertEqual(metrics["do_not_disclose_values"]["pair_f1"], 1.0)
        self.assertEqual(metrics["allowed_values"]["value_recall"], 1.0)
        self.assertEqual(metrics["allowed_values"]["pair_recall"], 0.0)
        self.assertEqual(metrics["allowed_values"]["key_accuracy_on_matched_values"], 0.0)

    def test_evaluate_protected_key_value_prediction(self) -> None:
        gold = build_protected_key_value_target(self.sample())
        prediction = {
            "policy_targets": {
                "protected_values": [
                    {"key": "email", "value": "Noah Baumann"},
                ],
            }
        }

        metrics = evaluate_prediction(gold, prediction)

        self.assertTrue(metrics["schema_valid"])
        self.assertIn("protected_values", metrics)
        self.assertNotIn("allowed_values", metrics)
        self.assertEqual(metrics["protected_values"]["value_recall"], 1.0)
        self.assertEqual(metrics["protected_values"]["pair_recall"], 0.0)

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

    def test_tokenize_record_preserves_target_labels_when_truncating(self) -> None:
        record = build_induction_record(self.sample())
        record["input"] = " ".join(["verylong"] * 200)

        tokenized = tokenize_record(TinyTokenizer(), record, max_length=32)

        self.assertEqual(len(tokenized["input_ids"]), 32)
        self.assertTrue(any(label != -100 for label in tokenized["labels"]))


if __name__ == "__main__":
    unittest.main()
