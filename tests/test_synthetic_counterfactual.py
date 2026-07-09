from __future__ import annotations

import unittest

from privacy_runtime.induction_data import validate_scoring_target
from privacy_runtime.induction_data import (
    validate_key_value_scoring_target,
    validate_protected_key_value_target,
)
from privacy_runtime.synthetic_counterfactual import (
    generate_synthetic_records,
    split_records_by_base_doc,
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


class SyntheticCounterfactualTests(unittest.TestCase):
    def test_same_source_document_has_counterfactual_targets(self) -> None:
        records = generate_synthetic_records(num_base_docs=1, policies_per_doc=4, seed=7)

        self.assertEqual({record["base_doc_id"] for record in records}, {"synth_00000"})
        source_inputs = {
            record["input"].split("\n\nPrivacy policy:", 1)[0] for record in records
        }
        self.assertEqual(len(source_inputs), 1)

        protected_sets = {
            tuple(record["target"]["scoring_targets"]["do_not_disclose_values"])
            for record in records
        }
        self.assertGreater(len(protected_sets), 1)

    def test_policy_hard_negatives_do_not_enter_protected_values(self) -> None:
        record = generate_synthetic_records(num_base_docs=1, policies_per_doc=1, seed=7)[0]
        values_by_category = {
            item["category"]: item["value"] for item in record["synthetic_values"]
        }
        protected_values = set(record["target"]["scoring_targets"]["do_not_disclose_values"])
        allowed_values = set(record["target"]["scoring_targets"]["allowed_values"])

        self.assertEqual(record["protected_categories"], ["email", "name", "phone"])
        self.assertIn(values_by_category["email"], protected_values)
        self.assertIn(values_by_category["name"], protected_values)
        self.assertIn(values_by_category["phone"], protected_values)
        self.assertNotIn(values_by_category["home_address"], protected_values)
        self.assertNotIn(values_by_category["grade"], protected_values)
        self.assertIn(values_by_category["home_address"], allowed_values)
        self.assertIn(values_by_category["grade"], allowed_values)

    def test_split_keeps_base_documents_disjoint(self) -> None:
        records = generate_synthetic_records(num_base_docs=20, policies_per_doc=4, seed=11)
        splits = split_records_by_base_doc(records, seed=11)

        split_ids = {
            split: {record["base_doc_id"] for record in split_records}
            for split, split_records in splits.items()
        }
        self.assertTrue(split_ids["train"].isdisjoint(split_ids["val"]))
        self.assertTrue(split_ids["train"].isdisjoint(split_ids["test"]))
        self.assertTrue(split_ids["val"].isdisjoint(split_ids["test"]))
        self.assertEqual(sum(len(records) for records in splits.values()), 80)

    def test_records_match_existing_schema_and_tokenizer(self) -> None:
        record = generate_synthetic_records(num_base_docs=1, policies_per_doc=1, seed=13)[0]

        valid, error = validate_scoring_target(record["target"])
        self.assertTrue(valid, error)
        self.assertIn("Source document:", record["input"])
        self.assertIn("Privacy policy:", record["input"])
        self.assertIn("Task instruction:", record["input"])

        tokenized = tokenize_record(TinyTokenizer(), record, max_length=64)
        self.assertEqual(len(tokenized["input_ids"]), 64)
        self.assertTrue(any(label != -100 for label in tokenized["labels"]))

    def test_key_value_records_use_categories_as_keys(self) -> None:
        record = generate_synthetic_records(
            num_base_docs=1,
            policies_per_doc=1,
            seed=7,
            target_schema="key_value",
        )[0]

        valid, error = validate_key_value_scoring_target(record["target"])
        self.assertTrue(valid, error)
        protected_entries = record["target"]["scoring_targets"]["do_not_disclose_values"]
        self.assertEqual(
            [(entry["key"], entry["value"]) for entry in protected_entries],
            [
                ("name", protected_entries[0]["value"]),
                ("email", protected_entries[1]["value"]),
                ("phone", protected_entries[2]["value"]),
            ],
        )
        self.assertEqual(record["target_schema"], "key_value")
        self.assertIn('"key":"name"', record["target_text"])

    def test_protected_only_mode_leaves_hard_negatives_out_of_target(self) -> None:
        record = generate_synthetic_records(
            num_base_docs=1,
            policies_per_doc=1,
            seed=7,
            target_schema="key_value",
            synthetic_mode="protected_only",
        )[0]
        values_by_category = {
            item["category"]: item["value"] for item in record["synthetic_values"]
        }
        allowed_entries = record["target"]["scoring_targets"]["allowed_values"]
        protected_entries = record["target"]["scoring_targets"]["do_not_disclose_values"]
        protected_values = {entry["value"] for entry in protected_entries}

        self.assertEqual(allowed_entries, [])
        self.assertEqual(record["synthetic_mode"], "protected_only")
        self.assertIn(values_by_category["home_address"], record["input"])
        self.assertIn(values_by_category["grade"], record["input"])
        self.assertNotIn(values_by_category["home_address"], protected_values)
        self.assertNotIn(values_by_category["grade"], protected_values)
        self.assertIn("Set allowed_values to an empty list", record["input"])

    def test_protected_key_value_schema_has_no_allowed_values(self) -> None:
        record = generate_synthetic_records(
            num_base_docs=1,
            policies_per_doc=1,
            seed=7,
            target_schema="protected_key_value",
            synthetic_mode="protected_only",
        )[0]

        valid, error = validate_protected_key_value_target(record["target"])
        self.assertTrue(valid, error)
        self.assertEqual(record["target_schema"], "protected_key_value")
        self.assertIn("policy_targets", record["target"])
        self.assertIn("protected_values", record["target"]["policy_targets"])
        self.assertNotIn("allowed_values", record["target_text"])
        self.assertIn("Do not output allowed_values", record["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
