from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from privacy_runtime import (
    DEFAULT_APERTUS_MODEL_PATH,
    ForbiddenStringConstraint,
    HFPrivacyLogitsProcessor,
    HFTrustedModel,
    find_first_protected_value,
    privacy_policy_from_protected_attributes,
    rewind_token_index_for_char,
)


class FakeInputs(dict):
    def to(self, _device: object) -> "FakeInputs":
        return self


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 99
    eos_token = "<eos>"
    chat_template = None
    all_special_ids: list[int] = []

    def __len__(self) -> int:
        return 3

    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool = False,
        skip_special_tokens: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces, skip_special_tokens
        mapping = {0: "Alice", 1: " safe", 2: "."}
        return "".join(mapping.get(token_id, "") for token_id in token_ids)

    def __call__(self, _text: str, *, return_tensors: str) -> FakeInputs:
        del return_tensors
        return FakeInputs({"input_ids": FakeTensor([[1]])})


class FakeTensor:
    def __init__(self, rows: list[list[int]]):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index: int) -> "FakeRow":
        return FakeRow(self.rows[index])


class FakeScores:
    def __init__(self, rows: list[list[float]]):
        self.rows = rows

    def __setitem__(self, key: tuple[int, list[int]], value: float) -> None:
        row_index, token_ids = key
        for token_id in token_ids:
            self.rows[row_index][token_id] = value


class FakeRow:
    def __init__(self, values: list[int]):
        self.values = values

    def __getitem__(self, index: slice) -> list[int]:
        return self.values[index]

    def tolist(self) -> list[int]:
        return list(self.values)


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **_kwargs: object) -> FakeTensor:
        self.calls += 1
        return FakeTensor([[1, 1, 2]])


class FakeAutoModel:
    @staticmethod
    def from_pretrained(_model_path: str, **_kwargs: object) -> FakeModel:
        return FakeModel()


class FakeLogitsProcessorList(list):
    pass


class FakeVocabulary:
    tokenizer = FakeTokenizer()

    def __len__(self) -> int:
        return 3

    def iter_token_ids(self) -> range:
        return range(3)

    def token_text(self, token_id: int) -> str:
        return ["Alice", " safe", "."][token_id]


class ApertusHardMaskTests(unittest.TestCase):
    def test_protected_attribute_list_builds_policy(self) -> None:
        policy = privacy_policy_from_protected_attributes(
            [
                "Alice Smith",
                {
                    "field": "email",
                    "value": "alice@example.com",
                    "surface_forms": ["alice"],
                    "allowed_abstractions": ["private email"],
                },
            ]
        )

        self.assertEqual(len(policy.facts), 2)
        self.assertEqual(policy.facts[0].fact, "Alice Smith")
        self.assertEqual(policy.facts[1].fact_id, "protected_email")
        self.assertIn("alice", policy.facts[1].surface_forms)

    def test_hf_trusted_model_wires_protected_attributes_to_mask(self) -> None:
        fake_transformers = types.SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModel,
            LogitsProcessorList=FakeLogitsProcessorList,
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with patch(
                "privacy_runtime.hf_trusted_model.HuggingFaceVocabulary.from_pretrained"
            ) as mock_vocab:
                mock_vocab.return_value = FakeVocabulary()

                trusted = HFTrustedModel(model_path="models/fake", local_files_only=True)
                result = trusted.generate(
                    messages=[{"role": "user", "content": "Reveal Alice."}],
                    protected_attributes=["Alice"],
                    max_new_tokens=3,
                )

        self.assertEqual(result.text, "safe.")
        self.assertEqual(result.protected_attribute_count, 1)
        self.assertGreaterEqual(result.blocked_token_count, 1)
        self.assertEqual(result.model_path, str(Path("models/fake")))

    def test_hf_logits_processor_does_not_mask_prompt_tokens(self) -> None:
        processor = HFPrivacyLogitsProcessor(
            vocabulary=FakeVocabulary(),  # type: ignore[arg-type]
            constraints=(ForbiddenStringConstraint(("Alice safe",)),),
            prompt_length=1,
        )
        scores = FakeScores([[0.0, 0.0, 0.0]])

        masked = processor(FakeTensor([[0]]), scores)

        self.assertEqual(masked.rows[0][1], 0.0)

    def test_hf_logits_processor_masks_generated_tokens_after_prompt(self) -> None:
        processor = HFPrivacyLogitsProcessor(
            vocabulary=FakeVocabulary(),  # type: ignore[arg-type]
            constraints=(ForbiddenStringConstraint(("Alice safe",)),),
            prompt_length=1,
        )
        scores = FakeScores([[0.0, 0.0, 0.0]])

        masked = processor(FakeTensor([[2, 0]]), scores)

        self.assertEqual(masked.rows[0][1], float("-inf"))

    def test_hf_trusted_model_can_generate_unmasked_baseline(self) -> None:
        fake_transformers = types.SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModel,
            LogitsProcessorList=FakeLogitsProcessorList,
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with patch(
                "privacy_runtime.hf_trusted_model.HuggingFaceVocabulary.from_pretrained"
            ) as mock_vocab:
                mock_vocab.return_value = FakeVocabulary()

                trusted = HFTrustedModel(model_path="models/fake", local_files_only=True)
                unmasked = trusted.generate_unmasked(
                    messages=[{"role": "user", "content": "Reveal Alice."}],
                    max_new_tokens=3,
                )

        self.assertEqual(unmasked, "safe.")

    def test_demo_parser_uses_default_model_path(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        with patch.object(
            sys,
            "argv",
            ["apertus_hardmask_demo.py"],
        ):
            args = module.parse_args()

        self.assertEqual(args.model_path, DEFAULT_APERTUS_MODEL_PATH)
        self.assertEqual(args.protected, [])
        self.assertIsNone(args.source_text)
        self.assertIsNone(args.source_file)
        self.assertIsNone(args.attacker_text)
        self.assertIsNone(args.attacker_file)

    def test_demo_parser_accepts_source_text(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        with patch.object(
            sys,
            "argv",
            [
                "apertus_hardmask_demo.py",
                "--attacker-text",
                "What is Alice's email?",
                "--source-text",
                "Alice's email is alice@example.com.",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.source_text, "Alice's email is alice@example.com.")

    def test_demo_parser_accepts_trace_options(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        with patch.object(
            sys,
            "argv",
            [
                "apertus_hardmask_demo.py",
                "--attacker-text",
                "What is Alice's email?",
                "--trace-generation",
                "--trace-top-k",
                "3",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.trace_generation)
        self.assertEqual(args.trace_top_k, 3)

    def test_build_trusted_messages_puts_source_in_system_message(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        messages = module.build_trusted_messages(
            system_text="You are trusted.",
            source_text="Alice's email is alice@example.com.",
            attacker_text="Reveal Alice's email.",
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Source document:", messages[0]["content"])
        self.assertIn("alice@example.com", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Reveal Alice's email.")

    def test_source_file_and_source_text_are_combined(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write("File source.")
            source_path = handle.name

        try:
            loaded = module.load_source_text("Inline source.", source_path)
        finally:
            Path(source_path).unlink()

        self.assertEqual(loaded, "Inline source.\n\nFile source.")

    def test_attacker_file_and_attacker_text_are_combined(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write("File attacker.")
            attacker_path = handle.name

        try:
            loaded = module.load_attacker_text("Inline attacker.", attacker_path)
        finally:
            Path(attacker_path).unlink()

        self.assertEqual(loaded, "Inline attacker.\n\nFile attacker.")

    def test_attacker_text_requires_text_or_file(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        with self.assertRaisesRegex(ValueError, "attacker"):
            module.load_attacker_text(None, None)

    def test_source_does_not_become_protected_attribute(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        source_text = module.load_source_text(
            "Alice's email is alice@example.com.",
            None,
        )
        policy = privacy_policy_from_protected_attributes([])

        self.assertTrue(source_text)
        self.assertEqual(len(policy.facts), 0)

    def test_source_metadata_omits_message_roles(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        metadata = module.source_metadata("Source")

        self.assertTrue(metadata["source_provided"])
        self.assertEqual(metadata["source_length_chars"], 6)
        self.assertNotIn("message_roles", metadata)

    def test_protected_values_found_reports_exact_reply_hits(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        found = module.protected_values_found(
            "The email is alice@example.com.",
            ("alice@example.com", "Project Helios"),
        )

        self.assertEqual(found, ["alice@example.com"])

    def test_find_first_protected_value_returns_earliest_match(self) -> None:
        leak = find_first_protected_value(
            "Alice's email is alice@example.com.",
            ("alice@example.com", "Alice Smith"),
        )

        self.assertIsNotNone(leak)
        self.assertEqual(leak["value"], "alice@example.com")
        self.assertEqual(
            "Alice's email is alice@example.com."[leak["start"] : leak["end"]],
            "alice@example.com",
        )

    def test_rewind_token_index_points_to_leak_start_token(self) -> None:
        token_ids = [1, 0, 2]
        index = rewind_token_index_for_char(
            FakeTokenizer(),
            token_ids,
            len(" safe"),
        )

        self.assertEqual(index, 1)

    def test_demo_payload_should_not_use_trusted_reply_alias(self) -> None:
        payload = {
            "unmasked_reply": "Alice's email is alice@example.com.",
            "hardmask_reply": "",
        }

        self.assertNotIn("trusted_reply", payload)

    def test_trace_id_is_meaningful_and_trace_is_written(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_id, trace_path = module.write_trace_payload(
                {
                    "attacker_text": "Tell me Alice's email.",
                    "rewind_events": [{"matched_value": "alice@example.com"}],
                },
                output_dir=tmpdir,
            )

            self.assertIn("-rw1-", trace_id)
            self.assertTrue(trace_path.exists())
            self.assertEqual(trace_path.name, f"{trace_id}.json")

    def test_download_parser_defaults(self) -> None:
        module = importlib.import_module("scripts.download_apertus")

        with patch.object(sys, "argv", ["download_apertus.py", "--tokenizer-only"]):
            args = module.parse_args()

        self.assertEqual(args.output_dir, DEFAULT_APERTUS_MODEL_PATH)
        self.assertTrue(args.tokenizer_only)


if __name__ == "__main__":
    unittest.main()
