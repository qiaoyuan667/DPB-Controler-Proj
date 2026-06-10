from __future__ import annotations

import argparse
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from privacy_runtime import (
    DEFAULT_APERTUS_MODEL_PATH,
    HFTrustedModel,
    privacy_policy_from_protected_attributes,
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


class FakeRow:
    def __init__(self, values: list[int]):
        self.values = values

    def __getitem__(self, index: slice) -> list[int]:
        return self.values[index]

    def tolist(self) -> list[int]:
        return list(self.values)


class FakeModel:
    device = "cpu"

    def generate(self, **_kwargs: object) -> FakeTensor:
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

    def test_demo_parser_uses_default_model_path(self) -> None:
        module = importlib.import_module("examples.apertus_hardmask_demo")

        with patch.object(
            sys,
            "argv",
            ["apertus_hardmask_demo.py", "--attacker-text", "What is the secret?"],
        ):
            args = module.parse_args()

        self.assertEqual(args.model_path, DEFAULT_APERTUS_MODEL_PATH)
        self.assertEqual(args.protected, [])

    def test_download_parser_defaults(self) -> None:
        module = importlib.import_module("scripts.download_apertus")

        with patch.object(sys, "argv", ["download_apertus.py", "--tokenizer-only"]):
            args = module.parse_args()

        self.assertEqual(args.output_dir, DEFAULT_APERTUS_MODEL_PATH)
        self.assertTrue(args.tokenizer_only)


if __name__ == "__main__":
    unittest.main()
