from __future__ import annotations

import importlib.util
import math
import types
import unittest

from privacy_runtime import HFCausalLMLikelihoodScorer


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is optional")
class HFLikelihoodTests(unittest.TestCase):
    def test_scores_only_completion_tokens(self) -> None:
        import torch

        class ToyTokenizer:
            bos_token_id = 0

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                mapping = {"P": [1], "ab": [2, 3]}
                return mapping[text]

            def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
                names = {2: "a", 3: "b"}
                return [names[token_id] for token_id in ids]

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(5, 2)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                batch, length = input_ids.shape
                logits = torch.zeros(batch, length, 5, device=input_ids.device)
                for position in range(length):
                    current = int(input_ids[0, position])
                    if current == 1:
                        logits[0, position, 2] = 2.0
                    elif current == 2:
                        logits[0, position, 3] = 3.0
                return types.SimpleNamespace(logits=logits)

        result = HFCausalLMLikelihoodScorer(ToyModel(), ToyTokenizer()).score("P", "ab")

        expected_a = 2.0 - math.log(math.exp(2.0) + 4.0)
        expected_b = 3.0 - math.log(math.exp(3.0) + 4.0)
        self.assertEqual(result.token_ids, (2, 3))
        self.assertEqual(result.tokens, ("a", "b"))
        self.assertEqual(result.prompt_token_count, 1)
        self.assertAlmostEqual(result.logprob, expected_a + expected_b, places=5)


if __name__ == "__main__":
    unittest.main()
