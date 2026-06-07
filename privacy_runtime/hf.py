from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .constraints import PrivacyLogitProcessor, TokenConstraint
from .tokenizer import HuggingFaceVocabulary


@dataclass
class HFPrivacyLogitsProcessor:
    """Callable compatible with HuggingFace ``generate``.

    Transformers accepts any callable with ``(input_ids, scores) -> scores`` in a
    LogitsProcessorList. This adapter decodes each batch row, asks the privacy
    processor which token ids are blocked, and sets their logits to ``-inf``.
    """

    vocabulary: HuggingFaceVocabulary
    constraints: tuple[TokenConstraint, ...]
    prompt_text: str = ""

    def __post_init__(self) -> None:
        self.processor = PrivacyLogitProcessor(
            vocabulary=self.vocabulary,
            constraints=self.constraints,
        )

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for row_index in range(input_ids.shape[0]):
            token_ids = input_ids[row_index].tolist()
            generated_text = self.vocabulary.tokenizer.decode(
                token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            if self.prompt_text and generated_text.startswith(self.prompt_text):
                generated_text = generated_text[len(self.prompt_text) :]

            decision = self.processor.blocked_tokens(generated_text)
            if decision.blocked_token_ids:
                blocked_ids = list(decision.blocked_token_ids)
                scores[row_index, blocked_ids] = -math.inf
        return scores

