from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrivacyLossResult:
    private_logprob: float
    public_logprob: float
    privacy_loss: float
    token_count: int
    normalized: bool


@dataclass
class CounterfactualPrivacyCostEstimator:
    """Candidate-level counterfactual privacy loss estimator.

    Computes log P(y | private context) - log P(y | counterfactual context).
    This is a practical privacy-loss score, not a complete DP guarantee.
    """

    model: Any
    tokenizer: Any
    normalize_by_tokens: bool = False

    def estimate(
        self,
        private_prompt: str,
        public_prompt: str,
        candidate: str,
    ) -> PrivacyLossResult:
        private_logprob, token_count = sequence_logprob(
            self.model,
            self.tokenizer,
            private_prompt,
            candidate,
        )
        public_logprob, public_token_count = sequence_logprob(
            self.model,
            self.tokenizer,
            public_prompt,
            candidate,
        )
        if token_count != public_token_count:
            raise ValueError("private/public token counts differ for the same candidate")

        loss = private_logprob - public_logprob
        if self.normalize_by_tokens and token_count > 0:
            loss = loss / token_count

        return PrivacyLossResult(
            private_logprob=private_logprob,
            public_logprob=public_logprob,
            privacy_loss=loss,
            token_count=token_count,
            normalized=self.normalize_by_tokens,
        )


def sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
) -> tuple[float, int]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "sequence_logprob requires torch. Install optional HF dependencies "
            "with: pip install -r requirements-hf.txt"
        ) from exc

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids:
        return 0.0, 0

    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]
        log_probs = F.log_softmax(logits, dim=-1)

    start = len(prompt_ids)
    total = 0.0
    for offset, token_id in enumerate(completion_ids):
        absolute_pos = start + offset
        if absolute_pos == 0:
            continue
        total += float(log_probs[absolute_pos - 1, token_id].item())

    return total, len(completion_ids)

