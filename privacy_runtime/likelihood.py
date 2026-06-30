from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SequenceLikelihood:
    """Teacher-forced likelihood for one fixed completion token sequence."""

    logprob: float
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    token_logprobs: tuple[float, ...]
    prompt_token_count: int

    def __post_init__(self) -> None:
        lengths = {len(self.token_ids), len(self.tokens), len(self.token_logprobs)}
        if len(lengths) != 1:
            raise ValueError("token ids, strings, and logprobs must have equal lengths")

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


class LikelihoodScorer(Protocol):
    def score(self, prompt: str, completion: str) -> SequenceLikelihood:
        ...


@dataclass
class HFCausalLMLikelihoodScorer:
    """Score fixed completions with a HuggingFace causal language model.

    Completion ids are tokenized independently and appended to prompt ids. This
    keeps the evaluated completion token sequence identical across private and
    counterfactual prompts, which is required for a valid likelihood ratio.
    """

    model: Any
    tokenizer: Any
    add_special_tokens_to_prompt: bool = False
    temperature: float = 1.0
    max_sequence_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def score(self, prompt: str, completion: str) -> SequenceLikelihood:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "HFCausalLMLikelihoodScorer requires torch. Install optional "
                "dependencies with: pip install -r requirements-hf.txt"
            ) from exc

        prompt_ids = tuple(
            int(token_id)
            for token_id in self.tokenizer.encode(
                prompt,
                add_special_tokens=self.add_special_tokens_to_prompt,
            )
        )
        completion_ids = tuple(
            int(token_id)
            for token_id in self.tokenizer.encode(
                completion,
                add_special_tokens=False,
            )
        )
        if not completion_ids:
            return SequenceLikelihood(0.0, (), (), (), len(prompt_ids))

        if not prompt_ids:
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_token_id is None:
                raise ValueError("a non-empty prompt or tokenizer BOS token is required")
            prompt_ids = (int(bos_token_id),)

        input_token_ids = prompt_ids + completion_ids
        if (
            self.max_sequence_tokens is not None
            and len(input_token_ids) > self.max_sequence_tokens
        ):
            raise ValueError(
                f"sequence has {len(input_token_ids)} tokens, exceeding "
                f"max_sequence_tokens={self.max_sequence_tokens}"
            )

        input_ids = torch.tensor([input_token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        input_device = _input_device(self.model)
        input_ids = input_ids.to(input_device)
        attention_mask = attention_mask.to(input_device)

        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[0]

        start = len(prompt_ids)
        prediction_logits = logits[start - 1 : len(input_token_ids) - 1].float()
        if self.temperature != 1.0:
            prediction_logits = prediction_logits / self.temperature
        target_ids = input_ids[0, start:].to(prediction_logits.device)
        selected_logits = prediction_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        normalizers = prediction_logits.logsumexp(dim=-1)
        token_logprobs_tensor = selected_logits - normalizers
        token_logprobs = tuple(
            float(value) for value in token_logprobs_tensor.detach().cpu().tolist()
        )

        tokens = _token_strings(self.tokenizer, completion_ids)
        return SequenceLikelihood(
            logprob=float(sum(token_logprobs)),
            token_ids=completion_ids,
            tokens=tokens,
            token_logprobs=token_logprobs,
            prompt_token_count=len(prompt_ids),
        )


def _input_device(model: Any) -> Any:
    try:
        embeddings = model.get_input_embeddings()
        device = embeddings.weight.device
        if getattr(device, "type", None) != "meta":
            return device
    except (AttributeError, RuntimeError):
        pass

    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise ValueError("could not determine model input device") from exc


def _token_strings(tokenizer: Any, token_ids: tuple[int, ...]) -> tuple[str, ...]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        converted = tokenizer.convert_ids_to_tokens(list(token_ids))
        if isinstance(converted, str):
            return (converted,)
        return tuple(str(token) for token in converted)
    return tuple(str(token_id) for token_id in token_ids)
