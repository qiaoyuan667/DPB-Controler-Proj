from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constraints import (
    DEFAULT_PRIVATE_REGEXES,
    ForbiddenRegexConstraint,
    ForbiddenStringConstraint,
)
from .hf import HFPrivacyLogitsProcessor
from .policy import PrivacyPolicy
from .protected_attributes import (
    ProtectedAttributeInput,
    privacy_policy_from_protected_attributes,
)
from .tokenizer import HuggingFaceVocabulary


DEFAULT_APERTUS_MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_APERTUS_MODEL_PATH = "models/apertus-8b-instruct-2509"


@dataclass(frozen=True)
class HFTrustedGenerationResult:
    text: str
    blocked_token_count: int
    protected_attribute_count: int
    model_path: str
    blocked_token_sample: tuple[tuple[int, str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class HFHardMaskTraceResult:
    text: str
    steps: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class HFRewindGenerationResult:
    text: str
    protected_attribute_count: int
    model_path: str
    steps: tuple[dict[str, Any], ...]
    rewind_events: tuple[dict[str, Any], ...]
    fallback_used: bool


@dataclass
class HFTrustedModel:
    """Local HuggingFace trusted-model backend with decoder-time hard masks."""

    model_path: str = DEFAULT_APERTUS_MODEL_PATH
    local_files_only: bool = True
    include_default_regexes: bool = True
    fallback_text: str = "I cannot disclose protected personal information."
    device_map: str | Mapping[str, Any] | None = "auto"
    torch_dtype: str | None = "auto"

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "HFTrustedModel requires transformers and torch. Install optional "
                "dependencies with: pip install -r requirements-hf.txt"
            ) from exc

        kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype

        self.vocabulary = HuggingFaceVocabulary.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
        )
        self.tokenizer = self.vocabulary.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        protected_attributes: Sequence[ProtectedAttributeInput] | None = None,
        policy: PrivacyPolicy | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        seed: int | None = None,
    ) -> HFTrustedGenerationResult:
        if policy is None:
            policy = privacy_policy_from_protected_attributes(
                tuple(protected_attributes or ()),
            )

        prompt_text = render_chat_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        try:
            from transformers import LogitsProcessorList
        except ImportError as exc:
            raise ImportError(
                "HFTrustedModel generation requires transformers. Install with: "
                "pip install -r requirements-hf.txt"
            ) from exc

        constraints = [
            ForbiddenStringConstraint(policy.all_forbidden_strings()),
        ]
        if self.include_default_regexes:
            constraints.append(ForbiddenRegexConstraint(DEFAULT_PRIVATE_REGEXES))

        privacy_processor = HFPrivacyLogitsProcessor(
            vocabulary=self.vocabulary,
            constraints=tuple(constraints),
            prompt_text=prompt_text,
            prompt_length=inputs["input_ids"].shape[1],
        )
        logits_processor = LogitsProcessorList([privacy_processor])
        initial_decision = privacy_processor.processor.blocked_tokens("")

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "logits_processor": logits_processor,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        if seed is not None:
            self._manual_seed(seed)

        output_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(
            generated_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        ).strip()

        return HFTrustedGenerationResult(
            text=text,
            blocked_token_count=len(initial_decision.blocked_token_ids),
            protected_attribute_count=len(policy.facts),
            model_path=str(Path(self.model_path)),
            blocked_token_sample=_blocked_token_sample(
                self.vocabulary,
                initial_decision.reasons,
                initial_decision.blocked_token_ids,
            ),
        )

    def generate_unmasked(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        seed: int | None = None,
    ) -> str:
        prompt_text = render_chat_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        if seed is not None:
            self._manual_seed(seed)

        output_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(
            generated_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        ).strip()

    def generate_with_rewind(
        self,
        *,
        messages: list[dict[str, str]],
        protected_attributes: Sequence[ProtectedAttributeInput] | None = None,
        policy: PrivacyPolicy | None = None,
        max_new_tokens: int = 256,
        top_k: int = 5,
        max_rewinds: int = 8,
        trace: bool = False,
        seed: int | None = None,
    ) -> HFRewindGenerationResult:
        if policy is None:
            policy = privacy_policy_from_protected_attributes(
                tuple(protected_attributes or ()),
            )
        if seed is not None:
            self._manual_seed(seed)

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "generate_with_rewind requires torch. Install optional "
                "dependencies with: pip install -r requirements-hf.txt"
            ) from exc

        prompt_text = render_chat_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        prompt_ids = inputs["input_ids"]
        generated_token_ids: list[int] = []
        state_bans: dict[tuple[int, ...], set[int]] = {}
        rewind_events: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        eos_token_id = self.tokenizer.eos_token_id
        fallback_used = False

        self.model.eval()
        with torch.no_grad():
            step = 0
            while step < max_new_tokens:
                step += 1
                input_ids = _append_generated_ids(
                    prompt_ids,
                    generated_token_ids,
                    torch=torch,
                )
                outputs = self.model(input_ids=input_ids)
                scores = outputs.logits[:, -1, :].clone()
                state = tuple(generated_token_ids)
                banned_ids = state_bans.get(state, set())
                for token_id in banned_ids:
                    scores[0, token_id] = -torch.inf

                probs = torch.softmax(scores[0], dim=-1)
                selected_token_id = int(torch.argmax(scores[0]).item())
                selected_prob = float(probs[selected_token_id].item())
                selected_token_text = self.vocabulary.token_text(selected_token_id)

                if trace:
                    steps.append(
                        {
                            "step": step,
                            "generated_text_before": self._decode_generated(
                                generated_token_ids,
                                skip_special_tokens=True,
                            ),
                            "state_banned_token_ids": sorted(banned_ids),
                            "top_after_state_bans": _top_token_records(
                                self.vocabulary,
                                probs,
                                top_k=top_k,
                            ),
                            "selected_token": {
                                "token_id": selected_token_id,
                                "text": selected_token_text,
                                "probability": selected_prob,
                                "is_eos": selected_token_id == eos_token_id,
                            },
                        }
                    )

                if selected_token_id == eos_token_id:
                    break

                generated_token_ids.append(selected_token_id)
                generated_text = self._decode_generated(
                    generated_token_ids,
                    skip_special_tokens=True,
                )
                leak = find_first_protected_value(
                    generated_text,
                    policy.all_forbidden_strings(),
                )
                if leak is None:
                    continue

                if len(rewind_events) >= max_rewinds:
                    generated_token_ids = self.tokenizer.encode(
                        self.fallback_text,
                        add_special_tokens=False,
                    )
                    fallback_used = True
                    break

                rewind_index = rewind_token_index_for_char(
                    self.tokenizer,
                    generated_token_ids,
                    leak["start"],
                )
                leaked_token_id = generated_token_ids[rewind_index]
                rewind_state = tuple(generated_token_ids[:rewind_index])
                state_bans.setdefault(rewind_state, set()).add(leaked_token_id)
                rewind_text = self._decode_generated(
                    list(rewind_state),
                    skip_special_tokens=True,
                )
                rewind_events.append(
                    {
                        "matched_value": leak["value"],
                        "text_before_rewind": generated_text,
                        "rewind_to_text": rewind_text,
                        "rewind_token_index": rewind_index,
                        "banned_token_id": leaked_token_id,
                        "banned_token_text": self.vocabulary.token_text(leaked_token_id),
                    }
                )
                generated_token_ids = list(rewind_state)

        final_text = self._decode_generated(
            generated_token_ids,
            skip_special_tokens=True,
        ).strip()
        return HFRewindGenerationResult(
            text=final_text,
            protected_attribute_count=len(policy.facts),
            model_path=str(Path(self.model_path)),
            steps=tuple(steps),
            rewind_events=tuple(rewind_events),
            fallback_used=fallback_used,
        )

    def trace_hardmask_generation(
        self,
        *,
        messages: list[dict[str, str]],
        protected_attributes: Sequence[ProtectedAttributeInput] | None = None,
        policy: PrivacyPolicy | None = None,
        max_new_tokens: int = 32,
        top_k: int = 5,
        seed: int | None = None,
    ) -> HFHardMaskTraceResult:
        if policy is None:
            policy = privacy_policy_from_protected_attributes(
                tuple(protected_attributes or ()),
            )
        if seed is not None:
            self._manual_seed(seed)

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "trace_hardmask_generation requires torch. Install optional "
                "dependencies with: pip install -r requirements-hf.txt"
            ) from exc

        prompt_text = render_chat_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        constraints = [
            ForbiddenStringConstraint(policy.all_forbidden_strings()),
        ]
        if self.include_default_regexes:
            constraints.append(ForbiddenRegexConstraint(DEFAULT_PRIVATE_REGEXES))
        privacy_processor = HFPrivacyLogitsProcessor(
            vocabulary=self.vocabulary,
            constraints=tuple(constraints),
            prompt_text=prompt_text,
            prompt_length=inputs["input_ids"].shape[1],
        )

        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        generated_token_ids: list[int] = []
        trace: list[dict[str, Any]] = []
        eos_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        with torch.no_grad():
            for step in range(1, max_new_tokens + 1):
                outputs = self.model(input_ids=input_ids)
                raw_scores = outputs.logits[:, -1, :]
                masked_scores = privacy_processor(input_ids, raw_scores.clone())

                raw_probs = torch.softmax(raw_scores[0], dim=-1)
                masked_probs = torch.softmax(masked_scores[0], dim=-1)
                raw_top = _top_token_records(self.vocabulary, raw_probs, top_k=top_k)
                masked_top = _top_token_records(
                    self.vocabulary,
                    masked_probs,
                    top_k=top_k,
                )
                selected_token_id = int(torch.argmax(masked_scores[0]).item())
                selected_prob = float(masked_probs[selected_token_id].item())
                raw_top_token_id = raw_top[0]["token_id"] if raw_top else None

                generated_text_before = self.tokenizer.decode(
                    generated_token_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=True,
                )
                generated_text_for_mask = self.tokenizer.decode(
                    generated_token_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
                decision = privacy_processor.processor.blocked_tokens(
                    generated_text_for_mask
                )
                trace.append(
                    {
                        "step": step,
                        "generated_text_before": generated_text_before,
                        "generated_text_for_mask": generated_text_for_mask,
                        "raw_top": raw_top,
                        "masked_top": masked_top,
                        "raw_top_was_masked": raw_top_token_id
                        in decision.blocked_token_ids
                        if raw_top_token_id is not None
                        else False,
                        "selected_token": {
                            "token_id": selected_token_id,
                            "text": self.vocabulary.token_text(selected_token_id),
                            "probability_after_mask": selected_prob,
                            "is_eos": selected_token_id == eos_token_id,
                        },
                        "blocked_token_count": len(decision.blocked_token_ids),
                        "blocked_top_reasons": list(
                            decision.reasons.get(int(raw_top_token_id), ())
                        )
                        if raw_top_token_id is not None
                        else [],
                    }
                )

                if selected_token_id == eos_token_id:
                    break

                generated_token_ids.append(selected_token_id)
                next_token = torch.tensor(
                    [[selected_token_id]],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, next_token], dim=1)

        text = self.tokenizer.decode(
            input_ids[0][prompt_len:],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        ).strip()
        return HFHardMaskTraceResult(text=text, steps=tuple(trace))

    def _decode_generated(
        self,
        generated_token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        return self.tokenizer.decode(
            generated_token_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=skip_special_tokens,
        )

    @staticmethod
    def _manual_seed(seed: int) -> None:
        try:
            import torch
        except ImportError:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines: list[str] = []
    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}:\n{message['content']}")
    lines.append("ASSISTANT:\n")
    return "\n\n".join(lines)


def _blocked_token_sample(
    vocabulary: HuggingFaceVocabulary,
    reasons: Mapping[int, tuple[str, ...]],
    blocked_token_ids: frozenset[int],
    *,
    limit: int = 20,
) -> tuple[tuple[int, str, tuple[str, ...]], ...]:
    out: list[tuple[int, str, tuple[str, ...]]] = []
    for token_id in sorted(blocked_token_ids)[:limit]:
        out.append((token_id, vocabulary.token_text(token_id), reasons.get(token_id, ())))
    return tuple(out)


def _top_token_records(
    vocabulary: HuggingFaceVocabulary,
    probs: Any,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    values, indices = probs.topk(max(1, top_k))
    records = []
    for value, token_id in zip(values.tolist(), indices.tolist(), strict=True):
        records.append(
            {
                "token_id": int(token_id),
                "text": vocabulary.token_text(int(token_id)),
                "probability": float(value),
            }
        )
    return records


def _append_generated_ids(
    prompt_ids: Any,
    generated_token_ids: list[int],
    *,
    torch: Any,
) -> Any:
    if not generated_token_ids:
        return prompt_ids
    next_tokens = torch.tensor(
        [generated_token_ids],
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    )
    return torch.cat([prompt_ids, next_tokens], dim=1)


def find_first_protected_value(
    text: str,
    protected_values: tuple[str, ...],
) -> dict[str, Any] | None:
    lowered = text.casefold()
    matches = []
    for value in protected_values:
        normalized = value.casefold()
        if not normalized:
            continue
        index = lowered.find(normalized)
        if index >= 0:
            matches.append({"value": value, "start": index, "end": index + len(value)})
    if not matches:
        return None
    return min(matches, key=lambda item: (int(item["start"]), -len(str(item["value"]))))


def rewind_token_index_for_char(
    tokenizer: Any,
    generated_token_ids: list[int],
    char_index: int,
) -> int:
    if not generated_token_ids:
        return 0
    for token_index in range(len(generated_token_ids)):
        prefix_text = tokenizer.decode(
            generated_token_ids[: token_index + 1],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
        if len(prefix_text) > char_index:
            return token_index
    return max(0, len(generated_token_ids) - 1)
