from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


class TokenVocabulary(Protocol):
    """Minimal tokenizer facade needed by privacy constraints."""

    def __len__(self) -> int:
        raise NotImplementedError

    def iter_token_ids(self) -> Iterable[int]:
        raise NotImplementedError

    def token_text(self, token_id: int) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class SimpleVocabulary:
    """A tiny tokenizer facade for logit-processor tests and demos.

    Real deployments should wrap the production tokenizer and expose token text
    for each token id.
    """

    tokens: tuple[str, ...]

    @classmethod
    def from_tokens(cls, tokens: list[str] | tuple[str, ...]) -> "SimpleVocabulary":
        return cls(tokens=tuple(tokens))

    def __len__(self) -> int:
        return len(self.tokens)

    def iter_token_ids(self) -> Iterable[int]:
        return range(len(self.tokens))

    def token_text(self, token_id: int) -> str:
        return self.tokens[token_id]

    def token_id(self, text: str) -> int:
        return self.tokens.index(text)


@dataclass(frozen=True)
class HuggingFaceVocabulary:
    """Adapter around a HuggingFace tokenizer.

    This keeps the rest of the privacy runtime independent from transformers.
    Token text is produced with ``decode([token_id])`` so byte-pair and
    SentencePiece space markers are normalized to the actual generated text.
    """

    tokenizer: Any
    include_special_tokens: bool = False
    _special_token_ids: frozenset[int] = field(init=False, repr=False)
    _token_text_cache: dict[int, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        object.__setattr__(self, "_special_token_ids", frozenset(special_ids))

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        local_files_only: bool = True,
        include_special_tokens: bool = False,
        **kwargs: Any,
    ) -> "HuggingFaceVocabulary":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceVocabulary requires the optional 'transformers' "
                "package. Install it with: pip install transformers"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            **kwargs,
        )
        return cls(
            tokenizer=tokenizer,
            include_special_tokens=include_special_tokens,
        )

    def __len__(self) -> int:
        return len(self.tokenizer)

    def iter_token_ids(self) -> Iterable[int]:
        for token_id in range(len(self)):
            if not self.include_special_tokens and token_id in self._special_token_ids:
                continue
            yield token_id

    def token_text(self, token_id: int) -> str:
        cached = self._token_text_cache.get(token_id)
        if cached is not None:
            return cached
        text = self.tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        self._token_text_cache[token_id] = text
        return text

    def token_ids_for_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)
