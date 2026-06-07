from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .policy import PrivacyPolicy, ProtectedFact


@dataclass(frozen=True)
class Replacement:
    fact_id: str
    original: str
    replacement: str


@dataclass(frozen=True)
class CounterfactualText:
    private_text: str
    public_text: str
    replacements: tuple[Replacement, ...]


def build_counterfactual_text(
    text: str,
    policy: PrivacyPolicy,
    *,
    placeholder_prefix: str = "PRIVATE",
) -> CounterfactualText:
    """Replace protected surface forms with safe abstractions or placeholders."""

    public_text = text
    replacements: list[Replacement] = []

    for fact in policy.facts:
        replacement = safe_replacement_for_fact(
            fact,
            placeholder_prefix=placeholder_prefix,
        )
        for surface in _longest_first(fact.all_surface_forms()):
            if not surface:
                continue
            pattern = re.compile(re.escape(surface), flags=re.IGNORECASE)
            if not pattern.search(public_text):
                continue
            public_text = pattern.sub(replacement, public_text)
            replacements.append(
                Replacement(
                    fact_id=fact.fact_id,
                    original=surface,
                    replacement=replacement,
                )
            )

    return CounterfactualText(
        private_text=text,
        public_text=public_text,
        replacements=tuple(replacements),
    )


def safe_replacement_for_fact(
    fact: ProtectedFact,
    *,
    placeholder_prefix: str = "PRIVATE",
) -> str:
    if fact.allowed_abstractions:
        return fact.allowed_abstractions[0]

    label = _label_from_fact_id(fact.fact_id)
    return f"[{placeholder_prefix}_{label}]"


def _longest_first(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (-len(value), value.casefold()))


def _label_from_fact_id(fact_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", fact_id).strip("_").upper()
    return cleaned or "FACT"

