from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ProtectedFact:
    """A user- or context-specific fact that should not freely leave the agent."""

    fact_id: str
    fact: str
    surface_forms: tuple[str, ...] = ()
    semantic_hints: tuple[str, ...] = ()
    allowed_abstractions: tuple[str, ...] = ()
    fact_type: str = "unspecified"
    privacy_scope: str = "value"
    counterfactual_values: tuple[str, ...] = ()
    default_budget: float = 0.0
    channel_budgets: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fact_id.strip():
            raise ValueError("fact_id must be non-empty")
        if not self.fact.strip():
            raise ValueError("fact must be non-empty")
        if self.default_budget < 0:
            raise ValueError("default_budget must be non-negative")
        if any(float(value) < 0 for value in self.channel_budgets.values()):
            raise ValueError("channel budgets must be non-negative")

    def all_surface_forms(self) -> tuple[str, ...]:
        values = [self.fact, *self.surface_forms]
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            key = value.strip().casefold()
            if key and key not in seen:
                seen.add(key)
                unique.append(value.strip())
        return tuple(unique)

    def budget_for(self, channel: str) -> float:
        return float(self.channel_budgets.get(channel, self.default_budget))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProtectedFact:
        return cls(
            fact_id=str(value["fact_id"]),
            fact=str(value["fact"]),
            surface_forms=_string_tuple(value.get("surface_forms")),
            semantic_hints=_string_tuple(value.get("semantic_hints")),
            allowed_abstractions=_string_tuple(value.get("allowed_abstractions")),
            fact_type=str(value.get("fact_type") or "unspecified"),
            privacy_scope=str(value.get("privacy_scope") or "value"),
            counterfactual_values=_string_tuple(value.get("counterfactual_values")),
            default_budget=float(value.get("default_budget", 0.0)),
            channel_budgets={
                str(channel): float(budget)
                for channel, budget in _mapping(
                    value.get("channel_budgets", {})
                ).items()
            },
        )


@dataclass(frozen=True)
class PrivacyPolicy:
    """Structured privacy policy induced from user context or provided manually."""

    facts: tuple[ProtectedFact, ...]
    forbidden_regexes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact_id values must be unique")

    def fact_by_id(self, fact_id: str) -> ProtectedFact:
        for fact in self.facts:
            if fact.fact_id == fact_id:
                return fact
        raise KeyError(f"unknown protected fact: {fact_id}")

    def all_forbidden_strings(self) -> tuple[str, ...]:
        strings: list[str] = []
        for fact in self.facts:
            strings.extend(fact.all_surface_forms())
        return tuple(strings)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PrivacyPolicy:
        raw_facts = value.get("facts", ())
        if not isinstance(raw_facts, (list, tuple)):
            raise ValueError("policy facts must be a list")
        return cls(
            facts=tuple(ProtectedFact.from_mapping(_mapping(item)) for item in raw_facts),
            forbidden_regexes=_string_tuple(value.get("forbidden_regexes")),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of strings")
    return tuple(str(item) for item in value if str(item).strip())
