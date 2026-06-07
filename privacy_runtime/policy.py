from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ProtectedFact:
    """A user- or context-specific fact that should not freely leave the agent."""

    fact_id: str
    fact: str
    surface_forms: tuple[str, ...] = ()
    semantic_hints: tuple[str, ...] = ()
    allowed_abstractions: tuple[str, ...] = ()
    default_budget: float = 0.0
    channel_budgets: Mapping[str, float] = field(default_factory=dict)

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


@dataclass(frozen=True)
class PrivacyPolicy:
    """Structured privacy policy induced from user context or provided manually."""

    facts: tuple[ProtectedFact, ...]
    forbidden_regexes: tuple[str, ...] = ()

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

