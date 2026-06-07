from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from .policy import PrivacyPolicy, ProtectedFact


@dataclass(frozen=True)
class VerificationResult:
    hard_violation: bool
    costs: Mapping[str, float]
    findings: tuple[str, ...] = ()

    @property
    def total_cost(self) -> float:
        return sum(self.costs.values())


@dataclass(frozen=True)
class LeakageVerifier:
    """Heuristic privacy-cost estimator.

    This is deliberately simple. Replace it with a local entailment model or an
    attacker posterior estimator when moving from prototype to research system.
    """

    policy: PrivacyPolicy
    exact_cost: float = 1.0
    semantic_cost: float = 0.35
    abstraction_cost: float = 0.05
    regex_cost: float = 1.0
    regex_flags: int = re.IGNORECASE
    compiled_regexes: tuple[re.Pattern[str], ...] = field(init=False)

    def __post_init__(self) -> None:
        compiled = tuple(
            re.compile(pattern, self.regex_flags)
            for pattern in self.policy.forbidden_regexes
        )
        object.__setattr__(self, "compiled_regexes", compiled)

    def verify(self, text: str) -> VerificationResult:
        findings: list[str] = []
        costs: dict[str, float] = {}
        hard_violation = False

        for fact in self.policy.facts:
            cost, fact_findings, fact_hard_violation = self._score_fact(text, fact)
            if cost > 0:
                costs[fact.fact_id] = cost
                findings.extend(fact_findings)
            hard_violation = hard_violation or fact_hard_violation

        for pattern in self.compiled_regexes:
            if pattern.search(text):
                hard_violation = True
                findings.append(f"regex leak: {pattern.pattern}")

        return VerificationResult(
            hard_violation=hard_violation,
            costs=costs,
            findings=tuple(findings),
        )

    def _score_fact(
        self, text: str, fact: ProtectedFact
    ) -> tuple[float, list[str], bool]:
        lowered = text.casefold()
        findings: list[str] = []
        cost = 0.0
        hard_violation = False

        for surface in fact.all_surface_forms():
            if surface.casefold() in lowered:
                findings.append(f"{fact.fact_id}: exact surface form '{surface}'")
                cost = max(cost, self.exact_cost)
                hard_violation = True

        for hint in fact.semantic_hints:
            if hint.casefold() in lowered:
                findings.append(f"{fact.fact_id}: semantic hint '{hint}'")
                cost = max(cost, self.semantic_cost)

        for abstraction in fact.allowed_abstractions:
            if abstraction.casefold() in lowered:
                findings.append(f"{fact.fact_id}: allowed abstraction '{abstraction}'")
                cost = max(cost, self.abstraction_cost)

        return cost, findings, hard_violation

