from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .policy import PrivacyPolicy


CostMap = Mapping[str, float]


@dataclass
class PrivacyAccountant:
    """Tracks sequential privacy spending for protected facts."""

    policy: PrivacyPolicy
    spent: dict[tuple[str, str], float] = field(default_factory=dict)

    def budget(self, fact_id: str, channel: str) -> float:
        return self.policy.fact_by_id(fact_id).budget_for(channel)

    def spent_for(self, fact_id: str, channel: str) -> float:
        return self.spent.get((fact_id, channel), 0.0)

    def remaining(self, fact_id: str, channel: str) -> float:
        return max(0.0, self.budget(fact_id, channel) - self.spent_for(fact_id, channel))

    def can_spend(self, costs: CostMap, channel: str) -> bool:
        for fact_id, cost in costs.items():
            if cost > self.remaining(fact_id, channel):
                return False
        return True

    def spend(self, costs: CostMap, channel: str) -> None:
        if not self.can_spend(costs, channel):
            raise ValueError("privacy budget exceeded")
        for fact_id, cost in costs.items():
            key = (fact_id, channel)
            self.spent[key] = self.spent.get(key, 0.0) + float(cost)

