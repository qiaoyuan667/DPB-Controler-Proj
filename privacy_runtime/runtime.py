from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .budget import PrivacyAccountant
from .verifier import LeakageVerifier, VerificationResult


@dataclass(frozen=True)
class CandidateAction:
    text: str
    utility: float
    channel: str = "final"
    recipient: str = "user"
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RejectedAction:
    candidate: CandidateAction
    verification: VerificationResult
    reason: str


@dataclass(frozen=True)
class RuntimeDecision:
    selected: CandidateAction | None
    verification: VerificationResult | None
    rejected: tuple[RejectedAction, ...]


@dataclass
class PrivacyRuntime:
    """Chooses privacy-valid agent actions under a sequential budget."""

    verifier: LeakageVerifier
    accountant: PrivacyAccountant

    def choose(self, candidates: list[CandidateAction]) -> RuntimeDecision:
        rejected: list[RejectedAction] = []

        for candidate in sorted(candidates, key=lambda item: item.utility, reverse=True):
            verification = self.verifier.verify(candidate.text)

            if verification.hard_violation:
                rejected.append(
                    RejectedAction(candidate, verification, "hard privacy violation")
                )
                continue

            if not self.accountant.can_spend(verification.costs, candidate.channel):
                rejected.append(
                    RejectedAction(candidate, verification, "privacy budget exceeded")
                )
                continue

            self.accountant.spend(verification.costs, candidate.channel)
            return RuntimeDecision(candidate, verification, tuple(rejected))

        return RuntimeDecision(None, None, tuple(rejected))

