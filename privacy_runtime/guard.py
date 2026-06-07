from __future__ import annotations

from dataclasses import dataclass

from .privacy_cost import CounterfactualPrivacyCostEstimator, PrivacyLossResult
from .rewriter import RewriteResult, TemplateAbstractionRewriter
from .semantic_verifier import SemanticLeakageVerifierV0, SemanticVerification
from .verifier import LeakageVerifier, VerificationResult


@dataclass(frozen=True)
class GuardDecision:
    original_text: str
    final_text: str
    exact_verification: VerificationResult
    semantic_verification: SemanticVerification
    rewrite: RewriteResult | None
    privacy_loss: PrivacyLossResult | None
    accepted: bool
    reasons: tuple[str, ...]


@dataclass
class PrivacyGuardV0:
    """Candidate-level v0 guard: verify, optionally estimate cost, then rewrite."""

    exact_verifier: LeakageVerifier
    semantic_verifier: SemanticLeakageVerifierV0
    rewriter: TemplateAbstractionRewriter
    cost_estimator: CounterfactualPrivacyCostEstimator | None = None
    privacy_loss_epsilon: float | None = None
    semantic_threshold: float = 0.5

    def protect(
        self,
        text: str,
        *,
        private_prompt: str | None = None,
        public_prompt: str | None = None,
    ) -> GuardDecision:
        exact = self.exact_verifier.verify(text)
        semantic = self.semantic_verifier.verify(text)
        privacy_loss = None
        reasons: list[str] = []

        if exact.hard_violation:
            reasons.append("exact_hard_violation")
        if semantic.max_score >= self.semantic_threshold:
            reasons.append("semantic_risk")

        if self.cost_estimator is not None and private_prompt and public_prompt:
            privacy_loss = self.cost_estimator.estimate(
                private_prompt,
                public_prompt,
                text,
            )
            if (
                self.privacy_loss_epsilon is not None
                and privacy_loss.privacy_loss > self.privacy_loss_epsilon
            ):
                reasons.append("privacy_loss_budget_exceeded")

        if reasons:
            rewrite = self.rewriter.rewrite(text, semantic)
            return GuardDecision(
                original_text=text,
                final_text=rewrite.text,
                exact_verification=exact,
                semantic_verification=semantic,
                rewrite=rewrite,
                privacy_loss=privacy_loss,
                accepted=False,
                reasons=tuple(reasons),
            )

        return GuardDecision(
            original_text=text,
            final_text=text,
            exact_verification=exact,
            semantic_verification=semantic,
            rewrite=None,
            privacy_loss=privacy_loss,
            accepted=True,
            reasons=(),
        )

