from __future__ import annotations

import unittest

from privacy_runtime import (
    LeakageVerifier,
    PrivacyGuardV0,
    PrivacyPolicy,
    ProtectedFact,
    SemanticLeakageVerifierV0,
    TemplateAbstractionRewriter,
    build_counterfactual_text,
)


class V0GuardTests(unittest.TestCase):
    def build_policy(self) -> PrivacyPolicy:
        return PrivacyPolicy(
            facts=(
                ProtectedFact(
                    fact_id="budget",
                    fact="budget ceiling is $2M",
                    surface_forms=("$2M", "2M"),
                    semantic_hints=("maximum budget", "cannot exceed"),
                    allowed_abstractions=("a constrained budget",),
                ),
            )
        )

    def test_counterfactual_replaces_protected_surface(self) -> None:
        policy = self.build_policy()

        result = build_counterfactual_text(
            "The budget ceiling is $2M for this negotiation.",
            policy,
        )

        self.assertIn("a constrained budget", result.public_text)
        self.assertNotIn("$2M", result.public_text)
        self.assertTrue(result.replacements)

    def test_semantic_verifier_catches_numeric_word_variant(self) -> None:
        verifier = SemanticLeakageVerifierV0(self.build_policy())

        result = verifier.verify("We cannot exceed 2 million dollars.")

        self.assertGreaterEqual(result.max_score, 0.85)
        self.assertTrue(
            any(finding.label == "numeric_word_reformulation" for finding in result.findings)
        )

    def test_guard_rewrites_exact_leak(self) -> None:
        policy = self.build_policy()
        guard = PrivacyGuardV0(
            exact_verifier=LeakageVerifier(policy),
            semantic_verifier=SemanticLeakageVerifierV0(policy),
            rewriter=TemplateAbstractionRewriter(policy),
        )

        decision = guard.protect("Our maximum budget is $2M.")

        self.assertFalse(decision.accepted)
        self.assertIn("a constrained budget", decision.final_text)
        self.assertNotIn("$2M", decision.final_text)

    def test_guard_falls_back_on_semantic_leak(self) -> None:
        policy = self.build_policy()
        guard = PrivacyGuardV0(
            exact_verifier=LeakageVerifier(policy),
            semantic_verifier=SemanticLeakageVerifierV0(policy),
            rewriter=TemplateAbstractionRewriter(policy),
        )

        decision = guard.protect("We cannot exceed 2 million dollars.")

        self.assertFalse(decision.accepted)
        self.assertIn("cannot disclose protected details", decision.final_text)


if __name__ == "__main__":
    unittest.main()

