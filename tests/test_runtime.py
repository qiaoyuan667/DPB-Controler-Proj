from __future__ import annotations

import unittest

from privacy_runtime import (
    CandidateAction,
    ForbiddenStringConstraint,
    LeakageVerifier,
    PrivacyAccountant,
    PrivacyLogitProcessor,
    PrivacyPolicy,
    PrivacyRuntime,
    ProtectedFact,
    SimpleVocabulary,
)


class PrivacyRuntimeTests(unittest.TestCase):
    def build_policy(self) -> PrivacyPolicy:
        return PrivacyPolicy(
            facts=(
                ProtectedFact(
                    fact_id="s1",
                    fact="budget ceiling is $2M",
                    surface_forms=("$2M", "2M", "two million"),
                    semantic_hints=("maximum budget", "budget ceiling"),
                    allowed_abstractions=("constrained budget",),
                    default_budget=0.10,
                ),
            )
        )

    def test_forbidden_string_constraint_blocks_completion(self) -> None:
        policy = self.build_policy()
        vocab = SimpleVocabulary.from_tokens(["$2M", "constrained", "."])
        processor = PrivacyLogitProcessor(
            vocabulary=vocab,
            constraints=(ForbiddenStringConstraint(policy.all_forbidden_strings()),),
        )

        decision = processor.blocked_tokens("Our maximum budget is ")

        self.assertIn(vocab.token_id("$2M"), decision.blocked_token_ids)
        self.assertNotIn(vocab.token_id("constrained"), decision.blocked_token_ids)

    def test_forbidden_string_constraint_blocks_split_completion(self) -> None:
        policy = self.build_policy()
        vocab = SimpleVocabulary.from_tokens([" million", " budget", "."])
        processor = PrivacyLogitProcessor(
            vocabulary=vocab,
            constraints=(ForbiddenStringConstraint(policy.all_forbidden_strings()),),
        )

        decision = processor.blocked_tokens("We cannot say two")

        self.assertIn(vocab.token_id(" million"), decision.blocked_token_ids)
        self.assertNotIn(vocab.token_id(" budget"), decision.blocked_token_ids)

    def test_forbidden_string_constraint_blocks_bpe_style_suffix(self) -> None:
        vocab = SimpleVocabulary.from_tokens(["ios", " plan", "."])
        processor = PrivacyLogitProcessor(
            vocabulary=vocab,
            constraints=(ForbiddenStringConstraint(("Project Helios",)),),
        )

        decision = processor.blocked_tokens("External message: Project Hel")

        self.assertIn(vocab.token_id("ios"), decision.blocked_token_ids)
        self.assertNotIn(vocab.token_id(" plan"), decision.blocked_token_ids)

    def test_forbidden_string_constraint_does_not_block_partial_prefix(self) -> None:
        vocab = SimpleVocabulary.from_tokens(["alice", "alice@example.com", " other"])
        processor = PrivacyLogitProcessor(
            vocabulary=vocab,
            constraints=(ForbiddenStringConstraint(("alice@example.com",)),),
        )

        decision = processor.blocked_tokens("")

        self.assertNotIn(vocab.token_id("alice"), decision.blocked_token_ids)
        self.assertIn(vocab.token_id("alice@example.com"), decision.blocked_token_ids)

    def test_forbidden_string_constraint_uses_token_boundary(self) -> None:
        vocab = SimpleVocabulary.from_tokens(["M", "ment", "M."])
        processor = PrivacyLogitProcessor(
            vocabulary=vocab,
            constraints=(ForbiddenStringConstraint(("$2M",)),),
        )

        decision = processor.blocked_tokens("budget is $2")

        self.assertIn(vocab.token_id("M"), decision.blocked_token_ids)
        self.assertIn(vocab.token_id("M."), decision.blocked_token_ids)
        self.assertNotIn(vocab.token_id("ment"), decision.blocked_token_ids)

    def test_runtime_prefers_highest_utility_allowed_candidate(self) -> None:
        policy = self.build_policy()
        runtime = PrivacyRuntime(
            verifier=LeakageVerifier(policy),
            accountant=PrivacyAccountant(policy),
        )

        decision = runtime.choose(
            [
                CandidateAction("Our maximum budget is $2M.", utility=0.99),
                CandidateAction("We have a constrained budget.", utility=0.80),
                CandidateAction("I cannot disclose that.", utility=0.50),
            ]
        )

        self.assertIsNotNone(decision.selected)
        self.assertEqual(decision.selected.text, "We have a constrained budget.")
        self.assertEqual(len(decision.rejected), 1)

    def test_budget_blocks_repeated_semantic_leakage(self) -> None:
        policy = self.build_policy()
        runtime = PrivacyRuntime(
            verifier=LeakageVerifier(policy, semantic_cost=0.08),
            accountant=PrivacyAccountant(policy),
        )

        first = runtime.choose(
            [CandidateAction("We will not discuss the maximum budget.", utility=1.0)]
        )
        second = runtime.choose(
            [CandidateAction("The budget ceiling is internal.", utility=1.0)]
        )

        self.assertIsNotNone(first.selected)
        self.assertIsNone(second.selected)
        self.assertEqual(second.rejected[0].reason, "privacy budget exceeded")


if __name__ == "__main__":
    unittest.main()
