from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from privacy_runtime import (
    CounterfactualBuilder,
    CounterfactualEstimationError,
    CounterfactualPrivacyCostEstimator,
    PrivacyAccountant,
    PrivacyPolicy,
    ProtectedFact,
    SequenceLikelihood,
)


PRIVATE_DOCUMENT = "Alice is 18 years old and works on Project Helios."
CANDIDATE = "She is newly adult."


def build_policy() -> PrivacyPolicy:
    return PrivacyPolicy(
        facts=(
            ProtectedFact(
                fact_id="age",
                fact="18 years old",
                surface_forms=("18",),
                allowed_abstractions=("an adult",),
                fact_type="age",
                counterfactual_values=("35 years old",),
                default_budget=3.0,
            ),
            ProtectedFact(
                fact_id="project",
                fact="Project Helios",
                allowed_abstractions=("an internal project",),
                fact_type="project_name",
                default_budget=1.0,
            ),
        )
    )


@dataclass
class RuleBasedScorer:
    different_candidate_tokens: bool = False

    def score(self, prompt: str, completion: str) -> SequenceLikelihood:
        if prompt == PRIVATE_DOCUMENT:
            values = (-0.1, -0.2, -0.3, -0.4)
        elif "[PRIVATE_AGE]" in prompt:
            values = (-1.1, -0.7, -0.3, -0.4)
        elif "an adult" in prompt:
            values = (-0.6, -0.5, -0.3, -0.4)
        elif "35 years old" in prompt:
            values = (-0.4, -0.3, -0.3, -0.4)
        elif "[PRIVATE_PROJECT_NAME]" in prompt:
            values = (-0.05, -0.05, -0.3, -0.4)
        elif "an internal project" in prompt:
            values = (-0.05, -0.05, -0.3, -0.4)
        else:
            raise AssertionError(f"unexpected prompt: {prompt}")

        token_ids = (1, 2, 3, 4)
        if self.different_candidate_tokens and "[PRIVATE_AGE]" in prompt:
            token_ids = (9, 2, 3, 4)
        return SequenceLikelihood(
            logprob=sum(values),
            token_ids=token_ids,
            tokens=("She", " is", " newly", " adult."),
            token_logprobs=values,
            prompt_token_count=10,
        )


class CounterfactualBuilderTests(unittest.TestCase):
    def test_builds_multiple_independent_interventions(self) -> None:
        result = CounterfactualBuilder().build(PRIVATE_DOCUMENT, build_policy())

        age = result.for_fact("age")
        self.assertEqual(
            [item.mode for item in age],
            ["placeholder", "abstraction", "substitution"],
        )
        self.assertTrue(all("Project Helios" in item.text for item in age))
        self.assertTrue(all("18" not in item.text for item in age))
        self.assertEqual(age[0].replacement, "[PRIVATE_AGE]")

    def test_marks_missing_target_invalid(self) -> None:
        result = CounterfactualBuilder().build(
            "No protected value appears here.",
            build_policy(),
        )

        self.assertFalse(result.valid_interventions)
        self.assertTrue(
            all("target_not_found" in item.validation_errors for item in result.interventions)
        )

    def test_detects_non_target_modification(self) -> None:
        policy = PrivacyPolicy(
            facts=(
                ProtectedFact("short", "Helios"),
                ProtectedFact("full", "Project Helios"),
            )
        )
        result = CounterfactualBuilder().build("Project Helios", policy)

        short = result.for_fact("short", valid_only=False)
        self.assertTrue(short)
        self.assertIn("non_target_modified:full", short[0].validation_errors)


class CounterfactualEstimatorTests(unittest.TestCase):
    def build_estimator(self, **kwargs: object) -> CounterfactualPrivacyCostEstimator:
        return CounterfactualPrivacyCostEstimator(
            scorer=RuleBasedScorer(),
            **kwargs,
        )

    def test_returns_per_fact_worst_case_cost_and_token_attribution(self) -> None:
        result = self.build_estimator().estimate_policy(
            PRIVATE_DOCUMENT,
            build_policy(),
            CANDIDATE,
        )

        age = result.for_fact("age")
        project = result.for_fact("project")
        self.assertAlmostEqual(age.signed_loss, 1.5)
        self.assertAlmostEqual(age.cost, 1.5)
        self.assertEqual(age.selected_intervention_id, "age:placeholder:1")
        self.assertEqual(len(age.interventions[0].token_losses), 4)
        self.assertAlmostEqual(age.interventions[0].token_losses[0].signed_loss, 1.0)
        self.assertAlmostEqual(project.cost, 0.0)
        self.assertEqual(dict(result.costs), {"age": age.cost, "project": 0.0})

    def test_mixture_aggregation_uses_sequence_probability_mixture(self) -> None:
        estimator = self.build_estimator(aggregation="mixture")
        result = estimator.estimate_policy(
            PRIVATE_DOCUMENT,
            build_policy(),
            CANDIDATE,
            fact_ids={"age"},
        )

        cf_logprobs = (-2.5, -1.8, -1.4)
        expected_mix = max(cf_logprobs) + math.log(
            sum(math.exp(value - max(cf_logprobs)) for value in cf_logprobs) / 3
        )
        self.assertAlmostEqual(result.for_fact("age").signed_loss, -1.0 - expected_mix)
        self.assertIsNone(result.for_fact("age").selected_intervention_id)

    def test_prompt_builder_keeps_fixed_runtime_context(self) -> None:
        seen: list[str] = []

        class RecordingScorer(RuleBasedScorer):
            def score(self, prompt: str, completion: str) -> SequenceLikelihood:
                self_prompt = prompt.removeprefix("TASK\n")
                seen.append(prompt)
                return super().score(self_prompt, completion)

        estimator = CounterfactualPrivacyCostEstimator(scorer=RecordingScorer())
        estimator.estimate_policy(
            PRIVATE_DOCUMENT,
            build_policy(),
            CANDIDATE,
            prompt_builder=lambda document: f"TASK\n{document}",
        )

        self.assertTrue(seen)
        self.assertTrue(all(prompt.startswith("TASK\n") for prompt in seen))

    def test_rejects_secret_reintroduced_by_serialized_policy(self) -> None:
        class ConstantScorer:
            def score(self, prompt: str, completion: str) -> SequenceLikelihood:
                return SequenceLikelihood(
                    logprob=-1.0,
                    token_ids=(1,),
                    tokens=("candidate",),
                    token_logprobs=(-1.0,),
                    prompt_token_count=10,
                )

        estimator = CounterfactualPrivacyCostEstimator(scorer=ConstantScorer())
        with self.assertRaisesRegex(
            CounterfactualEstimationError,
            "serialized policy text",
        ):
            estimator.estimate_policy(
                PRIVATE_DOCUMENT,
                build_policy(),
                CANDIDATE,
                prompt_builder=lambda document: f"{document}\nPolicy metadata: age=18",
                fact_ids={"age"},
            )

    def test_rejects_prompt_builder_that_ignores_document(self) -> None:
        class NeverCalledScorer:
            def score(self, prompt: str, completion: str) -> SequenceLikelihood:
                raise AssertionError("validation must happen before model scoring")

        estimator = CounterfactualPrivacyCostEstimator(scorer=NeverCalledScorer())
        with self.assertRaisesRegex(
            CounterfactualEstimationError,
            "ensure it uses the document",
        ):
            estimator.estimate_policy(
                PRIVATE_DOCUMENT,
                build_policy(),
                CANDIDATE,
                prompt_builder=lambda document: "A fixed prompt",
                fact_ids={"age"},
                allowed_prompt_occurrences={"age": 0},
            )

    def test_allows_explicitly_accounted_released_history_occurrence(self) -> None:
        class ConstantScorer:
            def score(self, prompt: str, completion: str) -> SequenceLikelihood:
                return SequenceLikelihood(
                    logprob=-1.0,
                    token_ids=(1,),
                    tokens=("candidate",),
                    token_logprobs=(-1.0,),
                    prompt_token_count=10,
                )

        estimator = CounterfactualPrivacyCostEstimator(scorer=ConstantScorer())
        result = estimator.estimate_policy(
            PRIVATE_DOCUMENT,
            build_policy(),
            CANDIDATE,
            prompt_builder=lambda document: f"{document}\nReleased history: age=18",
            fact_ids={"age"},
            allowed_prompt_occurrences={"age": 1},
        )

        self.assertEqual(result.for_fact("age").cost, 0.0)

    def test_policy_can_be_loaded_from_structured_mapping(self) -> None:
        policy = PrivacyPolicy.from_mapping(
            {
                "facts": [
                    {
                        "fact_id": "age",
                        "fact": "18",
                        "fact_type": "age",
                        "surface_forms": ["18 years old"],
                        "allowed_abstractions": ["an adult"],
                        "counterfactual_values": ["35"],
                        "default_budget": 2.0,
                    }
                ]
            }
        )

        self.assertEqual(policy.fact_by_id("age").counterfactual_values, ("35",))

    def test_rejects_mismatched_candidate_tokenization(self) -> None:
        estimator = CounterfactualPrivacyCostEstimator(
            scorer=RuleBasedScorer(different_candidate_tokens=True)
        )

        with self.assertRaises(CounterfactualEstimationError):
            estimator.estimate_policy(
                PRIVATE_DOCUMENT,
                build_policy(),
                CANDIDATE,
                fact_ids={"age"},
            )

    def test_costs_compose_with_existing_accountant(self) -> None:
        policy = build_policy()
        estimator = self.build_estimator()
        result = estimator.estimate_policy(PRIVATE_DOCUMENT, policy, CANDIDATE)
        accountant = PrivacyAccountant(policy)

        self.assertTrue(accountant.can_spend(result.costs, "final"))
        accountant.spend(result.costs, "final")
        self.assertAlmostEqual(accountant.spent_for("age", "final"), 1.5)
        self.assertAlmostEqual(accountant.remaining("age", "final"), 1.5)

    def test_strict_mode_rejects_missing_counterfactuals(self) -> None:
        estimator = self.build_estimator()
        with self.assertRaises(CounterfactualEstimationError):
            estimator.estimate_policy(
                "No protected value appears here.",
                build_policy(),
                CANDIDATE,
            )


if __name__ == "__main__":
    unittest.main()
