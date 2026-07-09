from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

from .counterfactual import (
    CounterfactualBuilder,
    CounterfactualIntervention,
)
from .likelihood import (
    HFCausalLMLikelihoodScorer,
    LikelihoodScorer,
    SequenceLikelihood,
)
from .policy import PrivacyPolicy


AggregationMode = Literal["max", "mixture", "mean"]
PromptBuilder = Callable[[str], str]


class CounterfactualEstimationError(ValueError):
    pass


@dataclass(frozen=True)
class PrivacyLossResult:
    """Backward-compatible single-counterfactual result."""

    private_logprob: float
    public_logprob: float
    privacy_loss: float
    token_count: int
    normalized: bool


@dataclass(frozen=True)
class TokenPrivacyLoss:
    position: int
    token_id: int
    token: str
    private_logprob: float
    counterfactual_logprob: float
    signed_loss: float


@dataclass(frozen=True)
class InterventionPrivacyLoss:
    fact_id: str
    intervention_id: str
    mode: str
    replacement: str
    private_logprob: float
    counterfactual_logprob: float
    signed_loss: float
    normalized_loss: float
    positive_token_mass: float
    token_losses: tuple[TokenPrivacyLoss, ...]


@dataclass(frozen=True)
class FactPrivacyLoss:
    fact_id: str
    aggregation: AggregationMode
    signed_loss: float
    cost: float
    normalized_loss: float
    normalized_cost: float
    selected_intervention_id: str | None
    interventions: tuple[InterventionPrivacyLoss, ...]


@dataclass(frozen=True)
class CounterfactualCostResult:
    candidate: str
    private_logprob: float
    token_count: int
    facts: tuple[FactPrivacyLoss, ...]
    invalid_interventions: tuple[CounterfactualIntervention, ...] = ()

    @property
    def costs(self) -> Mapping[str, float]:
        return {fact.fact_id: fact.cost for fact in self.facts}

    @property
    def normalized_costs(self) -> Mapping[str, float]:
        return {fact.fact_id: fact.normalized_cost for fact in self.facts}

    def for_fact(self, fact_id: str) -> FactPrivacyLoss:
        for fact in self.facts:
            if fact.fact_id == fact_id:
                return fact
        raise KeyError(f"no counterfactual result for fact: {fact_id}")


@dataclass
class CounterfactualPrivacyCostEstimator:
    """Per-fact, multiple-counterfactual privacy dependence estimator.

    The primary cost is the positive part of an unnormalized sequence
    log-likelihood ratio. ``normalized_cost`` is provided only as a diagnostic;
    sequential accounting should use ``costs`` so that losses compose in nats.
    """

    model: Any | None = None
    tokenizer: Any | None = None
    normalize_by_tokens: bool = False
    scorer: LikelihoodScorer | None = None
    builder: CounterfactualBuilder = field(default_factory=CounterfactualBuilder)
    aggregation: AggregationMode = "max"
    strict_counterfactuals: bool = True
    validate_counterfactual_prompts: bool = True

    def __post_init__(self) -> None:
        if self.aggregation not in {"max", "mixture", "mean"}:
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        if self.scorer is None:
            if self.model is None or self.tokenizer is None:
                raise ValueError("provide either scorer or both model and tokenizer")
            self.scorer = HFCausalLMLikelihoodScorer(self.model, self.tokenizer)

    def estimate(
        self,
        private_prompt: str,
        public_prompt: str,
        candidate: str,
    ) -> PrivacyLossResult:
        """Score one prompt pair for compatibility with the v0 guard."""

        private = self._score(private_prompt, candidate)
        public = self._score(public_prompt, candidate)
        _require_same_candidate_tokens(private, public)
        loss = private.logprob - public.logprob
        if self.normalize_by_tokens and private.token_count > 0:
            loss /= private.token_count
        return PrivacyLossResult(
            private_logprob=private.logprob,
            public_logprob=public.logprob,
            privacy_loss=loss,
            token_count=private.token_count,
            normalized=self.normalize_by_tokens,
        )

    def estimate_policy(
        self,
        private_document: str,
        policy: PrivacyPolicy,
        candidate: str,
        *,
        prompt_builder: PromptBuilder | None = None,
        fact_ids: set[str] | None = None,
        allowed_prompt_occurrences: Mapping[str, int] | None = None,
    ) -> CounterfactualCostResult:
        """Estimate independent privacy cost for every selected protected fact.

        ``prompt_builder`` must render a document into the exact Model A prompt
        while keeping policy text, task instruction, and released history fixed.
        The real protected value must not be serialized elsewhere in that prompt.
        """

        render = prompt_builder or _identity
        allowed_occurrences = allowed_prompt_occurrences or {}
        build_result = self.builder.build(private_document, policy)
        selected_fact_ids = (
            {fact.fact_id for fact in policy.facts}
            if fact_ids is None
            else set(fact_ids)
        )
        unknown = selected_fact_ids - {fact.fact_id for fact in policy.facts}
        if unknown:
            raise KeyError(f"unknown protected facts: {sorted(unknown)}")
        unknown_occurrence_facts = set(allowed_occurrences) - {
            fact.fact_id for fact in policy.facts
        }
        if unknown_occurrence_facts:
            raise KeyError(
                "unknown facts in allowed_prompt_occurrences: "
                f"{sorted(unknown_occurrence_facts)}"
            )

        interventions_by_fact: dict[str, tuple[CounterfactualIntervention, ...]] = {}
        for fact_id in selected_fact_ids:
            interventions = build_result.for_fact(fact_id, valid_only=True)
            if not interventions and self.strict_counterfactuals:
                invalid = build_result.for_fact(fact_id, valid_only=False)
                details = [item.validation_errors for item in invalid]
                raise CounterfactualEstimationError(
                    f"no valid counterfactual for fact {fact_id}: {details}"
                )
            interventions_by_fact[fact_id] = interventions

        private_prompt = render(private_document)
        rendered_by_fact: dict[
            str,
            tuple[tuple[CounterfactualIntervention, str], ...],
        ] = {}
        for fact in policy.facts:
            if fact.fact_id not in selected_fact_ids:
                continue
            rendered: list[tuple[CounterfactualIntervention, str]] = []
            for intervention in interventions_by_fact[fact.fact_id]:
                counterfactual_prompt = self._validated_prompt(
                    render(intervention.text),
                    policy=policy,
                    fact_id=fact.fact_id,
                    allowed_occurrences=allowed_occurrences.get(fact.fact_id, 0),
                )
                if counterfactual_prompt == private_prompt:
                    raise CounterfactualEstimationError(
                        "prompt_builder produced identical private and counterfactual "
                        f"prompts for fact {fact.fact_id}; ensure it uses the document"
                    )
                rendered.append((intervention, counterfactual_prompt))
            rendered_by_fact[fact.fact_id] = tuple(rendered)

        private = self._score(private_prompt, candidate)
        facts: list[FactPrivacyLoss] = []

        for fact in policy.facts:
            if fact.fact_id not in selected_fact_ids:
                continue
            rendered_interventions = rendered_by_fact[fact.fact_id]
            if not rendered_interventions:
                continue

            losses = tuple(
                self._score_intervention(
                    private=private,
                    intervention=intervention,
                    prompt=counterfactual_prompt,
                    candidate=candidate,
                )
                for intervention, counterfactual_prompt in rendered_interventions
            )
            facts.append(self._aggregate_fact(fact.fact_id, private, losses))

        return CounterfactualCostResult(
            candidate=candidate,
            private_logprob=private.logprob,
            token_count=private.token_count,
            facts=tuple(facts),
            invalid_interventions=build_result.invalid_interventions,
        )

    def _validated_prompt(
        self,
        prompt: str,
        *,
        policy: PrivacyPolicy,
        fact_id: str,
        allowed_occurrences: int,
    ) -> str:
        if allowed_occurrences < 0:
            raise ValueError("allowed prompt occurrences must be non-negative")
        if not self.validate_counterfactual_prompts:
            return prompt
        fact = policy.fact_by_id(fact_id)
        count = _count_non_overlapping_surfaces(prompt, fact.all_surface_forms())
        if count > allowed_occurrences:
            raise CounterfactualEstimationError(
                f"counterfactual prompt for {fact_id} retains {count} protected "
                f"occurrence(s); allowed={allowed_occurrences}. Keep raw values out "
                "of serialized policy text, or explicitly account for released history."
            )
        return prompt

    def _score_intervention(
        self,
        *,
        private: SequenceLikelihood,
        intervention: CounterfactualIntervention,
        prompt: str,
        candidate: str,
    ) -> InterventionPrivacyLoss:
        counterfactual = self._score(prompt, candidate)
        _require_same_candidate_tokens(private, counterfactual)
        token_losses = tuple(
            TokenPrivacyLoss(
                position=index,
                token_id=token_id,
                token=token,
                private_logprob=private_logprob,
                counterfactual_logprob=counterfactual_logprob,
                signed_loss=private_logprob - counterfactual_logprob,
            )
            for index, (
                token_id,
                token,
                private_logprob,
                counterfactual_logprob,
            ) in enumerate(
                zip(
                    private.token_ids,
                    private.tokens,
                    private.token_logprobs,
                    counterfactual.token_logprobs,
                )
            )
        )
        signed_loss = private.logprob - counterfactual.logprob
        normalized = signed_loss / private.token_count if private.token_count else 0.0
        return InterventionPrivacyLoss(
            fact_id=intervention.fact_id,
            intervention_id=intervention.intervention_id,
            mode=intervention.mode,
            replacement=intervention.replacement,
            private_logprob=private.logprob,
            counterfactual_logprob=counterfactual.logprob,
            signed_loss=signed_loss,
            normalized_loss=normalized,
            positive_token_mass=sum(max(0.0, item.signed_loss) for item in token_losses),
            token_losses=token_losses,
        )

    def _aggregate_fact(
        self,
        fact_id: str,
        private: SequenceLikelihood,
        losses: tuple[InterventionPrivacyLoss, ...],
    ) -> FactPrivacyLoss:
        selected_intervention_id: str | None = None
        if self.aggregation == "max":
            selected = max(losses, key=lambda item: item.signed_loss)
            signed_loss = selected.signed_loss
            selected_intervention_id = selected.intervention_id
        elif self.aggregation == "mean":
            signed_loss = sum(item.signed_loss for item in losses) / len(losses)
        else:
            mixture_logprob = _logmeanexp(
                tuple(item.counterfactual_logprob for item in losses)
            )
            signed_loss = private.logprob - mixture_logprob

        normalized = signed_loss / private.token_count if private.token_count else 0.0
        return FactPrivacyLoss(
            fact_id=fact_id,
            aggregation=self.aggregation,
            signed_loss=signed_loss,
            cost=max(0.0, signed_loss),
            normalized_loss=normalized,
            normalized_cost=max(0.0, normalized),
            selected_intervention_id=selected_intervention_id,
            interventions=losses,
        )

    def _score(self, prompt: str, candidate: str) -> SequenceLikelihood:
        assert self.scorer is not None
        return self.scorer.score(prompt, candidate)


def sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
) -> tuple[float, int]:
    """Backward-compatible wrapper around the HuggingFace scorer."""

    result = HFCausalLMLikelihoodScorer(model, tokenizer).score(prompt, completion)
    return result.logprob, result.token_count


def _require_same_candidate_tokens(
    private: SequenceLikelihood,
    counterfactual: SequenceLikelihood,
) -> None:
    if private.token_ids != counterfactual.token_ids:
        raise CounterfactualEstimationError(
            "private and counterfactual scorers used different candidate tokens"
        )


def _logmeanexp(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    maximum = max(values)
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values) / len(values)
    )


def _identity(value: str) -> str:
    return value


def _count_non_overlapping_surfaces(prompt: str, values: tuple[str, ...]) -> int:
    surfaces = sorted(
        {value for value in values if value},
        key=lambda value: (-len(value), value.casefold()),
    )
    if not surfaces:
        return 0
    pattern = re.compile(
        "|".join(re.escape(value) for value in surfaces),
        flags=re.IGNORECASE,
    )
    return sum(1 for _ in pattern.finditer(prompt))
