from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from .policy import PrivacyPolicy, ProtectedFact


@dataclass(frozen=True)
class Replacement:
    fact_id: str
    original: str
    replacement: str


@dataclass(frozen=True)
class CounterfactualText:
    private_text: str
    public_text: str
    replacements: tuple[Replacement, ...]


CounterfactualMode = Literal["placeholder", "abstraction", "substitution", "removal"]


@dataclass(frozen=True)
class CounterfactualIntervention:
    """One auditable intervention on exactly one protected fact."""

    fact_id: str
    intervention_id: str
    mode: CounterfactualMode
    replacement: str
    text: str
    replacements: tuple[Replacement, ...]
    valid: bool
    validation_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CounterfactualBuildResult:
    private_text: str
    interventions: tuple[CounterfactualIntervention, ...]

    def for_fact(
        self,
        fact_id: str,
        *,
        valid_only: bool = True,
    ) -> tuple[CounterfactualIntervention, ...]:
        return tuple(
            intervention
            for intervention in self.interventions
            if intervention.fact_id == fact_id
            and (intervention.valid or not valid_only)
        )

    @property
    def valid_interventions(self) -> tuple[CounterfactualIntervention, ...]:
        return tuple(item for item in self.interventions if item.valid)

    @property
    def invalid_interventions(self) -> tuple[CounterfactualIntervention, ...]:
        return tuple(item for item in self.interventions if not item.valid)


@dataclass(frozen=True)
class CounterfactualBuilder:
    """Build per-fact interventions while preserving all non-target facts."""

    placeholder_prefix: str = "PRIVATE"
    include_placeholder: bool = True
    include_abstractions: bool = True
    include_substitutions: bool = True
    include_removal: bool = False
    require_target_presence: bool = True
    preserve_non_target_facts: bool = True
    max_interventions_per_fact: int | None = None

    def __post_init__(self) -> None:
        if (
            self.max_interventions_per_fact is not None
            and self.max_interventions_per_fact <= 0
        ):
            raise ValueError("max_interventions_per_fact must be positive")

    def build(self, text: str, policy: PrivacyPolicy) -> CounterfactualBuildResult:
        interventions: list[CounterfactualIntervention] = []
        for fact in policy.facts:
            replacements = self._replacement_specs(fact)
            if self.max_interventions_per_fact is not None:
                replacements = replacements[: self.max_interventions_per_fact]

            for index, (mode, replacement) in enumerate(replacements, start=1):
                rendered, applied = _replace_fact_surfaces(text, fact, replacement)
                errors = self._validate(
                    private_text=text,
                    counterfactual_text=rendered,
                    target=fact,
                    policy=policy,
                    replacement=replacement,
                    replacements=applied,
                )
                interventions.append(
                    CounterfactualIntervention(
                        fact_id=fact.fact_id,
                        intervention_id=f"{fact.fact_id}:{mode}:{index}",
                        mode=mode,
                        replacement=replacement,
                        text=rendered,
                        replacements=applied,
                        valid=not errors,
                        validation_errors=tuple(errors),
                    )
                )

        return CounterfactualBuildResult(
            private_text=text,
            interventions=tuple(interventions),
        )

    def _replacement_specs(
        self,
        fact: ProtectedFact,
    ) -> list[tuple[CounterfactualMode, str]]:
        specs: list[tuple[CounterfactualMode, str]] = []
        if self.include_placeholder:
            specs.append(
                (
                    "placeholder",
                    typed_placeholder_for_fact(
                        fact,
                        placeholder_prefix=self.placeholder_prefix,
                    ),
                )
            )
        if self.include_abstractions:
            specs.extend(("abstraction", value) for value in fact.allowed_abstractions)
        if self.include_substitutions:
            specs.extend(("substitution", value) for value in fact.counterfactual_values)
        if self.include_removal:
            specs.append(("removal", ""))

        seen: set[str] = set()
        unique: list[tuple[CounterfactualMode, str]] = []
        for mode, replacement in specs:
            key = f"{mode}:{replacement.strip().casefold()}"
            if key not in seen:
                seen.add(key)
                unique.append((mode, replacement))
        return unique

    def _validate(
        self,
        *,
        private_text: str,
        counterfactual_text: str,
        target: ProtectedFact,
        policy: PrivacyPolicy,
        replacement: str,
        replacements: tuple[Replacement, ...],
    ) -> list[str]:
        errors: list[str] = []
        if self.require_target_presence and not replacements:
            errors.append("target_not_found")
        if replacement and _contains_any(replacement, target.all_surface_forms()):
            errors.append("replacement_contains_target")
        if _contains_any(counterfactual_text, target.all_surface_forms()):
            errors.append("target_residual")

        if self.preserve_non_target_facts:
            for fact in policy.facts:
                if fact.fact_id == target.fact_id:
                    continue
                before = _surface_counts(private_text, fact.all_surface_forms())
                after = _surface_counts(counterfactual_text, fact.all_surface_forms())
                if before != after:
                    errors.append(f"non_target_modified:{fact.fact_id}")
        return errors


def build_counterfactual_text(
    text: str,
    policy: PrivacyPolicy,
    *,
    placeholder_prefix: str = "PRIVATE",
) -> CounterfactualText:
    """Replace protected surface forms with safe abstractions or placeholders."""

    public_text = text
    replacements: list[Replacement] = []

    for fact in policy.facts:
        replacement = safe_replacement_for_fact(
            fact,
            placeholder_prefix=placeholder_prefix,
        )
        public_text, applied = _replace_fact_surfaces(public_text, fact, replacement)
        replacements.extend(applied)

    return CounterfactualText(
        private_text=text,
        public_text=public_text,
        replacements=tuple(replacements),
    )


def safe_replacement_for_fact(
    fact: ProtectedFact,
    *,
    placeholder_prefix: str = "PRIVATE",
) -> str:
    if fact.allowed_abstractions:
        return fact.allowed_abstractions[0]

    label = _label_from_fact_id(fact.fact_id)
    return f"[{placeholder_prefix}_{label}]"


def typed_placeholder_for_fact(
    fact: ProtectedFact,
    *,
    placeholder_prefix: str = "PRIVATE",
) -> str:
    label_source = fact.fact_type if fact.fact_type != "unspecified" else fact.fact_id
    label = _label_from_fact_id(label_source)
    return f"[{placeholder_prefix}_{label}]"


def _longest_first(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (-len(value), value.casefold()))


def _label_from_fact_id(fact_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", fact_id).strip("_").upper()
    return cleaned or "FACT"


def _replace_fact_surfaces(
    text: str,
    fact: ProtectedFact,
    replacement: str,
) -> tuple[str, tuple[Replacement, ...]]:
    surfaces = [value for value in _longest_first(fact.all_surface_forms()) if value]
    if not surfaces:
        return text, ()

    pattern = re.compile(
        "|".join(re.escape(surface) for surface in surfaces),
        flags=re.IGNORECASE,
    )
    applied: list[Replacement] = []

    def replace(match: re.Match[str]) -> str:
        applied.append(
            Replacement(
                fact_id=fact.fact_id,
                original=match.group(0),
                replacement=replacement,
            )
        )
        return replacement

    return pattern.sub(replace, text), tuple(applied)


def _contains_any(text: str, values: Iterable[str]) -> bool:
    lowered = text.casefold()
    return any(value and value.casefold() in lowered for value in values)


def _surface_counts(text: str, values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    lowered = text.casefold()
    return tuple(
        (value.casefold(), lowered.count(value.casefold()))
        for value in _longest_first(values)
        if value
    )
