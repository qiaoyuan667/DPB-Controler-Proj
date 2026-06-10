from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .policy import PrivacyPolicy, ProtectedFact


ProtectedAttributeInput = str | Mapping[str, Any]


@dataclass(frozen=True)
class ProtectedAttributePolicyOptions:
    """Defaults for turning external protected attributes into a policy."""

    fact_id_prefix: str = "protected"
    default_budget: float = 0.0
    channel_budgets: Mapping[str, float] = field(default_factory=dict)


def privacy_policy_from_protected_attributes(
    protected_attributes: Sequence[ProtectedAttributeInput],
    *,
    options: ProtectedAttributePolicyOptions | None = None,
) -> PrivacyPolicy:
    """Build a privacy policy from runtime-provided protected attributes.

    The minimal supported input is a list of strings. A structured mapping may
    also provide value/fact, field/fact_id, surface_forms, semantic_hints, and
    allowed_abstractions for future POLAR adapters.
    """

    opts = options or ProtectedAttributePolicyOptions()
    facts: list[ProtectedFact] = []

    for index, attribute in enumerate(protected_attributes, start=1):
        fact = _protected_fact_from_attribute(attribute, index=index, options=opts)
        if fact is not None:
            facts.append(fact)

    return PrivacyPolicy(facts=tuple(facts))


def _protected_fact_from_attribute(
    attribute: ProtectedAttributeInput,
    *,
    index: int,
    options: ProtectedAttributePolicyOptions,
) -> ProtectedFact | None:
    if isinstance(attribute, str):
        value = attribute.strip()
        if not value:
            return None
        return ProtectedFact(
            fact_id=f"{options.fact_id_prefix}_{index}",
            fact=value,
            default_budget=options.default_budget,
            channel_budgets=options.channel_budgets,
        )

    value = str(attribute.get("value") or attribute.get("fact") or "").strip()
    if not value:
        return None

    raw_fact_id = str(
        attribute.get("fact_id") or attribute.get("field") or f"{options.fact_id_prefix}_{index}"
    ).strip()
    fact_id = raw_fact_id or f"{options.fact_id_prefix}_{index}"
    if not fact_id.startswith(f"{options.fact_id_prefix}_") and "fact_id" not in attribute:
        fact_id = f"{options.fact_id_prefix}_{fact_id}"

    return ProtectedFact(
        fact_id=fact_id,
        fact=value,
        surface_forms=_string_tuple(attribute.get("surface_forms")),
        semantic_hints=_string_tuple(attribute.get("semantic_hints")),
        allowed_abstractions=_string_tuple(attribute.get("allowed_abstractions")),
        default_budget=float(attribute.get("default_budget", options.default_budget)),
        channel_budgets=_float_mapping(
            attribute.get("channel_budgets"),
            default=options.channel_budgets,
        ),
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Sequence):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return tuple(out)
    text = str(value).strip()
    return (text,) if text else ()


def _float_mapping(
    value: Any,
    *,
    default: Mapping[str, float],
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        return default
    return {str(key): float(item) for key, item in value.items()}
