from __future__ import annotations

import re
from dataclasses import dataclass

from .counterfactual import safe_replacement_for_fact
from .policy import PrivacyPolicy
from .semantic_verifier import SemanticVerification


@dataclass(frozen=True)
class RewriteResult:
    text: str
    changed: bool
    actions: tuple[str, ...]


@dataclass(frozen=True)
class TemplateAbstractionRewriter:
    """Deterministic v0 rewriter for exact leaks and high-risk semantic leaks."""

    policy: PrivacyPolicy
    semantic_refusal_threshold: float = 0.75
    fallback_text: str = (
        "I cannot disclose protected details, but I can provide the allowed "
        "task-relevant information in a generalized form."
    )

    def rewrite(
        self,
        text: str,
        semantic_verification: SemanticVerification | None = None,
    ) -> RewriteResult:
        rewritten = text
        actions: list[str] = []

        for fact in self.policy.facts:
            replacement = safe_replacement_for_fact(fact)
            for surface in sorted(
                fact.all_surface_forms(),
                key=lambda value: (-len(value), value.casefold()),
            ):
                if not surface:
                    continue
                pattern = re.compile(re.escape(surface), flags=re.IGNORECASE)
                if pattern.search(rewritten):
                    rewritten = pattern.sub(replacement, rewritten)
                    actions.append(f"replaced {fact.fact_id}: {surface}")

        if semantic_verification is not None:
            high_risk = [
                finding
                for finding in semantic_verification.findings
                if finding.score >= self.semantic_refusal_threshold
                and finding.label not in {"exact_surface", "allowed_abstraction"}
            ]
            if high_risk and rewritten == text:
                rewritten = self.fallback_text
                actions.append("semantic_fallback")

        rewritten = _cleanup_rewrite(rewritten)
        return RewriteResult(
            text=rewritten,
            changed=rewritten != text,
            actions=tuple(actions),
        )


def _cleanup_rewrite(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".")
    return text

