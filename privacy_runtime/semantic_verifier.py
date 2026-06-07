from __future__ import annotations

import re
from dataclasses import dataclass

from .policy import PrivacyPolicy, ProtectedFact


@dataclass(frozen=True)
class SemanticFinding:
    fact_id: str
    label: str
    score: float
    evidence: str


@dataclass(frozen=True)
class SemanticVerification:
    max_score: float
    findings: tuple[SemanticFinding, ...]

    @property
    def is_leak(self) -> bool:
        return self.max_score >= 0.5


@dataclass(frozen=True)
class SemanticLeakageVerifierV0:
    """Heuristic semantic leakage verifier.

    This v0 module is intentionally transparent and deterministic. It catches
    common paraphrase-ish leaks such as numeric reformulations and high keyword
    overlap. Replace it with an NLI or attacker-posterior model for paper-grade
    experiments.
    """

    policy: PrivacyPolicy
    exact_score: float = 1.0
    numeric_score: float = 0.85
    hint_score: float = 0.65
    overlap_threshold: float = 0.67
    overlap_score: float = 0.6
    abstraction_score: float = 0.05

    def verify(self, text: str) -> SemanticVerification:
        findings: list[SemanticFinding] = []
        for fact in self.policy.facts:
            findings.extend(self._verify_fact(text, fact))

        findings = _dedupe_findings(findings)
        max_score = max((finding.score for finding in findings), default=0.0)
        return SemanticVerification(max_score=max_score, findings=tuple(findings))

    def _verify_fact(self, text: str, fact: ProtectedFact) -> list[SemanticFinding]:
        lowered = text.casefold()
        normalized_text = _normalize_text(text)
        text_digits = _digits_only(text)
        findings: list[SemanticFinding] = []

        for abstraction in fact.allowed_abstractions:
            if abstraction and abstraction.casefold() in lowered:
                findings.append(
                    SemanticFinding(
                        fact_id=fact.fact_id,
                        label="allowed_abstraction",
                        score=self.abstraction_score,
                        evidence=abstraction,
                    )
                )

        for surface in fact.all_surface_forms():
            if surface and surface.casefold() in lowered:
                findings.append(
                    SemanticFinding(
                        fact_id=fact.fact_id,
                        label="exact_surface",
                        score=self.exact_score,
                        evidence=surface,
                    )
                )

            surface_digits = _digits_only(surface)
            if len(surface_digits) >= 3 and surface_digits in text_digits:
                findings.append(
                    SemanticFinding(
                        fact_id=fact.fact_id,
                        label="numeric_reformulation",
                        score=self.numeric_score,
                        evidence=surface_digits,
                    )
                )

            for variant in _numeric_word_variants(surface):
                if variant in normalized_text:
                    findings.append(
                        SemanticFinding(
                            fact_id=fact.fact_id,
                            label="numeric_word_reformulation",
                            score=self.numeric_score,
                            evidence=variant,
                        )
                    )

        for hint in fact.semantic_hints:
            if hint and _normalize_text(hint) in normalized_text:
                findings.append(
                    SemanticFinding(
                        fact_id=fact.fact_id,
                        label="semantic_hint",
                        score=self.hint_score,
                        evidence=hint,
                    )
                )

        overlap = _content_overlap(fact.fact, text)
        if overlap >= self.overlap_threshold:
            findings.append(
                SemanticFinding(
                    fact_id=fact.fact_id,
                    label="high_content_overlap",
                    score=self.overlap_score,
                    evidence=f"{overlap:.2f}",
                )
            )

        return findings


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "to",
    "we",
    "with",
}


def _normalize_text(text: str) -> str:
    text = text.casefold()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[^a-z0-9$+.@-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _digits_only(text: str) -> str:
    return re.sub(r"\D+", "", text)


def _content_words(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", _normalize_text(text)))
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}


def _content_overlap(source: str, candidate: str) -> float:
    source_words = _content_words(source)
    if not source_words:
        return 0.0
    candidate_words = _content_words(candidate)
    return len(source_words & candidate_words) / len(source_words)


def _numeric_word_variants(text: str) -> tuple[str, ...]:
    variants: list[str] = []

    money_match = re.search(r"\$?\b([0-9]+(?:\.[0-9]+)?)\s*([mMkK])\b", text)
    if money_match:
        amount = money_match.group(1).rstrip("0").rstrip(".")
        suffix = money_match.group(2).casefold()
        if suffix == "m":
            variants.extend([f"{amount} million", f"{amount} million dollars"])
        elif suffix == "k":
            variants.extend([f"{amount} thousand", f"{amount} thousand dollars"])

    return tuple(_normalize_text(variant) for variant in variants)


def _dedupe_findings(findings: list[SemanticFinding]) -> list[SemanticFinding]:
    seen: set[tuple[str, str, str]] = set()
    out: list[SemanticFinding] = []
    for finding in findings:
        key = (finding.fact_id, finding.label, finding.evidence)
        if key in seen:
            continue
        seen.add(key)
        out.append(finding)
    return out
