from __future__ import annotations

import re

from .policy import PrivacyPolicy, ProtectedFact


class HeuristicPolicyInducer:
    """Small placeholder for a future local SLM policy inducer.

    The research system should replace this with an on-device model that emits
    JSON. This heuristic version is useful for demos and unit tests.
    """

    _PATTERNS = (
        re.compile(r"\bdo not reveal\s+(?P<secret>[^.!\n]+)", re.IGNORECASE),
        re.compile(r"\bdon't reveal\s+(?P<secret>[^.!\n]+)", re.IGNORECASE),
        re.compile(r"\bnever disclose\s+(?P<secret>[^.!\n]+)", re.IGNORECASE),
        re.compile(r"\bsecret\s*:\s*(?P<secret>[^.!\n]+)", re.IGNORECASE),
        re.compile(r"\bprotected\s*:\s*(?P<secret>[^.!\n]+)", re.IGNORECASE),
    )

    def induce(self, text: str, *, default_budget: float = 0.25) -> PrivacyPolicy:
        facts: list[ProtectedFact] = []
        for index, secret in enumerate(self._extract_secrets(text), start=1):
            facts.append(
                ProtectedFact(
                    fact_id=f"s{index}",
                    fact=secret,
                    surface_forms=self._surface_variants(secret),
                    semantic_hints=self._semantic_hints(secret),
                    allowed_abstractions=self._allowed_abstractions(secret),
                    default_budget=default_budget,
                )
            )
        return PrivacyPolicy(facts=tuple(facts))

    def _extract_secrets(self, text: str) -> tuple[str, ...]:
        secrets: list[str] = []
        seen: set[str] = set()
        for pattern in self._PATTERNS:
            for match in pattern.finditer(text):
                secret = match.group("secret").strip(" .;:")
                key = secret.casefold()
                if secret and key not in seen:
                    seen.add(key)
                    secrets.append(secret)
        return tuple(secrets)

    def _surface_variants(self, secret: str) -> tuple[str, ...]:
        variants: list[str] = []
        money_match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)\s*([mMkK])\b", secret)
        if money_match:
            amount = money_match.group(1)
            suffix = money_match.group(2).casefold()
            if suffix == "m":
                variants.extend(
                    [
                        f"{amount}M",
                        f"{amount} million",
                        f"{amount} million dollars",
                    ]
                )
            elif suffix == "k":
                variants.extend(
                    [
                        f"{amount}K",
                        f"{amount} thousand",
                        f"{amount} thousand dollars",
                    ]
                )
        return tuple(variants)

    def _semantic_hints(self, secret: str) -> tuple[str, ...]:
        lowered = secret.casefold()
        hints: list[str] = []
        if "budget" in lowered:
            hints.extend(["budget ceiling", "maximum budget", "cannot exceed"])
        if "diagnos" in lowered or "disorder" in lowered:
            hints.extend(["diagnosis", "medical condition", "treatment"])
        return tuple(hints)

    def _allowed_abstractions(self, secret: str) -> tuple[str, ...]:
        lowered = secret.casefold()
        if "budget" in lowered:
            return ("constrained budget", "limited budget")
        if "diagnos" in lowered or "disorder" in lowered:
            return ("sensitive health context",)
        return ()

