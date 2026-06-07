from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Mapping

from .tokenizer import TokenVocabulary


DEFAULT_PRIVATE_REGEXES: tuple[str, ...] = (
    r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(?:\d[ -]?){13,19}\b",
    r"\bsk-[A-Za-z0-9_-]{16,}\b",
)


@dataclass(frozen=True)
class MaskDecision:
    blocked_token_ids: frozenset[int]
    reasons: Mapping[int, tuple[str, ...]]


class TokenConstraint:
    def blocked_tokens(
        self, generated_text: str, vocabulary: TokenVocabulary
    ) -> MaskDecision:
        raise NotImplementedError


@dataclass(frozen=True)
class _DfaNode:
    transitions: Mapping[str, int]
    fail: int
    outputs: tuple[str, ...]


class _ForbiddenStringDFA:
    """Aho-Corasick automaton over normalized text characters."""

    def __init__(self, targets: tuple[str, ...]) -> None:
        mutable_nodes: list[dict[str, object]] = [
            {"transitions": {}, "fail": 0, "outputs": []}
        ]
        for target in targets:
            state = 0
            for char in target:
                transitions = mutable_nodes[state]["transitions"]
                assert isinstance(transitions, dict)
                if char not in transitions:
                    transitions[char] = len(mutable_nodes)
                    mutable_nodes.append(
                        {"transitions": {}, "fail": 0, "outputs": []}
                    )
                state = int(transitions[char])
            outputs = mutable_nodes[state]["outputs"]
            assert isinstance(outputs, list)
            outputs.append(target)

        queue: list[int] = []
        root_transitions = mutable_nodes[0]["transitions"]
        assert isinstance(root_transitions, dict)
        for next_state in root_transitions.values():
            queue.append(int(next_state))

        while queue:
            current = queue.pop(0)
            transitions = mutable_nodes[current]["transitions"]
            assert isinstance(transitions, dict)
            for char, next_state_object in transitions.items():
                next_state = int(next_state_object)
                queue.append(next_state)

                fail_state = int(mutable_nodes[current]["fail"])
                while fail_state and char not in mutable_nodes[fail_state]["transitions"]:
                    fail_state = int(mutable_nodes[fail_state]["fail"])

                fail_transitions = mutable_nodes[fail_state]["transitions"]
                assert isinstance(fail_transitions, dict)
                mutable_nodes[next_state]["fail"] = int(fail_transitions.get(char, 0))

                outputs = mutable_nodes[next_state]["outputs"]
                fail_outputs = mutable_nodes[int(mutable_nodes[next_state]["fail"])]["outputs"]
                assert isinstance(outputs, list)
                assert isinstance(fail_outputs, list)
                outputs.extend(fail_outputs)

        self.nodes = tuple(
            _DfaNode(
                transitions=dict(node["transitions"]),
                fail=int(node["fail"]),
                outputs=tuple(node["outputs"]),
            )
            for node in mutable_nodes
        )

    def step(self, state: int, char: str) -> int:
        while state and char not in self.nodes[state].transitions:
            state = self.nodes[state].fail
        return self.nodes[state].transitions.get(char, 0)

    def state_after(self, text: str) -> int:
        state = 0
        for char in text:
            state = self.step(state, char)
        return state

    def matches_while_consuming(
        self, initial_state: int, text: str
    ) -> tuple[tuple[str, int], ...]:
        state = initial_state
        matches: list[tuple[str, int]] = []
        for index, char in enumerate(text):
            state = self.step(state, char)
            matches.extend((output, index) for output in self.nodes[state].outputs)
        return tuple(matches)


@dataclass(frozen=True)
class ForbiddenStringConstraint(TokenConstraint):
    forbidden_strings: tuple[str, ...]
    case_sensitive: bool = False
    max_state_chars: int | None = None
    require_token_boundary: bool = True
    _raw_targets_cache: tuple[str, ...] = field(init=False, repr=False)
    _normalized_targets_cache: tuple[str, ...] = field(init=False, repr=False)
    _target_lookup: Mapping[str, str] = field(init=False, repr=False)
    _dfa: _ForbiddenStringDFA | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw_targets = self._build_raw_targets()
        normalized_targets = tuple(self._normalize(target) for target in raw_targets)
        object.__setattr__(self, "_raw_targets_cache", raw_targets)
        object.__setattr__(self, "_normalized_targets_cache", normalized_targets)
        object.__setattr__(
            self,
            "_target_lookup",
            dict(zip(normalized_targets, raw_targets, strict=True)),
        )
        object.__setattr__(
            self,
            "_dfa",
            _ForbiddenStringDFA(normalized_targets) if normalized_targets else None,
        )

    def blocked_tokens(
        self, generated_text: str, vocabulary: TokenVocabulary
    ) -> MaskDecision:
        targets = self._normalized_targets_cache
        if not targets or self._dfa is None:
            return MaskDecision(frozenset(), {})

        max_target_len = max(len(target) for target in targets)
        state_chars = self.max_state_chars or max(1, max_target_len - 1)
        tail = self._normalize(generated_text)[-state_chars:]
        base_state = self._dfa.state_after(tail)
        blocked: set[int] = set()
        reasons: dict[int, tuple[str, ...]] = {}

        for token_id in vocabulary.iter_token_ids():
            token = self._normalize(vocabulary.token_text(token_id))
            if not token:
                continue
            matches = self._boundary_filtered_matches(
                token,
                self._dfa.matches_while_consuming(base_state, token),
            )
            if matches:
                blocked.add(token_id)
                raw_matches = tuple(self._target_lookup[match] for match in matches)
                reasons[token_id] = tuple(
                    f"would complete forbidden string '{match}'"
                    for match in raw_matches
                )

        return MaskDecision(frozenset(blocked), reasons)

    def _build_raw_targets(self) -> tuple[str, ...]:
        targets: list[str] = []
        seen: set[str] = set()
        for target in self.forbidden_strings:
            stripped = target.strip()
            key = self._normalize(stripped)
            if key and key not in seen:
                seen.add(key)
                targets.append(stripped)
        return tuple(targets)

    def _normalize(self, text: str) -> str:
        return text if self.case_sensitive else text.casefold()

    def _boundary_filtered_matches(
        self, token: str, matches: tuple[tuple[str, int], ...]
    ) -> tuple[str, ...]:
        if not self.require_token_boundary:
            return tuple(match for match, _ in matches)

        filtered: list[str] = []
        for match, end_index in matches:
            if not match[-1].isalnum():
                filtered.append(match)
                continue
            next_index = end_index + 1
            if next_index >= len(token) or not token[next_index].isalnum():
                filtered.append(match)
        return tuple(filtered)


@dataclass(frozen=True)
class ForbiddenRegexConstraint(TokenConstraint):
    patterns: tuple[str, ...]
    flags: int = re.IGNORECASE

    def blocked_tokens(
        self, generated_text: str, vocabulary: TokenVocabulary
    ) -> MaskDecision:
        compiled = tuple(re.compile(pattern, self.flags) for pattern in self.patterns)
        if not compiled:
            return MaskDecision(frozenset(), {})

        tail = generated_text[-256:]
        blocked: set[int] = set()
        reasons: dict[int, tuple[str, ...]] = {}

        for token_id in vocabulary.iter_token_ids():
            token = vocabulary.token_text(token_id)
            probe = tail + token
            matches = [pattern.pattern for pattern in compiled if pattern.search(probe)]
            if matches:
                blocked.add(token_id)
                reasons[token_id] = tuple(f"matches regex {pattern}" for pattern in matches)

        return MaskDecision(frozenset(blocked), reasons)


@dataclass(frozen=True)
class PrivacyLogitProcessor:
    """Applies hard token masks by setting blocked logits to -inf."""

    vocabulary: TokenVocabulary
    constraints: tuple[TokenConstraint, ...]

    def blocked_tokens(self, generated_text: str) -> MaskDecision:
        blocked: set[int] = set()
        reasons: dict[int, list[str]] = {}

        for constraint in self.constraints:
            decision = constraint.blocked_tokens(generated_text, self.vocabulary)
            blocked.update(decision.blocked_token_ids)
            for token_id, token_reasons in decision.reasons.items():
                reasons.setdefault(token_id, []).extend(token_reasons)

        return MaskDecision(
            blocked_token_ids=frozenset(blocked),
            reasons={key: tuple(value) for key, value in reasons.items()},
        )

    def apply(self, generated_text: str, logits: Mapping[int, float]) -> dict[int, float]:
        decision = self.blocked_tokens(generated_text)
        masked = dict(logits)
        for token_id in decision.blocked_token_ids:
            masked[token_id] = -math.inf
        return masked
