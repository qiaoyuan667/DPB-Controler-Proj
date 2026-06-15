from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj"}
COPULA_SLOT_DEPS = {"attr", "acomp", "oprd", "ROOT"}
OBJECT_DEPS = {"obj", "dobj", "iobj", "attr", "oprd", "xcomp", "ccomp"}
PREPOSITIONAL_OBJECT_DEPS = {"pobj", "obl"}
APPOSITION_DEPS = {"appos"}
PREDICATE_POS = {"VERB", "AUX"}

FIELD_LINE_RE = re.compile(
    r"(?i)(?:^|\b)(?:"
    r"email|e-mail|phone|telephone|name|date of birth|dob|birth date|"
    r"hospital id|patient id|id|employer|company"
    r")\s*(?:[:=-]|\*\*:\s*|:\s*\*\*)\s*$"
)

_NLP_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class DependencyRewindDecision:
    char_index: int
    rewind_reason: str
    dependency_available: bool
    fallback_strategy: str | None = None
    dependency_role: str | None = None
    governing_head: str | None = None
    governing_head_pos: str | None = None


def choose_dependency_rewind_start(
    text: str,
    leak_start: int,
    leak_end: int,
    *,
    model_name: str = "en_core_web_sm",
    doc: Any | None = None,
) -> DependencyRewindDecision:
    """Choose a rewind point using lightweight line rules plus dependency roles."""

    line_decision = _field_line_decision(text, leak_start)
    if line_decision is not None:
        return line_decision

    if doc is None:
        nlp = _load_spacy_model(model_name)
        if nlp is None:
            return _fallback_decision(
                text,
                leak_start,
                dependency_available=False,
                reason="parser_unavailable",
            )
        try:
            doc = nlp(text)
        except Exception:
            return _fallback_decision(
                text,
                leak_start,
                dependency_available=False,
                reason="parser_failed",
            )

    return choose_dependency_rewind_start_from_doc(text, leak_start, leak_end, doc)


def choose_dependency_rewind_start_from_doc(
    text: str,
    leak_start: int,
    leak_end: int,
    doc: Any,
) -> DependencyRewindDecision:
    tokens = _overlapping_tokens(doc, leak_start, leak_end)
    if not tokens:
        return _fallback_decision(
            text,
            leak_start,
            dependency_available=True,
            reason="no_overlapping_dependency_token",
        )

    token = _primary_token(tokens)
    role = str(getattr(token, "dep_", "") or "")

    if role in APPOSITION_DEPS:
        return _decision_for_token(
            char_index=_subtree_start(token),
            reason="apposition",
            token=token,
        )

    if role in SUBJECT_DEPS:
        return _decision_for_token(
            char_index=_sentence_or_clause_start(text, token, leak_start),
            reason="subject_like_protected_value",
            token=token,
        )

    if role in COPULA_SLOT_DEPS and _has_copula(token):
        subject_start = _slot_subject_start(token)
        if subject_start is None:
            subject_start = _sentence_or_clause_start(text, token, leak_start)
        return _decision_for_token(
            char_index=subject_start,
            reason="copula_slot_value",
            token=token,
        )

    if role in PREPOSITIONAL_OBJECT_DEPS:
        predicate = _governing_predicate_for_prepositional_object(token)
        return _decision_for_token(
            char_index=int(getattr(predicate, "idx", 0)),
            reason="prepositional_object",
            token=token,
            governing=predicate,
        )

    if role in OBJECT_DEPS:
        predicate = _governing_predicate(token)
        return _decision_for_token(
            char_index=int(getattr(predicate, "idx", 0)),
            reason="object_or_complement",
            token=token,
            governing=predicate,
        )

    return _fallback_decision(
        text,
        leak_start,
        dependency_available=True,
        reason="unknown_dependency_role",
        role=role,
        governing=token,
    )


def _load_spacy_model(model_name: str) -> Any | None:
    if model_name in _NLP_CACHE:
        return _NLP_CACHE[model_name]
    try:
        import spacy

        nlp = spacy.load(model_name)
    except Exception:
        return None
    _NLP_CACHE[model_name] = nlp
    return nlp


def _field_line_decision(
    text: str,
    leak_start: int,
) -> DependencyRewindDecision | None:
    line_start = text.rfind("\n", 0, leak_start) + 1
    prefix = text[line_start:leak_start]
    if FIELD_LINE_RE.search(prefix):
        return DependencyRewindDecision(
            char_index=line_start,
            rewind_reason="list_field",
            dependency_available=True,
            fallback_strategy=None,
        )
    return None


def _overlapping_tokens(doc: Any, leak_start: int, leak_end: int) -> list[Any]:
    tokens = []
    for token in doc:
        start = int(getattr(token, "idx", 0))
        end = start + len(str(getattr(token, "text", "")))
        if start < leak_end and end > leak_start:
            tokens.append(token)
    return tokens


def _primary_token(tokens: list[Any]) -> Any:
    for token in tokens:
        dep = str(getattr(token, "dep_", "") or "")
        if dep in SUBJECT_DEPS | COPULA_SLOT_DEPS | OBJECT_DEPS | PREPOSITIONAL_OBJECT_DEPS | APPOSITION_DEPS:
            return token
    return tokens[0]


def _decision_for_token(
    *,
    char_index: int,
    reason: str,
    token: Any,
    governing: Any | None = None,
) -> DependencyRewindDecision:
    governing = governing or getattr(token, "head", token)
    return DependencyRewindDecision(
        char_index=max(0, int(char_index)),
        rewind_reason=reason,
        dependency_available=True,
        fallback_strategy=None,
        dependency_role=str(getattr(token, "dep_", "") or "") or None,
        governing_head=str(getattr(governing, "text", "") or "") or None,
        governing_head_pos=str(getattr(governing, "pos_", "") or "") or None,
    )


def _fallback_decision(
    text: str,
    leak_start: int,
    *,
    dependency_available: bool,
    reason: str,
    role: str | None = None,
    governing: Any | None = None,
) -> DependencyRewindDecision:
    clause_start = _clause_start(text, leak_start)
    fallback_strategy = "clause" if clause_start < leak_start else "value"
    return DependencyRewindDecision(
        char_index=clause_start if clause_start < leak_start else leak_start,
        rewind_reason=reason,
        dependency_available=dependency_available,
        fallback_strategy=fallback_strategy,
        dependency_role=role,
        governing_head=str(getattr(governing, "text", "") or "") or None,
        governing_head_pos=str(getattr(governing, "pos_", "") or "") or None,
    )


def _sentence_or_clause_start(text: str, token: Any, leak_start: int) -> int:
    sent = getattr(token, "sent", None)
    sent_start = int(getattr(sent, "start_char", 0) or 0)
    clause_start = _clause_start(text, leak_start)
    return max(sent_start, clause_start)


def _clause_start(text: str, leak_start: int) -> int:
    candidates = [0]
    for pattern in (
        r"[\n.;!?]\s*",
        r",\s+(?:and|but|or|while|their|his|her|the|a|an|it|this|that)\s+",
        r"\)\s+",
    ):
        for match in re.finditer(pattern, text[:leak_start], flags=re.IGNORECASE):
            candidates.append(match.end())
    return max(candidates)


def _subtree_start(token: Any) -> int:
    starts = [int(getattr(token, "idx", 0))]
    try:
        starts.extend(int(getattr(child, "idx", 0)) for child in token.subtree)
    except Exception:
        pass
    return min(starts)


def _children(token: Any) -> list[Any]:
    try:
        return list(token.children)
    except Exception:
        return []


def _has_copula(token: Any) -> bool:
    head = getattr(token, "head", token)
    family = [token, head, *_children(token), *_children(head)]
    return any(
        str(getattr(item, "dep_", "") or "") == "cop"
        or str(getattr(item, "lemma_", "") or "").casefold() == "be"
        or str(getattr(item, "text", "") or "").casefold() in {"is", "are", "was", "were", "be"}
        for item in family
    )


def _slot_subject_start(token: Any) -> int | None:
    head = getattr(token, "head", token)
    for child in _children(head):
        if str(getattr(child, "dep_", "") or "") in SUBJECT_DEPS:
            return _subtree_start(child)
    return None


def _governing_predicate(token: Any) -> Any:
    current = getattr(token, "head", token)
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if str(getattr(current, "pos_", "") or "") in PREDICATE_POS:
            return current
        if getattr(current, "head", current) is current:
            return current
        current = getattr(current, "head", current)
    return token


def _governing_predicate_for_prepositional_object(token: Any) -> Any:
    head = getattr(token, "head", token)
    if str(getattr(head, "pos_", "") or "") == "ADP":
        return _governing_predicate(head)
    return _governing_predicate(token)
