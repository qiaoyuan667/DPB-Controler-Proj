from __future__ import annotations

import unittest
from unittest.mock import patch

from privacy_runtime.dependency_rewind import (
    choose_dependency_rewind_start,
    choose_dependency_rewind_start_from_doc,
)


class FakeSent:
    def __init__(self, start_char: int = 0) -> None:
        self.start_char = start_char


class FakeToken:
    def __init__(
        self,
        text: str,
        idx: int,
        dep: str,
        pos: str,
        *,
        sent: FakeSent | None = None,
    ) -> None:
        self.text = text
        self.idx = idx
        self.dep_ = dep
        self.pos_ = pos
        self.head: FakeToken = self
        self._children: list[FakeToken] = []
        self.sent = sent or FakeSent(0)

    @property
    def children(self) -> tuple["FakeToken", ...]:
        return tuple(self._children)

    @property
    def subtree(self) -> tuple["FakeToken", ...]:
        out = [self]
        for child in self._children:
            out.extend(child.subtree)
        return tuple(out)

    def set_head(self, head: "FakeToken") -> "FakeToken":
        self.head = head
        head._children.append(self)
        return self


class FakeDoc(list[FakeToken]):
    pass


class DependencyRewindTests(unittest.TestCase):
    def test_markdown_field_rewinds_to_line_start(self) -> None:
        text = "Summary\nEmail: noah@example.com"
        leak_start = text.index("noah")

        decision = choose_dependency_rewind_start(
            text,
            leak_start,
            leak_start + len("noah@example.com"),
            doc=FakeDoc(),
        )

        self.assertEqual(decision.rewind_reason, "list_field")
        self.assertEqual(decision.char_index, text.index("Email:"))

    def test_subject_like_value_rewinds_to_sentence_start(self) -> None:
        text = "Noah Baumann is the patient."
        sent = FakeSent(0)
        is_token = FakeToken("is", text.index("is"), "ROOT", "AUX", sent=sent)
        noah = FakeToken("Noah", 0, "nsubj", "PROPN", sent=sent).set_head(is_token)
        baumann = FakeToken("Baumann", 5, "flat", "PROPN", sent=sent).set_head(noah)
        doc = FakeDoc([noah, baumann, is_token])

        decision = choose_dependency_rewind_start_from_doc(
            text,
            0,
            len("Noah Baumann"),
            doc,
        )

        self.assertEqual(decision.rewind_reason, "subject_like_protected_value")
        self.assertEqual(decision.char_index, 0)

    def test_copula_slot_value_rewinds_to_slot_subject_start(self) -> None:
        text = "His email is 121@gmail.com."
        sent = FakeSent(0)
        is_token = FakeToken("is", text.index("is"), "ROOT", "AUX", sent=sent)
        email = FakeToken("email", text.index("email"), "nsubj", "NOUN", sent=sent)
        email.set_head(is_token)
        FakeToken("His", 0, "poss", "PRON", sent=sent).set_head(email)
        value = FakeToken(
            "121@gmail.com",
            text.index("121"),
            "attr",
            "NOUN",
            sent=sent,
        ).set_head(is_token)
        doc = FakeDoc([email, is_token, value])

        decision = choose_dependency_rewind_start_from_doc(
            text,
            text.index("121"),
            text.index("121") + len("121@gmail.com"),
            doc,
        )

        self.assertEqual(decision.rewind_reason, "copula_slot_value")
        self.assertEqual(decision.char_index, 0)
        self.assertEqual(decision.governing_head, "is")

    def test_object_value_rewinds_to_governing_predicate_start(self) -> None:
        text = "The note includes H-133326."
        includes = FakeToken("includes", text.index("includes"), "ROOT", "VERB")
        value = FakeToken("H-133326", text.index("H-"), "obj", "NOUN").set_head(
            includes
        )
        doc = FakeDoc([includes, value])

        decision = choose_dependency_rewind_start_from_doc(
            text,
            text.index("H-"),
            text.index("H-") + len("H-133326"),
            doc,
        )

        self.assertEqual(decision.rewind_reason, "object_or_complement")
        self.assertEqual(decision.char_index, text.index("includes"))

    def test_prepositional_object_rewinds_to_governing_predicate_start(self) -> None:
        text = "Contact him at +41."
        contact = FakeToken("Contact", 0, "ROOT", "VERB")
        at = FakeToken("at", text.index("at"), "prep", "ADP").set_head(contact)
        phone = FakeToken("+41", text.index("+41"), "pobj", "NUM").set_head(at)
        doc = FakeDoc([contact, at, phone])

        decision = choose_dependency_rewind_start_from_doc(
            text,
            text.index("+41"),
            text.index("+41") + len("+41"),
            doc,
        )

        self.assertEqual(decision.rewind_reason, "prepositional_object")
        self.assertEqual(decision.char_index, 0)

    def test_apposition_rewinds_to_appositive_span_start(self) -> None:
        text = "The patient, Noah Baumann, reports pain."
        patient = FakeToken("patient", text.index("patient"), "nsubj", "NOUN")
        noah = FakeToken("Noah", text.index("Noah"), "appos", "PROPN").set_head(
            patient
        )
        FakeToken("Baumann", text.index("Baumann"), "flat", "PROPN").set_head(noah)
        doc = FakeDoc([patient, noah])

        decision = choose_dependency_rewind_start_from_doc(
            text,
            text.index("Noah"),
            text.index("Noah") + len("Noah Baumann"),
            doc,
        )

        self.assertEqual(decision.rewind_reason, "apposition")
        self.assertEqual(decision.char_index, text.index("Noah"))

    def test_parser_unavailable_falls_back_to_clause_or_value(self) -> None:
        text = "The patient is stable, and email is noah@example.com"
        leak_start = text.index("noah")

        with patch("privacy_runtime.dependency_rewind._load_spacy_model") as load:
            load.return_value = None
            decision = choose_dependency_rewind_start(
                text,
                leak_start,
                leak_start + len("noah@example.com"),
            )

        self.assertFalse(decision.dependency_available)
        self.assertEqual(decision.fallback_strategy, "clause")
        self.assertEqual(decision.char_index, text.index("email"))


if __name__ == "__main__":
    unittest.main()
