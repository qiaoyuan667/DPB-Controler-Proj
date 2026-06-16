from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@dataclass(frozen=True)
class Predicate:
    label: str
    text: str
    probability: float
    level: str

    @property
    def ic_bits(self) -> float:
        return -math.log2(self.probability)


@dataclass(frozen=True)
class Finding:
    predicate: Predicate
    sentence: str
    span: str
    sentence_entailment: float
    span_entailment: float
    score_bits: float
    normalized_score: float


@dataclass(frozen=True)
class SpanTrace:
    predicate_label: str
    predicate: str
    sentence: str
    window: int
    spans: tuple[dict[str, float | str], ...]


@dataclass(frozen=True)
class SentenceCandidate:
    predicate_label: str
    predicate: str
    sentence: str
    sentence_entailment: float


class EntailmentScorer(Protocol):
    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        ...


class NLIScorer:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = {
            idx: label.lower()
            for idx, label in self.model.config.id2label.items()
        }

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        return {self.id2label[idx]: probs[idx] for idx in range(len(probs))}


def generate_predicates(
    protected_value: str,
    attribute_type: str,
    manual_predicates: list[str] | None = None,
    predicate_specs: list[Predicate] | None = None,
    value_only: bool = False,
    predicate_probability: float = 0.01,
    hypothesis_template: str | None = None,
) -> list[Predicate]:
    if predicate_specs:
        return predicate_specs

    if manual_predicates:
        return [
            Predicate(
                label=f"manual_{idx}",
                text=format_predicate_text(predicate, hypothesis_template),
                probability=predicate_probability,
                level="manual",
            )
            for idx, predicate in enumerate(manual_predicates, start=1)
        ]

    value = protected_value.strip()
    if value_only:
        return [
            Predicate(
                label="protected_value",
                text=format_predicate_text(value, hypothesis_template),
                probability=predicate_probability,
                level="value",
            )
        ]

    attr = attribute_type.casefold()

    if attr == "gender":
        return gender_predicates(value)
    if attr == "age":
        return age_predicates(value)
    if attr == "location":
        return location_predicates(value)

    return [
        Predicate(
            label="exact_value",
            text=format_predicate_text(value, hypothesis_template),
            probability=predicate_probability,
            level="exact",
        )
    ]


def format_predicate_text(value: str, hypothesis_template: str | None) -> str:
    if not hypothesis_template:
        return value
    return hypothesis_template.format(value=value)


def gender_predicates(value: str) -> list[Predicate]:
    v = value.casefold()
    if v in {"female", "woman", "girl", "she", "f"}:
        return [
            Predicate("gender_female", "The patient is female.", 0.50, "gender"),
            Predicate("woman", "The patient is a woman.", 0.50, "gender"),
            Predicate("she_pronouns", "The patient uses she/her pronouns.", 0.50, "pronoun"),
            Predicate("pregnancy", "The patient is pregnant.", 0.02, "medical_status"),
            Predicate("mother", "The patient is a mother.", 0.25, "family_role"),
        ]
    if v in {"male", "man", "boy", "he", "m"}:
        return [
            Predicate("gender_male", "The patient is male.", 0.50, "gender"),
            Predicate("man", "The patient is a man.", 0.50, "gender"),
            Predicate("he_pronouns", "The patient uses he/him pronouns.", 0.50, "pronoun"),
            Predicate("father", "The patient is a father.", 0.25, "family_role"),
        ]
    return [
        Predicate("gender_value", f"The patient has gender {value}.", 0.10, "gender")
    ]


def age_predicates(value: str) -> list[Predicate]:
    age_match = re.search(r"\d+", value)
    if not age_match:
        return [
            Predicate("age_value", f"The patient age is {value}.", 0.10, "age")
        ]

    age = int(age_match.group(0))
    predicates = [
        Predicate("exact_age", f"The patient is {age} years old.", 1 / 121, "exact"),
    ]

    if age < 18:
        predicates.extend(
            [
                Predicate("under_18", "The patient is under 18.", 18 / 121, "range"),
                Predicate("minor", "The patient is a minor.", 18 / 121, "category"),
            ]
        )
    if 13 <= age <= 19:
        predicates.append(
            Predicate("teenager", "The patient is a teenager.", 7 / 121, "category")
        )
    if 12 <= age <= 15:
        predicates.append(
            Predicate(
                "middle_school_age",
                "The patient is about 12 to 15 years old.",
                4 / 121,
                "range",
            )
        )
    if age >= 18:
        predicates.append(Predicate("adult", "The patient is an adult.", 103 / 121, "category"))

    return predicates


def location_predicates(value: str) -> list[Predicate]:
    key = normalized(value)
    known = {
        "zurich": [
            Predicate("city_zurich", "The patient is in Zurich.", 1 / 10000, "city"),
            Predicate("country_switzerland", "The patient is in Switzerland.", 50 / 10000, "country"),
            Predicate("continent_europe", "The patient is in Europe.", 2500 / 10000, "continent"),
        ],
        "shenzhen": [
            Predicate("city_shenzhen", "The patient is in Shenzhen.", 1 / 10000, "city"),
            Predicate("country_china", "The patient is in China.", 700 / 10000, "country"),
            Predicate("continent_asia", "The patient is in Asia.", 3500 / 10000, "continent"),
        ],
        "switzerland": [
            Predicate("country_switzerland", "The patient is in Switzerland.", 50 / 10000, "country"),
            Predicate("continent_europe", "The patient is in Europe.", 2500 / 10000, "continent"),
        ],
        "china": [
            Predicate("country_china", "The patient is in China.", 700 / 10000, "country"),
            Predicate("continent_asia", "The patient is in Asia.", 3500 / 10000, "continent"),
        ],
    }
    return known.get(
        key,
        [
            Predicate(
                "location_value",
                f"The patient is in {value}.",
                0.01,
                "location",
            )
        ],
    )


def split_sentences(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            lines.append(cleaned)

    pieces: list[str] = []
    for line in lines:
        line = (
            line.replace("\u3002", ".")
            .replace("\uff01", "!")
            .replace("\uff1f", "?")
        )
        pieces.extend(re.split(r"(?<=[.!?])\s+", line))
    return [piece.strip() for piece in pieces if piece.strip()]


def generate_spans(sentence: str, max_window: int = 5) -> list[str]:
    tokens = tokenize_for_spans(sentence)
    spans = [
        " ".join(tokens[start : start + window])
        for window in range(1, max_window + 1)
        for start in range(0, max(0, len(tokens) - window + 1))
    ]
    return dedupe_preserve_order(span for span in spans if len(span.strip()) >= 2)


def tokenize_for_spans(sentence: str) -> list[str]:
    return re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[\u4e00-\u9fff]",
        sentence,
    )


def analyze_text(
    text: str,
    predicates: list[Predicate],
    scorer: EntailmentScorer,
    attribute_type: str,
    sentence_threshold: float,
    span_threshold: float,
    max_span_window: int,
) -> tuple[list[Finding], list[SpanTrace], list[SentenceCandidate]]:
    findings: list[Finding] = []
    traces: list[SpanTrace] = []
    sentence_candidates: list[SentenceCandidate] = []
    max_ic = max_ic_for_attribute(attribute_type, predicates)

    for sentence in split_sentences(text):
        for predicate in predicates:
            sentence_scores = scorer.score(sentence, predicate.text)
            sentence_entailment = entailment_probability(sentence_scores)
            if sentence_entailment < sentence_threshold:
                continue
            sentence_candidates.append(
                SentenceCandidate(
                    predicate_label=predicate.label,
                    predicate=predicate.text,
                    sentence=sentence,
                    sentence_entailment=sentence_entailment,
                )
            )

            span_hits = find_minimal_spans(
                sentence=sentence,
                predicate=predicate,
                scorer=scorer,
                sentence_entailment=sentence_entailment,
                span_threshold=span_threshold,
                max_span_window=max_span_window,
                max_ic=max_ic,
            )
            findings.extend(span_hits.findings)
            traces.extend(span_hits.traces)

    sorted_findings = sorted(
        findings,
        key=lambda finding: (
            -finding.normalized_score,
            len(finding.span.split()),
            len(finding.span),
        ),
    )
    return sorted_findings, traces, sentence_candidates


@dataclass(frozen=True)
class MinimalSpanResult:
    findings: list[Finding]
    traces: list[SpanTrace]


def max_ic_for_attribute(attribute_type: str, predicates: list[Predicate]) -> float:
    attr = attribute_type.casefold()
    if attr == "gender":
        return 1.0
    if attr == "age":
        return math.log2(121)
    if attr == "location":
        return math.log2(10000)
    return max((predicate.ic_bits for predicate in predicates), default=1.0)


def find_minimal_spans(
    sentence: str,
    predicate: Predicate,
    scorer: EntailmentScorer,
    sentence_entailment: float,
    span_threshold: float,
    max_span_window: int,
    max_ic: float,
) -> MinimalSpanResult:
    tokens = tokenize_for_spans(sentence)
    if not tokens:
        return MinimalSpanResult(findings=[], traces=[])

    traces: list[SpanTrace] = []

    for window in range(1, min(max_span_window, len(tokens)) + 1):
        hits: list[Finding] = []
        span_scores_for_trace: list[dict[str, float | str]] = []
        for start in range(0, len(tokens) - window + 1):
            span = " ".join(tokens[start : start + window])
            span_scores = scorer.score(span, predicate.text)
            span_entailment = entailment_probability(span_scores)
            span_scores_for_trace.append(
                {
                    "span": span,
                    "entailment": round(span_entailment, 4),
                }
            )
            if span_entailment < span_threshold:
                continue

            score_bits = span_entailment * predicate.ic_bits
            normalized_score = min(
                1.0,
                span_entailment * (predicate.ic_bits / max_ic),
            )
            hits.append(
                Finding(
                    predicate=predicate,
                    sentence=sentence,
                    span=span,
                    sentence_entailment=sentence_entailment,
                    span_entailment=span_entailment,
                    score_bits=score_bits,
                    normalized_score=normalized_score,
                )
            )
        traces.append(
            SpanTrace(
                predicate_label=predicate.label,
                predicate=predicate.text,
                sentence=sentence,
                window=window,
                spans=tuple(span_scores_for_trace),
            )
        )
        if hits:
            return MinimalSpanResult(findings=hits, traces=traces)

    return MinimalSpanResult(findings=[], traces=traces)


def aggregate_findings(findings: list[Finding]) -> dict[str, float]:
    by_predicate: dict[str, float] = {}
    for finding in findings:
        key = finding.predicate.label
        by_predicate[key] = max(by_predicate.get(key, 0.0), finding.normalized_score)

    max_score = max(by_predicate.values(), default=0.0)
    return {
        "total_normalized_score": max_score,
        "max_normalized_score": max_score,
        "finding_count": float(len(findings)),
        "predicate_count": float(len(by_predicate)),
    }


def entailment_probability(scores: dict[str, float]) -> float:
    for label, prob in scores.items():
        if "entail" in label:
            return prob
    raise KeyError(f"Could not find entailment label in labels: {sorted(scores)}")


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def any_word(text: str, *words: str) -> bool:
    normalized_text = f" {normalized(text)} "
    return any(f" {normalized(word)} " in normalized_text for word in words)


def dedupe_preserve_order(items) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = normalized(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def finding_to_dict(finding: Finding) -> dict[str, object]:
    return {
        "predicate_label": finding.predicate.label,
        "predicate": finding.predicate.text,
        "predicate_level": finding.predicate.level,
        "predicate_probability": finding.predicate.probability,
        "ic_bits": finding.predicate.ic_bits,
        "sentence": finding.sentence,
        "span": finding.span,
        "sentence_entailment": finding.sentence_entailment,
        "span_entailment": finding.span_entailment,
        "score_bits": finding.score_bits,
        "normalized_score": finding.normalized_score,
    }


def trace_to_dict(trace: SpanTrace, span_limit: int) -> dict[str, object]:
    return {
        "predicate_label": trace.predicate_label,
        "predicate": trace.predicate,
        "sentence": trace.sentence,
        "window": trace.window,
        "spans": list(trace.spans[:span_limit]),
        "truncated": len(trace.spans) > span_limit,
    }


def sentence_candidate_to_dict(candidate: SentenceCandidate) -> dict[str, object]:
    return {
        "predicate_label": candidate.predicate_label,
        "predicate": candidate.predicate,
        "sentence": candidate.sentence,
        "sentence_entailment": candidate.sentence_entailment,
    }


def load_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    return (
        "I understand that you need specific information to complete the referral. "
        "However, I must respect the patient's privacy and confidentiality. "
        "She prefers a late morning appointment. "
        "The patient still needs guardian consent for treatment."
    )


def parse_manual_predicates(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    predicates = [item.strip() for item in raw.split(";")]
    return [predicate for predicate in predicates if predicate]


def parse_predicate_specs(raw: str | None) -> list[Predicate] | None:
    if not raw:
        return None

    predicates: list[Predicate] = []
    for idx, item in enumerate(raw.split(";"), start=1):
        item = item.strip()
        if not item:
            continue

        parts = [part.strip() for part in item.split("|")]
        text = parts[0]
        probability = float(parts[1]) if len(parts) >= 2 and parts[1] else 0.01
        label = parts[2] if len(parts) >= 3 and parts[2] else f"spec_{idx}"
        level = parts[3] if len(parts) >= 4 and parts[3] else "spec"
        predicates.append(
            Predicate(
                label=label,
                text=text,
                probability=probability,
                level=level,
            )
        )

    return predicates or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")
    parser.add_argument("--text-file")
    parser.add_argument("--protected-value", default="female")
    parser.add_argument("--attribute-type", default="gender")
    parser.add_argument("--predicates", help="Optional semicolon-separated predicate list.")
    parser.add_argument(
        "--predicate-specs",
        help=(
            "Optional semicolon-separated bare predicates with probability, "
            "format: text|probability|label|level. Example: female|0.5;pregnant|0.02"
        ),
    )
    parser.add_argument(
        "--value-only-predicate",
        action="store_true",
        help="Use the protected value itself as the only predicate.",
    )
    parser.add_argument(
        "--predicate-probability",
        type=float,
        default=0.01,
        help="Probability used for manual/value-only predicates when computing IC.",
    )
    parser.add_argument(
        "--hypothesis-template",
        help="Optional template for manual/value-only predicates, e.g. 'The patient is {value}.'",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sentence-threshold", type=float, default=0.35)
    parser.add_argument("--span-threshold", type=float, default=0.35)
    parser.add_argument("--max-span-window", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--show-span-trace",
        action="store_true",
        help="Show compact span-window scores for sentence/predicate pairs that pass sentence-level filtering.",
    )
    parser.add_argument(
        "--trace-span-limit",
        type=int,
        default=20,
        help="Maximum number of spans to print per window in --show-span-trace output.",
    )
    args = parser.parse_args()

    text = load_text(args)
    predicates = generate_predicates(
        args.protected_value,
        args.attribute_type,
        parse_manual_predicates(args.predicates),
        predicate_specs=parse_predicate_specs(args.predicate_specs),
        value_only=args.value_only_predicate,
        predicate_probability=args.predicate_probability,
        hypothesis_template=args.hypothesis_template,
    )

    try:
        scorer: EntailmentScorer = NLIScorer(args.model)
    except OSError as exc:
        raise SystemExit(
            "Could not load the NLI model. If this is the first run, download "
            "the model from Hugging Face or point --model to a local model path.\n"
            f"Model: {args.model}\n"
            f"Error: {exc}"
        ) from exc

    findings, traces, sentence_candidates = analyze_text(
        text=text,
        predicates=predicates,
        scorer=scorer,
        attribute_type=args.attribute_type,
        sentence_threshold=args.sentence_threshold,
        span_threshold=args.span_threshold,
        max_span_window=args.max_span_window,
    )
    aggregate = aggregate_findings(findings)

    result = {
        "protected_value": args.protected_value,
        "attribute_type": args.attribute_type,
        "backend": "nli",
        "model": args.model,
        "thresholds": {
            "sentence_entailment": args.sentence_threshold,
            "span_entailment": args.span_threshold,
        },
        "predicates": [
            {
                "label": predicate.label,
                "text": predicate.text,
                "probability": predicate.probability,
                "ic_bits": predicate.ic_bits,
                "level": predicate.level,
            }
            for predicate in predicates
        ],
        "aggregate": aggregate,
        "sentence_candidates": [
            sentence_candidate_to_dict(candidate)
            for candidate in sentence_candidates
        ],
        "top_findings": [
            finding_to_dict(finding)
            for finding in findings[: args.top_k]
        ],
    }
    if args.show_span_trace:
        result["span_trace"] = [
            trace_to_dict(trace, args.trace_span_limit)
            for trace in traces
        ]

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
