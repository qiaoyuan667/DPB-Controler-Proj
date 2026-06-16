from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@dataclass(frozen=True)
class PredicateProbe:
    answer: str
    predicate: str
    total_candidates: int
    remaining_candidates: int
    note: str

    @property
    def information_gain_bits(self) -> float:
        if self.remaining_candidates <= 0:
            raise ValueError("remaining_candidates must be positive")
        if self.total_candidates < self.remaining_candidates:
            raise ValueError("total_candidates must be >= remaining_candidates")
        return math.log2(self.total_candidates / self.remaining_candidates)


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


class HeuristicScorer:
    """Offline baseline for testing the score pipeline without downloading NLI."""

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        entailment = 0.05

        if "\u672a\u6210\u5e74" in premise and "\u672a\u6ee118\u5c81" in hypothesis:
            entailment = 0.95
        elif "14\u5c81" in premise and "\u672a\u6ee118\u5c81" in hypothesis:
            entailment = 0.98
        elif "\u672a\u6210\u5e74" in premise and "14\u5c81" in hypothesis:
            entailment = 0.12
        elif "\u5408\u6cd5\u996e\u9152" in premise and "\u672a\u6ee118\u5c81" in hypothesis:
            entailment = 0.60
        elif "\u76d1\u62a4\u4eba\u540c\u610f" in premise and "\u672a\u6ee118\u5c81" in hypothesis:
            entailment = 0.75
        elif "\u521a\u4e0a\u521d\u4e2d" in premise and "\u672a\u6ee118\u5c81" in hypothesis:
            entailment = 0.85
        elif "\u521a\u4e0a\u521d\u4e2d" in premise and "12\u523015\u5c81" in hypothesis:
            entailment = 0.70
        elif "\u5e7f\u4e1c\u7701" in premise and "\u6df1\u5733\u5e02" in hypothesis:
            entailment = 0.08
        elif "\u6df1\u5733\u5e02" in premise and "\u5e7f\u4e1c\u7701" in hypothesis:
            entailment = 0.95
        elif "\u5e7f\u4e1c\u7701" in premise and "\u4e2d\u56fd\u5883\u5185" in hypothesis:
            entailment = 0.90

        contradiction = 0.02
        neutral = max(0.0, 1.0 - entailment - contradiction)
        return {
            "contradiction": contradiction,
            "entailment": entailment,
            "neutral": neutral,
        }


def entailment_probability(scores: dict[str, float]) -> float:
    for label, prob in scores.items():
        if "entail" in label:
            return prob
    raise KeyError(f"Could not find entailment label in labels: {sorted(scores)}")


def default_probes() -> list[PredicateProbe]:
    return [
        PredicateProbe(
            answer="\u5979\u672a\u6210\u5e74\u3002",
            predicate="\u5979\u672a\u6ee118\u5c81\u3002",
            total_candidates=121,
            remaining_candidates=18,
            note="Partial semantic leak: exact age is not revealed, but age range is narrowed.",
        ),
        PredicateProbe(
            answer="\u5979\u672a\u6210\u5e74\u3002",
            predicate="\u597914\u5c81\u3002",
            total_candidates=121,
            remaining_candidates=1,
            note="Should usually be neutral: minor does not imply exact age 14.",
        ),
        PredicateProbe(
            answer="\u597914\u5c81\u3002",
            predicate="\u5979\u672a\u6ee118\u5c81\u3002",
            total_candidates=121,
            remaining_candidates=18,
            note="Exact age should entail the broader minor predicate.",
        ),
        PredicateProbe(
            answer="\u5979\u8fd8\u6ca1\u5230\u53ef\u4ee5\u5408\u6cd5\u996e\u9152\u7684\u5e74\u9f84\u3002",
            predicate="\u5979\u672a\u6ee118\u5c81\u3002",
            total_candidates=121,
            remaining_candidates=18,
            note="Drinking-age clue: this may imply under 18 in some jurisdictions, but is context-sensitive.",
        ),
        PredicateProbe(
            answer="\u5979\u4ecd\u7136\u9700\u8981\u76d1\u62a4\u4eba\u540c\u610f\u3002",
            predicate="\u5979\u672a\u6ee118\u5c81\u3002",
            total_candidates=121,
            remaining_candidates=18,
            note="Guardian-consent clue: likely under 18, but depends on the action requiring consent.",
        ),
        PredicateProbe(
            answer="\u5979\u521a\u4e0a\u521d\u4e2d\u3002",
            predicate="\u5979\u672a\u6ee118\u5c81\u3002",
            total_candidates=121,
            remaining_candidates=18,
            note="School-stage clue: junior middle school strongly suggests under 18.",
        ),
        PredicateProbe(
            answer="\u5979\u521a\u4e0a\u521d\u4e2d\u3002",
            predicate="\u5979\u5927\u7ea6\u572812\u523015\u5c81\u4e4b\u95f4\u3002",
            total_candidates=121,
            remaining_candidates=4,
            note="School-stage clue with a narrower age-range predicate.",
        ),
        PredicateProbe(
            answer="\u5979\u4f4f\u5728\u5e7f\u4e1c\u7701\u3002",
            predicate="\u5979\u4f4f\u5728\u6df1\u5733\u5e02\u3002",
            total_candidates=300,
            remaining_candidates=1,
            note="Should usually be neutral: province does not imply exact city.",
        ),
        PredicateProbe(
            answer="\u5979\u4f4f\u5728\u6df1\u5733\u5e02\u3002",
            predicate="\u5979\u4f4f\u5728\u5e7f\u4e1c\u7701\u3002",
            total_candidates=300,
            remaining_candidates=21,
            note="City-to-province implication; may require geography knowledge.",
        ),
        PredicateProbe(
            answer="\u5979\u4f4f\u5728\u5e7f\u4e1c\u7701\u3002",
            predicate="\u5979\u4f4f\u5728\u4e2d\u56fd\u5883\u5185\u3002",
            total_candidates=300,
            remaining_candidates=300,
            note="Low information gain if the prior domain is already Chinese cities.",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("nli", "heuristic"),
        default="nli",
        help="Use a real NLI model or a small offline baseline.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.backend == "nli":
        try:
            scorer = NLIScorer(args.model)
        except OSError as exc:
            raise SystemExit(
                "Could not load the NLI model. If this is the first run, allow "
                "Hugging Face model download, or use '--backend heuristic' to "
                "test the local scoring pipeline without a model.\n"
                f"Model: {args.model}\n"
                f"Error: {exc}"
            ) from exc
        print("backend: nli")
        print(f"model:   {args.model}")
    else:
        scorer = HeuristicScorer()
        print("backend: heuristic")
        print("model:   none")
    print()

    for idx, probe in enumerate(default_probes(), start=1):
        scores = scorer.score(probe.answer, probe.predicate)
        entailment = entailment_probability(scores)
        info_gain = probe.information_gain_bits
        weighted_score = entailment * info_gain

        print(f"[{idx}] {probe.note}")
        print(f"answer:    {probe.answer}")
        print(f"predicate: {probe.predicate}")
        print(f"scores:    {format_scores(scores)}")
        print(f"entailment:      {entailment:.4f}")
        print(f"info_gain_bits:  {info_gain:.4f}")
        print(f"weighted_score:  {weighted_score:.4f}")
        print()


def format_scores(scores: dict[str, float]) -> str:
    return ", ".join(
        f"{label}={prob:.4f}"
        for label, prob in sorted(scores.items())
    )


if __name__ == "__main__":
    main()
