from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime import (  # noqa: E402
    CounterfactualBuilder,
    CounterfactualPrivacyCostEstimator,
    HFCausalLMLikelihoodScorer,
    PrivacyPolicy,
)


DEFAULT_SYSTEM_TEMPLATE = """You are a trusted assistant.

Source document:
{document}

Task instruction:
{task}
"""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    category: str
    expected_leak: bool | None
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score per-fact counterfactual privacy cost with a local HF model."
    )
    parser.add_argument("--model", required=True, help="HF model id or local path.")
    parser.add_argument("--document", required=True, help="UTF-8 source document file.")
    parser.add_argument("--policy", required=True, help="Structured policy JSON file.")
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate", help="Candidate assistant response.")
    candidate.add_argument("--candidate-file", help="UTF-8 candidate response file.")
    candidate.add_argument(
        "--candidate-suite",
        help="JSON list of candidate_id/category/expected_leak/text records.",
    )
    parser.add_argument("--task", default="Answer the user's request safely.")
    parser.add_argument(
        "--system-template",
        help="UTF-8 template containing {document} and optionally {task}.",
    )
    parser.add_argument(
        "--history-json",
        help="Optional JSON list of prior chat messages with role/content fields.",
    )
    parser.add_argument(
        "--allowed-prompt-occurrence",
        action="append",
        default=[],
        metavar="FACT_ID=COUNT",
        help="Protected occurrences already present in released history.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["max", "mixture", "mean"],
        default="max",
    )
    parser.add_argument(
        "--counterfactual-modes",
        nargs="+",
        choices=["placeholder", "abstraction", "substitution", "removal"],
        default=["placeholder", "abstraction", "substitution"],
        help="Intervention families to include; useful for paper ablations.",
    )
    parser.add_argument("--max-interventions-per-fact", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-sequence-tokens", type=int)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map; use 'none' to disable Accelerate dispatch.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--top-tokens", type=int, default=8)
    parser.add_argument("--output", default="counterfactual_cost_result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_files_only = not args.allow_download
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install local-model dependencies with: pip install -r requirements-hf.txt"
        ) from exc

    document = Path(args.document).read_text(encoding="utf-8")
    policy_data = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    policy = PrivacyPolicy.from_mapping(policy_data)
    candidate_specs = _load_candidates(args)
    template = (
        Path(args.system_template).read_text(encoding="utf-8")
        if args.system_template
        else DEFAULT_SYSTEM_TEMPLATE
    )
    if "{document}" not in template:
        raise SystemExit("--system-template must contain {document}")
    history = _load_history(args.history_json)
    allowed_occurrences = _parse_allowed_occurrences(args.allowed_prompt_occurrence)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "dtype": _dtype(args.dtype, torch),
    }
    if args.device_map.casefold() != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    scorer = HFCausalLMLikelihoodScorer(
        model=model,
        tokenizer=tokenizer,
        temperature=args.temperature,
        max_sequence_tokens=args.max_sequence_tokens,
    )
    builder = CounterfactualBuilder(
        include_placeholder="placeholder" in args.counterfactual_modes,
        include_abstractions="abstraction" in args.counterfactual_modes,
        include_substitutions="substitution" in args.counterfactual_modes,
        include_removal="removal" in args.counterfactual_modes,
        max_interventions_per_fact=args.max_interventions_per_fact,
    )
    estimator = CounterfactualPrivacyCostEstimator(
        scorer=scorer,
        builder=builder,
        aggregation=args.aggregation,
    )

    def prompt_builder(current_document: str) -> str:
        system = template.replace("{document}", current_document).replace(
            "{task}", args.task
        )
        messages = [{"role": "system", "content": system}, *history]
        return render_chat_prompt(tokenizer, messages)

    entries: list[dict[str, Any]] = []
    for index, spec in enumerate(candidate_specs, start=1):
        if args.candidate_suite:
            print(
                f"[{index}/{len(candidate_specs)}] {spec.candidate_id} "
                f"category={spec.category} expected_leak={spec.expected_leak}"
            )
        result = estimator.estimate_policy(
            document,
            policy,
            spec.text,
            prompt_builder=prompt_builder,
            allowed_prompt_occurrences=allowed_occurrences,
        )
        result_payload = asdict(result)
        result_payload["costs"] = dict(result.costs)
        result_payload["normalized_costs"] = dict(result.normalized_costs)
        entries.append(
            {
                "candidate_id": spec.candidate_id,
                "category": spec.category,
                "expected_leak": spec.expected_leak,
                "result": result_payload,
            }
        )
        print_summary(result, top_tokens=args.top_tokens)

    if args.candidate_suite:
        payload = {
            "config": {
                "model": args.model,
                "document": args.document,
                "policy": args.policy,
                "aggregation": args.aggregation,
                "counterfactual_modes": args.counterfactual_modes,
                "temperature": args.temperature,
            },
            "rankings": _build_rankings(entries),
            "results": entries,
        }
    else:
        payload = entries[0]["result"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    lines = [f"{message['role'].upper()}:\n{message['content']}" for message in messages]
    lines.append("ASSISTANT:\n")
    return "\n\n".join(lines)


def print_summary(result: Any, *, top_tokens: int) -> None:
    print(f"candidate_tokens={result.token_count} private_logprob={result.private_logprob:.4f}")
    for fact in result.facts:
        print(
            f"{fact.fact_id}: cost={fact.cost:.4f} nats "
            f"normalized={fact.normalized_cost:.6f} aggregation={fact.aggregation}"
        )
        if fact.selected_intervention_id is not None:
            selected = next(
                item
                for item in fact.interventions
                if item.intervention_id == fact.selected_intervention_id
            )
        else:
            selected = max(fact.interventions, key=lambda item: item.signed_loss)
        ranked = sorted(
            selected.token_losses,
            key=lambda item: item.signed_loss,
            reverse=True,
        )[:top_tokens]
        token_summary = ", ".join(
            f"{item.token!r}:{item.signed_loss:.3f}" for item in ranked
        )
        print(f"  diagnostic_intervention={selected.intervention_id}")
        print(f"  top_token_losses={token_summary}")


def _load_history(path: str | None) -> list[dict[str, str]]:
    if path is None:
        return []
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise SystemExit("--history-json must contain a JSON list")
    history: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or "role" not in item or "content" not in item:
            raise SystemExit("every history message needs role and content")
        history.append({"role": str(item["role"]), "content": str(item["content"])})
    return history


def _load_candidates(args: argparse.Namespace) -> list[CandidateSpec]:
    if args.candidate is not None:
        return [CandidateSpec("candidate", "unspecified", None, args.candidate)]
    if args.candidate_file is not None:
        text = Path(args.candidate_file).read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit("--candidate-file is empty")
        return [CandidateSpec("candidate", "unspecified", None, text)]
    return load_candidate_suite(args.candidate_suite)


def load_candidate_suite(path: str | Path) -> list[CandidateSpec]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise SystemExit("--candidate-suite must contain a non-empty JSON list")

    specs: list[CandidateSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"candidate suite item {index} must be an object")
        candidate_id = str(item.get("candidate_id") or "").strip()
        text = str(item.get("text") or "").strip()
        category = str(item.get("category") or "unspecified").strip()
        expected_leak = item.get("expected_leak")
        if not candidate_id or not text:
            raise SystemExit(
                f"candidate suite item {index} needs candidate_id and text"
            )
        if candidate_id in seen_ids:
            raise SystemExit(f"duplicate candidate_id: {candidate_id}")
        if expected_leak is not None and not isinstance(expected_leak, bool):
            raise SystemExit(
                f"expected_leak for {candidate_id} must be true, false, or null"
            )
        seen_ids.add(candidate_id)
        specs.append(CandidateSpec(candidate_id, category, expected_leak, text))
    return specs


def _build_rankings(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    fact_ids = {
        fact_id
        for entry in entries
        for fact_id in entry["result"]["costs"]
    }
    rankings: dict[str, list[dict[str, Any]]] = {}
    for fact_id in sorted(fact_ids):
        rows = [
            {
                "candidate_id": entry["candidate_id"],
                "category": entry["category"],
                "expected_leak": entry["expected_leak"],
                "cost": entry["result"]["costs"][fact_id],
                "normalized_cost": entry["result"]["normalized_costs"][fact_id],
            }
            for entry in entries
        ]
        rankings[fact_id] = sorted(
            rows,
            key=lambda row: (row["cost"], row["normalized_cost"]),
            reverse=True,
        )
    return rankings


def _parse_allowed_occurrences(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        fact_id, separator, raw_count = value.partition("=")
        if not separator or not fact_id.strip():
            raise SystemExit(f"invalid --allowed-prompt-occurrence: {value}")
        count = int(raw_count)
        if count < 0:
            raise SystemExit("allowed occurrence count must be non-negative")
        parsed[fact_id.strip()] = count
    return parsed


def _dtype(name: str, torch: Any) -> Any:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


if __name__ == "__main__":
    main()
