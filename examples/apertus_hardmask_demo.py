from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime import (  # noqa: E402
    DEFAULT_APERTUS_MODEL_PATH,
    DEFAULT_PRIVATE_REGEXES,
    ForbiddenRegexConstraint,
    ForbiddenStringConstraint,
    HFTrustedModel,
    HuggingFaceVocabulary,
    PrivacyLogitProcessor,
    privacy_policy_from_protected_attributes,
)


DEFAULT_MAX_NEW_TOKENS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Apertus trusted-model hard-mask demo."
    )
    parser.add_argument("--model-path", default=DEFAULT_APERTUS_MODEL_PATH)
    parser.add_argument(
        "--protected",
        action="append",
        default=[],
        help="Protected attribute value. Repeat for multiple values.",
    )
    parser.add_argument(
        "--protected-json",
        default=None,
        help="Optional JSON file containing strings or structured protected attributes.",
    )
    parser.add_argument(
        "--source-text",
        default=None,
        help="Trusted source document text visible to the trusted model.",
    )
    parser.add_argument(
        "--source-file",
        default=None,
        help="UTF-8 text file containing trusted source document text.",
    )
    parser.add_argument("--attacker-text", default=None)
    parser.add_argument(
        "--attacker-file",
        default=None,
        help="UTF-8 text file containing the attacker/user message.",
    )
    parser.add_argument("--system-text", default="You are a helpful trusted assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--trace-generation",
        action="store_true",
        help="Trace greedy hard-mask decoding step by step.",
    )
    parser.add_argument(
        "--trace-top-k",
        type=int,
        default=5,
        help="Number of raw/masked top tokens to show per traced step.",
    )
    parser.add_argument(
        "--trace-output-dir",
        default="outputs/traces",
        help="Directory for detailed trace JSON files.",
    )
    parser.add_argument(
        "--rewind-strategy",
        choices=("value", "dependency"),
        default="value",
        help="Rewind to the protected value start or a dependency-derived slot boundary.",
    )
    parser.add_argument(
        "--dependency-model",
        default="en_core_web_sm",
        help="spaCy model used when --rewind-strategy dependency is selected.",
    )
    parser.add_argument(
        "--inspect-mask-only",
        action="store_true",
        help="Load tokenizer only and inspect blocked token ids without model generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    protected_attributes = list(args.protected)
    if args.protected_json:
        protected_attributes.extend(_load_protected_json(args.protected_json))

    policy = privacy_policy_from_protected_attributes(protected_attributes)
    source_text = load_source_text(args.source_text, args.source_file)
    attacker_text = load_attacker_text(args.attacker_text, args.attacker_file)
    messages = build_trusted_messages(
        system_text=args.system_text,
        source_text=source_text,
        attacker_text=attacker_text,
    )

    if args.inspect_mask_only:
        payload = inspect_mask_only(
            model_path=args.model_path,
            protected_attributes=protected_attributes,
            local_files_only=not args.allow_download,
        )
        payload["attacker_text"] = attacker_text
        payload.update(source_metadata(source_text))
        payload["protected_values_found_in_reply"] = []
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.do_sample:
        raise ValueError("rewind hard-mask demo currently supports greedy decoding only")

    trusted_model = HFTrustedModel(
        model_path=args.model_path,
        local_files_only=not args.allow_download,
    )
    unmasked_reply = trusted_model.generate_unmasked(
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )
    result = trusted_model.generate_with_rewind(
        messages=messages,
        policy=policy,
        max_new_tokens=args.max_new_tokens,
        top_k=args.trace_top_k,
        rewind_strategy=args.rewind_strategy,
        dependency_model=args.dependency_model,
        trace=args.trace_generation,
    )
    trace_payload = None
    if args.trace_generation:
        trace_payload = build_trace_payload(
            attacker_text=attacker_text,
            source_text=source_text,
            model_path=result.model_path,
            protected_attribute_count=result.protected_attribute_count,
            hardmask_reply=result.text,
            rewind_strategy=args.rewind_strategy,
            steps=list(result.steps),
            rewind_events=list(result.rewind_events),
            fallback_used=result.fallback_used,
        )
        trace_id, trace_path = write_trace_payload(
            trace_payload,
            output_dir=args.trace_output_dir,
        )

    payload = {
        "attacker_text": attacker_text,
        "unmasked_reply": unmasked_reply,
        "hardmask_reply": result.text,
        **source_metadata(source_text),
        "comparison": {
            "unmasked_protected_values_found": protected_values_found(
                unmasked_reply,
                policy.all_forbidden_strings(),
            ),
            "hardmask_protected_values_found": protected_values_found(
                result.text,
                policy.all_forbidden_strings(),
            ),
            "replies_differ": unmasked_reply != result.text,
        },
        "mask_summary": {
            "model_path": result.model_path,
            "protected_attribute_count": result.protected_attribute_count,
            "rewind_strategy": args.rewind_strategy,
            "rewind_event_count": len(result.rewind_events),
            "fallback_used": result.fallback_used,
        },
        "protected_values_found_in_reply": protected_values_found(
            result.text,
            policy.all_forbidden_strings(),
        ),
    }
    if trace_payload is not None:
        payload["hardmask_trace"] = {
            "trace_id": trace_id,
            "trace_path": str(trace_path),
            "rewind_strategy": args.rewind_strategy,
            "rewind_event_count": len(result.rewind_events),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def load_source_text(source_text: str | None, source_file: str | None) -> str:
    chunks: list[str] = []
    if source_text:
        chunks.append(source_text.strip())
    if source_file:
        chunks.append(Path(source_file).read_text(encoding="utf-8").strip())
    return "\n\n".join(chunk for chunk in chunks if chunk)


def load_attacker_text(attacker_text: str | None, attacker_file: str | None) -> str:
    chunks: list[str] = []
    if attacker_text:
        chunks.append(attacker_text.strip())
    if attacker_file:
        chunks.append(Path(attacker_file).read_text(encoding="utf-8").strip())
    text = "\n\n".join(chunk for chunk in chunks if chunk)
    if not text:
        raise ValueError("provide --attacker-text or --attacker-file")
    return text


def build_trusted_messages(
    *,
    system_text: str,
    source_text: str,
    attacker_text: str,
) -> list[dict[str, str]]:
    system_content = system_text
    if source_text:
        system_content = f"{system_content}\n\nSource document:\n{source_text}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": attacker_text},
    ]


def source_metadata(
    source_text: str,
) -> dict[str, object]:
    return {
        "source_provided": bool(source_text),
        "source_length_chars": len(source_text),
    }


def build_trace_payload(
    *,
    attacker_text: str,
    source_text: str,
    model_path: str,
    protected_attribute_count: int,
    hardmask_reply: str,
    rewind_strategy: str,
    steps: list[dict[str, object]],
    rewind_events: list[dict[str, object]],
    fallback_used: bool,
) -> dict[str, object]:
    return {
        "attacker_text": attacker_text,
        "source_provided": bool(source_text),
        "source_length_chars": len(source_text),
        "model_path": model_path,
        "protected_attribute_count": protected_attribute_count,
        "hardmask_trace_reply": hardmask_reply,
        "rewind_strategy": rewind_strategy,
        "steps": steps,
        "rewind_events": rewind_events,
        "fallback_used": fallback_used,
    }


def write_trace_payload(
    payload: dict[str, object],
    *,
    output_dir: str,
) -> tuple[str, Path]:
    trace_id = make_trace_id(
        attacker_text=str(payload.get("attacker_text") or ""),
        rewind_events=payload.get("rewind_events"),
    )
    trace_dir = Path(output_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{trace_id}.json"
    trace_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return trace_id, trace_path


def make_trace_id(attacker_text: str, rewind_events: object) -> str:
    event_count = len(rewind_events) if isinstance(rewind_events, list) else 0
    digest = hashlib.sha1(attacker_text.encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"trace-{timestamp}-rw{event_count}-{digest}"


def inspect_mask_only(
    *,
    model_path: str,
    protected_attributes: list[object],
    local_files_only: bool,
) -> dict[str, object]:
    policy = privacy_policy_from_protected_attributes(protected_attributes)
    vocabulary = HuggingFaceVocabulary.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    processor = PrivacyLogitProcessor(
        vocabulary=vocabulary,
        constraints=(
            ForbiddenStringConstraint(policy.all_forbidden_strings()),
            ForbiddenRegexConstraint(DEFAULT_PRIVATE_REGEXES),
        ),
    )
    decision = processor.blocked_tokens("")
    sample = []
    for token_id in sorted(decision.blocked_token_ids)[:20]:
        sample.append(
            {
                "token_id": token_id,
                "text": vocabulary.token_text(token_id),
                "reasons": list(decision.reasons.get(token_id, ())),
            }
        )
    return {
        "mode": "inspect-mask-only",
        "model_path": model_path,
        "protected_attribute_count": len(policy.facts),
        "blocked_token_count_at_start": len(decision.blocked_token_ids),
        "blocked_token_sample": sample,
    }


def _load_protected_json(path: str) -> list[object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--protected-json must contain a JSON array")
    return data


def protected_values_found(text: str, values: tuple[str, ...]) -> list[str]:
    lowered = text.casefold()
    found = []
    for value in values:
        if value.casefold() in lowered:
            found.append(value)
    return found


if __name__ == "__main__":
    main()
