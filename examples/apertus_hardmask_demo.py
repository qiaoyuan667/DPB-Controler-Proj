from __future__ import annotations

import argparse
import json
import os
import sys
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
    parser.add_argument("--attacker-text", required=True)
    parser.add_argument("--system-text", default="You are a helpful trusted assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
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
    messages = [
        {"role": "system", "content": args.system_text},
        {"role": "user", "content": args.attacker_text},
    ]

    if args.inspect_mask_only:
        payload = inspect_mask_only(
            model_path=args.model_path,
            protected_attributes=protected_attributes,
            local_files_only=not args.allow_download,
        )
        payload["attacker_text"] = args.attacker_text
        payload["protected_values_found_in_reply"] = []
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    trusted_model = HFTrustedModel(
        model_path=args.model_path,
        local_files_only=not args.allow_download,
    )
    result = trusted_model.generate(
        messages=messages,
        policy=policy,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    payload = {
        "attacker_text": args.attacker_text,
        "trusted_reply": result.text,
        "mask_summary": {
            "model_path": result.model_path,
            "protected_attribute_count": result.protected_attribute_count,
            "blocked_token_count_at_start": result.blocked_token_count,
            "blocked_token_sample": [
                {"token_id": token_id, "text": text, "reasons": list(reasons)}
                for token_id, text, reasons in result.blocked_token_sample
            ],
        },
        "protected_values_found_in_reply": protected_values_found(
            result.text,
            policy.all_forbidden_strings(),
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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
