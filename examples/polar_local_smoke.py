from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime import (  # noqa: E402
    ForbiddenStringConstraint,
    HFPrivacyLogitsProcessor,
    HuggingFaceVocabulary,
    LeakageVerifier,
    PrivacyGuardV0,
    SemanticLeakageVerifierV0,
    TemplateAbstractionRewriter,
)
from privacy_runtime.polar_adapter import (  # noqa: E402
    build_polar_model_a_messages,
    build_polar_policy,
    filter_polar_samples,
    load_polar_dataset,
    parse_polar_attacker_turns,
    score_polar_transcript,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small local HuggingFace smoke eval on POLAR/Anonymous-Bench."
    )
    parser.add_argument("--dataset", required=True, help="POLAR rendered JSON/JSONL file.")
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--defense",
        choices=["none", "hard-mask", "v0-guard", "hard-mask+v0-guard"],
        default="hard-mask",
    )
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--output", default="polar_results/local_smoke.json")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow HuggingFace to download tokenizer/model files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_files_only = not args.allow_download
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        from transformers import AutoModelForCausalLM, LogitsProcessorList
    except ImportError as exc:
        raise SystemExit(
            "This script requires transformers and torch. "
            "Install with: pip install -r requirements-hf.txt"
        ) from exc

    vocabulary = HuggingFaceVocabulary.from_pretrained(
        args.model,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=local_files_only,
    )
    tokenizer = vocabulary.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = filter_polar_samples(
        load_polar_dataset(args.dataset),
        domains=set(args.domains) if args.domains else None,
        limit=args.limit,
    )

    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        policy = build_polar_policy(sample)
        constraints = (ForbiddenStringConstraint(policy.all_forbidden_strings()),)
        guard = PrivacyGuardV0(
            exact_verifier=LeakageVerifier(policy),
            semantic_verifier=SemanticLeakageVerifierV0(policy),
            rewriter=TemplateAbstractionRewriter(policy),
        )
        transcript: list[dict[str, str]] = []
        guard_events: list[dict[str, Any]] = []

        for attacker_turn in parse_polar_attacker_turns(sample)[: args.max_turns]:
            transcript.append({"role": "B", "content": attacker_turn})
            messages = build_polar_model_a_messages(sample, transcript)
            prompt_text = render_chat_prompt(tokenizer, messages)
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            )
            prompt_text = tokenizer.decode(
                inputs["input_ids"][0],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )

            logits_processor = None
            if args.defense in {"hard-mask", "hard-mask+v0-guard"}:
                logits_processor = LogitsProcessorList(
                    [
                        HFPrivacyLogitsProcessor(
                            vocabulary=vocabulary,
                            constraints=constraints,
                            prompt_text=prompt_text,
                        )
                    ]
                )

            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            reply = tokenizer.decode(
                generated_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            ).strip()

            if args.defense in {"v0-guard", "hard-mask+v0-guard"}:
                decision = guard.protect(reply)
                guard_events.append(
                    {
                        "accepted": decision.accepted,
                        "reasons": list(decision.reasons),
                        "original_text": decision.original_text,
                        "final_text": decision.final_text,
                        "rewrite_actions": list(decision.rewrite.actions)
                        if decision.rewrite
                        else [],
                    }
                )
                reply = decision.final_text

            transcript.append({"role": "A", "content": reply})

        score = score_polar_transcript(sample, transcript)
        score["defense"] = args.defense
        score["transcript"] = transcript
        score["guard_events"] = guard_events
        results.append(score)
        print(
            f"[{index}/{len(samples)}] {score['sample_id']} "
            f"privacy={score['privacy_score']} utility={score['utility_score']} "
            f"leaks={score['leaked_count']}"
        )

    summary = summarize(results)
    payload = {"summary": summary, "results": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines: list[str] = []
    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}:\n{message['content']}")
    lines.append("ASSISTANT:\n")
    return "\n\n".join(lines)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    privacy_scores = [
        float(result["privacy_score"])
        for result in results
        if result.get("privacy_score") is not None
    ]
    utility_scores = [
        float(result["utility_score"])
        for result in results
        if result.get("utility_score") is not None
    ]
    example_scores = [
        float(result["example_score"])
        for result in results
        if result.get("example_score") is not None
    ]
    return {
        "num_examples": len(results),
        "privacy_score": mean(privacy_scores) if privacy_scores else None,
        "utility_score": mean(utility_scores) if utility_scores else None,
        "example_score": mean(example_scores) if example_scores else None,
        "leaked_count": sum(int(result["leaked_count"]) for result in results),
        "allowed_count": sum(int(result["allowed_count"]) for result in results),
    }


if __name__ == "__main__":
    main()
