from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime import (  # noqa: E402
    ForbiddenStringConstraint,
    HFPrivacyLogitsProcessor,
    HuggingFaceVocabulary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HuggingFace generation with privacy token masking."
    )
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument(
        "--prompt",
        default="Write one sentence. Our maximum budget is ",
        help="Prompt passed to the model.",
    )
    parser.add_argument(
        "--secret",
        action="append",
        required=True,
        help="Forbidden string. Repeat this flag for aliases.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download tokenizer/model files.",
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
            "This demo requires transformers and torch. Install with: "
            "pip install transformers torch"
        ) from exc

    vocabulary = HuggingFaceVocabulary.from_pretrained(
        args.model,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=local_files_only,
    )

    inputs = vocabulary.tokenizer(args.prompt, return_tensors="pt")
    logits_processor = LogitsProcessorList(
        [
            HFPrivacyLogitsProcessor(
                vocabulary=vocabulary,
                constraints=(ForbiddenStringConstraint(tuple(args.secret)),),
                prompt_length=inputs["input_ids"].shape[1],
            )
        ]
    )
    output_ids = model.generate(
        **inputs,
        logits_processor=logits_processor,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=vocabulary.tokenizer.eos_token_id,
    )
    print(
        vocabulary.tokenizer.decode(
            output_ids[0],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
    )


if __name__ == "__main__":
    main()
