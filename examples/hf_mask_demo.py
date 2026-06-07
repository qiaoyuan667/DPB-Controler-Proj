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
    HuggingFaceVocabulary,
    PrivacyLogitProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect which HuggingFace tokenizer ids are blocked."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Tokenizer name or local model path, e.g. gpt2 or ./models/llama.",
    )
    parser.add_argument(
        "--secret",
        action="append",
        required=True,
        help="Forbidden string. Repeat this flag for aliases.",
    )
    parser.add_argument(
        "--prefix",
        default="Our maximum budget is ",
        help="Already generated text used to compute next-token masks.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download tokenizer files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    vocabulary = HuggingFaceVocabulary.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
    )
    processor = PrivacyLogitProcessor(
        vocabulary=vocabulary,
        constraints=(ForbiddenStringConstraint(tuple(args.secret)),),
    )
    decision = processor.blocked_tokens(args.prefix)

    print(f"blocked {len(decision.blocked_token_ids)} token ids")
    for token_id in sorted(decision.blocked_token_ids)[:40]:
        token_text = vocabulary.token_text(token_id)
        print(f"{token_id}\t{token_text!r}\t{decision.reasons[token_id]}")


if __name__ == "__main__":
    main()
