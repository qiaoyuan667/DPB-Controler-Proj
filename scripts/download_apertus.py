from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_OUTPUT_DIR = "models/apertus-8b-instruct-2509"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Apertus 8B assets for local hard-mask experiments."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Download only tokenizer/config assets for token-mask inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "This script requires huggingface_hub. Install with: "
            "pip install -r requirements-hf.txt"
        ) from exc

    output_dir = Path(args.output_dir)
    allow_patterns = None
    if args.tokenizer_only:
        allow_patterns = [
            "tokenizer*",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "chat_template*",
            "config.json",
            "*.model",
            "*.tiktoken",
        ]

    print(f"Downloading {args.model_id} -> {output_dir}")
    if args.tokenizer_only:
        print("Mode: tokenizer/config assets only")

    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
