from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime.induction_data import detect_target_schema, read_jsonl, write_jsonl  # noqa: E402
from privacy_runtime.synthetic_counterfactual import (  # noqa: E402
    generate_synthetic_records,
    split_records_by_base_doc,
    summarize_synthetic_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build synthetic counterfactual policy-induction JSONL splits."
    )
    parser.add_argument(
        "--output-dir",
        default="data/induction/synthetic_counterfactual_v3_protected_key_value",
        help="Directory for train.jsonl, val.jsonl, test.jsonl, and metadata.json.",
    )
    parser.add_argument(
        "--polar-dir",
        default=None,
        help="Optional existing POLAR induction split dir to mix with synthetic records.",
    )
    parser.add_argument("--num-base-docs", type=int, default=250)
    parser.add_argument("--policies-per-doc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--target-schema",
        choices=["value", "key_value", "protected_key_value"],
        default="protected_key_value",
        help="Use value-only, scoring key-value, or runtime protected-only key-value targets.",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=["full_allowed", "protected_only", "sparse_allowed"],
        default="protected_only",
        help=(
            "Synthetic target style. full_allowed keeps v1 behavior; "
            "protected_only sets allowed_values to []; sparse_allowed keeps only "
            "a few task-relevant allowed values."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    polar_dir = Path(args.polar_dir) if args.polar_dir else None
    if polar_dir and output_dir.resolve() == polar_dir.resolve():
        raise ValueError("--output-dir must differ from --polar-dir to avoid overwriting POLAR splits")

    synthetic_records = generate_synthetic_records(
        num_base_docs=args.num_base_docs,
        policies_per_doc=args.policies_per_doc,
        seed=args.seed,
        target_schema=args.target_schema,
        synthetic_mode=args.synthetic_mode,
    )
    synthetic_splits = split_records_by_base_doc(
        synthetic_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_splits = {split: list(records) for split, records in synthetic_splits.items()}
    polar_splits: dict[str, int] = {}
    if polar_dir:
        for split in ("train", "val", "test"):
            polar_records = read_jsonl(polar_dir / f"{split}.jsonl")
            if polar_records:
                polar_schema = detect_target_schema(polar_records[0].get("target", {}))
                if args.target_schema != polar_schema:
                    raise ValueError(
                        f"--target-schema {args.target_schema} cannot mix with "
                        f"a POLAR split using {polar_schema}. Rebuild POLAR with "
                        "scripts/build_polar_induction_dataset.py using the same "
                        "--target-schema."
                    )
            polar_splits[split] = len(polar_records)
            output_splits[split] = polar_records + output_splits[split]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in output_splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)

    metadata = summarize_synthetic_splits(
        synthetic_splits,
        seed=args.seed,
        num_base_docs=args.num_base_docs,
        policies_per_doc=args.policies_per_doc,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        mixed_with_polar=polar_dir is not None,
        polar_dir=str(polar_dir) if polar_dir else None,
        polar_splits=polar_splits,
        target_schema=args.target_schema,
        synthetic_mode=args.synthetic_mode,
    )
    metadata["synthetic_splits"] = metadata.pop("splits")
    metadata["splits"] = {split: len(records) for split, records in output_splits.items()}

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
