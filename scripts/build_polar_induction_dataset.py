from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime.induction_data import (  # noqa: E402
    DEFAULT_REPAIRED_POLAR_PATH,
    build_induction_record,
    filter_p1_samples,
    load_polar_repaired,
    split_records_by_domain,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build minimal POLAR P1 policy-induction train/val/test JSONL files."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_REPAIRED_POLAR_PATH,
        help="Path to privacy_benchmark_rendered_repaired.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/induction/p1_scoring_targets",
        help="Directory for train.jsonl, val.jsonl, test.jsonl, and metadata.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_polar_repaired(args.input)
    p1_samples = filter_p1_samples(samples)
    records = [build_induction_record(sample) for sample in p1_samples]
    splits = split_records_by_domain(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records)

    metadata = {
        "input": str(args.input),
        "filter": {
            "metadata.privacy_level": 1,
            "metadata.privacy_type": "explicit_field_constraints",
        },
        "target_schema": {
            "scoring_targets": {
                "allowed_values": "list[str]",
                "do_not_disclose_values": "list[str]",
            }
        },
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "total_samples": len(samples),
        "p1_samples": len(records),
        "splits": {key: len(value) for key, value in splits.items()},
        "domains": {
            split: dict(Counter(record["domain"] for record in split_records))
            for split, split_records in splits.items()
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

