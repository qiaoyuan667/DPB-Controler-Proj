from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_runtime.induction_data import (  # noqa: E402
    evaluate_prediction,
    extract_json_object,
    pretty_json_dumps,
    read_jsonl,
    summarize_evaluations,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate minimal POLAR induction predictions."
    )
    parser.add_argument("--predictions", required=True, help="Predictions JSONL path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional summary JSON path. Defaults to <predictions>.summary.json.",
    )
    parser.add_argument(
        "--details-output",
        default=None,
        help="Optional detailed JSONL path. Defaults to <predictions>.details.jsonl.",
    )
    return parser.parse_args()


def _gold_from_row(row: dict) -> dict:
    if isinstance(row.get("gold"), dict):
        return row["gold"]
    if isinstance(row.get("target"), dict):
        return row["target"]
    raise ValueError(f"prediction row {row.get('sample_id', '<unknown>')} has no gold/target object")


def _prediction_from_row(row: dict) -> tuple[dict | None, str]:
    if isinstance(row.get("prediction"), dict):
        return row["prediction"], ""
    prediction_text = str(row.get("prediction_text") or row.get("output") or "")
    return extract_json_object(prediction_text)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    details = []
    for row in rows:
        prediction, parse_error = _prediction_from_row(row)
        metrics = evaluate_prediction(_gold_from_row(row), prediction)
        details.append(
            {
                "sample_id": row.get("sample_id"),
                "domain": row.get("domain"),
                "parsed": prediction is not None,
                "parse_error": parse_error,
                "metrics": metrics,
            }
        )

    summary = summarize_evaluations(details)
    summary["predictions"] = str(args.predictions)

    summary_path = Path(args.output or f"{args.predictions}.summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    details_path = Path(args.details_output or f"{args.predictions}.details.jsonl")
    write_jsonl(details_path, details)

    print(pretty_json_dumps(summary))


if __name__ == "__main__":
    main()

