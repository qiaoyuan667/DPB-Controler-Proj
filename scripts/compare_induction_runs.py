from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple induction evaluation summary JSON files."
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        help="Summary JSON files produced by eval_induction_predictions.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def safe_get(summary: dict[str, Any], *keys: str) -> Any:
    value: Any = summary
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def run_name(path: str) -> str:
    name = Path(path).name
    suffix = ".jsonl.summary.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    if name.endswith(".summary.json"):
        return name[: -len(".summary.json")]
    return Path(path).stem


def load_row(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return {
        "run": run_name(path),
        "num_examples": safe_get(summary, "num_examples"),
        "parse_rate": safe_get(summary, "parse_rate"),
        "schema_valid_rate": safe_get(summary, "schema_valid_rate"),
        "protected_precision": safe_get(
            summary,
            "do_not_disclose_values",
            "precision",
        ),
        "protected_recall": safe_get(
            summary,
            "do_not_disclose_values",
            "recall",
        ),
        "protected_f1": safe_get(summary, "do_not_disclose_values", "f1"),
        "protected_exact": safe_get(
            summary,
            "do_not_disclose_values",
            "exact_match_rate",
        ),
        "allowed_precision": safe_get(summary, "allowed_values", "precision"),
        "allowed_recall": safe_get(summary, "allowed_values", "recall"),
        "allowed_f1": safe_get(summary, "allowed_values", "f1"),
        "allowed_exact": safe_get(summary, "allowed_values", "exact_match_rate"),
        "exact_set_match": safe_get(summary, "exact_set_match_rate"),
        "path": path,
    }


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def print_markdown_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run",
        "num_examples",
        "parse_rate",
        "schema_valid_rate",
        "protected_recall",
        "protected_f1",
        "allowed_recall",
        "allowed_f1",
        "exact_set_match",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    import csv

    columns = [
        "run",
        "num_examples",
        "parse_rate",
        "schema_valid_rate",
        "protected_precision",
        "protected_recall",
        "protected_f1",
        "protected_exact",
        "allowed_precision",
        "allowed_recall",
        "allowed_f1",
        "allowed_exact",
        "exact_set_match",
        "path",
    ]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> None:
    args = parse_args()
    rows = [load_row(path) for path in args.summaries]
    print_markdown_table(rows)
    if args.output:
        write_csv(args.output, rows)


if __name__ == "__main__":
    main()

