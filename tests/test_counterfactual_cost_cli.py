from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.counterfactual_cost_hf import _build_rankings, load_candidate_suite


class CounterfactualCostCLITests(unittest.TestCase):
    def test_load_candidate_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "candidate_id": "leak",
                            "category": "direct_leak",
                            "expected_leak": True,
                            "text": "The protected value is 18.",
                        },
                        {
                            "candidate_id": "safe",
                            "category": "safe_utility",
                            "expected_leak": False,
                            "text": "The application is ready.",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            specs = load_candidate_suite(path)

        self.assertEqual([spec.candidate_id for spec in specs], ["leak", "safe"])
        self.assertEqual([spec.expected_leak for spec in specs], [True, False])

    def test_rankings_sort_by_raw_cost(self) -> None:
        entries = [
            {
                "candidate_id": "safe",
                "category": "safe_utility",
                "expected_leak": False,
                "result": {
                    "costs": {"age": 0.2},
                    "normalized_costs": {"age": 0.02},
                },
            },
            {
                "candidate_id": "leak",
                "category": "direct_leak",
                "expected_leak": True,
                "result": {
                    "costs": {"age": 2.0},
                    "normalized_costs": {"age": 0.1},
                },
            },
        ]

        rankings = _build_rankings(entries)

        self.assertEqual(
            [row["candidate_id"] for row in rankings["age"]],
            ["leak", "safe"],
        )


if __name__ == "__main__":
    unittest.main()
