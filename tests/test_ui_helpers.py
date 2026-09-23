"""Tests for starter prompts and checked-figure CSV export."""
import csv
from io import StringIO
import unittest

from ui import SAMPLE_STARTERS, checked_figures_csv, starter_prompts


class UiHelpers(unittest.TestCase):
    def test_sample_starters_are_checked_figure_questions(self):
        prompts = starter_prompts(sample=True)
        self.assertEqual(prompts, SAMPLE_STARTERS)
        self.assertTrue(any("free cash flow" in p for p in prompts))
        self.assertTrue(any("return on equity" in p for p in prompts))
        self.assertTrue(any("current ratio" in p for p in prompts))
        self.assertTrue(any("growth" in p for p in prompts))

    def test_checked_figures_csv_exports_verified_rows(self):
        history = [{
            "q": "What was revenue in fiscal 2025?",
            "a": "Revenue …",
            "sources": [],
            "research": {
                "verification": {
                    "status": "verified",
                    "facts": [{
                        "metric": "revenue", "year": 2025, "value": "416161000000",
                        "currency": "USD", "scale": "millions", "raw_value": "416,161",
                        "row": "Total net sales 416,161 391,035", "source_id": "S1", "page": 32,
                    }],
                    "calculations": [],
                }
            },
        }, {
            "q": "Why did revenue rise?",
            "a": "…",
            "sources": [],
            "research": {"verification": {"status": "not_checked", "facts": []}},
        }]
        rows = list(csv.reader(StringIO(checked_figures_csv(history))))
        self.assertEqual(rows[0][0], "question")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][2], "revenue")
        self.assertEqual(rows[1][4], "416161000000")


if __name__ == "__main__":
    unittest.main()
