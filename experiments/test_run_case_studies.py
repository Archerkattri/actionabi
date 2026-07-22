from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.run_case_studies import assemble_case_report, write_manifest


class CaseStudyReportTest(unittest.TestCase):
    def test_ambiguous_outcome_does_not_invent_coverage_or_converter(self) -> None:
        report = assemble_case_report(
            "aloha",
            {
                "outcome": "absolute_episode_relative_equivalence",
                "best": [{"target": "absolute", "nrmse": 0.01}],
                "source_sha256": "a" * 64,
                "rows": 100,
                "episodes": 10,
            },
            runtime_seconds=1.25,
        )

        self.assertEqual(report["equivalence_fields"]["target"], "ambiguous")
        self.assertIsNone(report["empirical_coverage"])
        self.assertEqual(report["converter_status"], "blocked")
        self.assertEqual(report["probe_utility"]["status"], "candidate_required")
        self.assertIn("missing sensor modality", report["evidence_assessment"])

    def test_manifest_is_stable_when_runtime_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            base = {"dataset": "pusht", "source_sha256": "b" * 64}
            write_manifest(first, [{**base, "runtime_seconds": 1.0}])
            write_manifest(second, [{**base, "runtime_seconds": 9.0}])

            first_manifest = (first / "manifest.json").read_text(encoding="utf-8")
            second_manifest = (second / "manifest.json").read_text(encoding="utf-8")

        self.assertEqual(first_manifest, second_manifest)


if __name__ == "__main__":
    unittest.main()
