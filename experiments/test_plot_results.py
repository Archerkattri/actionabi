from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class PlotResultsTest(unittest.TestCase):
    def test_benchmark_summary_applies_cuda_headline_gate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            source = temporary / "benchmark.json"
            output = temporary / "summary.md"
            source.write_text(
                json.dumps(
                    {
                        "gpu": "test GPU",
                        "cpu": "test CPU",
                        "compiler": "test compiler",
                        "workloads": [
                            {
                                "hypotheses": 128,
                                "residual_evaluations": 10_000_000,
                                "speedup_over_multicore_cpu": 4.9,
                                "multicore_cpu_ms": {"median": 4.0},
                                "cuda_transfer_inclusive_ms": {"median": 1.0},
                                "cuda_kernel_only_ms": {"median": 0.5},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python",
                    root / "experiments" / "plot_results.py",
                    "benchmark",
                    "--input",
                    source,
                    "--output",
                    output,
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            rendered = output.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("experimental", rendered)
        self.assertIn("4.90", rendered)
        self.assertNotIn("headline gate: passed", rendered)

    def test_case_summary_preserves_unsupported_fields(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            source = temporary / "aloha.json"
            output = temporary / "cases.md"
            source.write_text(
                json.dumps(
                    {
                        "dataset": "aloha",
                        "outcome": "absolute_episode_relative_equivalence",
                        "episodes": 50,
                        "rows": 20_000,
                        "matched_expected_outcome": True,
                        "equivalence_fields": {"target": "ambiguous"},
                        "converter_status": "blocked",
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python",
                    root / "experiments" / "plot_results.py",
                    "cases",
                    "--input",
                    source,
                    "--output",
                    output,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            rendered = output.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("aloha", rendered)
        self.assertIn("ambiguous", rendered)
        self.assertIn("blocked", rendered)


if __name__ == "__main__":
    unittest.main()
