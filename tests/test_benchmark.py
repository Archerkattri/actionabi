from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class BenchmarkContractTest(unittest.TestCase):
    def test_smoke_benchmark_emits_transfer_inclusive_comparison(self) -> None:
        root = Path(__file__).resolve().parents[1]
        binary = root / "build-cuda" / "score_benchmark"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark.json"
            completed = subprocess.run(
                [
                    binary,
                    "--output",
                    output,
                    "--hypotheses",
                    "8",
                    "--total-evaluations",
                    "1024",
                    "--warmups",
                    "1",
                    "--measurements",
                    "2",
                    "--dimension",
                    "4",
                    "--rows-per-episode",
                    "16",
                    "--cpu-threads",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["workloads"][0]["hypotheses"], 8)
        self.assertEqual(report["workloads"][0]["measurements"], 2)
        self.assertEqual(report["workloads"][0]["dimension"], 4)
        self.assertEqual(report["workloads"][0]["rows_per_episode"], 16)
        self.assertEqual(report["workloads"][0]["cpu_threads"], 2)
        self.assertIn("multicore_cpu_ms", report["workloads"][0])
        self.assertIn("single_thread_cpu_ms", report["workloads"][0])
        self.assertIn("cuda_transfer_inclusive_ms", report["workloads"][0])
        self.assertIn("cuda_kernel_only_ms", report["workloads"][0])
        self.assertIn("speedup_over_multicore_cpu", report["workloads"][0])
        self.assertIn("compiler", report)
        self.assertIn("cpu", report)
        self.assertIn("estimated_peak_device_bytes", report["workloads"][0])
        self.assertEqual(len(report["workloads"][0]["multicore_cpu_samples_ms"]), 2)
        self.assertEqual(len(report["workloads"][0]["cuda_transfer_samples_ms"]), 2)


if __name__ == "__main__":
    unittest.main()
