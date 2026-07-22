from __future__ import annotations

import re
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class CliContractTest(unittest.TestCase):
    def test_version_reports_semver_and_git_revision(self) -> None:
        binary = Path(__file__).resolve().parents[1] / "build" / "actionabi"

        completed = subprocess.run(
            [binary, "--version"], check=False, capture_output=True, text=True
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertRegex(
            completed.stdout.strip(),
            re.compile(r"^ActionABI 0\.1\.0 \(git (?:[0-9a-f]{7,40}|unknown)\)$"),
        )

    def test_infer_writes_an_evidence_report_for_supplied_hypotheses(self) -> None:
        binary = Path(__file__).resolve().parents[1] / "build" / "actionabi"
        metadata = {
            "record_type": "metadata",
            "schema_version": "1.0",
            "source_filename": "source.parquet",
            "source_sha256": "a" * 64,
            "extraction_date": "2026-07-18",
            "state_columns": ["q0"],
            "state_units": ["rad"],
        }
        contract = {
            "schema_version": "1.0",
            "target": "delta_position",
            "space": "joint",
            "frame": "unspecified",
            "permutation": [0],
            "sign": [1],
            "scale": [1.0],
            "lag_steps": 0,
            "gripper_inverted": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "trajectory.jsonl"
            records = [metadata]
            for episode in range(2):
                records.extend(
                    [
                        {"record_type": "sample", "episode_id": episode, "t_ns": 0,
                         "state": [0.0], "action": [0.1]},
                        {"record_type": "sample", "episode_id": episode, "t_ns": 1_000_000_000,
                         "state": [0.1], "action": [0.0]},
                    ]
                )
            trajectory.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            output = root / "report.json"

            completed = subprocess.run(
                [binary, "infer", "--input", trajectory, "--contract", contract_path,
                 "--output", output],
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(report["backend"], "cpu")
        self.assertEqual(report["ranked_hypotheses"][0]["score"]["heldout_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
