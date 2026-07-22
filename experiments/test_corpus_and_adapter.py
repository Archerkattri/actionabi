from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "adapters" / "lerobot_to_jsonl.py"
CORPUS_PATH = ROOT / "experiments" / "corpus.yaml"


def load_adapter():
    spec = importlib.util.spec_from_file_location("lerobot_to_jsonl", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load adapter module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CorpusConfigTest(unittest.TestCase):
    def test_corpus_freezes_seed_and_expected_outcomes(self) -> None:
        config = yaml.safe_load(CORPUS_PATH.read_text(encoding="utf-8"))

        self.assertEqual(config["seed"], 20260718)
        self.assertEqual(config["datasets"]["synthetic"]["train_episodes"], 200)
        self.assertEqual(config["datasets"]["synthetic"]["test_episodes"], 100)
        self.assertEqual(config["datasets"]["pusht"]["expected"], "unique_absolute")
        self.assertEqual(
            config["datasets"]["aloha"]["expected"],
            "absolute_episode_relative_equivalence",
        )
        self.assertEqual(config["datasets"]["ur5"]["expected"], "partial_cartesian")


class AdapterRecordTest(unittest.TestCase):
    def test_metadata_preserves_source_provenance_and_state_semantics(self) -> None:
        adapter = load_adapter()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episode.parquet"
            source.write_bytes(b"frozen-source")

            metadata = adapter.build_metadata(
                source=source,
                state_columns=("joint_0", "joint_1"),
                state_units=("rad", "rad"),
                extraction_date=date(2026, 7, 18),
            )

        self.assertEqual(metadata["record_type"], "metadata")
        self.assertEqual(metadata["schema_version"], "1.0")
        self.assertEqual(metadata["source_filename"], "episode.parquet")
        self.assertEqual(
            metadata["source_sha256"], hashlib.sha256(b"frozen-source").hexdigest()
        )
        self.assertEqual(metadata["extraction_date"], "2026-07-18")
        self.assertEqual(metadata["state_columns"], ["joint_0", "joint_1"])
        self.assertEqual(metadata["state_units"], ["rad", "rad"])

    def test_sample_record_has_canonical_fields_and_nanosecond_timestamp(self) -> None:
        adapter = load_adapter()

        record = adapter.build_sample_record(
            episode_id=7,
            timestamp_seconds=0.125,
            state=[1.0, 2.0],
            action=[-0.5, 0.25],
        )

        self.assertEqual(
            record,
            {
                "record_type": "sample",
                "episode_id": 7,
                "t_ns": 125_000_000,
                "state": [1.0, 2.0],
                "action": [-0.5, 0.25],
            },
        )

    def test_jsonl_writer_places_exactly_one_metadata_record_first(self) -> None:
        adapter = load_adapter()
        metadata = {"record_type": "metadata", "schema_version": "1.0"}
        samples = [
            {
                "record_type": "sample",
                "episode_id": 0,
                "t_ns": 0,
                "state": [0.0],
                "action": [1.0],
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectory.jsonl"
            adapter.write_jsonl(output, metadata, samples)
            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(records, [metadata, *samples])

    def test_cli_help_does_not_require_pyarrow(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ADAPTER_PATH), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--input", completed.stdout)
        self.assertIn("--output", completed.stdout)
        self.assertIn("--state-key", completed.stdout)
        self.assertIn("--action-key", completed.stdout)


if __name__ == "__main__":
    unittest.main()
