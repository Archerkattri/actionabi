from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "adapters" / "lerobot_to_jsonl.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("lerobot_to_jsonl", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load adapter module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AdapterDirectoryIntegrationTest(unittest.TestCase):
    def test_directory_shard_converts_to_canonical_jsonl(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("PyArrow is installed through the experiments extra")

        adapter = load_adapter()
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "data" / "chunk-000" / "file-000.parquet"
            shard.parent.mkdir(parents=True)
            parquet.write_table(
                pa.table(
                    {
                        "observation.state": [[0.1, 0.2], [0.3, 0.4]],
                        "action": [[1.0], [2.0]],
                        "timestamp": [0.0, 0.1],
                        "episode_index": [3, 3],
                        "video": [b"unused-0", b"unused-1"],
                    }
                ),
                shard,
            )

            output = Path(directory) / "trajectory.jsonl"
            self.assertEqual(
                adapter.main(
                    [
                        "--input",
                        directory,
                        "--output",
                        str(output),
                        "--extraction-date",
                        "2026-10-12",
                    ]
                ),
                0,
            )
            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["record_type"], "metadata")
        self.assertEqual(records[0]["state_columns"], ["state_0", "state_1"])
        self.assertEqual(records[1]["state"], [0.1, 0.2])
        self.assertEqual(records[2]["action"], [2.0])

if __name__ == "__main__":
    unittest.main()
