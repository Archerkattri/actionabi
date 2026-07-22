from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ReleaseAssetsTest(unittest.TestCase):
    def test_required_release_files_exist_and_claim_is_bounded(self) -> None:
        for name in ("README.md", "LICENSE", "CITATION.cff"):
            self.assertTrue((ROOT / name).is_file(), name)
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        self.assertIn("equivalence set", readme)
        self.assertIn("does not", readme)
        self.assertIn("experimental", readme)
        # REPRODUCING.md was consolidated into README.md; assert the reproduction
        # guide and claim ledger survived the merge rather than a standalone file.
        self.assertIn("## reproducing", readme)
        self.assertIn("cpu build and falsification tests", readme)
        self.assertIn("do not claim (actionabi)", readme)

    def test_ci_has_two_cpu_compilers_and_opt_in_backends(self) -> None:
        workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        jobs = workflow["jobs"]
        self.assertEqual(set(jobs["cpu"]["strategy"]["matrix"]["compiler"]), {"gcc", "clang"})
        self.assertIn("if", jobs["cuda"])
        self.assertIn("if", jobs["pinocchio"])


if __name__ == "__main__":
    unittest.main()
