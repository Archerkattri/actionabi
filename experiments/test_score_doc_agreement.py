from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.score_doc_agreement import (
    DOCUMENTATION_LABELS,
    actionabi_target_readout,
    aggregate,
    build_report,
    score_dataset,
    score_field,
)


class TargetReadoutTest(unittest.TestCase):
    def test_outcomes_map_to_status_and_retained_values(self) -> None:
        self.assertEqual(
            actionabi_target_readout("unique_absolute"),
            {"status": "identified", "values": ["absolute"]},
        )
        self.assertEqual(
            actionabi_target_readout("absolute_episode_relative_equivalence")["status"],
            "ambiguous",
        )
        self.assertIn(
            "absolute",
            actionabi_target_readout("absolute_episode_relative_equivalence")["values"],
        )
        self.assertEqual(
            actionabi_target_readout("partial_cartesian")["status"], "partially_identified"
        )
        self.assertEqual(
            actionabi_target_readout("report_without_unique_requirement")["status"],
            "unsupported",
        )


class ScoreFieldTest(unittest.TestCase):
    def test_unlabeled_is_not_scored(self) -> None:
        self.assertEqual(score_field("scale", None, {"status": "unsupported"}), "unlabeled")

    def test_identified_agreement_and_contradiction(self) -> None:
        self.assertEqual(
            score_field("target", "absolute", {"status": "identified", "values": ["absolute"]}),
            "agreement",
        )
        self.assertEqual(
            score_field("target", "delta", {"status": "identified", "values": ["absolute"]}),
            "contradiction",
        )

    def test_ambiguous_set_containing_truth_is_consistent_not_contradiction(self) -> None:
        self.assertEqual(
            score_field(
                "target", "absolute",
                {"status": "ambiguous", "values": ["absolute", "episode_relative"]},
            ),
            "equivalence_consistent",
        )
        # A truth EXCLUDED from the equivalence set is a contradiction (false exclusion).
        self.assertEqual(
            score_field(
                "target", "velocity",
                {"status": "ambiguous", "values": ["absolute", "episode_relative"]},
            ),
            "contradiction",
        )

    def test_abstention_on_documented_field_is_consistent(self) -> None:
        self.assertEqual(
            score_field("frame", "base", {"status": "unsupported", "values": []}),
            "abstention_consistent",
        )

    def test_partial_frame_discrepancy_is_flagged_not_certified(self) -> None:
        # UR5-style: partial cartesian, best-fit tool frame, documented world -> discrepant.
        self.assertEqual(
            score_field(
                "frame", "world",
                {"status": "partially_identified", "values": [], "best_fit": "tool"},
            ),
            "partial_discrepant",
        )
        # Same partial identification but compatible base/world direction -> consistent.
        self.assertEqual(
            score_field(
                "frame", "base",
                {"status": "partially_identified", "values": [], "best_fit": "world"},
            ),
            "partial_consistent",
        )


class EndToEndTest(unittest.TestCase):
    def _write_results(self, root: Path) -> None:
        fixtures = {
            "pusht": "unique_absolute",
            "aloha": "absolute_episode_relative_equivalence",
            "ur5": "partial_cartesian",
            "droid": "report_without_unique_requirement",
            "hydra": "report_without_unique_requirement",
            "xarm": "report_without_unique_requirement",
        }
        for dataset, outcome in fixtures.items():
            equivalence = {
                key: ("partially_identified" if (outcome == "partial_cartesian" and key == "frame") else "unsupported")
                for key in ("target", "permutation", "sign", "scale", "lag", "frame", "gripper")
            }
            (root / f"{dataset}.json").write_text(
                json.dumps({"outcome": outcome, "equivalence_fields": equivalence}),
                encoding="utf-8",
            )

    def test_no_contradiction_and_one_correct_unique_certification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_results(root)
            report = build_report(root)
            agg = report["aggregate"]
            # ActionABI never contradicts documentation across the six real datasets.
            self.assertEqual(agg["contradictions"], 0)
            # Exactly one unique field certification (PushT target=absolute), and it is correct.
            self.assertEqual(agg["unique_field_certifications"], 1)
            self.assertEqual(agg["unique_certifications_correct"], 1)
            self.assertEqual(agg["false_unique_certifications"], 0)
            # PushT target is a clean agreement; ALOHA target retains truth as equivalence.
            self.assertEqual(
                report["datasets"]["pusht"]["fields"]["target"]["verdict"], "agreement"
            )
            self.assertEqual(
                report["datasets"]["aloha"]["fields"]["target"]["verdict"],
                "equivalence_consistent",
            )
            # UR5 frame is a flagged partial discrepancy, not a contradiction.
            self.assertEqual(
                report["datasets"]["ur5"]["fields"]["frame"]["verdict"], "partial_discrepant"
            )

    def test_every_label_has_source_and_quote(self) -> None:
        for dataset, labels in DOCUMENTATION_LABELS.items():
            for field in ("target", "frame", "permutation", "sign", "gripper"):
                entry = labels[field]
                self.assertIn("url", entry, f"{dataset}.{field} missing url")
                self.assertIn("quote", entry, f"{dataset}.{field} missing quote")

    def test_aggregate_counts_only_scored_labels(self) -> None:
        scored = {
            "x": {
                "hf_repo": "r",
                "outcome": "unique_absolute",
                "fields": {
                    "target": {"verdict": "agreement"},
                    "frame": {"verdict": "unlabeled"},
                    "gripper": {"verdict": "not_applicable"},
                },
            }
        }
        agg = aggregate(scored)
        self.assertEqual(agg["scored_field_labels"], 1)
        self.assertEqual(agg["verdict_counts"]["agreement"], 1)


if __name__ == "__main__":
    unittest.main()
