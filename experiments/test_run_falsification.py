from __future__ import annotations

import unittest
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.run_falsification import (
    Hypothesis,
    apply_real_dataset_gate,
    generate_episode,
    minimum_score_tie,
    parse_dataset_spec,
    score_hypothesis,
)


class ReferenceScorerTest(unittest.TestCase):
    def test_direct_cli_help_works_with_external_dependency_path(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "/tmp/actionabi-pydeps"

        completed = subprocess.run(
            [sys.executable, "experiments/run_falsification.py", "--help"],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def assert_true_hypothesis_wins(self, hypothesis: Hypothesis) -> None:
        commands = np.array(
            [
                [0.20, -0.10],
                [0.35, 0.40],
                [-0.15, 0.25],
                [0.50, -0.30],
            ],
            dtype=np.float64,
        )
        episode = generate_episode(
            hypothesis,
            semantic_commands=commands,
            initial_state=np.array([0.05, -0.02]),
            dt_seconds=0.1,
        )
        alternatives = [
            Hypothesis("absolute", (0, 1), (1, 1), (1.0, 1.0), 0),
            Hypothesis("delta", (0, 1), (1, 1), (1.0, 1.0), 0),
            Hypothesis("velocity", (0, 1), (1, 1), (1.0, 1.0), 0),
            Hypothesis("delta", (1, 0), (1, -1), (0.01, 2.0), 2),
            hypothesis,
        ]

        scores = {candidate: score_hypothesis([episode], candidate) for candidate in alternatives}
        tied = minimum_score_tie(scores, absolute_tolerance=1e-12, relative_tolerance=1e-9)

        self.assertIn(hypothesis, tied)
        self.assertAlmostEqual(scores[hypothesis], 0.0)

    def test_absolute_target_is_recovered(self) -> None:
        self.assert_true_hypothesis_wins(
            Hypothesis("absolute", (0, 1), (1, 1), (1.0, 1.0), 0)
        )

    def test_delta_target_is_recovered(self) -> None:
        self.assert_true_hypothesis_wins(
            Hypothesis("delta", (0, 1), (1, 1), (1.0, 1.0), 0)
        )

    def test_velocity_target_is_recovered(self) -> None:
        self.assert_true_hypothesis_wins(
            Hypothesis("velocity", (0, 1), (1, 1), (1.0, 1.0), 0)
        )

    def test_permutation_sign_scale_and_lag_are_recovered_together(self) -> None:
        self.assert_true_hypothesis_wins(
            Hypothesis("delta", (1, 0), (1, -1), (0.01, 2.0), 2)
        )

    def test_huber_loss_limits_a_single_outlier(self) -> None:
        hypothesis = Hypothesis("delta", (0,), (1,), (1.0,), 0)
        episode = generate_episode(
            hypothesis,
            semantic_commands=np.array([[0.1], [0.2], [0.3]]),
            initial_state=np.array([0.0]),
            dt_seconds=0.1,
        )
        corrupted_states = episode.states.copy()
        corrupted_states[-1, 0] += 1000.0
        corrupted = episode.with_states(corrupted_states)

        loss = score_hypothesis([corrupted], hypothesis, huber_delta=1.0)

        self.assertGreater(loss, 100.0)
        self.assertLess(loss, 400.0)

    def test_minimum_score_tie_never_uses_candidate_order_to_break_a_tie(self) -> None:
        first = Hypothesis("absolute", (0,), (1,), (1.0,), 0)
        second = Hypothesis("delta", (0,), (1,), (1.0,), 0)
        scores = {first: 1.0, second: 1.0 + 5e-7}

        tied = minimum_score_tie(
            scores,
            absolute_tolerance=1e-6,
            relative_tolerance=0.0,
        )

        self.assertEqual(tied, frozenset({first, second}))

    def test_dataset_spec_parses_name_and_path(self) -> None:
        name, path = parse_dataset_spec("pusht=/tmp/pusht.parquet")

        self.assertEqual(name, "pusht")
        self.assertEqual(str(path), "/tmp/pusht.parquet")

    def test_real_gate_requires_every_expected_outcome(self) -> None:
        expected = {
            "pusht": "unique_absolute",
            "aloha": "absolute_episode_relative_equivalence",
            "ur5": "partial_cartesian",
        }
        evaluated = {
            "pusht": {"outcome": "unique_absolute"},
            "aloha": {"outcome": "absolute_episode_relative_equivalence"},
            "ur5": {"outcome": "partial_cartesian"},
        }

        report = apply_real_dataset_gate(expected, evaluated)

        self.assertTrue(report["passed"])
        evaluated["ur5"] = {"outcome": "report_without_unique_requirement"}
        self.assertFalse(apply_real_dataset_gate(expected, evaluated)["passed"])


if __name__ == "__main__":
    unittest.main()
