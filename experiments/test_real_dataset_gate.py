from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from experiments.real_dataset_gate import (
    PassiveDataset,
    _parquet_sources,
    classify_dataset,
    rank_point_hypotheses,
)


def test_lerobot_directory_excludes_metadata_parquet(tmp_path: Path) -> None:
    data_file = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    metadata_file = tmp_path / "meta" / "tasks.parquet"
    data_file.parent.mkdir(parents=True)
    metadata_file.parent.mkdir(parents=True)
    data_file.touch()
    metadata_file.touch()

    assert _parquet_sources(tmp_path) == [data_file]


def point_dataset(*, episode_relative: bool = False) -> PassiveDataset:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[int] = []
    timestamps: list[float] = []
    for episode in range(10):
        initial = np.array([0.25, -0.5]) if episode_relative else np.array(
            [0.03 * episode, -0.02 * episode]
        )
        targets = np.array(
            [[0.1, -0.2], [0.4, 0.3], [-0.3, 0.2], [0.2, -0.4]], dtype=np.float64
        )
        trajectory = np.vstack([initial, targets])
        states.extend(trajectory)
        actions.extend(np.vstack([targets, targets[-1]]))
        episodes.extend([episode] * len(trajectory))
        timestamps.extend(np.arange(len(trajectory), dtype=float) * 0.1)
    return PassiveDataset(
        states=np.asarray(states),
        actions=np.asarray(actions),
        episode_ids=np.asarray(episodes),
        timestamps_seconds=np.asarray(timestamps),
    )


class PointHypothesisGateTest(unittest.TestCase):
    def test_varying_resets_make_absolute_target_unique(self) -> None:
        dataset = point_dataset(episode_relative=False)

        ranked = rank_point_hypotheses(dataset, max_lag=2)
        outcome = classify_dataset("pusht", dataset, ranked)

        self.assertEqual(ranked[0].target, "absolute")
        self.assertEqual(outcome["outcome"], "unique_absolute")

    def test_fixed_resets_preserve_absolute_episode_relative_equivalence(self) -> None:
        dataset = point_dataset(episode_relative=True)

        ranked = rank_point_hypotheses(dataset, max_lag=2)
        outcome = classify_dataset("aloha", dataset, ranked)

        absolute = min(result.nrmse for result in ranked if result.target == "absolute")
        relative = min(
            result.nrmse for result in ranked if result.target == "episode_relative"
        )
        self.assertAlmostEqual(absolute, relative, places=12)
        self.assertEqual(
            outcome["outcome"], "absolute_episode_relative_equivalence"
        )

    def test_delta_control_with_fixed_resets_is_not_false_equivalence(self) -> None:
        # Regression for the hub-audit gate defect (observed on berkeley_rpt): a delta-controlled
        # dataset with near-constant resets makes absolute ~ episode_relative (intercept absorbs the
        # static pose) even though DELTA fits far better. The equivalence branch must not fire and
        # exclude the strictly-better delta target -- it must fall through to a non-unique report.
        rng = np.random.default_rng(7)
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        episodes: list[int] = []
        timestamps: list[float] = []
        for episode in range(12):
            state = np.array([0.5, -0.5])  # fixed reset across every episode
            deltas = rng.uniform(-0.02, 0.02, size=(5, 2))
            traj = [state.copy()]
            for delta in deltas:
                state = state + delta  # true delta (lag-1) control
                traj.append(state.copy())
            states.extend(traj)
            actions.extend(np.vstack([deltas, deltas[-1]]))
            episodes.extend([episode] * len(traj))
            timestamps.extend(np.arange(len(traj), dtype=float) * 0.1)
        dataset = PassiveDataset(
            states=np.asarray(states),
            actions=np.asarray(actions),
            episode_ids=np.asarray(episodes),
            timestamps_seconds=np.asarray(timestamps),
        )

        ranked = rank_point_hypotheses(dataset, max_lag=2)
        outcome = classify_dataset("aloha", dataset, ranked)  # name must not force equivalence

        best = {
            target: min(r.nrmse for r in ranked if r.target == target)
            for target in ("absolute", "delta", "episode_relative")
        }
        self.assertLess(best["delta"], best["absolute"])  # delta genuinely fits better
        self.assertNotEqual(outcome["outcome"], "absolute_episode_relative_equivalence")

    def test_dimension_mismatch_with_predictable_translation_is_partial_cartesian(self) -> None:
        episode_ids = np.repeat(np.arange(10), 6)
        timestamps = np.tile(np.arange(6) * 0.1, 10)
        actions = np.zeros((60, 7), dtype=np.float64)
        states = np.zeros((60, 8), dtype=np.float64)
        states[:, 6] = 1.0  # xyzw identity quaternion
        for episode in range(10):
            start = episode * 6
            commands = np.array(
                [[0.02, 0.01, -0.01], [0.01, -0.02, 0.03], [-0.01, 0.01, 0.02],
                 [0.03, -0.01, 0.01], [0.0, 0.02, -0.02]]
            )
            actions[start : start + 5, :3] = commands
            for step, command in enumerate(commands):
                states[start + step + 1, :3] = states[start + step, :3] + command
        dataset = PassiveDataset(states, actions, episode_ids, timestamps)

        outcome = classify_dataset("ur5", dataset, [])

        self.assertEqual(outcome["outcome"], "partial_cartesian")
        self.assertLess(outcome["cartesian_translation_nrmse"], 1e-10)

    def test_tool_frame_translation_is_recognized_across_changing_orientations(self) -> None:
        episode_ids = np.repeat(np.arange(10), 6)
        timestamps = np.tile(np.arange(6) * 0.1, 10)
        actions = np.zeros((60, 7), dtype=np.float64)
        states = np.zeros((60, 8), dtype=np.float64)
        commands = np.array(
            [[0.02, 0.01, -0.01], [0.01, -0.02, 0.03], [-0.01, 0.01, 0.02],
             [0.03, -0.01, 0.01], [0.0, 0.02, -0.02]]
        )
        for episode in range(10):
            start = episode * 6
            angle = episode * np.pi / 5
            quaternion = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle), 0.0],
                 [np.sin(angle), np.cos(angle), 0.0],
                 [0.0, 0.0, 1.0]]
            )
            states[start : start + 6, 3:7] = quaternion
            actions[start : start + 5, :3] = commands
            for step, command in enumerate(commands):
                states[start + step + 1, :3] = (
                    states[start + step, :3] + rotation @ command
                )
        dataset = PassiveDataset(states, actions, episode_ids, timestamps)

        outcome = classify_dataset("ur5", dataset, [])

        self.assertEqual(outcome["outcome"], "partial_cartesian")
        self.assertEqual(outcome["cartesian_translation_frame"], "tool_xyzw")
        self.assertLess(outcome["cartesian_translation_nrmse"], 1e-10)


if __name__ == "__main__":
    unittest.main()
