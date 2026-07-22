#!/usr/bin/env python3
"""Unit tests for the systematic-response-bias robustness guard.

Covers the two required regimes:
  * biased traces  -> misspecification detected, positive slack, NO false unique;
  * unbiased traces -> not flagged, ~zero slack, calibration unchanged (no
    coverage loss).
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from experiments.bias_robust import bias_guard, channel_bias_diagnostic
    from experiments.score_labeled_traces import (
        heldout_channel_residuals,
        paired_equivalent,
    )
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from bias_robust import bias_guard, channel_bias_diagnostic
    from score_labeled_traces import heldout_channel_residuals, paired_equivalent


_DIM = 6


def _build_episodes(gain: float, *, seed: int, episodes: int = 6, rows: int = 40,
                    noise: float = 0.0):
    """Episodes under an identity true contract (delta, lag 0), where the
    measured response is ``gain * command`` (gain 1.0 = unbiased, 0.6 = the
    bridge's systematic under-tracking). Optional zero-mean state noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        action = rng.uniform(-0.5, 0.5, size=(rows, _DIM))
        state = np.zeros((rows, _DIM))
        for row in range(rows - 1):
            increment = gain * action[row]
            if noise > 0.0:
                increment = increment + rng.normal(0.0, noise, size=_DIM)
            state[row + 1] = state[row] + increment
        out.append({"state": state, "action": action})
    return out


def _contract(scale: float):
    return {
        "target": "delta_position",
        "lag": 0,
        "permutation": tuple(range(_DIM)),
        "sign": (1,) * _DIM,
        "scale": (scale,) * _DIM,
    }


class ChannelDiagnosticTests(unittest.TestCase):
    def test_zero_mean_noise_is_not_misspecified(self):
        rng = np.random.default_rng(0)
        residuals = rng.normal(0.0, 0.05, size=400)
        commands = rng.uniform(-0.5, 0.5, size=400)
        diagnostic = channel_bias_diagnostic(residuals, commands)
        self.assertFalse(diagnostic["misspecified"])
        self.assertLess(diagnostic["bias_loss"], 1e-3)

    def test_constant_offset_is_flagged(self):
        rng = np.random.default_rng(1)
        commands = rng.uniform(-0.5, 0.5, size=400)
        residuals = 0.3 + rng.normal(0.0, 0.02, size=400)  # non-zero mean
        diagnostic = channel_bias_diagnostic(residuals, commands)
        self.assertTrue(diagnostic["misspecified"])
        self.assertGreater(abs(diagnostic["t_mean"]), 4.0)
        self.assertGreater(diagnostic["bias_loss"], 0.0)

    def test_command_correlated_bias_is_flagged(self):
        rng = np.random.default_rng(2)
        commands = rng.uniform(-0.5, 0.5, size=400)
        residuals = -0.1 * commands + rng.normal(0.0, 0.01, size=400)
        diagnostic = channel_bias_diagnostic(residuals, commands)
        self.assertTrue(diagnostic["misspecified"])
        self.assertGreater(abs(diagnostic["t_slope"]), 4.0)
        self.assertGreater(diagnostic["bias_loss"], 0.0)

    def test_degenerate_inputs_are_safe(self):
        diagnostic = channel_bias_diagnostic(np.zeros(2), np.zeros(2))
        self.assertFalse(diagnostic["misspecified"])
        self.assertEqual(diagnostic["bias_loss"], 0.0)


class EpisodeGuardTests(unittest.TestCase):
    def test_unbiased_response_not_flagged_zero_slack(self):
        episodes = _build_episodes(gain=1.0, seed=10, noise=0.01)
        residuals, commands = heldout_channel_residuals(episodes, _contract(1.0))
        guard = bias_guard(residuals, commands)
        self.assertFalse(guard["misspecified"])
        self.assertLess(guard["slack"], 1e-3)

    def test_biased_response_is_flagged_positive_slack(self):
        # 0.6x under-tracking; the grid-best contract is scale 0.5, leaving a
        # command-correlated residual -> systematic structure the guard must see.
        episodes = _build_episodes(gain=0.6, seed=11, noise=0.01)
        residuals, commands = heldout_channel_residuals(episodes, _contract(0.5))
        guard = bias_guard(residuals, commands)
        self.assertTrue(guard["misspecified"])
        self.assertGreater(guard["slack"], 0.0)


class EquivalenceWideningTests(unittest.TestCase):
    def test_slack_prevents_a_systematic_bias_split(self):
        # Candidate is uniformly worse by a fixed systematic margin d.
        rng = np.random.default_rng(3)
        keys = [(0, r) for r in range(200)]
        argmin = {k: float(v) for k, v in zip(keys, rng.uniform(0.1, 0.2, size=200))}
        d = 0.05
        candidate = {k: argmin[k] + d for k in keys}
        # No slack: the systematic gap makes them look distinguishable (split).
        self.assertFalse(
            paired_equivalent(argmin, candidate, rng=np.random.default_rng(4), slack=0.0)
        )
        # Bias slack >= d: the candidate is retained (fail-closed widening).
        self.assertTrue(
            paired_equivalent(argmin, candidate, rng=np.random.default_rng(4), slack=d)
        )

    def test_slack_zero_recovers_noise_only_behavior(self):
        rng = np.random.default_rng(5)
        keys = [(0, r) for r in range(200)]
        argmin = {k: float(v) for k, v in zip(keys, rng.uniform(0.1, 0.2, size=200))}
        # A candidate that is genuinely equivalent (zero-mean gap) stays in.
        candidate = {k: argmin[k] + rng.normal(0.0, 0.01) for k in keys}
        self.assertTrue(
            paired_equivalent(argmin, candidate, rng=np.random.default_rng(6), slack=0.0)
        )


if __name__ == "__main__":
    unittest.main()
