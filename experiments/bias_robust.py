#!/usr/bin/env python3
"""Robustness of ActionABI's calibrated equivalence sets to systematic
response-model bias.

Motivation
----------
ActionABI certifies a contract *unique* only when a paired bootstrap over held-out
residual rows cannot distinguish the loss-argmin from its confusable neighbours.
That bootstrap calibrates against **sampling noise** and implicitly assumes the
best hypothesis's held-out residuals are zero-mean. On real dynamics that
assumption fails: the ``pd_ee_delta_pose`` controller systematically *under-tracks*
large commands (the labeled-trace bridge measured a command-magnitude-dependent
gain of ~0.6x). This systematic component is shared by every finite-grammar
hypothesis, so a significance-calibrated test resolves it as if it were signal and

  (a) certifies the loss-optimal (biased-scale) contract as UNIQUE, and
  (b) excludes the TRUE contract from the equivalence set.

Both are failures the bridge report flagged (finding 5): "bootstrap-noise
calibration does not guard against systematic response-model bias."

Mechanism (fail-closed)
-----------------------
We estimate a systematic-bias bound from the RESIDUAL STRUCTURE of the best
hypothesis, per output channel, and use it two ways:

1. **Bias-inflated equivalence threshold.** A candidate is equivalent to the
   argmin if it is not worse by more than a *bias slack* --- the per-row loss
   attributable to the estimated systematic (non-noise) residual component of the
   argmin. Concretely, for each channel we regress the argmin residual on the
   decoded command, ``r ~= a + b * command``; the fitted part ``a + b*command`` is
   the systematic component a correctly specified zero-mean model would not incur.
   The slack is the mean Huber loss of that fitted part, averaged over channels.
   The paired-bootstrap acceptance test is applied to ``gap - slack`` instead of
   ``gap``, so systematic-bias-sized loss gaps no longer split contracts apart.

2. **Model-misspecification guard.** We flag misspecification when the argmin's
   residuals show *statistically resolvable* systematic structure --- a non-zero
   mean (``|t_mean| > t_crit``) or a command-correlated slope (``|t_slope| >
   t_crit``). Under a detected flag the calibrated decision **abstains**: it never
   emits a unique certification, preferring a wider set. This is fail-closed: on
   misspecified dynamics we would rather abstain than certify a false unique.

Under genuine zero-mean noise both the slack and the flag collapse to their no-op
values (slack -> ~0, flag -> False), so unbiased behaviour is unchanged and no
coverage is lost. Degenerate inputs (n < 3, zero command variance, zero residual
variance) are treated as *not misspecified* with zero slack.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_HUBER_DELTA = 1.0
# Two-sided significance threshold on the t-like statistics. 4.0 (~ p<1e-4 for
# large n) is deliberately conservative about DECLARING misspecification so that
# genuinely unbiased noisy data is not spuriously flagged into abstention; a real
# controller bias like the bridge's ~0.6x under-tracking clears it by a wide
# margin.
_T_CRIT = 4.0


def _huber(residual: np.ndarray, delta: float = _HUBER_DELTA) -> np.ndarray:
    magnitude = np.abs(residual)
    return np.where(magnitude <= delta, 0.5 * residual**2,
                    delta * (magnitude - 0.5 * delta))


def channel_bias_diagnostic(residuals: np.ndarray, commands: np.ndarray, *,
                            t_crit: float = _T_CRIT,
                            huber_delta: float = _HUBER_DELTA) -> dict[str, float | bool]:
    """Diagnose systematic bias in one channel's held-out residuals.

    ``residuals`` are ``command - observable`` for the BEST hypothesis on that
    channel; ``commands`` are the corresponding decoded command values. Returns a
    per-channel diagnostic including the systematic-bias loss bound
    (``bias_loss``) and a ``misspecified`` flag.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    commands = np.asarray(commands, dtype=np.float64)
    n = int(residuals.size)
    zero = {
        "n": n, "mean_residual": 0.0, "t_mean": 0.0, "slope": 0.0,
        "t_slope": 0.0, "noise_std": 0.0, "bias_loss": 0.0, "misspecified": False,
    }
    if n < 3:
        return zero

    mean_residual = float(residuals.mean())
    resid_std = float(residuals.std(ddof=1))
    # Non-zero-mean test (constant systematic offset).
    t_mean = mean_residual / (resid_std / np.sqrt(n)) if resid_std > 0.0 else 0.0

    # Command-correlated test (magnitude-dependent systematic bias).
    xbar = float(commands.mean())
    ybar = mean_residual
    sxx = float(((commands - xbar) ** 2).sum())
    if sxx > 0.0:
        slope = float(((commands - xbar) * (residuals - ybar)).sum() / sxx)
        intercept = ybar - slope * xbar
        fitted = intercept + slope * commands
        about_fit = residuals - fitted
        dof = n - 2
        s2 = float((about_fit ** 2).sum() / dof) if dof > 0 else 0.0
        se_slope = np.sqrt(s2 / sxx) if s2 > 0.0 else 0.0
        t_slope = slope / se_slope if se_slope > 0.0 else 0.0
        noise_std = float(np.sqrt(s2))
    else:
        # No command variation: only the constant offset is identifiable.
        slope, intercept, t_slope = 0.0, ybar, 0.0
        fitted = np.full_like(residuals, ybar)
        noise_std = resid_std

    # Systematic-bias loss bound: the Huber loss of the fitted (non-noise) part.
    bias_loss = float(_huber(fitted, huber_delta).mean())
    misspecified = bool(abs(t_mean) > t_crit or abs(t_slope) > t_crit)
    return {
        "n": n, "mean_residual": mean_residual, "t_mean": float(t_mean),
        "slope": slope, "t_slope": float(t_slope), "noise_std": noise_std,
        "bias_loss": bias_loss, "misspecified": misspecified,
    }


def bias_guard(channel_residuals: Sequence[np.ndarray],
               channel_commands: Sequence[np.ndarray], *,
               t_crit: float = _T_CRIT,
               huber_delta: float = _HUBER_DELTA) -> dict[str, object]:
    """Aggregate per-channel diagnostics into an equivalence-set bias guard.

    Returns ``slack`` (per-row-equivalent, mean-over-channels systematic-bias loss
    to inflate the equivalence threshold by) and ``misspecified`` (True if ANY
    channel shows resolvable systematic structure -> fail-closed abstention).
    """
    diagnostics = [
        channel_bias_diagnostic(r, c, t_crit=t_crit, huber_delta=huber_delta)
        for r, c in zip(channel_residuals, channel_commands)
    ]
    if not diagnostics:
        return {"slack": 0.0, "misspecified": False, "channels": []}
    slack = float(np.mean([d["bias_loss"] for d in diagnostics]))
    misspecified = any(d["misspecified"] for d in diagnostics)
    return {"slack": slack, "misspecified": misspecified, "channels": diagnostics}
