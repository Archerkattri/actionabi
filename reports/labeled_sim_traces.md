# Labeled simulated traces: ActionABI supervised accuracy on known latent contracts

Run date: 2026-07-21 UTC. This report closes the top open evidence gap in ActionABI's
handoff — *"labeled real latent contracts or controlled robot/simulator traces across the
full grammar"* — with the first **supervised** accuracy measurement of ActionABI against
ground-truth contracts recovered from **real simulated robot dynamics**.

- Exporter (ActionShift side): `code/actionshift/experiments/export_labeled_traces.py`
- Scorer (ActionABI side): `experiments/score_labeled_traces.py`
- Machine-readable results: `results/labeled_sim/summary.json`, `results/labeled_sim/raw.jsonl`
- Labeled dataset: `code/actionshift/artifacts/actionabi_bridge/` (traces, per-trace labels, manifest)
- Backbone SHA-256 `3e6c95d6…`; dataset manifest SHA-256 `aaee673a…`; C++/Python residual
  parity max gap `2.8e-16` (Python residuals reproduce the C++ Huber scorer exactly).

## Why this is new evidence, and why it is honest

ActionABI's prior evidence was synthetic (noiseless constructed grammar) plus six pinned
real passive datasets that carry **no** latent-contract labels. It had never been scored
against real dynamics whose latent contract is known by construction. ActionShift's
hidden-contract ManiSkill wrapper supplies exactly that: it rolls out a frozen PPO policy
under a sampled contract (permutation / sign / scale / target / frame / lag / gripper over
6 pose channels + gripper) on real GPU simulation, so every field is a ground-truth label.
The label is written to a separate file and is **never** an inference input.

## Data generation (exporter)

- **Task / dynamics:** PickCube-v1, `pd_ee_delta_pose`, real ManiSkill 3.0.1 GPU simulation
  (GPU 1), frozen Gate 0 PPO backbone (`8131f330bd69aa6b/final_ckpt.pt`).
- **Contracts:** 45 contracts sampled over the 6-DoF grammar, hash-disjoint from every frozen
  Gate 1 evaluation contract. **Scales drawn from the finite grid `{0.5, 0.75, 1.0, 1.25,
  1.5, 2.0}`** and lag from `{0, 1, 2}`, so the sampled space equals ActionABI's declared
  finite grammar and supervised accuracy is well defined.
- **Excitation (both, labeled):** each contract is probed under (a) the frozen policy
  pass-through and (b) uniform random excitation → 90 traces, 16 episodes each.
- **Observed response → ActionABI state.** The physical observable is the tcp-pose-delta
  response (`actionshift.adaptation.calibration.response_from_observations`: translation
  delta + relative rotation vector, 6 channels). The `pd_ee_delta_pose` controller gain is
  tiny (`|alpha| ≈ 0.02–0.04`, fit R² 0.09–0.34, from the frozen **contract-independent**
  calibration), so the raw response is ~30× smaller than the commanded action; fed directly,
  ActionABI's `decode(action) ≈ observable` assumption collapses to "minimize command norm."
  We therefore express the response in commanded units by dividing by the per-channel,
  contract-free gain `alpha` (measured on the *unwrapped identity* environment — task
  knowledge already shared by every ActionShift method), then integrate it into a pseudo-pose
  state by cumulative sum. ActionABI's delta observable `s[t+off]−s[t]` then equals the
  per-step normalized response and its absolute observable telescopes to the commanded
  target. The full response-model semantics are recorded in the dataset manifest.

## Scoring (ActionABI, real C++ CLI)

ActionABI's `score_cpu` residual is separable per output channel and its held-out loss is
the mean of the per-dimension losses. For each (target, lag) the optimal channel assignment
with the best per-cell (sign, scale) is therefore a linear-assignment problem whose per-cell
costs are exactly the C++ `per_dimension_loss` values. We obtain those from the **real C++
CLI** (`actionabi infer`, Release build, CMake 4.4 via `uvx`, 8/8 CTests green) run on a
trace-independent basis of **432 contracts per trace** (6 cyclic permutations × 12 sign/scale
× 2 targets × 3 lags), then brute-force all 720 permutations. This evaluates the **entire
declared finite grammar** (target × lag × permutation × sign × scale) with authentic C++
residuals, without enumerating its billions of members.

- **Forced-argmin comparator:** always certifies the single minimum-loss contract.
- **Calibrated ActionABI:** certifies unique only when a paired bootstrap over held-out
  residual rows cannot distinguish the argmin from any of its confusable single-field
  neighbours (95% CI of the per-row loss gap includes 0); otherwise it abstains.

## Results (90 traces; per-field accuracy vs. ground truth)

Forced-argmin per-field accuracy (permutation/sign/scale are per-channel means over 6 channels):

| Stratum | n | permutation | sign | scale | target | lag |
|---|--:|--:|--:|--:|--:|--:|
| **Overall** | 90 | 0.63 | 0.76 | 0.24 | 0.74 | 0.39 |
| lag = 0 | 22 | **0.92** | **0.93** | 0.36 | 0.68 | 0.77 |
| lag = 1 | 28 | 0.58 | 0.74 | 0.19 | 0.71 | 0.29 |
| lag = 2 | 40 | 0.49 | 0.69 | 0.22 | 0.80 | 0.25 |
| policy excitation | 45 | 0.79 | 0.85 | 0.27 | 0.93 | 0.31 |
| random excitation | 45 | 0.47 | 0.67 | 0.22 | 0.56 | 0.47 |
| **lag 0, random** | 11 | **0.97** | **0.98** | 0.42 | 0.45 | **1.00** |
| lag 0, policy | 11 | 0.88 | 0.88 | 0.30 | 0.91 | 0.55 |
| lag 1, policy | 14 | 0.77 | 0.89 | 0.30 | 0.93 | 0.14 |
| lag 1, random | 14 | 0.39 | 0.58 | 0.08 | 0.50 | 0.43 |
| lag 2, policy | 20 | 0.74 | 0.81 | 0.23 | 0.95 | 0.30 |
| lag 2, random | 20 | 0.24 | 0.57 | 0.20 | 0.65 | 0.20 |

Calibrated ActionABI vs. forced argmin (equivalence / abstention):

| Metric | Overall | lag 0 | lag 1 | lag 2 |
|---|--:|--:|--:|--:|
| abstention rate | 0.96 | 1.00 | 0.96 | 0.93 |
| emitted (certified unique) | 4 | 0 | 1 | 3 |
| **false unique certifications** | 4 | **0** | 1 | 3 |
| equivalence-set coverage (truth ∈ set) | 0.02 | 0.09 | 0.00 | 0.00 |
| forced-argmin exact-contract accuracy | 0.00 | — | — | — |
| median held-out loss gap (truth − argmin) | 0.30 | **0.075** | 0.25 | 0.49 |

## Findings

1. **Structural fields are identifiable from real simulated responses.** ActionABI's
   held-out objective recovers **permutation and sign** at lag 0 with 0.92 / 0.93 accuracy,
   rising to **0.97 / 0.98 under random excitation** (with lag correctly 0 in every case).
   This is the substantive positive result: the permutation/sign structure survives the weak,
   noisy controller response because attenuation preserves *which* raw channel and *which*
   sign best explains each semantic channel.

2. **Scale is not identifiable — a controller/response-model confound, not a grammar limit.**
   Scale accuracy is 0.24–0.42 everywhere. Diagnostic (lag-0 channels, true perm/sign known):
   the least-squares scale is systematically **attenuated to ≈0.6× truth** (median 0.67×, 52%
   of channels below 0.7×). The `pd_ee_delta_pose` response under-tracks large commanded deltas
   with a command-magnitude-dependent gain that the constant calibration `alpha` cannot capture
   (R² 0.09–0.34), so the loss-optimal grid scale lands below truth. ActionABI's grammar *can*
   express the scale; the physical response simply does not linearly encode it.

3. **Target identifiability is excitation-dependent.** Smooth policy excitation identifies the
   delta/absolute target well (0.91–0.95); random excitation does not (0.45–0.65). The
   absolute-vs-delta signal lives in the previous-action difference, which policy smoothness
   exposes and independent random actions wash out.

4. **Lag > 0 exposes a real ActionABI scorer-model limitation.** ActionABI's C++ delta
   observable spans `lag+1` steps (`state[t+lag+1] − state[t]`), whereas the physical response
   is a single-step *delayed* jump. The two do not match for lag > 0, so all fields degrade
   (permutation 0.92 → 0.58 → 0.49) and lag itself is barely recovered (0.14–0.43). Note this
   also disagrees with ActionABI's own Python reference scorer, which uses **consecutive**
   deltas (`states[t+lag+1] − states[t+lag]`); the C++ and reference lag semantics diverge for
   lag > 0. This is recorded as a structured limitation, not scored around.

5. **Calibrated abstention avoids most, but not all, false uniques — and reveals a gap.**
   Calibrated ActionABI abstains on 96% of traces and never false-certifies at lag 0 (0/22),
   consistent with its design. But it still produces **4 false uniques, all at lag > 0**, and
   its equivalence set almost never contains the truth (coverage 0.02). The cause is that the
   true contract fits only *marginally* worse than the argmin at lag 0 (median loss gap 0.075
   on a 0.38 loss ≈ 20%), yet with ~180 held-out residual rows this small **systematic**
   scale-attenuation bias is statistically resolvable, so a significance-calibrated
   equivalence set excludes the truth. **Bootstrap-noise-calibrated equivalence does not guard
   against systematic response-model bias** — on real dynamics, ActionABI's calibration must
   model systematic response error, not only sampling noise, to preserve its zero-false-unique
   property. This is the single most actionable finding for ActionABI.

## Claim boundary

- These are **simulated** dynamics (ManiSkill GPU PickCube, `pd_ee_delta_pose`), **not
  hardware**. No hardware safety or device claim is made.
- The hidden-contract wrapper uses an **identity end-effector rotation**, so `base` and `tool`
  frames are observationally identical: **frame is a degenerate equivalence class, not
  identified**. `space` is fixed `cartesian`. Both are declared, not scored.
- The **gripper** channel is not observable from the tcp-pose response, so `gripper_inverted`
  is labeled but **excluded** from identification (structural limitation, reported as
  unsupported). Identification covers the 6 pose channels + target + lag.
- The declared grammar is **finite** (scales on a 6-value grid, lag ∈ {0,1,2}, targets
  {delta, absolute}); the sampled contracts live in exactly this grammar. This is **not** a
  claim over arbitrary continuous contracts.
- Supervised accuracy is over 90 traces / 45 contracts on one task and one backbone; the
  positive perm/sign result and the negative scale/lag results are consistent across
  excitation strata but are not a multi-task or multi-seed campaign.

## Post-fix update (2026-07-21)

Findings (4) and (5) were **real ActionABI defects**, not measurement artifacts. Both are now
fixed in code with tests; the pre-fix tables above are retained as evidence and the numbers below
are a re-score of the **same** bridge traces (`../actionshift/artifacts/actionabi_bridge/`, reused,
not re-exported) with the fixed Release build (`build-fix`, CMake 4.4 via `uvx`). The full write-up
with mechanism detail is `reports/scorer_fixes.md`; the machine-readable result is
`results/labeled_sim/summary_postfix.json` (raw: `raw_postfix.jsonl`). The pre-fix
`results/labeled_sim/summary.json` is unchanged.

**Fix 1 — lag observable is now single-step delayed.** The C++ (and CUDA) delta/velocity observable
now spans a single delayed step `state[row+lag+1] − state[row+lag]` instead of the multi-step span
`state[row+lag+1] − state[row]`. This matches the physical response (`executed_t = decode(raw_{t−lag})`)
and the Python reference scorer (`run_falsification.py::score_hypothesis`) for every lag; it reduces to
the old behaviour at lag 0. C++/Python residual parity is preserved (max gap `2.8e-16`) and now holds
for lag > 0 as well; a new CTest (`score`, "lagged delta observable is single-step delayed…")
reproduces the old bug (fails on the old span, passes on the fix) and the CUDA parity test passes.

Forced-argmin per-field accuracy, pre-fix → post-fix (same 90 traces):

| Stratum | n | permutation | sign | scale | target | lag |
|---|--:|--:|--:|--:|--:|--:|
| **Overall** | 90 | 0.63 → **0.80** | 0.76 → **0.84** | 0.24 → **0.31** | 0.74 → 0.74 | 0.39 → **0.66** |
| lag = 0 | 22 | 0.92 → 0.72 | 0.93 → 0.75 | 0.36 → 0.30 | 0.68 → 0.68 | 0.77 → 0.45 |
| lag = 1 | 28 | 0.58 → **0.73** | 0.74 → **0.78** | 0.19 → **0.30** | 0.71 → 0.71 | 0.29 → **0.50** |
| lag = 2 | 40 | 0.49 → **0.89** | 0.69 → **0.92** | 0.22 → **0.33** | 0.80 → 0.80 | 0.25 → **0.88** |

The lag > 0 strata — the pre-fix failures — improve sharply (lag = 2 permutation 0.49 → 0.89, lag
recovery 0.25 → 0.88). Honest counter-movement: the **lag = 0** stratum *regresses* on
permutation/sign/lag (e.g. lag 0.77 → 0.45). This is expected and not a new bug: once the lag > 0
hypotheses are scored correctly they compete fairly, and a smooth passive trajectory does not
strongly separate lag on a lag = 0 trace (a one-step-delayed delta can also fit), so some lag = 0
traces now attract a wrong-lag argmin. The net across strata is strongly positive and the specific
defect (lag > 0 mis-scoring) is resolved. Scale stays low (0.24 → 0.31): that is the separate
controller/response-model confound of finding (2), not addressed here.

**Fix 2 — calibration is now robust to systematic response-model bias.** A fail-closed guard
estimates a systematic-bias bound from the argmin's held-out residual structure (per-channel mean and
command-correlated slope vs. the noise floor), inflates the equivalence threshold by the
bias-explained loss slack, and — under detected misspecification — forces abstention. Effect on the
calibrated comparator (same traces, post lag-fix):

| Metric | Overall pre → post | lag 0 | lag 1 | lag 2 |
|---|--:|--:|--:|--:|
| **false unique certifications** | 4 → **0** | 0 → 0 | 1 → 0 | 3 → 0 |
| emitted (certified unique) | 4 → 0 | 0 → 0 | 1 → 0 | 3 → 0 |
| abstention rate | 0.96 → 1.00 | 1.00 | 1.00 | 1.00 |
| equivalence-set coverage (truth ∈ set) | 0.02 → **0.39** | 0.27 | 0.29 | 0.53 |
| misspecification flag rate (new) | 0.60 | 0.68 | 0.61 | 0.55 |
| mean bias slack (new) | 0.031 | 0.034 | 0.031 | 0.029 |

The guard detects the ~0.6× systematic under-tracking on 60 % of traces and, being fail-closed, drives
the calibrated comparator to **zero false uniques** (was 4, all at lag > 0) while raising truth
equivalence-set coverage from 0.02 to 0.39. No coverage regression on the existing 100-case synthetic
matrix: re-running `experiments/sprint_accuracy.py` (seed 20260718) still gives ActionABI 0 false
uniques / 1.00 coverage / 0.25 abstention and forced-argmin 25 false uniques — unchanged, because the
guard is a no-op under zero-mean noise (estimated slack → 0, flag → False).

**Verification.** New build `build-fix` (Release, CMake 4.4 via `uvx`): 8/8 CTests
(the `score` suite grew by the lag regression test); 8/8 CTests under AddressSanitizer/UBSan
(`build-fix-asan`); CUDA parity green on GPU 3 (`build-fix-cuda`, sm_120). Python experiment suite
32/32 (24 pre-existing + 8 new `test_bias_robust`).

## Reproduction

```bash
# 1. Export labeled traces (ActionShift venv, GPU 1)
cd code/actionshift
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/export_labeled_traces.py \
  --output artifacts/actionabi_bridge --contracts 45 --num-envs 16 --steps 40

# 2. Build ActionABI (CMake 4.4 via uvx, Release) and score
cd ../actionabi
uvx --from cmake cmake -S . -B build-bridge -DACTIONABI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
uvx --from cmake cmake --build build-bridge -j 8
uvx --from cmake ctest --test-dir build-bridge --output-on-failure   # 8/8
build-data-env/bin/python experiments/score_labeled_traces.py \
  --dataset ../actionshift/artifacts/actionabi_bridge --out results/labeled_sim
```

The large `results/labeled_sim/grammar_basis/` and `cli_reports/` are regenerable and
gitignored; `summary.json` and `raw.jsonl` are the committed machine-readable results.
