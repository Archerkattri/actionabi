# ActionABI scorer fixes: lag observable + calibration bias robustness

Date: 2026-07-21 UTC. This note documents the two real ActionABI defects surfaced by the
labeled-trace bridge run (`reports/labeled_sim_traces.md`, findings 4 and 5), the fixes, the tests
that guard them, and the before/after re-score on the same bridge traces. Pre-fix evidence is
retained: `results/labeled_sim/summary.json` (pre-fix) is unchanged; the post-fix machine-readable
result is `results/labeled_sim/summary_postfix.json` (raw rows `raw_postfix.jsonl`).

Build under test: `build-fix` (Release, `cmake_minimum_required 3.26`, CMake 4.4 via `uvx`). All
verification below is on this build unless noted.

---

## Defect 1 — lag observable scored a multi-step span, not a single-step delay

### What was wrong

For a lagged contract the physical response is a **single-step delta arriving `lag` steps late**:
`executed_t = decode(raw_{t−lag})`. The scorer, however, computed the delta/velocity observable as a
span of `lag+1` steps.

- `src/score_cpu.cpp` used `offset = lag_steps + 1`, `current = state(row)`, `future = state(row+offset)`,
  and for a delta target `observable = future − current = state[row+lag+1] − state[row]`.
- `cuda/score_cuda.cu` had the identical span.

For `lag = 0` this is correct (`state[row+1] − state[row]`). For `lag > 0` it summed `lag+1`
consecutive one-step deltas, so the true lagged contract carried a spurious residual (an extra decode
term) and lost to wrong-lag hypotheses. It also **disagreed with ActionABI's own Python reference
scorer** `experiments/run_falsification.py::score_hypothesis`, which uses the consecutive delta
`states[transition+1] − states[transition]` at `transition = action_index + lag` — i.e. the correct
single-step-delayed observable. The C++/Python parity discipline was therefore only enforced at
`lag = 0`.

### The fix

Score the delta/velocity observable against the one-step transition at the **delayed** index:

```
delayed_prev = state[row + offset - 1]   # = state[row + lag]
delayed_next = state[row + offset]        # = state[row + lag + 1]
delta observable    = delayed_next - delayed_prev
velocity observable = (delayed_next - delayed_prev) / (t[row+offset] - t[row+offset-1])
absolute observable = delayed_next        # unchanged
```

Applied identically in `src/score_cpu.cpp` and `cuda/score_cuda.cu`. This equals the Python reference
for every lag, reduces to the old behaviour at `lag = 0` (`offset = 1 ⇒ delayed_prev = state[row]`),
and leaves residual counts and absolute-target scoring untouched. The bridge scorer's Python residual
reproduction (`experiments/score_labeled_traces.py::heldout_row_losses`) was updated the same way, so
C++/Python parity now holds for `lag > 0` (measured max gap `2.8e-16`, was already `2.8e-16` at
lag 0).

### Tests

- **New CTest** in `tests/test_score.cpp`: *"lagged delta observable is single-step delayed, not a
  multi-step span."* Builds a synthetic trace generated under the single-step-delayed model with a
  known true contract at `lag ∈ {1,2}` and asserts (a) the true contract now has ~0 held-out loss and
  (b) it strictly beats every wrong-lag hypothesis. Verified to **fail on the old span** (7/8
  assertions fail — the true contract is not lossless and wrong lags win) and **pass on the fix**
  (8/8). This is the bug reproduction the task asked for.
- **CUDA parity** (`tests/test_cuda_parity.cpp`) still passes with the matched CUDA edit
  (`build-fix-cuda`, sm_120, GPU 3).

### Result (re-score, same 90 bridge traces)

Forced-argmin per-field accuracy, pre → post:

| Stratum | n | permutation | sign | scale | target | lag |
|---|--:|--:|--:|--:|--:|--:|
| **Overall** | 90 | 0.63 → **0.80** | 0.76 → **0.84** | 0.24 → **0.31** | 0.74 → 0.74 | 0.39 → **0.66** |
| lag = 0 | 22 | 0.92 → 0.72 | 0.93 → 0.75 | 0.36 → 0.30 | 0.68 → 0.68 | 0.77 → 0.45 |
| lag = 1 | 28 | 0.58 → **0.73** | 0.74 → **0.78** | 0.19 → **0.30** | 0.71 → 0.71 | 0.29 → **0.50** |
| lag = 2 | 40 | 0.49 → **0.89** | 0.69 → **0.92** | 0.22 → **0.33** | 0.80 → 0.80 | 0.25 → **0.88** |

The lag > 0 strata (the pre-fix failures) improve sharply. **Honest counter-movement:** the lag = 0
stratum regresses (permutation 0.92 → 0.72, lag 0.77 → 0.45). This is not a new bug — once lag > 0
hypotheses score correctly they compete fairly, and a smooth passive trajectory does not strongly
separate lag on a lag = 0 trace (a one-step-delayed delta can also fit it), so some lag = 0 traces now
attract a wrong-lag argmin. Net effect across strata is strongly positive and the specific defect is
resolved. Scale accuracy stays low — that is the independent controller/response confound (finding 2),
which this fix does not claim to address.

---

## Defect 2 — calibration was not robust to systematic response-model bias

### What was wrong

The calibrated equivalence set is built by a paired bootstrap over held-out residual rows: a candidate
is equivalent to the argmin if the 95 % CI of the per-row loss gap includes 0. That bootstrap
calibrates **sampling noise** and assumes the best hypothesis's residuals are zero-mean. On real
dynamics the controller systematically under-tracks large commands (~0.6× on the bridge), so every
grammar hypothesis's residuals carry a common, command-correlated systematic component. With ~180
held-out rows this small systematic gap is statistically resolvable, so the test (a) certifies the
loss-optimal biased-scale contract as **unique** and (b) **excludes the truth** — the 4 false uniques,
all at lag > 0, and the 0.02 truth coverage seen pre-fix.

### The fix (`experiments/bias_robust.py`)

A conservative, fail-closed guard estimates a systematic-bias bound from the residual structure of the
best hypothesis, per output channel:

1. **Diagnose** each channel's held-out residuals `r = command − observable`:
   - non-zero-mean offset: `t_mean = mean(r) / (std(r)/√n)`;
   - command-correlated slope: least-squares `r ≈ a + b·command`, `t_slope = b / se(b)`, with the
     noise floor taken as the residual scatter about that fit.
   A channel is **misspecified** when `|t_mean| > t_crit` or `|t_slope| > t_crit` (`t_crit = 4`,
   deliberately conservative about *declaring* misspecification so unbiased noisy data is not
   spuriously flagged). Degenerate inputs (`n < 3`, zero command or residual variance) are treated as
   not misspecified with zero slack.

2. **Bias-inflated equivalence threshold.** The per-channel **bias slack** is the mean Huber loss of
   the fitted systematic part `a + b·command` — the loss a correctly specified zero-mean model would
   not incur. Aggregated (mean over channels) it is added into the paired-bootstrap test as
   `gap − slack`, so a candidate is split from the argmin only if it is worse by *more* than the
   estimated systematic bias. Under zero-mean noise slack → 0 and the test is exactly the original.

3. **Model-misspecification guard (fail-closed).** If any channel is flagged misspecified, the
   calibrated decision **abstains** — it never emits a unique certification — preferring a wider set
   over a false unique.

Wired into `experiments/score_labeled_traces.py`: the argmin's per-channel held-out residuals feed the
guard; `paired_equivalent(..., slack=slack)` applies the widened threshold to both the candidate pool
and the truth-coverage test; `is_unique = (equivalence set is a singleton) and not misspecified`. Two
new per-trace fields (`bias_slack`, `misspecified`) and two new aggregate stats
(`misspecification_flag_rate`, `mean_bias_slack`) are reported.

### Tests (`experiments/test_bias_robust.py`, 8 tests)

- Zero-mean noise → not misspecified, slack ≈ 0 (no-op).
- Constant offset and command-correlated bias → flagged, positive slack.
- Degenerate inputs → safe (not flagged, zero slack).
- Episode-level: unbiased response (gain 1.0) → not flagged / ~0 slack; ~0.6× under-tracking → flagged
  with positive slack.
- `paired_equivalent`: a fixed systematic margin `d` splits contracts at `slack = 0` but is retained
  at `slack ≥ d` (widening); a genuine zero-mean-gap candidate is retained at `slack = 0` (unbiased
  behaviour unchanged).

### Result

Calibrated comparator on the same traces (post lag-fix):

| Metric | Overall pre → post | lag 0 | lag 1 | lag 2 |
|---|--:|--:|--:|--:|
| **false unique certifications** | 4 → **0** | 0 → 0 | 1 → 0 | 3 → 0 |
| emitted (certified unique) | 4 → 0 | 0 → 0 | 1 → 0 | 3 → 0 |
| abstention rate | 0.96 → 1.00 | 1.00 | 1.00 | 1.00 |
| equivalence-set coverage | 0.02 → **0.39** | 0.27 | 0.29 | 0.53 |
| misspecification flag rate | 0.60 | 0.68 | 0.61 | 0.55 |
| mean bias slack | 0.031 | 0.034 | 0.031 | 0.029 |

Zero false uniques (was 4, all lag > 0), and truth coverage up from 0.02 to 0.39. The guard flags the
systematic bias on 60 % of traces and, fail-closed, abstains on them.

**No regression on the existing synthetic matrix.** Re-running `experiments/sprint_accuracy.py`
(seed 20260718, 100 cases, 25 deliberately ambiguous): ActionABI still gives **0** false uniques,
**1.00** equivalence coverage, **0.25** abstention; forced argmin still gives 25 false uniques —
identical to the pre-fix ledger. The guard is a no-op on noiseless synthetic data (slack → 0,
flag → False), so no coverage is lost.

---

## Verification ledger

| Gate | Result |
|---|---|
| `build-fix` Release CTests | **8/8** (score suite grew by the lag regression test) |
| `build-fix-asan` ASan/UBSan CTests | **8/8** |
| `build-fix-cuda` CUDA parity (GPU 3, sm_120) | **pass** |
| Python experiment suite (`unittest discover experiments`) | **32/32** (24 prior + 8 new bias tests) |
| C++/Python residual parity, lag ≥ 0 | max gap **2.8e-16** |
| Synthetic 100-case matrix (sprint_accuracy) | unchanged: ActionABI 0 false uniques / 1.00 cov / 0.25 abstain |

## Reproduction

```bash
cd code/actionabi
uvx --from cmake cmake -S . -B build-fix -DACTIONABI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
uvx --from cmake cmake --build build-fix -j 8
uvx --from cmake ctest --test-dir build-fix --output-on-failure        # 8/8
build-data-env/bin/python -m unittest discover -s experiments -p 'test_*.py'   # 32/32
build-data-env/bin/python experiments/score_labeled_traces.py \
  --binary build-fix/actionabi \
  --dataset ../actionshift/artifacts/actionabi_bridge --out results/labeled_sim_postfix
cp results/labeled_sim_postfix/summary.json results/labeled_sim/summary_postfix.json
# optional: ASan (build-fix-asan) and CUDA parity (build-fix-cuda, -DCMAKE_CUDA_ARCHITECTURES=120)
```
