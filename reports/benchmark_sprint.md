# ActionABI real-data and scaling sprint

Run date: 2026-07-18/19 UTC  
Seed: `20260718`  
GPU: NVIDIA GeForce RTX 5090  
CPU: AMD Ryzen Threadripper PRO 7975WX 32-Cores  
Build: Release, GNU 12.2.0, float64 scorer

## Accuracy and calibration

The supervised matrix contains 100 independently generated finite-grammar cases. Twenty-five cases
have deliberately uninformative zero excitation, so multiple contracts are observationally
equivalent. Dataset documentation is never supplied to an inference method.

| Method | Field accuracy (target / lag) | Equivalence coverage | False unique | Abstention | Brier / ECE |
|---|---:|---:|---:|---:|---:|
| Metadata only | n/a / n/a | 0.00 | 0 | 1.00 | 0.00 / 0.00 |
| Forced residual argmin | 0.82 / 0.82 | 1.00 | 25 | 0.00 | 0.25 / 0.25 |
| ActionABI calibrated set | 1.00 / 1.00 | 1.00 | 0 | 0.25 | 0.00 / 0.00 |

The calibration target is a *valid unique certification*. ActionABI abstains on all 25 constructed
equivalence cases; forced argmin converts every one into a false unique answer. Accuracy for an
abstaining method is calculated only where it emitted the relevant field, while equivalence coverage
is calculated across every labeled case.

## Real passive evidence

| Frozen dataset | Episodes | Rows | Passive outcome |
|---|---:|---:|---|
| PushT | 206 | 25,650 | unique absolute within tested point grammar |
| ALOHA insertion | 50 | 20,000 | absolute / episode-relative equivalence |
| Berkeley UR5 | 1,000 | 97,939 | partial Cartesian translation evidence |
| DROID 100 | 100 | 32,212 | no unique requirement |
| Stanford HYDRA | 570 | 358,234 | no unique requirement |
| xArm lift | 800 | 20,000 | no unique requirement |

Every row count and local Parquet SHA-256 is retained in `results/sprint/accuracy/summary.json`.
These datasets do not label their latent controller contracts, so they are excluded from supervised
accuracy. The outcomes are held-out passive-transition evidence, not complete contract identification.
The loader was corrected during this sprint to hash and load only LeRobot `data/` Parquet shards,
excluding incompatible `meta/` tables.

## Performance matrix

Each cell uses five warmups and 30 measurements. Times are medians in milliseconds. `H` is both the
hypothesis count and CUDA batch axis; `R` is rows per episode across four episodes; `D` is action/state
dimension; `T` is the explicit multicore worker count.

| H | R | D | T | Single CPU | Multi CPU | CUDA incl. transfer | CUDA kernel | End-to-end speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 256 | 2 | 64 | 2.323 | 2.275 | 0.319 | 0.047 | 7.133x |
| 1,024 | 256 | 2 | 64 | 18.855 | 2.382 | 0.649 | 0.222 | 3.668x |
| 8,192 | 256 | 2 | 64 | 150.496 | 6.700 | 3.909 | 2.734 | 1.714x |
| 16,384 | 256 | 2 | 64 | 300.553 | 11.642 | 5.464 | 3.541 | 2.131x |
| 1,024 | 64 | 2 | 64 | 4.593 | 2.272 | 0.429 | 0.049 | 5.292x |
| 1,024 | 1,024 | 2 | 64 | 75.433 | 3.809 | 2.242 | 1.656 | 1.699x |
| 1,024 | 256 | 7 | 64 | 27.183 | 2.462 | 1.864 | 1.347 | 1.321x |
| 1,024 | 256 | 14 | 64 | 44.331 | 2.649 | 4.569 | 3.929 | 0.580x |
| 1,024 | 256 | 2 | 1 | 18.924 | 19.104 | 0.689 | 0.222 | 27.731x |
| 1,024 | 256 | 2 | 8 | 18.753 | 2.677 | 0.650 | 0.221 | 4.120x |

The result is workload-dependent. CUDA is useful for low-dimensional batches, but it is slower than
64-thread CPU at 14 dimensions. It therefore still **fails the preregistered 5x transfer-inclusive
promotion gate**. Kernel-only figures are diagnostic and are not substituted for end-to-end latency.
Float32 is structurally inapplicable because the public scorer is float64-only; the report does not
invent a dtype comparison.

## Claim verdict and limitations

- Supported: calibrated equivalence sets eliminate false unique certification in the constructed
  ambiguous cases while preserving full coverage in this finite synthetic grammar.
- Supported: the six pinned passive datasets produce reproducible, hash-bound evidence outcomes.
- Not supported: arbitrary-contract recovery, semantics outside the finite grammar, or unique labels
  for passive real logs.
- Failed gate: general CUDA promotion under the 5x transfer-inclusive criterion.
- Missing: labeled real latent contracts, active device probes under device-specific safety bounds,
  torque/controller internals, deployment shift, and clinical/device certification.

Raw append-only outputs are in `results/sprint/accuracy/raw.jsonl` and
`results/sprint/performance/raw.jsonl`; compact summaries are adjacent.
