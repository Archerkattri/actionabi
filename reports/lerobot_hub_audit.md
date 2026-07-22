# LeRobot Hub convention audit — recovering action conventions from trajectory evidence

Run date: 2026-07-21 UTC. CPU-only (threads capped at 4).
Machine-readable results: `results/hub_audit.json` (+ `results/hub/<dataset>.json` per dataset).
Baseline it extends: `reports/documentation_agreement.md` (the six pinned datasets, 27 field-labels).

## What this is

ActionABI is an evidence-first auditor for undocumented robot action tensors: given timestamped
trajectories and a finite declared grammar, it reports which action-contract fields the passive
evidence **identifies**, which it can only **retain as an equivalence set**, and which it must
**abstain** on — and it never forces a unique answer the data does not support.

This report points ActionABI at **35 popular LeRobot-format datasets** on the Hugging Face Hub
(6 pinned baseline + **29 newly audited**), spanning six ecosystems: Open X-Embodiment (OXE)
ports, native ALOHA (ACT), SO-100/101 leader-follower teleop, LIBERO, MetaWorld, and the
gym PushT/xArm datasets. For each dataset we (1) recover the action convention from trajectory
evidence with **zero documentation in scope**, (2) collect the **documented** convention from the
dataset's pinned config and the upstream spec (URL + verbatim quote), and (3) score agreement
per field. The payload is the **flagged inconsistencies** and the **cross-ecosystem convention
clusters**.

## Standing discipline (unchanged, test-enforced)

**Documentation is a scoring label only, never an inference input.** The passive evidence gate
(`experiments/real_dataset_gate.evaluate_dataset`) and the C++ scorer (`build-fix/actionabi infer`)
run with no access to any documentation; the scorer (`experiments/run_hub_audit.py`, reusing the
unit-tested `score_field` core) only *compares* frozen outcomes to labels. This is the same
discipline as the 27-label baseline and is guarded by `experiments/test_score_doc_agreement.py`.

## Method

- **Selection & pinning.** Datasets were chosen from the `lerobot` org for ecosystem diversity and
  card/config documentation. Every dataset is pinned to an immutable commit SHA (recorded in
  `results/hub_audit.json`; short SHAs in the table below). The 6 pinned baseline revisions match
  `results/hub_audit.json` exactly.
- **Disk.** Only trajectory Parquet shards were pulled (video is separate); metadata gives feature
  specs for free. **Total new download = 383 MB** across 35 datasets — far under the 60 GB cap. Large
  many-shard datasets used a bounded per-dataset subset (≤48 shards / ≤210 MB; the gate needs only a
  few hundred episodes). CPU-only, `max_workers=8` downloads, `OMP_NUM_THREADS=4`.
- **Adapter.** `adapters/lerobot_to_jsonl.py` was extended **additively** to accept a LeRobot v3.0
  dataset directory (globbing `data/**/*.parquet`, excluding `meta/`), to project only the four
  trajectory columns (never loading inline image/video blobs), and to hash a multi-shard source
  order-stably. Existing adapter tests still pass unchanged.
- **Two engines.** The Python passive gate produces the audited **outcome** (identical engine to the
  baseline). On square (point-target) datasets the **FIXED C++ CLI** (`build-fix`, the
  lag-observable + bias-robust build from `reports/scorer_fixes.md`) independently scores
  {absolute, delta, velocity} declared contracts as corroboration (its min-held-out-loss target is
  reported as `CLI min-loss`).
- **Two documentation layers, recorded per dataset.** (a) the literal `action.names` in the pinned
  `meta/info.json`; (b) the upstream convention (OXE/RLDS TFDS catalog, ACT/ALOHA, LIBERO,
  MetaWorld, gym-xarm) with a verbatim quote. Full quotes + URLs live in
  `experiments/hub_documentation.py` and `results/hub_audit.json`.

## Audit table

Verdict vocabulary: `agreement` (unique identification = label), `equivalence_consistent`
(equivalence set contains the label — truth retained, not resolved), `partial_consistent`
(partial direction consistent), `partial_discrepant` (partial best-fit disagrees — **flagged**,
not a certification), `abstention_consistent` (honest abstention on a documented field),
`contradiction` (identified/retained set excludes the label — a false certification), `unlabeled`.

### Baseline (6 pinned datasets, from reports/documentation_agreement.md)

| Dataset (repo@rev) | Eco | dim a/s | Documented target | ActionABI recovered | Target verdict |
|---|---|---|---|---|---|
| lerobot/pusht@7628202 | gym-pusht | 2/2 | absolute 2D EE (world) | unique_absolute | agreement |
| lerobot/aloha_sim_insertion_scripted@8ab6609 | ALOHA | 14/14 | absolute 14-DoF joint | absolute/episode_relative equivalence | equivalence_consistent |
| lerobot/berkeley_autolab_ur5@c4e26a6 | OXE-port | 7/8 | delta EE cartesian (world_vector) | partial_cartesian (frame tool@0.14) | partial_consistent; frame partial_discrepant |
| lerobot/droid_100@87301a2 | OXE/DROID | 7/7 | 6-DoF EE velocity (base) | report (no unique req.) | abstention_consistent |
| lerobot/stanford_hydra_dataset@ff06383 | OXE-port | 7/8 | delta EE (3 pos+3 orient) | report (no unique req.) | abstention_consistent |
| lerobot/xarm_lift_medium@79efb0e | gym-xarm | 4/4 | delta EE displacement + gripper | report (no unique req.) | abstention_consistent |

### New Hub audit (29 datasets)

| Dataset (repo@rev) | Eco | dim a/s | Documented target (config names) | ActionABI outcome | CLI min-loss | Target verdict |
|---|---|---|---|---|---|---|
| lerobot/aloha_mobile_cabinet@7a752b3 | ALOHA | 14/14 | absolute (semantic) | absolute episode relative equivalence | absolute | equivalence_consistent |
| lerobot/aloha_sim_insertion_human@cc571a3 | ALOHA | 14/14 | absolute (semantic) | absolute episode relative equivalence | absolute | equivalence_consistent |
| lerobot/aloha_sim_transfer_cube_human@6a43d50 | ALOHA | 14/14 | absolute (semantic) | absolute episode relative equivalence | absolute | equivalence_consistent |
| lerobot/aloha_static_battery@06dc3da | ALOHA | 14/14 | absolute (motor_N) | report without unique requirement | absolute | abstention_consistent |
| lerobot/aloha_static_coffee@b144896 | ALOHA | 14/14 | absolute (semantic) | report without unique requirement | absolute | abstention_consistent |
| lerobot/aloha_static_cups_open@d793c96 | ALOHA | 14/14 | absolute (semantic) | report without unique requirement | absolute | abstention_consistent |
| lerobot/libero@a1aaacb | LIBERO | 7/8 | delta (opaque) | partial cartesian | — | partial_consistent |
| lerobot/libero_10@551d7d8 | LIBERO | 7/8 | delta (semantic) | partial cartesian | — | partial_consistent |
| lerobot/metaworld_mt50@a59f742 | MetaWorld | 4/4 | delta (semantic) | report without unique requirement | delta | abstention_consistent |
| lerobot/austin_buds_dataset@d9f9289 | OXE-port | 7/24 | delta (motor_N) | report without unique requirement | — | abstention_consistent |
| lerobot/berkeley_cable_routing@20a7774 | OXE-port | 7/8 | velocity (motor_N) | report without unique requirement | — | abstention_consistent |
| lerobot/berkeley_fanuc_manipulation@51bdefb | OXE-port | 7/8 | delta (motor_N) | report without unique requirement | — | abstention_consistent |
| lerobot/berkeley_mvp@076135b | OXE-port | 8/15 | delta (motor_N) | report without unique requirement | — | abstention_consistent |
| lerobot/berkeley_rpt@692eff9 | OXE-port | 8/8 | delta (motor_N) | report without unique requirement | delta | abstention_consistent |
| lerobot/jaco_play@265773f | OXE-port | 7/8 | delta (motor_N) | partial cartesian | — | partial_consistent |
| lerobot/nyu_door_opening_surprising_effectiveness@4ae8986 | OXE-port | 7/8 | velocity (motor_N) | partial cartesian | — | partial_consistent |
| lerobot/nyu_franka_play_dataset@3b2367d | OXE-port | 15/13 | delta (motor_N) | partial cartesian | — | partial_consistent |
| lerobot/roboturk@d38c919 | OXE-port | 7/8 | delta (motor_N) | partial cartesian | — | partial_consistent |
| lerobot/stanford_kuka_multimodal_dataset@93927d1 | OXE-port | 7/7 | None (motor_N) | report without unique requirement | delta | unlabeled |
| lerobot/taco_play@15f69c9 | OXE-port | 7/7 | absolute (motor_N) | report without unique requirement | delta | abstention_consistent |
| lerobot/toto@51f5a8d | OXE-port | 7/8 | delta (motor_N) | report without unique requirement | — | abstention_consistent |
| lerobot/ucsd_kitchen_dataset@fd7751e | OXE-port | 8/21 | None (motor_N) | report without unique requirement | — | unlabeled |
| lerobot/ucsd_pick_and_place_dataset@5af7ee8 | OXE-port | 4/7 | velocity (motor_N) | report without unique requirement | — | abstention_consistent |
| lerobot/utaustin_mutex@a880eae | OXE-port | 7/8 | delta (motor_N) | report without unique requirement | — | abstention_consistent |
| lerobot/svla_so100_pickplace@728583b | SO-100/101 | 6/6 | absolute (semantic) | unique absolute | absolute | agreement |
| lerobot/svla_so100_sorting@13870ca | SO-100/101 | 6/6 | absolute (semantic) | unique absolute | absolute | agreement |
| lerobot/svla_so101_pickplace@f641879 | SO-100/101 | 6/6 | absolute (semantic) | unique absolute | absolute | agreement |
| lerobot/pusht-subtask@184ff45 | gym-pusht | 2/2 | absolute (motor_N) | unique absolute | absolute | agreement |
| lerobot/xarm_push_medium@50352b1 | gym-xarm | 3/4 | delta (motor_N) | partial cartesian | — | partial_consistent |


**Config-name column:** `semantic` = the pinned config documents the action channels (ALOHA joint
names, SO-100/101 `.pos`, LIBERO_10 `x,y,z,roll,pitch,yaw,gripper`, MetaWorld `x,y,z,gripper`);
`motor_N` = LeRobot v3.0 stripped the semantic OXE/RLDS names to generic `motor_0..N` (convention
erased at config level); `opaque` = a single `actions` name.

## Headline

Over **139 documented field-labels** across 35 datasets (27 baseline + 112 new):

| Metric | Baseline (6) | New (29) | Combined (35) |
|---|---:|---:|---:|
| Contradictions (ActionABI asserts a convention documentation refutes) | 0 | **0** | **0** |
| Unique field certifications | 1 | 4 | 5 |
| — documentation-correct | 1 | 4 | **5 / 5** |
| — false unique certifications | 0 | **0** | **0** |
| Equivalence-consistent (truth retained) | 1 | 3 | 4 |
| Partial-consistent | 1 | 11 | 12 |
| Flagged partial discrepancies (frame) | 1 | 2 | 3 |
| Abstention-consistent | 23 | 92 | 115 |
| Unlabeled / not-applicable | 3 | 33 | 36 |

**Zero contradictions with documentation on any field of any of the 35 datasets; all 5 unique
real-data certifications are documentation-correct; zero false unique certifications.** Every field
where ActionABI abstained is one documentation confirms it *could* have claimed — the abstentions
are honest, not vacuous. This reproduces, at ~6× the dataset count, the abstention property proven
synthetically (`reports/benchmark_sprint.md`) and on the pinned six.

Note this required a genuine **ActionABI defect fix** (below): a *pre-fix* run produced **one
contradiction** (berkeley_rpt), root-caused to a gate bug — not a documentation error — and fixed.

## Flagged inconsistencies — per-case analysis

### FLAG 0 (fixed defect) — berkeley_rpt: gate mislabeled a delta dataset as absolute/episode-relative

The first full run produced a **contradiction** on `lerobot/berkeley_rpt` (8-DoF, square): documented
target **delta** ("[7 delta joint pos, 1x gripper binary state]", TFDS), but the gate emitted
`absolute_episode_relative_equivalence`, whose retained set {absolute, episode_relative} **excludes**
delta.

Investigation (is it doc / adapter / ActionABI?): **ActionABI's own evidence says delta.** The
point-hypothesis held-out fits are delta = velocity = **0.353** nRMSE vs absolute **0.926**,
episode_relative **0.923** — delta fits *far* better. The FIXED C++ scorer independently ranks
**delta 0.030 < velocity 0.047 < absolute 0.521**. So the documentation is correct *and* matches
ActionABI's best fit — the contradiction was an **ActionABI gate defect**, not a doc error.

Root cause: `classify_dataset`'s equivalence branch fired on the coincidence
`absolute ≈ episode_relative` (a near-static, near-constant-reset joint trajectory makes those two
fit *each other* because the affine intercept absorbs the static pose) **without checking that
absolute was actually the best-fit target**. It therefore certified a 2-way set that excluded the
strictly-better delta.

Fix (`experiments/real_dataset_gate.py`): the equivalence branch now requires the overall best-fit
target to be absolute or episode_relative. Post-fix, berkeley_rpt correctly returns
`report_without_unique_requirement` (delta fits best but the finite grammar cannot *uniquely*
certify it), i.e. honest abstention → `abstention_consistent`, contradiction gone. Guarded by a new
regression test `test_delta_control_with_fixed_resets_is_not_false_equivalence`. **All 6 pinned
outcomes are preserved** (re-verified on the matched revisions; ALOHA still fires equivalence
because absolute *is* its best-fit family), and the 41-test Python suite stays green (42 with the new
test). The same fix also removed the old dataset-name hack (`name == "aloha"`) that had gated the
branch, making the equivalence outcome name-agnostic for the ALOHA family.

### FLAG 1 — taco_play: evidence-vs-documentation target tension (delta, not the documented absolute)

`lerobot/taco_play` (7-DoF, square). The source (TFDS `taco_play`) carries **three** candidate
action fields — `actions` (**absolute** gripper pose), `rel_actions_gripper` (tool-frame delta),
`rel_actions_world` (base-frame delta) — and the LeRobot config stripped names to `motor_N`, so
**which field was shipped is undocumented**. Our label used the primary `actions` field → absolute
(medium confidence).

ActionABI evidence points the other way: the gate abstains (report), and the FIXED C++ scorer
**decisively prefers delta** (held-out loss delta **0.090** < velocity 0.336 < absolute 0.505).
Judgment: the LeRobot conversion almost certainly shipped a **delta** (`rel_actions_*`) field, not
the absolute one — the passive+CLI evidence *disambiguates* the source's 3-way ambiguity. This is
recorded as a `target_evidence_documentation_tension`, **not** a contradiction, because the gate made
no unique certification (it abstained). It is the audit working as intended: surfacing that a card's
nominal convention likely does not match the shipped tensor.

### FLAGS 2–3 — libero & libero_10: partial frame discrepancy (best-fit world vs documented tool)

Both LIBERO ports (7-DoF, non-square) yield `partial_cartesian` with a **world** best-fit translation
frame (nRMSE 0.30), against our **tool** label (low confidence, from robosuite OSC_POSE). Recorded as
`partial_discrepant` — a *partial, uncertified* fit, never a unique frame claim. Judgment: this most
likely reflects a **weak documentation label** plus genuine passive frame degeneracy, not an
ActionABI error — robosuite OSC_POSE position deltas are conventionally applied in the world/base
frame, which agrees with ActionABI's world best-fit. Notably this points **opposite** to the pinned
UR5 flag (best-fit *tool* vs documented *world*): the two frame flags disagree in direction, which is
itself evidence that **passive diagonal-affine frame identification is under-determined** and that
frame adjudication needs an active on-robot probe (as `reports/labeled_sim_traces.md` already states).
Neither is a certification.

## Cross-ecosystem convention clusters

The audit quantifies, across 20+ in-the-wild datasets, the convention fragmentation the recon
(git history: `docs/weakness_sota_recon.md §2`; documented-convention sources summarized in the
Related work section of README.md) found anecdotally.

### Target convention by ecosystem

| Ecosystem | n | Documented target(s) | Config names | ActionABI outcomes |
|---|---:|---|---|---|
| ALOHA | 6 | **absolute** joint ×6 | semantic (5) / `motor_N` (1) | 3 equivalence, 3 report — all consistent w/ absolute |
| SO-100/101 | 3 | **absolute** joint ×3 | semantic | 3 `unique_absolute` → **agreement** |
| gym-pusht | 1 | **absolute** 2D EE | `motor_N` | 1 `unique_absolute` → **agreement** |
| gym-xarm | 1 | **delta** EE | `motor_N` | 1 partial_cartesian → partial_consistent |
| LIBERO | 2 | **delta** EE | semantic (1) / opaque (1) | 2 partial_cartesian (frame flagged) |
| MetaWorld | 1 | **delta** EE | semantic | 1 report; CLI corroborates delta |
| **OXE-port** | **15** | **9 delta · 3 velocity · 1 absolute · 2 undocumented** | **`motor_N` (all 15)** | 4 partial_cartesian, 11 report |

**Key finding — "OXE" is not one convention.** The 15 OXE ports do **not** share an action space:
9 delta, 3 velocity (cable_routing, nyu_door, ucsd_pick), 1 nominally absolute (taco — but evidence
says delta), 2 undocumented/mismatched (see below); layouts range from `world_vector`+`rotation_delta`
(ur5, roboturk, toto) to bare `[dx,dy,dz,droll,dpitch,dyaw]` (fanuc), delta **joint** (mvp, rpt),
mixed joint-velocity+EE-delta (nyu_franka, 15-dim), to velocity (cable_routing). **All 15 lost their
semantic RLDS names to `motor_N` in the LeRobot v3.0 conversion** — the config-level convention
documentation is uniformly erased for OXE ports, forcing reliance on the upstream TFDS spec.

Two OXE ports are **documentation-absent/mismatched at the tensor level**:
`stanford_kuka_multimodal_dataset` — TFDS `action` is 4-dim absolute EE but the LeRobot tensor is
**7-dim** (the documented convention cannot describe the shipped action; CLI evidence says delta,
0.0008); `ucsd_kitchen_dataset` — TFDS states EE position+orientation but not delta-vs-absolute.

### Gripper polarity cluster (the ALOHA-vs-OXE disagreement, quantified)

Among datasets with a documented gripper channel, **at least five mutually-incompatible polarity
conventions coexist**:

| Convention | Datasets | Meaning |
|---|---|---|
| ALOHA joint-gripper | 6 ALOHA (+ 3 SO-100/101 joint channels) | 0 = closed, 1 = open |
| OXE `gripper_closedness_action` | ur5, jaco, roboturk, nyu_door | **1 = close, −1/0 = open** |
| OXE binary state | berkeley_mvp, berkeley_rpt | 1 = closed, 0 = open |
| OXE `open_gripper` (bool) | toto | **True = open** (inverted *naming*) |
| taco `actions` gripper | taco_play | **−1 = open, 1 = close** |
| robosuite continuous | libero, libero_10, metaworld | continuous in [−1, 1] |

ALOHA's `0 = closed / 1 = open` is **literally inverted** from the OXE closedness convention
`1 = close`. Gripper polarity is therefore a real, silently-varying axis in the wild — exactly
ActionABI's premise. (ActionABI **abstains on gripper for every real dataset**, so this is a
*documentation-side* cluster, not a recovery — reported honestly as such.)

## CLI corroboration (square datasets)

On all 13 square datasets the FIXED C++ scorer's min-held-out-loss target agrees with the gate and the
documentation wherever both are pinned: ALOHA/SO-100/101/pusht-subtask → **absolute** (decisively;
delta/velocity 10–270× worse), metaworld/berkeley_rpt/stanford_kuka → **delta**. The one divergence is
taco_play (FLAG 1). This is an independent, compiled-engine cross-check of the Python gate — not a
separate certification.

## Honest limits

- **Passive under-determination is the dominant outcome, by design.** 115 of 139 labels are honest
  abstentions and 12 are partial-consistent; ActionABI *uniquely* identified a field on only 5 of 35
  datasets (all correct). On the delta/velocity Cartesian datasets it declines to name fields
  documentation *does* pin — confirming that passive trajectory evidence alone under-determines the
  action contract even when the convention is externally known, and bounding ActionABI's positive
  reach.
- **Documentation can be wrong, stale, incomplete, or not describe the shipped tensor** (taco field
  ambiguity; kuka dim mismatch; LIBERO frame). Labels are partial: scale (units, not gains) and lag
  are unlabeled everywhere; frame labels are low/medium confidence; permutation/sign are the
  canonical-order *implication* of the layout, the weakest tier.
- **Delta ≡ velocity under (near-)constant dt.** The diagonal-affine translation fit cannot separate a
  per-step delta from a velocity command (they differ by the scalar dt the affine absorbs), so
  `partial_cartesian` retains both and velocity-documented datasets score partial-*consistent*, not
  discrepant — an explicit, honest equivalence, not a claim.
- **Agreement with documentation is not agreement with dynamics.** A documented convention is a
  human-written label, not a measured latent contract; this is weaker than the labeled-simulation
  supervised accuracy (`reports/labeled_sim_traces.md`) and stronger than unlabeled real outcomes.
- **Frame adjudication needs active probes.** The three frame flags (ur5, libero, libero_10) point in
  inconsistent directions at partial nRMSE — passive frame identification is degenerate here.

## Reproduction

```bash
cd code/actionabi
# 1. harvest metadata + pin revisions, then download bounded trajectory subsets (383 MB total)
#    (helper scripts write /tmp/hub_audit/{harvest.json,data/}; revisions frozen in results/hub_audit.json)
# 2. build the FIXED C++ CLI used for corroboration
uvx --from cmake cmake -S . -B build-fix -DACTIONABI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
uvx --from cmake cmake --build build-fix -j 4
# 3. run the audit (passive gate + CLI corroboration + per-field scoring)
OMP_NUM_THREADS=4 build-data-env/bin/python -m experiments.run_hub_audit \
  --data-root /tmp/hub_audit/data --harvest /tmp/hub_audit/harvest.json \
  --binary build-fix/actionabi --out-dir results/hub --audit-json results/hub_audit.json
# 4. tests (gate fix + adapter variant + baseline discipline)
build-data-env/bin/python -m unittest discover -s experiments -p 'test_*.py'
```

Outputs: `results/hub_audit.json` (aggregate + `flagged_cases` + `ecosystem_clusters` + per-dataset
fields with source URLs/quotes), `results/hub/<dataset>.json`. Documentation labels + verbatim quotes:
`experiments/hub_documentation.py`.
