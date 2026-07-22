# Documentation-agreement scoring of ActionABI's real-data outcomes

Run date: 2026-07-21 UTC.
Scorer: `experiments/score_doc_agreement.py` (unit-tested: `experiments/test_score_doc_agreement.py`, 9/9).
Machine-readable output: `results/doc_agreement.json`.
Scored inputs: the already-computed passive outcomes in `results/{pusht,aloha,ur5,droid,hydra,xarm}.json`.

This report builds the closest **hardware-free real-data validation** available for ActionABI.
The six pinned real passive datasets carry no latent-contract ground truth in their trajectories,
but their action-space conventions **are documented** (dataset cards, original papers, RLDS feature
specs). We treat each documented convention as a (possibly partial) ground-truth **label** over
ActionABI's grammar and score ActionABI's frozen outcomes against it, per field.

## Standing discipline (unchanged)

**Documentation is a scoring label only, never an inference input.** The outcomes scored here were
produced by ActionABI's passive evaluator (`experiments/real_dataset_gate.py`) with zero access to
any documentation. This module only *compares* the frozen outcomes to the labels; no outcome is
recomputed with a label in scope. The frozen outcomes were re-verified on the current build (see
"Currency check" below) and reproduce the committed `results/*.json` exactly.

## Documented conventions (label sources)

Each label cites an authoritative source with a verbatim quote (full list with per-field quotes in
`results/doc_agreement.json` and `experiments/score_doc_agreement.py::DOCUMENTATION_LABELS`).

| Dataset | HF repo | Documented action convention | target | frame | source |
|---|---|---|---|---|---|
| PushT | `lerobot/pusht` | 2D **absolute** end-effector target position (cartesian, workspace/world) | absolute | world | LeRobot PushT docs; Diffusion Policy (Chi et al. 2023, arXiv:2303.04137) |
| ALOHA insertion | `lerobot/aloha_sim_insertion_scripted` | **absolute** 14-DoF target **joint** positions | absolute | n/a (joint) | ACT (Zhao et al., RSS 2023, `roboticsproceedings.org/rss19/p016.pdf`) |
| Berkeley UR5 | `lerobot/berkeley_autolab_ur5` | **delta** cartesian EE: `world_vector`=delta XYZ, `rotation_delta`=delta RPY, gripper 1=close/-1=open | delta | world | TFDS `berkeley_autolab_ur5`; Open X-Embodiment |
| DROID-100 | `lerobot/droid_100` | 6-DoF EE cartesian **velocity** + 1-DoF gripper velocity, base frame | velocity | base | DROID (Khazatsky et al. 2024, arXiv:2403.12945) |
| Stanford HYDRA | `lerobot/stanford_hydra_dataset` | 7-D **delta** EE: 3× pos delta + 3× euler orient delta + gripper | delta | tool | TFDS `stanford_hydra_dataset_converted_externally_to_rlds` |
| xArm lift | `lerobot/xarm_lift_medium` | 4-D **delta** EE displacement (x,y,z) + gripper | delta | base | gym-xarm / LeRobot (`action_dim=4`) |

Confidence is recorded per field: target labels are `high` except xArm (`medium`); frame labels are
`medium`/`low` (frames are the least explicitly documented); scale and lag are **unlabeled** for every
dataset (documentation gives physical units and fixed teleop rates, not an ActionABI-comparable
relative gain or a commanded action lag).

## Per-dataset verdicts

| Dataset | ActionABI outcome | target verdict | frame verdict | other fields |
|---|---|---|---|---|
| PushT | `unique_absolute` | **agreement** (identified `absolute` = documented) | abstention_consistent | perm/sign abstention_consistent; gripper n/a |
| ALOHA | `absolute_episode_relative_equivalence` | **equivalence_consistent** (documented `absolute` ∈ retained set {absolute, episode_relative}) | unlabeled (joint) | perm abstention_consistent; sign/gripper unlabeled |
| UR5 | `partial_cartesian` | **partial_consistent** (cartesian delta direction) | **partial_discrepant** (best-fit `tool` vs documented `world`) | perm/sign/gripper abstention_consistent |
| DROID-100 | `report_without_unique_requirement` | abstention_consistent | abstention_consistent | perm/sign/gripper abstention_consistent |
| HYDRA | `report_without_unique_requirement` | abstention_consistent | abstention_consistent | perm/sign/gripper abstention_consistent |
| xArm | `report_without_unique_requirement` | abstention_consistent | abstention_consistent | perm/sign/gripper abstention_consistent |

Verdict vocabulary: `agreement` (unique identification matches label), `equivalence_consistent`
(equivalence set contains the label — truth retained, not resolved), `partial_consistent` (partial
direction consistent with label), `partial_discrepant` (partial direction disagrees with label —
flagged, **not** a certification), `abstention_consistent` (abstained on a documented field — honest,
no claim), `contradiction` (identified/retained set disagrees with or excludes the label — a false
certification would land here), `unlabeled`/`not_applicable`.

## Aggregate statistics

Across **27 documented field-labels** scored over the six datasets (5 candidate fields × 6, minus 3
that documentation does not pin comparably: PushT gripper [no DoF], ALOHA frame [joint] and gripper
[joint-embedded]):

| Metric | Value |
|---|---:|
| Contradictions (ActionABI asserts a convention documentation refutes) | **0 / 27** |
| Unique field certifications made by ActionABI | 1 |
| — of which documentation-correct | **1 / 1** (PushT target = absolute) |
| — false unique certifications | **0** |
| Equivalence-consistent (truth retained, non-unique) | 1 (ALOHA target) |
| Partial-consistent | 1 (UR5 target) |
| Flagged partial discrepancy (not a certification) | 1 (UR5 frame) |
| Abstention-consistent (honest abstention on a documented field) | 23 |

**Headline: zero contradictions with documentation on any field of any dataset; ActionABI's single
unique real-data certification (PushT = absolute) is documentation-correct; zero false unique
certifications.** Every field where ActionABI abstained is confirmed by documentation to be a field
it *could* in principle have claimed — so the abstentions are honest, not vacuous.

## The one flagged discrepancy (UR5 frame)

ActionABI's partial cartesian evidence for UR5 best-fit a **tool** end-effector frame
(`cartesian_translation_frame = tool_xyzw`, nRMSE 0.1377), whereas the dataset's translation channel
is documented as `world_vector` (world/base frame). This is recorded as `partial_discrepant`, **not**
a contradiction, because ActionABI only *partially* identified the frame (`equivalence_fields.frame =
partially_identified`) and never uniquely certified `tool`. The passive fit therefore surfaces a
genuine frame ambiguity and retains it as ambiguity rather than emitting a false unique frame claim —
which is exactly the designed behavior. The discrepancy is worth flagging: either the "world_vector"
label is applied in a rotated frame in this LeRobot conversion, or the diagonal-affine passive fit is
frame-degenerate here; distinguishing them needs an active on-robot probe, not passive data.

## Currency check

The committed `results/*.json` predate the 2026-07-21 C++/CUDA lag-observable fix
(`reports/scorer_fixes.md`), but the six real-dataset outcomes are produced by the pure-Python
`real_dataset_gate.evaluate_dataset`, which that fix did not touch. Re-running the evaluator on the
pinned parquet (build-data-env) reproduces all six committed outcomes exactly
(`pusht→unique_absolute, aloha→absolute_episode_relative_equivalence, ur5→partial_cartesian,
droid/hydra/xarm→report_without_unique_requirement`). Outputs are current, not stale.

## Honest limits

- **Documentation can be wrong, stale, or incomplete.** A label is a human-written convention, not a
  measured latent contract. The UR5 frame discrepancy above is a concrete example of documentation
  and passive evidence disagreeing; we cannot adjudicate it from passive data.
- **Labels are partial.** scale (units, not gains) and lag (undocumented) are unlabeled everywhere;
  frame labels are low/medium confidence; permutation/sign labels are the canonical-ordering
  *implication* of the feature layout, not an explicit statement — so their abstention-consistent
  verdicts are the weakest tier of evidence and are reported as such.
- **Abstention-consistent is not identification.** 23 of 27 verdicts are honest abstentions: on the
  three delta/velocity cartesian datasets (DROID, HYDRA, xArm) ActionABI declines to identify fields
  that documentation *does* pin. This confirms the core thesis — passive trajectory evidence alone
  under-determines the action contract even when the convention is externally known — but it also
  bounds ActionABI's positive reach: it uniquely identified exactly one field (PushT absolute) across
  six real datasets.
- **This is agreement with documentation, not with dynamics.** It is weaker than the labeled-simulation
  supervised accuracy in `reports/labeled_sim_traces.md` (ground-truth latent contracts by
  construction) and stronger than the previously unlabeled real outcomes.

## What this adds over the unlabeled outcomes

Before this scoring, the six real outcomes were self-consistent but externally **unvalidated**: there
was no way to tell whether `unique_absolute` on PushT was correct or a lucky false certification, or
whether the equivalence/abstention verdicts were the *right* verdicts. The documentation labels supply
that missing external check and yield three concrete additions:

1. **A false-certification audit on real data.** ActionABI's one unique real-data certification is
   confirmed correct, and it made zero false unique certifications — the same abstention property
   proven synthetically (`reports/benchmark_sprint.md`) now holds against documented real conventions.
2. **Calibrated abstention is validated as honest.** Every abstention lands on a field documentation
   confirms was knowable, and ALOHA's equivalence set is shown to *retain* the documented truth rather
   than exclude it — evidence the calibration neither over-claims nor falsely rules out the truth.
3. **A new, actionable discrepancy** (UR5 tool-vs-world frame) that only external labels could surface,
   and which points to a specific active-probe experiment.
