#!/usr/bin/env python3
"""Score ActionABI's already-computed real-data outcomes against DOCUMENTED conventions.

The six pinned real passive datasets (PushT, ALOHA insertion, Berkeley UR5, DROID-100,
Stanford HYDRA, xArm lift) carry no latent-contract ground truth in their trajectories, but
their action-space conventions ARE documented in dataset cards / original papers / RLDS
feature specs. This driver builds the closest hardware-free real-data validation: it treats
each documented convention as a (possibly partial) ground-truth contract LABEL over
ActionABI's grammar, and scores ActionABI's ALREADY-COMPUTED passive outcomes
(``results/<dataset>.json``) against those labels per field.

Standing discipline (do not violate)
-------------------------------------
Documentation is a LABEL used only for *scoring*. It is never an inference input. The
ActionABI outcomes scored here were computed by the passive evaluator with zero access to
any documentation; this module only compares the frozen outcomes to the labels. No outcome
is recomputed with a label in scope.

Verdict vocabulary (per field)
------------------------------
- ``agreement``            ActionABI uniquely identified a value equal to the documented label.
- ``equivalence_consistent`` ActionABI returned an equivalence set that CONTAINS the label
                           (truth retained; not uniquely resolved) - no contradiction.
- ``partial_consistent``   ActionABI partially identified a direction consistent with the label.
- ``partial_discrepant``   ActionABI partially identified, but its best-fit direction disagrees
                           with the label (flagged; still not a unique certification).
- ``abstention_consistent`` ActionABI abstained (unsupported) on a documented field - an honest
                           abstention that makes no claim and cannot contradict the label.
- ``contradiction``        ActionABI identified / retained a set that DISAGREES with or EXCLUDES
                           the documented label. (A false certification lives here.)
- ``unlabeled`` / ``not_applicable`` documentation gives no comparable value for this field.

Honest limits
-------------
Documentation can itself be wrong, stale, or incomplete; several labels are partial
(units/scale and commanded lag are essentially undocumented and left unlabeled). A field the
documentation does not pin is scored ``unlabeled`` rather than assumed. Agreement with
documentation is agreement with a human-written convention, not with a measured latent
contract - it is weaker than the labeled-simulation supervised evidence in
``reports/labeled_sim_traces.md`` and stronger than the previously unlabeled real outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Documentation labels. Each field label carries the documented value (or None = unlabeled),
# an authoritative source URL, a verbatim quote, and a confidence tag. These are LABELS for
# scoring only; they never enter ActionABI inference.
#
# ActionABI grammar fields: target in {absolute, delta, velocity} (TargetKind), frame in
# {base, world, tool, none}, plus permutation, sign, scale, lag, gripper.
# --------------------------------------------------------------------------------------

DOCUMENTATION_LABELS: dict[str, dict[str, Any]] = {
    "pusht": {
        "hf_repo": "lerobot/pusht",
        "target": {
            "value": "absolute",
            "confidence": "high",
            "url": "https://huggingface.co/docs/lerobot/il_sim",
            "quote": (
                "The action at each timestep is the target 2D position of the (cylindrical) "
                "end-effector used to prod the T-shaped block; absolute position/orientation "
                "actions are used for the diffusion policy on PushT."
            ),
        },
        "frame": {
            "value": "world",
            "confidence": "medium",
            "url": "https://arxiv.org/abs/2303.04137",
            "quote": (
                "Diffusion Policy (Chi et al. 2023) PushT: the 2D end-effector target position "
                "is expressed in the fixed workspace/world image frame."
            ),
        },
        "permutation": {"value": "identity", "confidence": "medium", "url": "https://huggingface.co/datasets/lerobot/pusht",
                        "quote": "action features are the (x, y) end-effector coordinates in canonical order."},
        "sign": {"value": "positive", "confidence": "medium", "url": "https://huggingface.co/datasets/lerobot/pusht",
                 "quote": "action axes follow the canonical +x/+y workspace convention."},
        "gripper": {"value": None, "confidence": "n/a", "url": "https://huggingface.co/datasets/lerobot/pusht",
                    "quote": "PushT has no gripper degree of freedom."},
        "scale": {"value": None, "confidence": "n/a", "url": "", "quote": "documented in pixels, not an ActionABI relative gain."},
        "lag": {"value": None, "confidence": "n/a", "url": "", "quote": "no commanded action lag is documented."},
    },
    "aloha": {
        "hf_repo": "lerobot/aloha_sim_insertion_scripted",
        "target": {
            "value": "absolute",
            "confidence": "high",
            "url": "https://www.roboticsproceedings.org/rss19/p016.pdf",
            "quote": (
                "ACT (Zhao et al., RSS 2023) predicts the target joint positions for the next "
                "k timesteps; actions are target (absolute) joint angles commanded to the "
                "position-controlled 14-DoF bimanual ViperX arms."
            ),
        },
        "frame": {"value": None, "confidence": "n/a", "url": "https://tonyzhaozh.github.io/aloha/",
                  "quote": "action is 14-DoF joint-space; no Cartesian reference frame applies."},
        "permutation": {"value": "identity", "confidence": "medium", "url": "https://tonyzhaozh.github.io/aloha/",
                        "quote": "14 joint angles in canonical left-then-right arm order."},
        "sign": {"value": "positive", "confidence": "low", "url": "https://tonyzhaozh.github.io/aloha/",
                 "quote": "joint angles follow each servo's native sign convention."},
        "gripper": {"value": None, "confidence": "low", "url": "https://tonyzhaozh.github.io/aloha/",
                    "quote": "gripper is one of the 14 joint dimensions, not a separate inversion flag."},
        "scale": {"value": None, "confidence": "n/a", "url": "", "quote": "documented in radians, not an ActionABI relative gain."},
        "lag": {"value": None, "confidence": "n/a", "url": "", "quote": "no commanded action lag is documented."},
    },
    "ur5": {
        "hf_repo": "lerobot/berkeley_autolab_ur5",
        "target": {
            "value": "delta",
            "confidence": "high",
            "url": "https://www.tensorflow.org/datasets/catalog/berkeley_autolab_ur5",
            "quote": (
                "action.world_vector = 'delta change in XYZ'; action.rotation_delta = 'delta "
                "change in roll, pitch, yaw' - i.e. a delta (relative) Cartesian end-effector "
                "action (Open X-Embodiment berkeley_autolab_ur5)."
            ),
        },
        "frame": {
            "value": "world",
            "confidence": "medium",
            "url": "https://www.tensorflow.org/datasets/catalog/berkeley_autolab_ur5",
            "quote": (
                "the translation component is named 'world_vector' (delta change in XYZ), "
                "indicating a world/base reference frame for the delta."
            ),
        },
        "permutation": {"value": "identity", "confidence": "medium", "url": "https://www.tensorflow.org/datasets/catalog/berkeley_autolab_ur5",
                        "quote": "action = [world_vector(3), rotation_delta(3), gripper] in canonical order."},
        "sign": {"value": "positive", "confidence": "low", "url": "https://www.tensorflow.org/datasets/catalog/berkeley_autolab_ur5",
                 "quote": "world_vector / rotation_delta follow the standard +XYZ / +RPY convention."},
        "gripper": {
            "value": "documented",
            "confidence": "high",
            "url": "https://www.tensorflow.org/datasets/catalog/berkeley_autolab_ur5",
            "quote": "gripper_closedness_action: 1 = close gripper, -1 = open gripper, 0 = no change.",
        },
        "scale": {"value": None, "confidence": "n/a", "url": "", "quote": "documented in meters/radians, not an ActionABI relative gain."},
        "lag": {"value": None, "confidence": "n/a", "url": "", "quote": "no commanded action lag is documented."},
    },
    "droid": {
        "hf_repo": "lerobot/droid_100",
        "target": {
            "value": "velocity",
            "confidence": "high",
            "url": "https://arxiv.org/abs/2403.12945",
            "quote": (
                "DROID (Khazatsky et al. 2024): the environment exposes a 7-DoF action space - "
                "6-DoF end-effector Cartesian velocity control and a 1-DoF velocity command for "
                "the parallel gripper aperture."
            ),
        },
        "frame": {
            "value": "base",
            "confidence": "medium",
            "url": "https://arxiv.org/abs/2403.12945",
            "quote": "end-effector pose and velocity are expressed in the robot base frame (6D).",
        },
        "permutation": {"value": "identity", "confidence": "medium", "url": "https://droid-dataset.github.io/",
                        "quote": "action = [cartesian_velocity(6), gripper_velocity(1)] in canonical order."},
        "sign": {"value": "positive", "confidence": "low", "url": "https://droid-dataset.github.io/",
                 "quote": "velocity axes follow the standard base-frame +XYZ/+RPY convention."},
        "gripper": {"value": "documented", "confidence": "medium", "url": "https://arxiv.org/abs/2403.12945",
                    "quote": "1-DoF velocity command controls the aperture of the parallel gripper."},
        "scale": {"value": None, "confidence": "n/a", "url": "", "quote": "documented in m/s and rad/s, not an ActionABI relative gain."},
        "lag": {"value": None, "confidence": "n/a", "url": "", "quote": "no commanded action lag is documented."},
    },
    "hydra": {
        "hf_repo": "lerobot/stanford_hydra_dataset",
        "target": {
            "value": "delta",
            "confidence": "high",
            "url": "https://www.tensorflow.org/datasets/catalog/stanford_hydra_dataset_converted_externally_to_rlds",
            "quote": (
                "action (7,) float32 = 3x end-effector positional delta, 3x EEF orientation "
                "delta in euler angle, 1x close gripper - a delta (relative) Cartesian EEF action."
            ),
        },
        "frame": {
            "value": "tool",
            "confidence": "low",
            "url": "https://www.tensorflow.org/datasets/catalog/stanford_hydra_dataset_converted_externally_to_rlds",
            "quote": (
                "the delta is an end-effector positional/orientation delta; the frame (tool vs "
                "base) is not stated unambiguously in the RLDS feature doc."
            ),
        },
        "permutation": {"value": "identity", "confidence": "medium", "url": "https://www.tensorflow.org/datasets/catalog/stanford_hydra_dataset_converted_externally_to_rlds",
                        "quote": "action = [pos_delta(3), orient_delta(3), gripper(1)] in canonical order."},
        "sign": {"value": "positive", "confidence": "low", "url": "https://www.tensorflow.org/datasets/catalog/stanford_hydra_dataset_converted_externally_to_rlds",
                 "quote": "deltas follow the standard +XYZ/+RPY convention."},
        "gripper": {"value": "documented", "confidence": "medium", "url": "https://www.tensorflow.org/datasets/catalog/stanford_hydra_dataset_converted_externally_to_rlds",
                    "quote": "1x close gripper channel."},
        "scale": {"value": None, "confidence": "n/a", "url": "", "quote": "documented in meters/radians, not an ActionABI relative gain."},
        "lag": {"value": None, "confidence": "n/a", "url": "", "quote": "no commanded action lag is documented."},
    },
    "xarm": {
        "hf_repo": "lerobot/xarm_lift_medium",
        "target": {
            "value": "delta",
            "confidence": "medium",
            "url": "https://github.com/huggingface/gym-xarm",
            "quote": (
                "gym-xarm / LeRobot xarm_lift exposes a 4-D action (action_dim = 4): a delta "
                "end-effector displacement (x, y, z) plus a gripper command, in [-1, 1]."
            ),
        },
        "frame": {
            "value": "base",
            "confidence": "low",
            "url": "https://github.com/huggingface/gym-xarm",
            "quote": "the delta end-effector displacement is applied in the simulator base frame; not stated explicitly.",
        },
        "permutation": {"value": "identity", "confidence": "medium", "url": "https://github.com/huggingface/gym-xarm",
                        "quote": "action = [dx, dy, dz, gripper] in canonical order."},
        "sign": {"value": "positive", "confidence": "low", "url": "https://github.com/huggingface/gym-xarm",
                 "quote": "displacement axes follow the standard +XYZ convention."},
        "gripper": {"value": "documented", "confidence": "low", "url": "https://github.com/huggingface/gym-xarm",
                    "quote": "one gripper command channel in the 4-D action."},
        "scale": {"value": None, "confidence": "n/a", "url": "", "quote": "documented as normalized [-1,1], not an ActionABI relative gain."},
        "lag": {"value": None, "confidence": "n/a", "url": "", "quote": "no commanded action lag is documented."},
    },
}

# ActionABI's best-fitting Cartesian frame for datasets where it only PARTIALLY identified a
# frame. Provenance: reproduced by re-running experiments/real_dataset_gate.evaluate_dataset on
# the pinned parquet (build-data-env), which reports `cartesian_translation_frame`. This is
# derived ActionABI evidence, not a documentation input. UR5 best-fit = tool frame (nrmse
# 0.1377), which disagrees with the documented 'world_vector' name -> flagged as discrepant.
ACTIONABI_FRAME_BESTFIT: dict[str, str] = {"ur5": "tool"}

_SCORED_FIELDS = ("target", "frame", "permutation", "sign", "gripper")

# Cartesian frame families treated as compatible for the partial-frame check. A fixed-base
# arm's base and world frames coincide; the tool (end-effector) frame is distinct.
_FRAME_FAMILY = {"base": "base_world", "world": "base_world", "tool": "tool"}


def actionabi_target_readout(outcome: str) -> dict[str, Any]:
    """Map a frozen real-data outcome to the target field's status and retained value set."""
    if outcome == "unique_absolute":
        return {"status": "identified", "values": ["absolute"]}
    if outcome == "absolute_episode_relative_equivalence":
        # episode_relative is an affine variant of absolute within an episode; both retained.
        return {"status": "ambiguous", "values": ["absolute", "episode_relative"]}
    if outcome == "partial_cartesian":
        # Cartesian translation (position-delta) evidence, target not uniquely resolved.
        return {"status": "partially_identified", "values": ["delta"]}
    return {"status": "unsupported", "values": []}


def actionabi_field_readout(result: dict[str, Any], dataset: str, field: str) -> dict[str, Any]:
    """Extract ActionABI's frozen per-field status / retained values from a result JSON."""
    if field == "target":
        readout = actionabi_target_readout(str(result["outcome"]))
    else:
        status = str(result["equivalence_fields"][field])
        values: list[str] = []
        readout = {"status": status, "values": values}
    if field == "frame" and dataset in ACTIONABI_FRAME_BESTFIT:
        readout["best_fit"] = ACTIONABI_FRAME_BESTFIT[dataset]
    return readout


def _frame_compatible(doc_value: str, best_fit: str) -> bool:
    return _FRAME_FAMILY.get(doc_value, doc_value) == _FRAME_FAMILY.get(best_fit, best_fit)


def score_field(field: str, doc_value: Any, readout: dict[str, Any]) -> str:
    """Return the verdict for one documented field against one ActionABI field readout."""
    if doc_value is None:
        return "unlabeled"
    status = readout["status"]
    values = readout.get("values", [])
    if status == "not_applicable":
        return "not_applicable"
    if status == "identified":
        return "agreement" if doc_value in values else "contradiction"
    if status == "ambiguous":
        return "equivalence_consistent" if doc_value in values else "contradiction"
    if status == "partially_identified":
        if field == "frame":
            best_fit = readout.get("best_fit")
            if best_fit is not None and not _frame_compatible(str(doc_value), best_fit):
                return "partial_discrepant"
            return "partial_consistent"
        # Non-frame partial (e.g. target=delta direction): consistent unless it names a
        # different value than the label.
        if values and doc_value not in values:
            return "partial_discrepant"
        return "partial_consistent"
    if status in ("unsupported", "insufficient"):
        return "abstention_consistent"
    raise ValueError(f"unknown ActionABI field status: {status!r}")


def score_dataset(dataset: str, result: dict[str, Any]) -> dict[str, Any]:
    labels = DOCUMENTATION_LABELS[dataset]
    fields: dict[str, Any] = {}
    for field in _SCORED_FIELDS:
        label = labels[field]
        readout = actionabi_field_readout(result, dataset, field)
        verdict = score_field(field, label["value"], readout)
        entry = {
            "documented_value": label["value"],
            "documentation_confidence": label["confidence"],
            "actionabi_status": readout["status"],
            "actionabi_values": readout.get("values", []),
            "verdict": verdict,
            "source_url": label["url"],
            "source_quote": label["quote"],
        }
        if "best_fit" in readout:
            entry["actionabi_best_fit"] = readout["best_fit"]
        fields[field] = entry
    return {
        "hf_repo": labels["hf_repo"],
        "outcome": result["outcome"],
        "fields": fields,
    }


def aggregate(scored: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    scored_labels = 0
    for dataset in scored.values():
        for field in dataset["fields"].values():
            verdict = field["verdict"]
            counts[verdict] = counts.get(verdict, 0) + 1
            if verdict not in ("unlabeled", "not_applicable"):
                scored_labels += 1
    contradictions = counts.get("contradiction", 0)
    unique_certifications = counts.get("agreement", 0) + counts.get("contradiction", 0)
    return {
        "scored_field_labels": scored_labels,
        "verdict_counts": dict(sorted(counts.items())),
        "contradictions": contradictions,
        "unique_field_certifications": unique_certifications,
        "unique_certifications_correct": counts.get("agreement", 0),
        "false_unique_certifications": contradictions,
        "abstention_or_partial_consistent": (
            counts.get("abstention_consistent", 0)
            + counts.get("equivalence_consistent", 0)
            + counts.get("partial_consistent", 0)
        ),
        "flagged_partial_discrepancies": counts.get("partial_discrepant", 0),
    }


def build_report(results_dir: Path) -> dict[str, Any]:
    scored: dict[str, Any] = {}
    for dataset in DOCUMENTATION_LABELS:
        result = json.loads((results_dir / f"{dataset}.json").read_text(encoding="utf-8"))
        scored[dataset] = score_dataset(dataset, result)
    return {
        "schema_version": "1.0",
        "discipline": "documentation is a scoring label only; never an inference input",
        "verdict_vocabulary": [
            "agreement",
            "equivalence_consistent",
            "partial_consistent",
            "partial_discrepant",
            "abstention_consistent",
            "contradiction",
            "unlabeled",
            "not_applicable",
        ],
        "datasets": scored,
        "aggregate": aggregate(scored),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/doc_agreement.json"))
    arguments = parser.parse_args(argv)
    report = build_report(arguments.results)
    arguments.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
