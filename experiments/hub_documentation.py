#!/usr/bin/env python3
"""Documented action-space conventions for the LeRobot Hub audit (LABELS ONLY).

Every entry is a *scoring label*, never an inference input. Each field carries the
documented value (or ``None`` = unlabeled), an authoritative source URL, a verbatim quote,
a confidence tag, and an ecosystem tag. Sources are the per-revision LeRobot ``meta/info.json``
feature spec (config-level) and the upstream Open X-Embodiment / RLDS TFDS catalog, ACT/ALOHA,
LIBERO, MetaWorld, and gym-xarm / Diffusion-Policy specs (convention-level).

Grammar fields scored: ``target`` in {absolute, delta, velocity}; ``frame`` in
{world, base, tool}; ``permutation``; ``sign``; ``gripper``.

Two documentation layers are recorded and they can disagree with each other -- that
disagreement is itself an audit finding:

* ``config_action_names`` -- the literal ``action.names`` in the pinned ``meta/info.json``.
  For most OXE ports LeRobot v3.0 stripped these to generic ``motor_0..N`` (semantic
  convention erased at config level); native datasets (ALOHA, SO-100/101, LIBERO_10,
  MetaWorld) keep semantic names.
* the per-field ``target``/``frame``/``gripper`` labels below, taken from the upstream spec.
"""

from __future__ import annotations

from typing import Any

TFDS = "https://www.tensorflow.org/datasets/catalog"

# Per-ecosystem gripper polarity conventions (documentation-side; ActionABI abstains on
# gripper for all real data, so this is a DOCUMENTED-convention cluster, not a recovery).
GRIPPER_CONVENTIONS = {
    "OXE-closedness": "gripper_closedness_action: 1=close, -1/0=open (Open X-Embodiment RT-1 lineage)",
    "OXE-openpose": "gripper channel -1=open, +1=close (taco_play rel/abs actions)",
    "OXE-open_gripper-bool": "open_gripper boolean: True=open (TOTO) -- inverted naming vs closedness",
    "OXE-binary-closed": "binary gripper state: 1=closed, 0=open (berkeley MVP/RPT)",
    "ALOHA": "gripper is a joint dimension; ACT/ALOHA convention 0=closed, 1=open",
    "SO-100/101": "gripper is a joint position channel (main_gripper / gripper.pos)",
    "LIBERO": "gripper continuous in [-1, 1] (OSC_POSE controller, robosuite)",
    "MetaWorld": "gripper continuous in [-1, 1] (4th action dim)",
}


def _f(value, confidence, url, quote) -> dict[str, Any]:
    return {"value": value, "confidence": confidence, "url": url, "quote": quote}


_UNLABELED_SCALE = _f(None, "n/a", "", "documented in physical units, not an ActionABI relative gain.")
_UNLABELED_LAG = _f(None, "n/a", "", "no commanded action lag is documented.")


def _entry(hf_repo, ecosystem, config_names, target, frame, permutation, sign, gripper) -> dict[str, Any]:
    return {
        "hf_repo": hf_repo,
        "ecosystem": ecosystem,
        "config_action_names": config_names,
        "target": target,
        "frame": frame,
        "permutation": permutation,
        "sign": sign,
        "gripper": gripper,
        "scale": _UNLABELED_SCALE,
        "lag": _UNLABELED_LAG,
    }


# Shorthand builders for the two most common label shapes.
def _perm_identity(url):
    return _f("identity", "medium", url, "action channels documented in canonical (source) order.")


def _sign_positive(url):
    return _f("positive", "low", url, "axes follow the source's native +XYZ/+RPY (or servo) sign convention.")


HUB_DOCUMENTATION_LABELS: dict[str, dict[str, Any]] = {
    # ---------------- OXE ports (converted_externally_to_rlds -> LeRobot) ----------------
    "berkeley_cable_routing": _entry(
        "lerobot/berkeley_cable_routing", "OXE-port",
        "motor_0..motor_6 (semantic RLDS names stripped by LeRobot v3.0 conversion)",
        _f("velocity", "high", f"{TFDS}/berkeley_cable_routing",
           "action: world_vector = 'Velocity in XYZ', rotation_delta = 'Angular velocity about the z axis' -> Cartesian velocity command."),
        _f("world", "medium", f"{TFDS}/berkeley_cable_routing",
           "the translation channel is named world_vector (velocity in XYZ) -> world/base frame."),
        _perm_identity(f"{TFDS}/berkeley_cable_routing"), _sign_positive(f"{TFDS}/berkeley_cable_routing"),
        _f(None, "n/a", f"{TFDS}/berkeley_cable_routing", "cable-routing action has no gripper channel."),
    ),
    "berkeley_fanuc_manipulation": _entry(
        "lerobot/berkeley_fanuc_manipulation", "OXE-port",
        "motor_0..motor_6 (RLDS names stripped)",
        _f("delta", "high", f"{TFDS}/berkeley_fanuc_manipulation",
           "action: 'Robot action, consists of [dx, dy, dz] and [droll, dpitch, dyaw]' -> delta end-effector pose."),
        _f("tool", "low", f"{TFDS}/berkeley_fanuc_manipulation",
           "delta end-effector pose; base-vs-tool frame not stated in the RLDS feature doc."),
        _perm_identity(f"{TFDS}/berkeley_fanuc_manipulation"), _sign_positive(f"{TFDS}/berkeley_fanuc_manipulation"),
        _f(None, "n/a", f"{TFDS}/berkeley_fanuc_manipulation", "6-DoF action only; gripper is observed, not actuated."),
    ),
    "berkeley_mvp": _entry(
        "lerobot/berkeley_mvp", "OXE-port",
        "motor_0..motor_7 (RLDS names stripped)",
        _f("delta", "high", f"{TFDS}/berkeley_mvp_converted_externally_to_rlds",
           "action(8): 'Robot action, consists of [7 delta joint pos, 1x gripper binary state]' -> delta JOINT positions."),
        _f(None, "n/a", f"{TFDS}/berkeley_mvp_converted_externally_to_rlds", "joint-space action; no Cartesian frame applies."),
        _perm_identity(f"{TFDS}/berkeley_mvp_converted_externally_to_rlds"),
        _sign_positive(f"{TFDS}/berkeley_mvp_converted_externally_to_rlds"),
        _f("documented", "high", f"{TFDS}/berkeley_mvp_converted_externally_to_rlds",
           "1x gripper binary state (1 = closed, 0 = open)."),
    ),
    "berkeley_rpt": _entry(
        "lerobot/berkeley_rpt", "OXE-port",
        "motor_0..motor_7 (RLDS names stripped)",
        _f("delta", "high", f"{TFDS}/berkeley_rpt_converted_externally_to_rlds",
           "action(8): 'Robot action, consists of [7 delta joint pos, 1x gripper binary state]' -> delta JOINT positions."),
        _f(None, "n/a", f"{TFDS}/berkeley_rpt_converted_externally_to_rlds", "joint-space action; no Cartesian frame applies."),
        _perm_identity(f"{TFDS}/berkeley_rpt_converted_externally_to_rlds"),
        _sign_positive(f"{TFDS}/berkeley_rpt_converted_externally_to_rlds"),
        _f("documented", "high", f"{TFDS}/berkeley_rpt_converted_externally_to_rlds",
           "1x gripper binary state (1 = closed, 0 = open)."),
    ),
    "jaco_play": _entry(
        "lerobot/jaco_play", "OXE-port",
        "motor_0..motor_6 (RLDS names stripped)",
        _f("delta", "high", f"{TFDS}/jaco_play",
           "action: world_vector(3) + gripper_closedness_action(1) + terminate_episode(3); world_vector is a 3D motion delta (no rotation channel)."),
        _f("world", "medium", f"{TFDS}/jaco_play", "translation channel named world_vector -> world frame."),
        _perm_identity(f"{TFDS}/jaco_play"), _sign_positive(f"{TFDS}/jaco_play"),
        _f("documented", "high", f"{TFDS}/jaco_play", "gripper_closedness_action (OXE closedness convention)."),
    ),
    "nyu_door_opening_surprising_effectiveness": _entry(
        "lerobot/nyu_door_opening_surprising_effectiveness", "OXE-port",
        "motor_0..motor_6 (RLDS names stripped)",
        _f("velocity", "high", f"{TFDS}/nyu_door_opening_surprising_effectiveness",
           "action: world_vector='Velocity in XYZ', rotation_delta='Angular velocity around x,y,z' + gripper_closedness -> Cartesian velocity."),
        _f("world", "low", f"{TFDS}/nyu_door_opening_surprising_effectiveness",
           "world_vector naming; frame origin not explicitly stated."),
        _perm_identity(f"{TFDS}/nyu_door_opening_surprising_effectiveness"),
        _sign_positive(f"{TFDS}/nyu_door_opening_surprising_effectiveness"),
        _f("documented", "medium", f"{TFDS}/nyu_door_opening_surprising_effectiveness",
           "gripper_closedness_action (0.0=open, 1.0=closed)."),
    ),
    "nyu_franka_play_dataset": _entry(
        "lerobot/nyu_franka_play_dataset", "OXE-port",
        "motor_0..motor_14 (RLDS names stripped)",
        _f("delta", "medium", f"{TFDS}/nyu_franka_play_dataset_converted_externally_to_rlds",
           "action(15): '[7x joint velocities, 3x EE delta xyz, 3x EE delta rpy, 1x gripper position, 1x terminate]' -- MIXED joint-velocity + EE-delta."),
        _f(None, "n/a", f"{TFDS}/nyu_franka_play_dataset_converted_externally_to_rlds",
           "mixed joint-velocity + EE-delta action; single Cartesian frame does not apply."),
        _f(None, "low", f"{TFDS}/nyu_franka_play_dataset_converted_externally_to_rlds",
           "mixed layout; canonical permutation not a clean single-space claim."),
        _sign_positive(f"{TFDS}/nyu_franka_play_dataset_converted_externally_to_rlds"),
        _f("documented", "low", f"{TFDS}/nyu_franka_play_dataset_converted_externally_to_rlds", "1x gripper position channel."),
    ),
    "roboturk": _entry(
        "lerobot/roboturk", "OXE-port",
        "motor_0..motor_6 (RLDS names stripped)",
        _f("delta", "high", f"{TFDS}/roboturk",
           "action: world_vector(3) + rotation_delta(3) + gripper_closedness_action(1); rotation_delta named as a delta -> delta EE."),
        _f("world", "medium", f"{TFDS}/roboturk", "translation channel named world_vector -> world/base frame."),
        _perm_identity(f"{TFDS}/roboturk"), _sign_positive(f"{TFDS}/roboturk"),
        _f("documented", "high", f"{TFDS}/roboturk", "gripper_closedness_action (OXE closedness convention)."),
    ),
    "stanford_kuka_multimodal_dataset": _entry(
        "lerobot/stanford_kuka_multimodal_dataset", "OXE-port",
        "motor_0..motor_6 (RLDS names stripped)",
        # DOCUMENTATION MISMATCH: TFDS 'action' is 4-dim ([3x EEF position,1x gripper]) but the
        # LeRobot tensor is 7-dim -> the TFDS label cannot be applied to the LeRobot action.
        _f(None, "n/a", f"{TFDS}/stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
           "TFDS action is 4-dim '[3x EEF position, 1x gripper open/close]' but the LeRobot action tensor is 7-dim -> the documented convention does not describe the shipped action; treated as documentation-absent."),
        _f(None, "n/a", "", "documentation dimension mismatch (see target)."),
        _f(None, "n/a", "", "documentation dimension mismatch."),
        _f(None, "n/a", "", "documentation dimension mismatch."),
        _f(None, "n/a", "", "documentation dimension mismatch."),
    ),
    "taco_play": _entry(
        "lerobot/taco_play", "OXE-port",
        "motor_0..motor_6 (RLDS names stripped; source has THREE candidate action fields)",
        # Source has actions(absolute), rel_actions_gripper(tool delta), rel_actions_world(base delta).
        _f("absolute", "medium", f"{TFDS}/taco_play",
           "action.actions(7): 'Absolute desired values for gripper pose (x,y,z,yaw,pitch,roll,gripper)'; source also carries rel_actions_gripper/world deltas -> which field LeRobot used is not documented at config level."),
        _f("base", "low", f"{TFDS}/taco_play", "the absolute 'actions' field is in robot base frame (rel_actions_world also base)."),
        _perm_identity(f"{TFDS}/taco_play"), _sign_positive(f"{TFDS}/taco_play"),
        _f("documented", "medium", f"{TFDS}/taco_play", "gripper channel: -1=open, 1=close."),
    ),
    "toto": _entry(
        "lerobot/toto", "OXE-port",
        "motor_0..motor_6 (RLDS names stripped)",
        _f("delta", "high", f"{TFDS}/toto",
           "action: world_vector(3) + rotation_delta(3) + open_gripper(bool); rotation_delta named as delta -> delta EE (Franka)."),
        _f("world", "medium", f"{TFDS}/toto", "translation channel named world_vector -> world frame."),
        _perm_identity(f"{TFDS}/toto"), _sign_positive(f"{TFDS}/toto"),
        _f("documented", "medium", f"{TFDS}/toto", "open_gripper boolean (True=open) -- inverted naming vs closedness convention."),
    ),
    "ucsd_kitchen_dataset": _entry(
        "lerobot/ucsd_kitchen_dataset", "OXE-port",
        "motor_0..motor_7 (RLDS names stripped)",
        _f(None, "n/a", f"{TFDS}/ucsd_kitchen_dataset_converted_externally_to_rlds",
           "action(8): 'end-effector position and orientation, gripper open/close and episode termination' -- delta-vs-absolute NOT stated -> documentation-absent on target."),
        _f(None, "low", f"{TFDS}/ucsd_kitchen_dataset_converted_externally_to_rlds", "Cartesian EE action; frame not stated."),
        _perm_identity(f"{TFDS}/ucsd_kitchen_dataset_converted_externally_to_rlds"),
        _sign_positive(f"{TFDS}/ucsd_kitchen_dataset_converted_externally_to_rlds"),
        _f("documented", "low", f"{TFDS}/ucsd_kitchen_dataset_converted_externally_to_rlds", "1x gripper open/close channel."),
    ),
    "ucsd_pick_and_place_dataset": _entry(
        "lerobot/ucsd_pick_and_place_dataset", "OXE-port",
        "motor_0..motor_3 (RLDS names stripped)",
        _f("velocity", "high", f"{TFDS}/ucsd_pick_and_place_dataset_converted_externally_to_rlds",
           "action(4): 'Robot action, consists of [3x gripper velocities, 1x gripper open/close torque]' -> velocity."),
        _f("tool", "low", f"{TFDS}/ucsd_pick_and_place_dataset_converted_externally_to_rlds",
           "xArm end-effector (gripper) velocities; base-vs-tool frame not stated."),
        _perm_identity(f"{TFDS}/ucsd_pick_and_place_dataset_converted_externally_to_rlds"),
        _sign_positive(f"{TFDS}/ucsd_pick_and_place_dataset_converted_externally_to_rlds"),
        _f("documented", "low", f"{TFDS}/ucsd_pick_and_place_dataset_converted_externally_to_rlds", "1x gripper open/close torque."),
    ),
    "utaustin_mutex": _entry(
        "lerobot/utaustin_mutex", "OXE-port",
        "motor_0..motor_6 (RLDS names stripped)",
        _f("delta", "high", f"{TFDS}/utaustin_mutex",
           "action(7): 'Robot action, consists of [6x end effector delta pose, 1x gripper position]' -> delta EE."),
        _f("tool", "low", f"{TFDS}/utaustin_mutex", "6x EE delta pose; base-vs-tool frame not stated."),
        _perm_identity(f"{TFDS}/utaustin_mutex"), _sign_positive(f"{TFDS}/utaustin_mutex"),
        _f("documented", "low", f"{TFDS}/utaustin_mutex", "1x gripper position channel."),
    ),
    "austin_buds_dataset": _entry(
        "lerobot/austin_buds_dataset", "OXE-port",
        "motor_0..motor_6 (RLDS names stripped)",
        _f("delta", "high", f"{TFDS}/austin_buds_dataset_converted_externally_to_rlds",
           "action(7): 'Robot action, consists of [6x end effector delta pose, 1x gripper position]' -> delta EE."),
        _f("tool", "low", f"{TFDS}/austin_buds_dataset_converted_externally_to_rlds", "6x EE delta pose; frame not stated."),
        _perm_identity(f"{TFDS}/austin_buds_dataset_converted_externally_to_rlds"),
        _sign_positive(f"{TFDS}/austin_buds_dataset_converted_externally_to_rlds"),
        _f("documented", "low", f"{TFDS}/austin_buds_dataset_converted_externally_to_rlds", "1x gripper position channel."),
    ),
    # ---------------- ALOHA family (native LeRobot; ACT/ALOHA) ----------------
    **{
        name: _entry(
            f"lerobot/{name}", "ALOHA",
            names,
            _f("absolute", "high", "https://www.roboticsproceedings.org/rss19/p016.pdf",
               "ACT/ALOHA commands ABSOLUTE target joint positions to the position-controlled 14-DoF bimanual ViperX arms."),
            _f(None, "n/a", "https://tonyzhaozh.github.io/aloha/", "14-DoF joint-space action; no Cartesian frame applies."),
            _f("identity", "medium", "https://tonyzhaozh.github.io/aloha/",
               "14 joint angles in canonical left-then-right arm order."),
            _sign_positive("https://tonyzhaozh.github.io/aloha/"),
            _f(None, "low", "https://tonyzhaozh.github.io/aloha/",
               "gripper is a joint dimension (ALOHA 0=closed,1=open), not a separate inversion flag."),
        )
        for name, names in {
            "aloha_sim_transfer_cube_human": "left_waist..left_gripper,right_waist..right_gripper (semantic joint names)",
            "aloha_sim_insertion_human": "left_waist..left_gripper,right_waist..right_gripper (semantic joint names)",
            "aloha_static_coffee": "left_waist..left_gripper,right_waist..right_gripper (semantic joint names)",
            "aloha_static_cups_open": "left_waist..left_gripper,right_waist..right_gripper (semantic joint names)",
            "aloha_mobile_cabinet": "left_waist..left_gripper,right_waist..right_gripper (14-DoF arms; Mobile-ALOHA base dims NOT in this action)",
            "aloha_static_battery": "motor_0..motor_13 (semantic joint names STRIPPED despite ALOHA robot -- intra-ecosystem config inconsistency)",
        }.items()
    },
    # ---------------- SO-100/101 (native LeRobot teleop, leader-follower) ----------------
    **{
        name: _entry(
            f"lerobot/{name}", "SO-100/101",
            names,
            _f("absolute", "medium", "https://huggingface.co/docs/lerobot/so101",
               "SO-100/101 leader-follower teleoperation records ABSOLUTE follower joint positions (the '.pos' / joint-name channels are position targets)."),
            _f(None, "n/a", f"https://huggingface.co/datasets/lerobot/{name}", "6-DoF joint-space action; no Cartesian frame applies."),
            _f("identity", "medium", f"https://huggingface.co/datasets/lerobot/{name}",
               "6 joint positions in canonical shoulder->gripper order."),
            _sign_positive(f"https://huggingface.co/datasets/lerobot/{name}"),
            _f(None, "low", f"https://huggingface.co/datasets/lerobot/{name}",
               "gripper is a joint position channel, not a separate inversion flag."),
        )
        for name, names in {
            "svla_so100_pickplace": "main_shoulder_pan,main_shoulder_lift,main_elbow_flex,main_wrist_flex,main_wrist_roll,main_gripper",
            "svla_so101_pickplace": "shoulder_pan.pos,shoulder_lift.pos,elbow_flex.pos,wrist_flex.pos,wrist_roll.pos,gripper.pos ('.pos' documents position control)",
            "svla_so100_sorting": "main_shoulder_pan,main_shoulder_lift,main_elbow_flex,main_wrist_flex,main_wrist_roll,main_gripper",
        }.items()
    },
    # ---------------- gym (PushT / xArm) ----------------
    "pusht-subtask": _entry(
        "lerobot/pusht-subtask", "gym-pusht",
        "motor_0,motor_1 (2D end-effector target)",
        _f("absolute", "high", "https://huggingface.co/docs/lerobot/il_sim",
           "PushT action is the ABSOLUTE 2D target position of the cylindrical end-effector (same convention as lerobot/pusht)."),
        _f("world", "medium", "https://arxiv.org/abs/2303.04137", "2D EE target in the fixed workspace/world frame."),
        _f("identity", "medium", "https://huggingface.co/datasets/lerobot/pusht-subtask", "(x, y) in canonical order."),
        _sign_positive("https://huggingface.co/datasets/lerobot/pusht-subtask"),
        _f(None, "n/a", "https://huggingface.co/datasets/lerobot/pusht-subtask", "PushT has no gripper degree of freedom."),
    ),
    "xarm_push_medium": _entry(
        "lerobot/xarm_push_medium", "gym-xarm",
        "motor_0,motor_1,motor_2 (3D end-effector displacement; no gripper)",
        _f("delta", "medium", "https://github.com/huggingface/gym-xarm",
           "gym-xarm push exposes a 3-D delta end-effector displacement (x,y,z) in [-1,1]; the push task has no gripper action."),
        _f("base", "low", "https://github.com/huggingface/gym-xarm", "delta EE displacement in the simulator base frame; not stated explicitly."),
        _f("identity", "medium", "https://github.com/huggingface/gym-xarm", "action = [dx, dy, dz] in canonical order."),
        _sign_positive("https://github.com/huggingface/gym-xarm"),
        _f(None, "n/a", "https://github.com/huggingface/gym-xarm", "the push task action has no gripper channel."),
    ),
    # ---------------- LIBERO (ports) ----------------
    "libero_10": _entry(
        "lerobot/libero_10", "LIBERO",
        "x,y,z,roll,pitch,yaw,gripper (semantic EE-axis names retained in config)",
        _f("delta", "high", "https://huggingface.co/datasets/lerobot/libero_10",
           "config action.names = [x,y,z,roll,pitch,yaw,gripper]; LIBERO uses the robosuite OSC_POSE controller -> delta end-effector pose + gripper."),
        _f("tool", "low", "https://libero-project.github.io/", "OSC_POSE delta is applied in the controller/EE frame; robosuite default."),
        _f("identity", "high", "https://huggingface.co/datasets/lerobot/libero_10", "action = [x,y,z,roll,pitch,yaw,gripper] in canonical order (config-documented)."),
        _sign_positive("https://huggingface.co/datasets/lerobot/libero_10"),
        _f("documented", "medium", "https://libero-project.github.io/", "gripper continuous in [-1,1] (robosuite/OSC)."),
    ),
    "libero": _entry(
        "lerobot/libero", "LIBERO",
        "'actions' (single opaque name -- config-level convention absent)",
        _f("delta", "medium", "https://libero-project.github.io/",
           "LIBERO uses the robosuite OSC_POSE controller -> 6-D delta EE pose + gripper; the LeRobot config only labels the 7-vector 'actions' (opaque)."),
        _f("tool", "low", "https://libero-project.github.io/", "OSC_POSE delta in the controller/EE frame."),
        _f("identity", "low", "https://huggingface.co/datasets/lerobot/libero", "config action name is the opaque 'actions'; canonical order assumed."),
        _sign_positive("https://huggingface.co/datasets/lerobot/libero"),
        _f("documented", "low", "https://libero-project.github.io/", "gripper continuous in [-1,1]."),
    ),
    # ---------------- MetaWorld ----------------
    "metaworld_mt50": _entry(
        "lerobot/metaworld_mt50", "MetaWorld",
        "x,y,z,gripper (semantic EE-axis names retained in config)",
        _f("delta", "high", "https://github.com/Farama-Foundation/Metaworld",
           "MetaWorld action = 4-D: 3-D delta end-effector position (x,y,z) + gripper, normalized to [-1,1]."),
        _f("base", "medium", "https://github.com/Farama-Foundation/Metaworld", "delta EE position in the world/base frame."),
        _f("identity", "high", "https://huggingface.co/datasets/lerobot/metaworld_mt50", "action = [x,y,z,gripper] in canonical order (config-documented)."),
        _sign_positive("https://huggingface.co/datasets/lerobot/metaworld_mt50"),
        _f("documented", "medium", "https://github.com/Farama-Foundation/Metaworld", "4th dim = gripper, continuous in [-1,1]."),
    ),
}
