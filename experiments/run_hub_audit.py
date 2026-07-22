#!/usr/bin/env python3
"""LeRobot Hub convention audit driver.

For each pinned LeRobot dataset:

1. run ActionABI's passive evidence gate (``real_dataset_gate.evaluate_dataset``) -- the SAME
   engine as the six-dataset baseline -- with ZERO documentation in scope, and
2. optionally corroborate the target readout on square (point-target) datasets with the FIXED
   C++ ``actionabi infer`` binary over {absolute, delta, velocity} declared contracts, and
3. score the frozen outcome per field against the documented convention
   (``hub_documentation.HUB_DOCUMENTATION_LABELS``) using the tested ``score_field`` core.

Discipline: documentation is a scoring LABEL only, never an inference input. Steps 1-2 never
see any documentation; step 3 only compares.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from experiments.real_dataset_gate import evaluate_dataset
    from experiments.score_doc_agreement import score_field, _SCORED_FIELDS
    from experiments.hub_documentation import HUB_DOCUMENTATION_LABELS, GRIPPER_CONVENTIONS
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from real_dataset_gate import evaluate_dataset
    from score_doc_agreement import score_field, _SCORED_FIELDS
    from hub_documentation import HUB_DOCUMENTATION_LABELS, GRIPPER_CONVENTIONS

REPO_ROOT = Path(__file__).resolve().parent.parent

# Map the gate's cartesian_translation_frame readout to an ActionABI grammar frame value.
_FRAME_BESTFIT_MAP = {"world": "world", "tool_xyzw": "tool", "tool_wxyz": "tool"}


def equivalence_fields_from_outcome(outcome: str) -> dict[str, str]:
    """Same status mapping as run_case_studies.assemble_case_report (kept identical)."""
    if outcome == "unique_absolute":
        target_status = "identified"
    elif outcome == "absolute_episode_relative_equivalence":
        target_status = "ambiguous"
    elif outcome == "partial_cartesian":
        target_status = "partially_identified"
    else:
        target_status = "unsupported"
    return {
        "target": target_status,
        "permutation": "unsupported",
        "sign": "unsupported",
        "scale": "unsupported",
        "lag": "unsupported",
        "frame": "partially_identified" if outcome == "partial_cartesian" else "unsupported",
        "gripper": "unsupported",
    }


def actionabi_target_readout(outcome: str) -> dict[str, Any]:
    if outcome == "unique_absolute":
        return {"status": "identified", "values": ["absolute"]}
    if outcome == "absolute_episode_relative_equivalence":
        return {"status": "ambiguous", "values": ["absolute", "episode_relative"]}
    if outcome == "partial_cartesian":
        # The diagonal-affine translation fit cannot distinguish a per-step position delta
        # from a velocity command: under (near-)constant dt they differ only by the scalar dt,
        # which the affine scale absorbs. Both non-absolute Cartesian targets are therefore
        # retained (consistent), never separated -- honest under-determination, not a claim.
        return {"status": "partially_identified", "values": ["delta", "velocity"]}
    return {"status": "unsupported", "values": []}


def field_readout(evaluated: dict[str, Any], field: str, best_fit_frame: str | None) -> dict[str, Any]:
    outcome = str(evaluated["outcome"])
    if field == "target":
        return actionabi_target_readout(outcome)
    eq = equivalence_fields_from_outcome(outcome)
    readout: dict[str, Any] = {"status": eq[field], "values": []}
    if field == "frame" and best_fit_frame is not None:
        readout["best_fit"] = best_fit_frame
    return readout


def make_contract(target: str, dim: int) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "target": target,
        "space": "joint",
        "frame": "unspecified",
        "permutation": list(range(dim)),
        "sign": [1] * dim,
        "scale": [1.0] * dim,
        "lag_steps": 0,
        "gripper_inverted": False,
    }


def cli_corroborate(binary: Path, data_dir: Path, dim: int, work: Path) -> dict[str, Any] | None:
    """Run the FIXED C++ scorer over {absolute,delta,velocity} contracts on a square dataset."""
    work.mkdir(parents=True, exist_ok=True)
    jsonl = work / "traj.jsonl"
    adapter = REPO_ROOT / "adapters" / "lerobot_to_jsonl.py"
    import sys
    result = subprocess.run(
        [sys.executable, str(adapter), "--input", str(data_dir), "--output", str(jsonl)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"error": "adapter_failed", "detail": result.stderr[-300:]}
    targets = {"absolute": "absolute_position", "delta": "delta_position", "velocity": "velocity"}
    contract_paths = []
    for short, cname in targets.items():
        cpath = work / f"contract_{short}.json"
        cpath.write_text(json.dumps(make_contract(cname, dim)))
        contract_paths.append((short, cpath))
    out = work / "evidence.json"
    cmd = [str(binary), "infer", "--input", str(jsonl), "--output", str(out)]
    for _, cpath in contract_paths:
        cmd += ["--contract", str(cpath)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        return {"error": "cli_failed", "detail": run.stderr[-300:]}
    report = json.loads(out.read_text())
    losses = {}
    for entry in report["ranked_hypotheses"]:
        losses[entry["contract"]["target"]] = entry["score"]["heldout_loss"]
    inv = {"absolute_position": "absolute", "delta_position": "delta", "velocity": "velocity"}
    losses_short = {inv[k]: v for k, v in losses.items()}
    best = min(losses_short, key=losses_short.get)
    return {"heldout_loss": losses_short, "min_loss_target": best}


def audit_dataset(
    name: str, data_dir: Path, meta: dict[str, Any], binary: Path | None, work_root: Path
) -> dict[str, Any]:
    start = time.perf_counter()
    evaluated = evaluate_dataset(name, data_dir)
    elapsed = time.perf_counter() - start
    outcome = str(evaluated["outcome"])
    best_fit_frame = None
    if outcome == "partial_cartesian":
        raw_frame = str(evaluated.get("cartesian_translation_frame", ""))
        best_fit_frame = _FRAME_BESTFIT_MAP.get(raw_frame)

    labels = HUB_DOCUMENTATION_LABELS[name]
    fields: dict[str, Any] = {}
    for field in _SCORED_FIELDS:
        label = labels[field]
        readout = field_readout(evaluated, field, best_fit_frame)
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

    square = int(evaluated.get("state_dimension", meta.get("st_shape", [0])[0])) == \
        int(evaluated.get("action_dimension", meta.get("act_shape", [0])[0]))
    cli = None
    if binary is not None and square:
        dim = int(meta["act_shape"][0])
        cli = cli_corroborate(binary, data_dir, dim, work_root / name)

    return {
        "dataset": name,
        "hf_repo": labels["hf_repo"],
        "revision": meta["revision"],
        "ecosystem": labels["ecosystem"],
        "config_action_names": labels["config_action_names"],
        "action_dim": meta.get("act_shape", [None])[0],
        "state_dim": meta.get("st_shape", [None])[0],
        "episodes_evaluated": evaluated.get("episodes"),
        "rows_evaluated": evaluated.get("rows"),
        "outcome": outcome,
        "cartesian_translation_nrmse": evaluated.get("cartesian_translation_nrmse"),
        "cartesian_translation_frame": evaluated.get("cartesian_translation_frame"),
        "actionabi_best_fit_frame": best_fit_frame,
        "ranked_best": evaluated.get("best", []),
        "equivalence_fields": equivalence_fields_from_outcome(outcome),
        "cli_corroboration": cli,
        "fields": fields,
        "runtime_seconds": elapsed,
        "converter_status": "blocked",
    }


def aggregate(datasets: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    scored = 0
    for d in datasets.values():
        for f in d["fields"].values():
            v = f["verdict"]
            counts[v] = counts.get(v, 0) + 1
            if v not in ("unlabeled", "not_applicable"):
                scored += 1
    contradictions = counts.get("contradiction", 0)
    return {
        "datasets_audited": len(datasets),
        "scored_field_labels": scored,
        "verdict_counts": dict(sorted(counts.items())),
        "contradictions": contradictions,
        "unique_field_certifications": counts.get("agreement", 0) + contradictions,
        "unique_certifications_correct": counts.get("agreement", 0),
        "false_unique_certifications": contradictions,
        "equivalence_consistent": counts.get("equivalence_consistent", 0),
        "partial_consistent": counts.get("partial_consistent", 0),
        "flagged_partial_discrepancies": counts.get("partial_discrepant", 0),
        "abstention_consistent": counts.get("abstention_consistent", 0),
    }


def flagged_cases(datasets: dict[str, Any]) -> list[dict[str, Any]]:
    """Two flag classes: (1) partial frame discrepancies (ActionABI best-fit disagrees with
    the documented frame; never a certification), and (2) evidence-vs-documentation target
    tensions where the gate abstained on target yet the C++ scorer decisively (best held-out
    loss < 0.5x the runner-up) prefers a target different from the documented one."""
    flags: list[dict[str, Any]] = []
    for name, d in datasets.items():
        for field, fd in d["fields"].items():
            if fd["verdict"] == "partial_discrepant":
                flags.append({
                    "type": "frame_discrepancy", "dataset": name, "field": field,
                    "documented": fd["documented_value"], "documentation_confidence": fd["documentation_confidence"],
                    "actionabi_best_fit": fd.get("actionabi_best_fit"),
                    "nrmse": d.get("cartesian_translation_nrmse"),
                    "note": "partial (uncertified) frame fit; passive diagonal-affine frame identification is degenerate -- needs an active on-robot probe to adjudicate.",
                })
        cli = d.get("cli_corroboration")
        doc_t = d["fields"]["target"]["documented_value"]
        if cli and "min_loss_target" in cli and doc_t is not None:
            losses = cli["heldout_loss"]
            ordered = sorted(losses.values())
            decisive = len(ordered) >= 2 and ordered[0] < 0.5 * ordered[1]
            if decisive and cli["min_loss_target"] != doc_t and d["fields"]["target"]["verdict"] != "agreement":
                flags.append({
                    "type": "target_evidence_documentation_tension", "dataset": name,
                    "documented_target": doc_t, "documentation_confidence": d["fields"]["target"]["documentation_confidence"],
                    "cli_min_loss_target": cli["min_loss_target"], "cli_heldout_loss": losses,
                    "gate_outcome": d["outcome"],
                    "note": "the C++ scorer decisively prefers a target the documentation does not name; the gate abstained (no unique certification), so this is a surfaced tension, not a false certification.",
                })
    return flags


def ecosystem_clusters(datasets: dict[str, Any]) -> dict[str, Any]:
    clusters: dict[str, dict[str, Any]] = {}
    for name, d in datasets.items():
        eco = d["ecosystem"]
        c = clusters.setdefault(eco, {"datasets": [], "documented_targets": {}, "outcomes": {}})
        c["datasets"].append(name)
        t = d["fields"]["target"]["documented_value"]
        c["documented_targets"][str(t)] = c["documented_targets"].get(str(t), 0) + 1
        c["outcomes"][d["outcome"]] = c["outcomes"].get(d["outcome"], 0) + 1
    return clusters


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/tmp/hub_audit/data"))
    p.add_argument("--harvest", type=Path, default=Path("/tmp/hub_audit/harvest.json"))
    p.add_argument("--binary", type=Path, default=REPO_ROOT / "build-fix" / "actionabi")
    p.add_argument("--work", type=Path, default=Path("/tmp/hub_audit/cli_work"))
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "hub")
    p.add_argument("--audit-json", type=Path, default=REPO_ROOT / "results" / "hub_audit.json")
    p.add_argument("--no-cli", action="store_true", help="skip C++ CLI corroboration")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    harvest = json.loads(args.harvest.read_text())
    by_name = {repo.split("/")[-1]: rec for repo, rec in harvest.items()}
    binary = None if args.no_cli else (args.binary if args.binary.exists() else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, Any] = {}
    for name in HUB_DOCUMENTATION_LABELS:
        data_dir = args.data_root / name
        if not data_dir.exists():
            print(f"SKIP {name}: no data at {data_dir}")
            continue
        meta = by_name.get(name, {})
        try:
            report = audit_dataset(name, data_dir, meta, binary, args.work)
        except Exception as e:  # noqa: BLE001 - record and continue the audit
            print(f"ERROR {name}: {e}")
            continue
        datasets[name] = report
        (args.out_dir / f"{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        cli = report.get("cli_corroboration")
        cli_s = f" cli={cli['min_loss_target']}" if cli and "min_loss_target" in cli else ""
        tv = report["fields"]["target"]["verdict"]
        print(f"{name:44s} {report['outcome']:38s} target={tv:22s}{cli_s}")

    out = {
        "schema_version": "1.0",
        "discipline": "documentation is a scoring label only; never an inference input",
        "gripper_conventions_by_ecosystem": GRIPPER_CONVENTIONS,
        "aggregate": aggregate(datasets),
        "flagged_cases": flagged_cases(datasets),
        "ecosystem_clusters": ecosystem_clusters(datasets),
        "datasets": datasets,
    }
    args.audit_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("\n=== AGGREGATE ===")
    print(json.dumps(out["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
