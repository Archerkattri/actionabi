#!/usr/bin/env python3
"""Run the frozen passive-evidence matrix and preserve auditable limitations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import yaml

try:
    from experiments.real_dataset_gate import evaluate_dataset
    from experiments.run_falsification import parse_dataset_spec
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from real_dataset_gate import evaluate_dataset
    from run_falsification import parse_dataset_spec


def assemble_case_report(
    name: str, evaluated: dict[str, Any], *, runtime_seconds: float
) -> dict[str, Any]:
    outcome = str(evaluated["outcome"])
    if outcome == "unique_absolute":
        target_status = "identified"
        probe_status = "not_required_for_target"
    elif outcome == "absolute_episode_relative_equivalence":
        target_status = "ambiguous"
        probe_status = "candidate_required"
    elif outcome == "partial_cartesian":
        target_status = "partially_identified"
        probe_status = "candidate_required"
    else:
        target_status = "unsupported"
        probe_status = "insufficient_observability"
    evidence = {
        "strong evidence": (
            "held-out passive transition fit for the reported outcome"
            if outcome != "report_without_unique_requirement"
            else "none sufficient for a unique contract"
        ),
        "weak evidence": "finite diagonal-affine grammar and a single validation corpus",
        "missing population": "robots, operators, and tasks outside this dataset revision",
        "missing sensor modality": "torque, controller internals, and calibrated frame metadata",
        "missing validation setting": "active on-robot separating probes and deployment shift",
        "clinical / device relevance": "not evaluated; no clinical or safety certification",
    }
    return {
        "schema_version": "1.0",
        "dataset": name,
        "outcome": outcome,
        "source_filename": evaluated.get("source_filename"),
        "source_sha256": evaluated["source_sha256"],
        "episodes": evaluated.get("episodes"),
        "rows": evaluated.get("rows"),
        "ranked_hypotheses": evaluated.get("best", []),
        "equivalence_fields": {
            "target": target_status,
            "permutation": "unsupported",
            "sign": "unsupported",
            "scale": "unsupported",
            "lag": "unsupported",
            "frame": "partially_identified" if outcome == "partial_cartesian" else "unsupported",
            "gripper": "unsupported",
        },
        "empirical_coverage": None,
        "coverage_note": "No labeled real-data contract ground truth; synthetic calibration does not transfer automatically.",
        "worst_residuals": evaluated.get("worst_residuals", []),
        "runtime_seconds": runtime_seconds,
        "probe_utility": {
            "status": probe_status,
            "note": "A probe is a proposal only; execute it only after device-specific bounds are supplied.",
        },
        "converter_status": "blocked",
        "evidence_assessment": evidence,
    }


def write_manifest(output_directory: Path, reports: list[dict[str, Any]]) -> None:
    stable = [
        {
            "dataset": report["dataset"],
            "source_sha256": report["source_sha256"],
            "outcome": report.get("outcome"),
        }
        for report in sorted(reports, key=lambda item: str(item["dataset"]))
    ]
    manifest = {"schema_version": "1.0", "datasets": stable}
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="parquet file or directory; repeat for every frozen dataset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    expected = {
        name: str(settings["expected"])
        for name, settings in config["datasets"].items()
        if name != "synthetic"
    }
    dataset_paths = dict(parse_dataset_spec(specification) for specification in arguments.dataset)
    missing = sorted(set(expected) - set(dataset_paths))
    unexpected = sorted(set(dataset_paths) - set(expected))
    if missing or unexpected:
        raise ValueError(f"dataset matrix mismatch: missing={missing}, unexpected={unexpected}")
    arguments.out.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for name in sorted(dataset_paths):
        start = time.perf_counter()
        evaluated = evaluate_dataset(name, dataset_paths[name])
        elapsed = time.perf_counter() - start
        report = assemble_case_report(name, evaluated, runtime_seconds=elapsed)
        report["expected_outcome"] = expected[name]
        report["matched_expected_outcome"] = report["outcome"] == expected[name]
        reports.append(report)
        (arguments.out / f"{name}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    write_manifest(arguments.out, reports)
    summary = {
        "schema_version": "1.0",
        "seed": int(config["seed"]),
        "passed": all(report["matched_expected_outcome"] for report in reports),
        "datasets": [report["dataset"] for report in reports],
    }
    (arguments.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
