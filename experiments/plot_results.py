#!/usr/bin/env python3
"""Render deterministic, dependency-free ActionABI result summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_benchmarks(paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    metadata = {
        key: reports[0].get(key, "unknown")
        for key in ("gpu", "cpu", "compiler", "build_type", "cpu_threads")
    }
    workloads = [workload for report in reports for workload in report["workloads"]]
    workloads.sort(key=lambda workload: int(workload["hypotheses"]))
    return metadata, workloads


def render_benchmark(paths: list[Path]) -> str:
    metadata, workloads = _load_benchmarks(paths)
    best_speedup = max(float(workload["speedup_over_multicore_cpu"]) for workload in workloads)
    gate_passed = best_speedup >= 5.0
    verdict = (
        "CUDA headline gate: passed."
        if gate_passed
        else "CUDA headline gate: failed; CUDA remains experimental."
    )
    lines = [
        "# ActionABI backend benchmark",
        "",
        verdict,
        "",
        f"- CPU: {metadata['cpu']}",
        f"- GPU: {metadata['gpu']}",
        f"- Compiler: {metadata['compiler']}",
        f"- Build type: {metadata['build_type']}",
        f"- Hardware concurrency: {metadata['cpu_threads']}",
        "- Gate: transfer-inclusive CUDA median must be at least 5× faster than multicore CPU.",
        "",
        "| Hypotheses | Residual evaluations | Single CPU ms | Multicore CPU ms | CUDA total ms | CUDA kernel ms | Speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for workload in workloads:
        single = workload.get("single_thread_cpu_ms", {}).get("median", float("nan"))
        lines.append(
            "| {hypotheses} | {residual_evaluations} | {single:.3f} | {multi:.3f} | "
            "{cuda:.3f} | {kernel:.3f} | {speedup:.2f}× |".format(
                hypotheses=workload["hypotheses"],
                residual_evaluations=workload["residual_evaluations"],
                single=float(single),
                multi=float(workload["multicore_cpu_ms"]["median"]),
                cuda=float(workload["cuda_transfer_inclusive_ms"]["median"]),
                kernel=float(workload["cuda_kernel_only_ms"]["median"]),
                speedup=float(workload["speedup_over_multicore_cpu"]),
            )
        )
    return "\n".join(lines) + "\n"


def render_cases(paths: list[Path]) -> str:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reports.sort(key=lambda report: str(report["dataset"]))
    lines = [
        "# ActionABI frozen case studies",
        "",
        "Real-data outcomes are passive-evidence audits, not universal contract certifications.",
        "",
        "| Dataset | Episodes | Rows | Outcome | Target field | Converter | Expected |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for report in reports:
        lines.append(
            "| {dataset} | {episodes} | {rows} | {outcome} | {target} | {converter} | {matched} |".format(
                dataset=report["dataset"],
                episodes=report.get("episodes", "unknown"),
                rows=report.get("rows", "unknown"),
                outcome=report["outcome"],
                target=report["equivalence_fields"]["target"],
                converter=report["converter_status"],
                matched="yes" if report.get("matched_expected_outcome") else "no",
            )
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="render benchmark JSON")
    benchmark.add_argument("--input", type=Path, nargs="+", required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    cases = subparsers.add_parser("cases", help="render case-study JSON")
    cases.add_argument("--input", type=Path, nargs="+", required=True)
    cases.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.command == "benchmark":
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(render_benchmark(arguments.input), encoding="utf-8")
    elif arguments.command == "cases":
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(render_cases(arguments.input), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
