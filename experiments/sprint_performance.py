#!/usr/bin/env python3
"""Run and normalize ActionABI CPU/CUDA scaling measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def latency_quantiles(samples_ms: Iterable[float]) -> dict[str, int | float]:
    values = np.asarray(list(samples_ms), dtype=np.float64)
    if len(values) == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("latency samples must be nonempty, finite, and nonnegative")
    return {
        "count": int(len(values)),
        "p10_ms": float(np.quantile(values, 0.1)),
        "median_ms": float(np.quantile(values, 0.5)),
        "p90_ms": float(np.quantile(values, 0.9)),
    }


def stable_job_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def sprint_matrix(
    *,
    hypotheses: Sequence[int],
    rows: Sequence[int],
    dimensions: Sequence[int],
    threads: Sequence[int],
) -> list[dict[str, Any]]:
    if not hypotheses or not rows or not dimensions or not threads:
        raise ValueError("every performance axis must be nonempty")
    if any(value <= 0 for axis in (hypotheses, rows, dimensions, threads) for value in axis):
        raise ValueError("performance axes must contain positive integers")
    anchor = {
        "hypotheses": 1024 if 1024 in hypotheses else hypotheses[0],
        "rows_per_episode": 256 if 256 in rows else rows[0],
        "dimension": 2 if 2 in dimensions else dimensions[0],
        "cpu_threads": max(threads),
        "dtype": "float64",
    }
    candidates: list[dict[str, Any]] = []
    for key, values in (
        ("hypotheses", hypotheses),
        ("rows_per_episode", rows),
        ("dimension", dimensions),
        ("cpu_threads", threads),
    ):
        for value in values:
            candidates.append({**anchor, key: value})
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique[stable_job_id(candidate)] = candidate
    return list(unique.values())


def _measurements(workload: dict[str, Any], key: str) -> list[float]:
    value = workload.get(key)
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def run_workload(binary: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    command = [
        str(binary), "--output", str(output),
        "--hypotheses", str(config["hypotheses"]),
        "--total-evaluations", str(config["total_evaluations"]),
        "--warmups", str(config["warmups"]),
        "--measurements", str(config["measurements"]),
        "--dimension", str(config["dimension"]),
        "--rows-per-episode", str(config["rows_per_episode"]),
        "--cpu-threads", str(config["cpu_threads"]),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return {"status": "failed", "returncode": completed.returncode, "stderr": completed.stderr[-4000:]}
    report = json.loads(output.read_text(encoding="utf-8"))
    workload = report["workloads"][0]
    normalized: dict[str, Any] = {"status": "completed", "source": report, "config": config}
    sample_keys = {
        "single_thread_cpu_ms": "single_thread_cpu_samples_ms",
        "multicore_cpu_ms": "multicore_cpu_samples_ms",
        "cuda_kernel_only_ms": "cuda_kernel_samples_ms",
        "cuda_transfer_inclusive_ms": "cuda_transfer_samples_ms",
    }
    for key, sample_key in sample_keys.items():
        samples = _measurements(workload, sample_key)
        if samples:
            normalized[key] = latency_quantiles(samples)
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build-cuda/score_benchmark"))
    parser.add_argument("--out", type=Path, default=Path("results/sprint/performance"))
    parser.add_argument("--hypotheses", default="128,1024,8192,16384")
    parser.add_argument("--rows", default="64,256,1024")
    parser.add_argument("--dimensions", default="2,7,14")
    parser.add_argument("--threads", default="1,8,64")
    parser.add_argument("--total-evaluations", type=int, default=10_000_000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--measurements", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    raw_path = arguments.out / "raw.jsonl"
    records: list[dict[str, Any]] = []
    matrix = sprint_matrix(
        hypotheses=tuple(int(item) for item in arguments.hypotheses.split(",")),
        rows=tuple(int(item) for item in arguments.rows.split(",")),
        dimensions=tuple(int(item) for item in arguments.dimensions.split(",")),
        threads=tuple(int(item) for item in arguments.threads.split(",")),
    )
    for scientific_config in matrix:
        config = {
            **scientific_config,
            "total_evaluations": arguments.total_evaluations,
            "warmups": arguments.warmups,
            "measurements": arguments.measurements,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        job_id = stable_job_id(config)
        with tempfile.NamedTemporaryFile(suffix=".json") as temporary:
            record = run_workload(arguments.binary, Path(temporary.name), config)
        record["job_id"] = job_id
        records.append(record)
        with raw_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, sort_keys=True) + "\n")
    successful = [record for record in records if record["status"] == "completed"]
    summary = {
        "schema_version": "1.0",
        "jobs": len(records),
        "completed": len(successful),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "axis_interpretation": {
            "trajectory_length": "rows_per_episode across four episodes",
            "batch": "hypothesis count is the CUDA batch axis",
            "dtype": "float64 measured; float32 structurally inapplicable because the public scorer is float64-only",
            "thread_count": "explicit worker count; single-thread scorer is separately timed",
        },
        "inapplicable": [{"axis": "dtype", "value": "float32", "reason": "no float32 scorer exists"}],
        "results": successful,
    }
    (arguments.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"jobs": len(records), "completed": len(successful)}))
    return 0 if len(successful) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
