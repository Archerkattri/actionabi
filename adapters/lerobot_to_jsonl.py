#!/usr/bin/env python3
"""Convert one LeRobot parquet shard to ActionABI's canonical JSONL format."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source(path: Path) -> str:
    """Hash a single parquet file, or a directory of shards (order-stable)."""
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for source in _parquet_sources(path):
        digest.update(source.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(source)))
    return digest.hexdigest()


def build_metadata(
    *,
    source: Path,
    state_columns: Sequence[str],
    state_units: Sequence[str],
    extraction_date: date,
) -> dict[str, Any]:
    if len(state_columns) != len(state_units):
        raise ValueError("state_columns and state_units must have equal length")
    if not state_columns:
        raise ValueError("at least one state column is required")
    return {
        "record_type": "metadata",
        "schema_version": "1.0",
        "source_filename": source.name,
        "source_sha256": sha256_source(source),
        "extraction_date": extraction_date.isoformat(),
        "state_columns": list(state_columns),
        "state_units": list(state_units),
    }


def _finite_vector(values: Sequence[float], field: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must be a nonempty finite vector")
    return result


def build_sample_record(
    *,
    episode_id: int,
    timestamp_seconds: float,
    state: Sequence[float],
    action: Sequence[float],
) -> dict[str, Any]:
    if episode_id < 0:
        raise ValueError("episode_id must be nonnegative")
    if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
        raise ValueError("timestamp_seconds must be finite and nonnegative")
    return {
        "record_type": "sample",
        "episode_id": int(episode_id),
        "t_ns": round(timestamp_seconds * 1_000_000_000),
        "state": _finite_vector(state, "state"),
        "action": _finite_vector(action, "action"),
    }


def write_jsonl(
    output: Path,
    metadata: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
) -> None:
    if metadata.get("record_type") != "metadata":
        raise ValueError("first record must be metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        destination.write(json.dumps(dict(metadata), sort_keys=True) + "\n")
        for sample in samples:
            if sample.get("record_type") != "sample":
                raise ValueError("all trajectory records after metadata must be samples")
            destination.write(json.dumps(dict(sample), sort_keys=True) + "\n")


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parquet_sources(path: Path) -> list[Path]:
    """Resolve a single parquet file, or a LeRobot v3.0 dataset dir, to source files.

    Additive format variant: LeRobot codebase v3.0 packs episodes into many shard files
    under ``data/chunk-*/file-*.parquet`` (with metadata parquet under ``meta/``). Passing a
    directory globs the data shards only, excluding the ``meta/`` tables, and returns them in
    a stable sorted order so the concatenated trajectory is deterministic.
    """
    if path.is_file():
        return [path]
    data_root = path / "data" if (path / "data").is_dir() else path
    sources = sorted(p for p in data_root.rglob("*.parquet"))
    if not sources:
        raise ValueError(f"no parquet files found under {path}")
    return sources


def _read_parquet_rows(
    path: Path, columns: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "reading parquet requires PyArrow; install the adapter extra"
        ) from error
    sources = _parquet_sources(path)
    # Project to just the trajectory columns when they are all present, so inline image /
    # video blob columns (LeRobot v3.0 can embed them in the data shard) are never loaded.
    projection = None
    if columns:
        available = set(parquet.read_schema(sources[0]).names)
        if set(columns) <= available:
            projection = list(columns)
    return parquet.read_table(sources, columns=projection).to_pylist()


def _records_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    state_key: str,
    action_key: str,
    timestamp_key: str,
    episode_key: str,
) -> list[dict[str, Any]]:
    records = []
    for row_number, row in enumerate(rows):
        missing = {
            key
            for key in (state_key, action_key, timestamp_key, episode_key)
            if key not in row
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"row {row_number} is missing columns: {names}")
        records.append(
            build_sample_record(
                episode_id=int(row[episode_key]),
                timestamp_seconds=float(row[timestamp_key]),
                state=row[state_key],
                action=row[action_key],
            )
        )
    if not records:
        raise ValueError("input parquet contains no rows")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="LeRobot parquet shard")
    parser.add_argument("--output", type=Path, required=True, help="canonical JSONL output")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--timestamp-key", default="timestamp")
    parser.add_argument("--episode-key", default="episode_index")
    parser.add_argument(
        "--state-columns",
        help="comma-separated state names; defaults to state_0,state_1,...",
    )
    parser.add_argument(
        "--state-units",
        help="comma-separated units; defaults to unknown for every state component",
    )
    parser.add_argument(
        "--extraction-date",
        type=date.fromisoformat,
        default=date.today(),
        metavar="YYYY-MM-DD",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _read_parquet_rows(
        args.input,
        columns=[args.state_key, args.action_key, args.timestamp_key, args.episode_key],
    )
    samples = _records_from_rows(
        rows,
        state_key=args.state_key,
        action_key=args.action_key,
        timestamp_key=args.timestamp_key,
        episode_key=args.episode_key,
    )
    state_dimension = len(samples[0]["state"])
    state_columns = (
        _parse_csv(args.state_columns)
        if args.state_columns
        else tuple(f"state_{index}" for index in range(state_dimension))
    )
    state_units = (
        _parse_csv(args.state_units)
        if args.state_units
        else ("unknown",) * state_dimension
    )
    metadata = build_metadata(
        source=args.input,
        state_columns=state_columns,
        state_units=state_units,
        extraction_date=args.extraction_date,
    )
    write_jsonl(args.output, metadata, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
