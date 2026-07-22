"""Held-out passive-evidence checks for ActionABI's week-one real datasets."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


PointTarget = Literal["absolute", "delta", "velocity", "episode_relative"]


@dataclass(frozen=True)
class PassiveDataset:
    states: np.ndarray
    actions: np.ndarray
    episode_ids: np.ndarray
    timestamps_seconds: np.ndarray

    def __post_init__(self) -> None:
        row_count = len(self.states)
        if self.states.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("states and actions must be rank-two")
        if not (
            len(self.actions)
            == len(self.episode_ids)
            == len(self.timestamps_seconds)
            == row_count
        ):
            raise ValueError("all passive dataset arrays must have equal row counts")
        if row_count == 0:
            raise ValueError("passive dataset must not be empty")
        if not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.actions)):
            raise ValueError("states and actions must be finite")


@dataclass(frozen=True)
class PointResult:
    target: PointTarget
    lag: int
    nrmse: float
    median_scale: float


def _episode_split(dataset: PassiveDataset) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(dataset.episode_ids)
    if len(episodes) < 2:
        raise ValueError("held-out scoring requires at least two episodes")
    split = min(len(episodes) - 1, max(1, int(0.7 * len(episodes))))
    return episodes[:split], episodes[split:]


def _paired_indices(dataset: PassiveDataset, lag: int) -> np.ndarray:
    if lag <= 0:
        raise ValueError("real-dataset lag must be positive")
    if lag >= len(dataset.states):
        return np.empty(0, dtype=np.int64)
    indices = np.arange(len(dataset.states) - lag)
    return indices[dataset.episode_ids[:-lag] == dataset.episode_ids[lag:]]


def _initial_states(dataset: PassiveDataset) -> np.ndarray:
    starts = np.r_[True, dataset.episode_ids[1:] != dataset.episode_ids[:-1]]
    start_indices = np.maximum.accumulate(np.where(starts, np.arange(len(starts)), 0))
    return dataset.states[start_indices]


def _fit_diagonal_affine(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    coefficients = np.empty((features.shape[1], 2), dtype=np.float64)
    for component in range(features.shape[1]):
        design = np.column_stack([features[:, component], np.ones(len(features))])
        coefficients[component] = np.linalg.lstsq(
            design, targets[:, component], rcond=None
        )[0]
    return coefficients


def _predict_diagonal_affine(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return features * coefficients[:, 0] + coefficients[:, 1]


def _normalized_rmse(
    targets: np.ndarray, predictions: np.ndarray, training_targets: np.ndarray
) -> float:
    scale = np.std(training_targets, axis=0)
    varying = scale > 1e-9
    if not np.any(varying):
        return float(np.sqrt(np.mean((predictions - targets) ** 2)))
    normalized = (predictions[:, varying] - targets[:, varying]) / scale[varying]
    return float(np.sqrt(np.mean(normalized**2)))


def rank_point_hypotheses(
    dataset: PassiveDataset, *, max_lag: int = 6
) -> list[PointResult]:
    if dataset.states.shape[1] != dataset.actions.shape[1]:
        return []
    train_episodes, test_episodes = _episode_split(dataset)
    initial = _initial_states(dataset)
    results: list[PointResult] = []
    for lag in range(1, max_lag + 1):
        indices = _paired_indices(dataset, lag)
        if len(indices) == 0:
            continue
        future = dataset.states[indices + lag]
        current = dataset.states[indices]
        action = dataset.actions[indices]
        elapsed = dataset.timestamps_seconds[indices + lag] - dataset.timestamps_seconds[indices]
        valid_time = elapsed > 0
        train = np.isin(dataset.episode_ids[indices], train_episodes) & valid_time
        test = np.isin(dataset.episode_ids[indices], test_episodes) & valid_time
        if not np.any(train) or not np.any(test):
            continue
        representations: dict[PointTarget, tuple[np.ndarray, np.ndarray]] = {
            "absolute": (action, future),
            "delta": (action, future - current),
            "velocity": (action * elapsed[:, None], future - current),
            "episode_relative": (action, future - initial[indices]),
        }
        for target, (features, observable) in representations.items():
            coefficients = _fit_diagonal_affine(features[train], observable[train])
            prediction = _predict_diagonal_affine(features[test], coefficients)
            results.append(
                PointResult(
                    target=target,
                    lag=lag,
                    nrmse=_normalized_rmse(
                        observable[test], prediction, observable[train]
                    ),
                    median_scale=float(np.median(coefficients[:, 0])),
                )
            )
    return sorted(results, key=lambda result: (result.nrmse, result.target, result.lag))


def _rotation_matrices_xyzw(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    normalized = quaternions / np.maximum(norms, 1e-12)
    x, y, z, w = normalized.T
    matrices = np.empty((len(normalized), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def _cartesian_translation_evidence(
    dataset: PassiveDataset, max_lag: int = 6
) -> tuple[float, str]:
    if dataset.states.shape[1] < 3 or dataset.actions.shape[1] < 3:
        return math.inf, "unsupported"
    train_episodes, test_episodes = _episode_split(dataset)
    best = math.inf
    best_frame = "unsupported"
    for lag in range(1, max_lag + 1):
        indices = _paired_indices(dataset, lag)
        if len(indices) == 0:
            continue
        world_delta = dataset.states[indices + lag, :3] - dataset.states[indices, :3]
        features = dataset.actions[indices, :3]
        train = np.isin(dataset.episode_ids[indices], train_episodes)
        test = np.isin(dataset.episode_ids[indices], test_episodes)
        if not np.any(train) or not np.any(test):
            continue
        targets = {"world": world_delta}
        if dataset.states.shape[1] >= 7:
            quaternion = dataset.states[indices, 3:7]
            if np.all(np.linalg.norm(quaternion, axis=1) > 1e-6):
                for order, xyzw in (
                    ("xyzw", quaternion),
                    ("wxyz", quaternion[:, [1, 2, 3, 0]]),
                ):
                    rotation = _rotation_matrices_xyzw(xyzw)
                    targets[f"tool_{order}"] = np.einsum(
                        "nij,nj->ni", rotation.transpose(0, 2, 1), world_delta
                    )
        for frame, target in targets.items():
            coefficients = _fit_diagonal_affine(features[train], target[train])
            prediction = _predict_diagonal_affine(features[test], coefficients)
            error = _normalized_rmse(target[test], prediction, target[train])
            if error < best:
                best = error
                best_frame = frame
    return best, best_frame


def classify_dataset(
    name: str, dataset: PassiveDataset, ranked: list[PointResult]
) -> dict[str, Any]:
    normalized_name = name.lower()
    if dataset.states.shape[1] != dataset.actions.shape[1]:
        cartesian_error, cartesian_frame = _cartesian_translation_evidence(dataset)
        outcome = "partial_cartesian" if cartesian_error < 0.35 else "report_without_unique_requirement"
        return {
            "outcome": outcome,
            "state_dimension": int(dataset.states.shape[1]),
            "action_dimension": int(dataset.actions.shape[1]),
            "cartesian_translation_nrmse": cartesian_error,
            "cartesian_translation_frame": cartesian_frame,
        }
    if not ranked:
        return {"outcome": "report_without_unique_requirement"}
    best_by_target = {
        target: min(result.nrmse for result in ranked if result.target == target)
        for target in ("absolute", "delta", "velocity", "episode_relative")
    }
    absolute_relative_gap = abs(
        best_by_target["absolute"] - best_by_target["episode_relative"]
    )
    equivalence_tolerance = max(1e-8, 0.01 * max(best_by_target["absolute"], 1e-8))
    # Absolute vs episode-relative equivalence is a name-agnostic observational property:
    # when per-episode initial states are ~constant, a diagonal-affine fit cannot separate an
    # absolute target from an episode-relative one, so both are retained (no forced unique).
    # (Historically this branch was gated to name == "aloha"; the gate was a hack. Removing it
    # does not change any pinned-6 outcome -- non-aloha point datasets have varying resets, so
    # their absolute/episode-relative gap exceeds the tolerance and they fall through unchanged.)
    #
    # GUARD (hub-audit fix): the absolute~episode_relative coincidence must not be reported as an
    # {absolute, episode_relative} equivalence when a delta/velocity target actually fits BETTER.
    # A delta-controlled dataset with near-static, near-constant-reset joints makes absolute and
    # episode_relative fit each other closely (intercept absorbs the static pose) even though
    # delta fits far better. Firing the branch there would EXCLUDE the truly-best delta target and
    # emit a false equivalence certification (observed on berkeley_rpt: delta nRMSE 0.35 vs
    # absolute 0.93). Require the overall best-fit target to be absolute or episode_relative so the
    # retained set cannot exclude a strictly-better delta/velocity fit.
    best_target = min(best_by_target, key=best_by_target.get)
    if absolute_relative_gap <= equivalence_tolerance and best_target in (
        "absolute",
        "episode_relative",
    ):
        outcome = "absolute_episode_relative_equivalence"
    else:
        next_nonabsolute = min(best_by_target["delta"], best_by_target["velocity"])
        decisive_margin = max(0.02, 0.1 * max(best_by_target["absolute"], 1e-8))
        if (
            ranked[0].target == "absolute"
            and next_nonabsolute - best_by_target["absolute"] > decisive_margin
            and best_by_target["episode_relative"] - best_by_target["absolute"]
            > decisive_margin
        ):
            outcome = "unique_absolute"
        else:
            outcome = "report_without_unique_requirement"
    return {
        "outcome": outcome,
        "state_dimension": int(dataset.states.shape[1]),
        "action_dimension": int(dataset.actions.shape[1]),
        "best": [
            {
                "target": result.target,
                "lag": result.lag,
                "nrmse": result.nrmse,
                "median_scale": result.median_scale,
            }
            for result in ranked[:6]
        ],
    }


def _parquet_sources(path: Path) -> Path | list[Path]:
    if path.is_file():
        return path
    data_root = path / "data" if (path / "data").is_dir() else path
    sources = sorted(data_root.rglob("*.parquet"))
    if not sources:
        raise ValueError(f"no parquet files found under {path}")
    return sources


def load_lerobot_parquet(path: Path) -> PassiveDataset:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("real-dataset gate requires PyArrow") from error
    sources = _parquet_sources(path)
    table = parquet.read_table(
        sources,
        columns=["observation.state", "action", "episode_index", "timestamp"],
    )
    return PassiveDataset(
        states=np.asarray(table["observation.state"].to_pylist(), dtype=np.float64),
        actions=np.asarray(table["action"].to_pylist(), dtype=np.float64),
        episode_ids=np.asarray(table["episode_index"].to_numpy()),
        timestamps_seconds=np.asarray(table["timestamp"].to_numpy(), dtype=np.float64),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    sources = _parquet_sources(path)
    assert isinstance(sources, list)
    files = sources
    for file_path in files:
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def evaluate_dataset(name: str, path: Path) -> dict[str, Any]:
    dataset = load_lerobot_parquet(path)
    ranked = rank_point_hypotheses(dataset)
    report = classify_dataset(name, dataset, ranked)
    report["source_filename"] = path.name
    report["source_sha256"] = sha256_path(path)
    report["episodes"] = int(len(np.unique(dataset.episode_ids)))
    report["rows"] = int(len(dataset.states))
    return report
