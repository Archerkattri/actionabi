#!/usr/bin/env python3
"""Reference CPU falsification harness for ActionABI's finite contract grammar."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import yaml

try:
    from experiments.real_dataset_gate import evaluate_dataset
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from real_dataset_gate import evaluate_dataset


Target = Literal["absolute", "delta", "velocity"]


@dataclass(frozen=True)
class Hypothesis:
    target: Target
    permutation: tuple[int, ...]
    sign: tuple[int, ...]
    scale: tuple[float, ...]
    lag_steps: int

    def __post_init__(self) -> None:
        dimension = len(self.permutation)
        if dimension == 0:
            raise ValueError("hypothesis must contain at least one component")
        if len(self.sign) != dimension or len(self.scale) != dimension:
            raise ValueError("hypothesis component dimensions must match")
        if sorted(self.permutation) != list(range(dimension)):
            raise ValueError("permutation must be bijective")
        if any(value not in (-1, 1) for value in self.sign):
            raise ValueError("sign values must be -1 or 1")
        if any(not math.isfinite(value) or value <= 0 for value in self.scale):
            raise ValueError("scale values must be finite and positive")
        if self.lag_steps < 0:
            raise ValueError("lag_steps must be nonnegative")


@dataclass(frozen=True)
class Episode:
    states: np.ndarray
    actions: np.ndarray
    timestamps_seconds: np.ndarray

    def __post_init__(self) -> None:
        if self.states.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("states and actions must be rank-two arrays")
        if self.states.shape[1] != self.actions.shape[1]:
            raise ValueError("state and action dimensions must match")
        if len(self.timestamps_seconds) != len(self.states):
            raise ValueError("timestamps must align with states")
        if len(self.states) < 2:
            raise ValueError("episode must contain at least two states")
        if not np.all(np.diff(self.timestamps_seconds) > 0):
            raise ValueError("timestamps must be strictly increasing")
        if not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.actions)):
            raise ValueError("episode arrays must be finite")

    def with_states(self, states: np.ndarray) -> Episode:
        return replace(self, states=states)


def _decode_actions(actions: np.ndarray, hypothesis: Hypothesis) -> np.ndarray:
    semantic = actions[:, hypothesis.permutation].copy()
    semantic *= np.asarray(hypothesis.sign)
    semantic *= np.asarray(hypothesis.scale)
    return semantic


def _encode_actions(semantic: np.ndarray, hypothesis: Hypothesis) -> np.ndarray:
    raw = np.empty_like(semantic, dtype=np.float64)
    for semantic_index, raw_index in enumerate(hypothesis.permutation):
        raw[:, raw_index] = semantic[:, semantic_index] / (
            hypothesis.sign[semantic_index] * hypothesis.scale[semantic_index]
        )
    return raw


def generate_episode(
    hypothesis: Hypothesis,
    *,
    semantic_commands: np.ndarray,
    initial_state: np.ndarray,
    dt_seconds: float,
) -> Episode:
    """Generate a noiseless controlled trace under one declared hypothesis."""
    commands = np.asarray(semantic_commands, dtype=np.float64)
    initial = np.asarray(initial_state, dtype=np.float64)
    if commands.ndim != 2 or initial.shape != (commands.shape[1],):
        raise ValueError("initial_state must match the command component dimension")
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive")

    actions = _encode_actions(commands, hypothesis)
    transition_count = len(actions) + hypothesis.lag_steps
    states = np.empty((transition_count + 1, commands.shape[1]), dtype=np.float64)
    states[0] = initial
    for transition in range(transition_count):
        states[transition + 1] = states[transition]
        action_index = transition - hypothesis.lag_steps
        if action_index < 0 or action_index >= len(commands):
            continue
        command = commands[action_index]
        if hypothesis.target == "absolute":
            states[transition + 1] = command
        elif hypothesis.target == "delta":
            states[transition + 1] += command
        else:
            states[transition + 1] += command * dt_seconds
    timestamps = np.arange(transition_count + 1, dtype=np.float64) * dt_seconds
    return Episode(states=states, actions=actions, timestamps_seconds=timestamps)


def _huber(residual: np.ndarray, delta: float) -> np.ndarray:
    magnitude = np.abs(residual)
    return np.where(magnitude <= delta, 0.5 * residual**2, delta * (magnitude - 0.5 * delta))


def score_hypothesis(
    episodes: Sequence[Episode], hypothesis: Hypothesis, *, huber_delta: float = 1.0
) -> float:
    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
    losses: list[np.ndarray] = []
    for episode in episodes:
        if episode.actions.shape[1] != len(hypothesis.permutation):
            raise ValueError("episode and hypothesis dimensions must match")
        decoded = _decode_actions(episode.actions, hypothesis)
        for action_index, command in enumerate(decoded):
            transition = action_index + hypothesis.lag_steps
            if transition + 1 >= len(episode.states):
                continue
            before = episode.states[transition]
            after = episode.states[transition + 1]
            if hypothesis.target == "absolute":
                observable = after
            elif hypothesis.target == "delta":
                observable = after - before
            else:
                dt = (
                    episode.timestamps_seconds[transition + 1]
                    - episode.timestamps_seconds[transition]
                )
                observable = (after - before) / dt
            losses.append(_huber(command - observable, huber_delta))
    if not losses:
        return math.inf
    return float(np.mean(np.concatenate([loss.ravel() for loss in losses])))


def minimum_score_tie(
    scores: dict[Hypothesis, float], *, absolute_tolerance: float, relative_tolerance: float
) -> frozenset[Hypothesis]:
    if not scores:
        raise ValueError("at least one hypothesis score is required")
    best = min(scores.values())
    tolerance = max(absolute_tolerance, relative_tolerance * abs(best))
    return frozenset(
        hypothesis for hypothesis, score in scores.items() if score <= best + tolerance
    )


def _candidate_grammar(dimension: int) -> list[Hypothesis]:
    permutations = tuple(itertools.permutations(range(dimension)))
    signs = tuple(itertools.product((-1, 1), repeat=dimension))
    scale_vectors = tuple(itertools.product((0.01, 1.0, 2.0), repeat=dimension))
    return [
        Hypothesis(target, permutation, sign, scale, lag)
        for target in ("absolute", "delta", "velocity")
        for permutation in permutations
        for sign in signs
        for scale in scale_vectors
        for lag in range(3)
    ]


def run_synthetic_gate(seed: int, trials: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    grammar = _candidate_grammar(2)
    covered = 0
    for _ in range(trials):
        truth = grammar[int(rng.integers(len(grammar)))]
        episode = generate_episode(
            truth,
            semantic_commands=rng.uniform(-0.5, 0.5, size=(12, 2)),
            initial_state=rng.uniform(-0.1, 0.1, size=2),
            dt_seconds=0.1,
        )
        scores = {candidate: score_hypothesis([episode], candidate) for candidate in grammar}
        tied = minimum_score_tie(
            scores, absolute_tolerance=1e-12, relative_tolerance=1e-9
        )
        covered += truth in tied

    zero_commands = np.zeros((8, 2), dtype=np.float64)
    ambiguous_trace = generate_episode(
        Hypothesis("delta", (0, 1), (1, 1), (1.0, 1.0), 0),
        semantic_commands=zero_commands,
        initial_state=np.zeros(2),
        dt_seconds=0.1,
    )
    ambiguous_scores = {
        candidate: score_hypothesis([ambiguous_trace], candidate) for candidate in grammar
    }
    ambiguous_set = minimum_score_tie(
        ambiguous_scores, absolute_tolerance=1e-12, relative_tolerance=1e-9
    )
    return {
        "trials": trials,
        "covered": covered,
        "coverage": covered / trials,
        "constructed_equivalence_size": len(ambiguous_set),
        "false_unique_count": int(len(ambiguous_set) == 1),
    }


def parse_dataset_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("dataset must use NAME=/path/to/file.parquet")
    name, raw_path = value.split("=", 1)
    normalized_name = name.strip().lower()
    path = Path(raw_path.strip())
    if not normalized_name or not raw_path.strip():
        raise ValueError("dataset name and path must be nonempty")
    return normalized_name, path


def apply_real_dataset_gate(
    expected: dict[str, str], evaluated: dict[str, dict[str, object]]
) -> dict[str, object]:
    checks = {
        name: {
            "expected": outcome,
            "observed": evaluated.get(name, {}).get("outcome", "missing"),
            "matched": evaluated.get(name, {}).get("outcome") == outcome,
        }
        for name, outcome in expected.items()
    }
    return {
        "passed": all(check["matched"] for check in checks.values()),
        "checks": checks,
        "datasets": evaluated,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="real LeRobot parquet shard; repeat for each frozen dataset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    synthetic_config = config["datasets"]["synthetic"]
    trials = args.trials or int(synthetic_config["train_episodes"]) + int(
        synthetic_config["test_episodes"]
    )
    synthetic = run_synthetic_gate(int(config["seed"]), trials)
    synthetic_passed = (
        synthetic["coverage"] >= 0.95 and synthetic["false_unique_count"] == 0
    )
    dataset_paths = dict(parse_dataset_spec(specification) for specification in args.dataset)
    evaluated = {
        name: evaluate_dataset(name, path) for name, path in dataset_paths.items()
    }
    expected = {
        name: str(settings["expected"])
        for name, settings in config["datasets"].items()
        if name != "synthetic"
    }
    real_gate = apply_real_dataset_gate(expected, evaluated)
    passed = synthetic_passed and bool(real_gate["passed"])
    report = {
        "schema_version": "1.0",
        "seed": int(config["seed"]),
        "synthetic": synthetic,
        "real_dataset_gate": real_gate,
        "decision": "continue" if passed else (
            "needs_real_dataset_gate" if synthetic_passed and not dataset_paths else "stop"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2 if report["decision"] == "needs_real_dataset_gate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
