#!/usr/bin/env python3
"""Run auditable ActionABI accuracy/calibration comparisons.

Real passive datasets have no latent-contract ground truth, so this driver records
their evidence and abstentions but deliberately excludes them from supervised
accuracy. Synthetic labels are generated independently of all documentation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from experiments.real_dataset_gate import evaluate_dataset
    from experiments.run_falsification import (
        _candidate_grammar,
        generate_episode,
        minimum_score_tie,
        score_hypothesis,
    )
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from real_dataset_gate import evaluate_dataset
    from run_falsification import (
        _candidate_grammar,
        generate_episode,
        minimum_score_tie,
        score_hypothesis,
    )


def _mean(values: Sequence[bool | float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def accuracy_summary(
    cases: Iterable[Mapping[str, Any]], *, fields: Sequence[str]
) -> dict[str, Any]:
    materialized = list(cases)
    field_scores: dict[str, float | None] = {}
    for field in fields:
        comparable = [
            case["prediction"][field] == case["truth"][field]
            for case in materialized
            if field in case.get("truth", {}) and field in case.get("prediction", {})
        ]
        field_scores[field] = _mean(comparable)

    labeled_sets = [case for case in materialized if case.get("equivalence_set")]
    covered = [
        bool(set(case.get("predicted_set", ())) & set(case["equivalence_set"]))
        for case in labeled_sets
    ]
    false_unique = sum(
        case.get("status") == "unique"
        and (
            len(case.get("equivalence_set", ())) != 1
            or not bool(set(case.get("predicted_set", ())) & set(case["equivalence_set"]))
        )
        for case in labeled_sets
    )
    abstained = [
        case.get("status") in {"abstain", "unsupported"} for case in materialized
    ]
    return {
        "cases": len(materialized),
        "per_field_accuracy": field_scores,
        "equivalence_set_coverage": _mean(covered),
        "false_unique_certifications": int(false_unique),
        "abstention_rate": _mean(abstained),
    }


def calibration_metrics(
    observations: Iterable[tuple[float, bool]], *, bins: int = 10
) -> dict[str, float | int | None]:
    values = list(observations)
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not values:
        return {"count": 0, "brier_score": None, "expected_calibration_error": None}
    for confidence, _ in values:
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
    brier = float(np.mean([(confidence - float(correct)) ** 2 for confidence, correct in values]))
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            (confidence, correct)
            for confidence, correct in values
            if lower <= confidence <= upper and (index == bins - 1 or confidence < upper)
        ]
        if bucket:
            mean_confidence = float(np.mean([item[0] for item in bucket]))
            accuracy = float(np.mean([item[1] for item in bucket]))
            ece += len(bucket) / len(values) * abs(mean_confidence - accuracy)
    return {"count": len(values), "brier_score": brier, "expected_calibration_error": ece}


def converter_metrics(cases: Iterable[Mapping[str, Any]]) -> dict[str, int | float | None]:
    materialized = list(cases)
    eligible = [case for case in materialized if case.get("eligible")]
    unsafe = [case for case in materialized if not case.get("safe")]
    return {
        "eligible": len(eligible),
        "coverage": _mean([bool(case.get("emitted")) for case in eligible]),
        "unsafe_refusal_rate": _mean([not bool(case.get("emitted")) for case in unsafe]),
        "unsafe_emissions": sum(bool(case.get("emitted")) for case in unsafe),
    }


def probe_metrics(cases: Iterable[Mapping[str, Any]]) -> dict[str, int | float | None]:
    required = [case for case in cases if case.get("required")]
    return {
        "required": len(required),
        "mean_count_when_required": _mean([float(case.get("executed", 0)) for case in required]),
        "total_displacement": float(sum(float(case.get("displacement", 0.0)) for case in required)),
        "resolution_rate": _mean([bool(case.get("resolved")) for case in required]),
    }


def _hypothesis_id(hypothesis: Any) -> str:
    return (
        f"{hypothesis.target}|p={','.join(map(str, hypothesis.permutation))}"
        f"|s={','.join(map(str, hypothesis.sign))}|z={','.join(map(str, hypothesis.scale))}"
        f"|l={hypothesis.lag_steps}"
    )


def synthetic_records(seed: int, trials: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grammar = _candidate_grammar(2)
    records: list[dict[str, Any]] = []
    for trial in range(trials):
        truth = grammar[int(rng.integers(len(grammar)))]
        ambiguous = trial % 4 == 0
        episode = generate_episode(
            truth,
            semantic_commands=(
                np.zeros((12, 2))
                if ambiguous
                else rng.uniform(-0.5, 0.5, size=(12, 2))
            ),
            initial_state=(
                np.zeros(2) if ambiguous else rng.uniform(-0.1, 0.1, size=2)
            ),
            dt_seconds=0.1,
        )
        scores = {candidate: score_hypothesis([episode], candidate) for candidate in grammar}
        tied = minimum_score_tie(scores, absolute_tolerance=1e-12, relative_tolerance=1e-9)
        ordered = sorted(scores, key=lambda candidate: (scores[candidate], _hypothesis_id(candidate)))
        truth_id = _hypothesis_id(truth)
        equivalence_set = [_hypothesis_id(item) for item in sorted(tied, key=_hypothesis_id)]
        if truth_id not in equivalence_set:
            raise RuntimeError("synthetic truth fell outside its calibrated equivalence set")
        for method, selected, status in (
            ("metadata_only", [], "abstain"),
            ("uncalibrated_argmin", [ordered[0]], "unique"),
            ("actionabi", sorted(tied, key=_hypothesis_id), "unique" if len(tied) == 1 else "abstain"),
        ):
            prediction = selected[0] if len(selected) == 1 else None
            records.append(
                {
                    "source": "synthetic",
                    "trial": trial,
                    "method": method,
                    "truth": {"target": truth.target, "lag": truth.lag_steps},
                    "equivalence_set": equivalence_set,
                    "prediction": (
                        {"target": prediction.target, "lag": prediction.lag_steps}
                        if prediction is not None else {}
                    ),
                    "predicted_set": [_hypothesis_id(item) for item in selected],
                    "status": status,
                    "confidence": 1.0 if status == "unique" else 0.0,
                }
            )
    return records


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/sprint/accuracy"))
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    records = synthetic_records(arguments.seed, arguments.trials)
    real_evidence: list[dict[str, Any]] = []
    for specification in arguments.dataset:
        name, raw_path = specification.split("=", 1)
        evaluated = evaluate_dataset(name.strip().lower(), Path(raw_path))
        real_evidence.append(
            {
                "dataset": name.strip().lower(),
                "source_sha256": evaluated["source_sha256"],
                "rows": evaluated["rows"],
                "episodes": evaluated["episodes"],
                "outcome": evaluated["outcome"],
                "accuracy_exclusion": "latent contract is not labeled in passive data",
            }
        )
    _write_jsonl(arguments.out / "raw.jsonl", records)
    methods: dict[str, Any] = {}
    for method in ("metadata_only", "uncalibrated_argmin", "actionabi"):
        subset = [record for record in records if record["method"] == method]
        correctness = [
            (
                float(record["confidence"]),
                len(record["equivalence_set"]) == 1
                and bool(set(record["predicted_set"]) & set(record["equivalence_set"])),
            )
            for record in subset
        ]
        methods[method] = {
            **accuracy_summary(subset, fields=("target", "lag")),
            "calibration": calibration_metrics(correctness),
        }
    summary = {
        "schema_version": "1.0",
        "seed": arguments.seed,
        "synthetic_trials": arguments.trials,
        "methods": methods,
        "real_passive_evidence": real_evidence,
        "claim_boundary": "supervised accuracy is synthetic-only; real logs provide unlabeled passive evidence",
    }
    (arguments.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
