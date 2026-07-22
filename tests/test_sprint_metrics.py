from __future__ import annotations

import pytest

from experiments.sprint_accuracy import (
    accuracy_summary,
    calibration_metrics,
    converter_metrics,
    probe_metrics,
    synthetic_records,
)
from experiments.sprint_performance import latency_quantiles, sprint_matrix


def test_accuracy_counts_fields_equivalence_false_unique_and_abstention() -> None:
    cases = [
        {
            "truth": {"target": "delta", "lag": 1},
            "equivalence_set": ["delta@1"],
            "prediction": {"target": "delta", "lag": 1},
            "predicted_set": ["delta@1"],
            "status": "unique",
        },
        {
            "truth": {"target": "absolute", "lag": 0},
            "equivalence_set": ["absolute@0", "episode_relative@0"],
            "prediction": {"target": "absolute", "lag": 2},
            "predicted_set": ["absolute@0"],
            "status": "unique",
        },
        {
            "truth": {"target": "velocity", "lag": 2},
            "equivalence_set": ["velocity@2"],
            "prediction": {},
            "predicted_set": [],
            "status": "abstain",
        },
    ]

    summary = accuracy_summary(cases, fields=("target", "lag"))

    assert summary["per_field_accuracy"] == {"target": 1.0, "lag": 0.5}
    assert summary["equivalence_set_coverage"] == pytest.approx(2 / 3)
    assert summary["false_unique_certifications"] == 1
    assert summary["abstention_rate"] == pytest.approx(1 / 3)


def test_calibration_has_hand_computed_brier_and_ece() -> None:
    metrics = calibration_metrics([(0.9, True), (0.8, False), (0.2, False)], bins=2)

    assert metrics["brier_score"] == pytest.approx((0.1**2 + 0.8**2 + 0.2**2) / 3)
    assert metrics["expected_calibration_error"] == pytest.approx(
        (1 / 3) * 0.2 + (2 / 3) * 0.35
    )


def test_converter_and_probe_metrics_include_refusals_and_displacement() -> None:
    converters = converter_metrics(
        [
            {"eligible": True, "emitted": True, "safe": True},
            {"eligible": True, "emitted": False, "safe": False},
            {"eligible": False, "emitted": False, "safe": False},
        ]
    )
    probes = probe_metrics(
        [
            {"required": True, "executed": 2, "displacement": 0.3, "resolved": True},
            {"required": True, "executed": 1, "displacement": 0.1, "resolved": False},
            {"required": False, "executed": 0, "displacement": 0.0, "resolved": True},
        ]
    )

    assert converters == {
        "eligible": 2,
        "coverage": 0.5,
        "unsafe_refusal_rate": 1.0,
        "unsafe_emissions": 0,
    }
    assert probes == {
        "required": 2,
        "mean_count_when_required": 1.5,
        "total_displacement": 0.4,
        "resolution_rate": 0.5,
    }


def test_latency_quantiles_are_reported_without_interpolation_surprises() -> None:
    assert latency_quantiles([1.0, 2.0, 3.0, 4.0, 100.0]) == {
        "count": 5,
        "p10_ms": 1.4,
        "median_ms": 3.0,
        "p90_ms": pytest.approx(61.6),
    }


def test_calibrated_method_abstains_on_constructed_equivalence() -> None:
    records = synthetic_records(20260718, 4)
    argmin = [record for record in records if record["method"] == "uncalibrated_argmin"]
    actionabi = [record for record in records if record["method"] == "actionabi"]

    assert accuracy_summary(argmin, fields=("target",))["false_unique_certifications"] >= 1
    calibrated = accuracy_summary(actionabi, fields=("target",))
    assert calibrated["false_unique_certifications"] == 0
    assert calibrated["abstention_rate"] >= 0.25


def test_performance_matrix_sweeps_each_supported_axis() -> None:
    matrix = sprint_matrix(
        hypotheses=(128, 1024), rows=(64, 256), dimensions=(2, 7), threads=(1, 8)
    )

    assert {item["hypotheses"] for item in matrix} == {128, 1024}
    assert {item["rows_per_episode"] for item in matrix} == {64, 256}
    assert {item["dimension"] for item in matrix} == {2, 7}
    assert {item["cpu_threads"] for item in matrix} == {1, 8}
    assert {item["dtype"] for item in matrix} == {"float64"}
    assert len({tuple(sorted(item.items())) for item in matrix}) == len(matrix)
