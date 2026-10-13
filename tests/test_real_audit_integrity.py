"""Trajectory boundary and projected-column regressions for the public-data audit."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from adapters.lerobot_to_jsonl import _read_parquet_rows
from experiments import run_hub_audit as hub
from experiments.real_dataset_gate import PassiveDataset
from experiments.run_hub_audit import HUB_DOCUMENTATION_LABELS, main


@pytest.mark.parametrize("episodes,times,message", [
    ([0, 1, 0], [0., 0., 1.], "contiguous"),
    ([0, 0, 1], [0., 0., 0.], "increasing"),
    ([0, 0, 1], [1., 0., 0.], "increasing"),
    ([0, 0, 1], [0., float("nan"), 0.], "finite"),
    ([0, 0, 1], [0., 1., float("inf")], "finite"),
])
def test_temporal_evidence_rejects_invalid_episode_boundaries(episodes, times, message):
    with pytest.raises(ValueError, match=message):
        PassiveDataset(np.zeros((3, 1)), np.zeros((3, 1)),
                       np.array(episodes), np.array(times))


def test_time_can_reset_between_independent_episodes():
    PassiveDataset(np.zeros((4, 1)), np.zeros((4, 1)),
                   np.array([0, 0, 1, 1]), np.array([0., 1., 0., 1.]))


def test_missing_projected_columns_fail_before_any_blob_is_read(tmp_path, monkeypatch):
    import pyarrow as pa
    from pyarrow import parquet

    shard = tmp_path / "incomplete.parquet"
    parquet.write_table(pa.table({"action": [[0.]], "large_blob": [b"unused"]}), shard)
    reads = []
    monkeypatch.setattr(parquet, "read_table", lambda *args, **kwargs: reads.append(args))
    with pytest.raises(ValueError, match="missing columns"):
        _read_parquet_rows(shard, ["action", "observation.state"])
    assert reads == []


def test_missing_datasets_are_reported_and_do_not_return_success(tmp_path: Path):
    harvest = tmp_path / "harvest.json"
    harvest.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "report.json"
    assert main([
        "--harvest", str(harvest), "--data-root", str(tmp_path / "missing"),
        "--out-dir", str(tmp_path / "reports"), "--audit-json", str(report_path), "--no-cli",
    ]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["requested_datasets"] == report["missing_datasets"] == len(HUB_DOCUMENTATION_LABELS)
    assert report["aggregate"]["datasets_audited"] == 0
    assert all(item["status"] == "MISSING_INPUT" for item in report["availability"].values())


def _audit_arguments(tmp_path, *, work=None, out_dir=None, audit_json=None):
    harvest = tmp_path / "harvest.json"
    harvest.write_text("{}", encoding="utf-8")
    return [
        "--harvest", str(harvest), "--data-root", str(tmp_path / "data"),
        "--work", str(work or tmp_path / "work"),
        "--out-dir", str(out_dir or tmp_path / "reports"),
        "--audit-json", str(audit_json or tmp_path / "report.json"), "--no-cli",
    ]


@pytest.mark.parametrize("destination", ["work", "out_dir", "audit_json"])
def test_driver_rejects_dataset_tree_destinations_before_any_write(tmp_path, destination):
    unsafe = tmp_path / "data" / "new-output"
    with pytest.raises(ValueError, match="source dataset tree"):
        main(_audit_arguments(tmp_path, **{destination: unsafe}))
    assert not unsafe.exists()
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("existing", ["directory", "file"])
def test_driver_rejects_work_reuse_before_any_write(tmp_path, existing):
    work = tmp_path / "work"
    if existing == "directory":
        work.mkdir()
        sentinel = work / "contract_absolute.json"
    else:
        sentinel = work
    sentinel.write_bytes(b"original evidence")
    with pytest.raises(ValueError, match="reuse or overwrite"):
        main(_audit_arguments(tmp_path))
    assert sentinel.read_bytes() == b"original evidence"
    assert not (tmp_path / "reports").exists()


def test_driver_rejects_repository_source_work_and_normalized_dataset_paths(tmp_path):
    with pytest.raises(ValueError, match="repository source tree"):
        main(_audit_arguments(tmp_path, work=hub.REPO_ROOT / "experiments" / "new-cli-work"))
    work = tmp_path / "other" / ".." / "data" / "new-cli-work"
    with pytest.raises(ValueError, match="source dataset tree"):
        main(_audit_arguments(tmp_path, work=work))
    assert not (tmp_path / "other").exists()


def _symlink(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")


def test_driver_rejects_work_through_a_dataset_symlink(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    alias = tmp_path / "alias"
    _symlink(alias, source, directory=True)
    with pytest.raises(ValueError, match="source dataset tree"):
        main(_audit_arguments(tmp_path, work=alias / "new-cli-work"))
    assert list(source.iterdir()) == []


def test_driver_protects_a_dataset_referent_outside_the_data_root(tmp_path):
    source = tmp_path / "external-dataset"
    source.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    _symlink(data / next(iter(HUB_DOCUMENTATION_LABELS)), source, directory=True)
    with pytest.raises(ValueError, match="source dataset tree"):
        main(_audit_arguments(tmp_path, work=source / "new-cli-work"))
    assert list(source.iterdir()) == []


@pytest.mark.parametrize("destination", ["work", "out_dir", "audit_json"])
def test_driver_refuses_dangling_destination_symlinks(tmp_path, destination):
    link = tmp_path / "dangling"
    target = tmp_path / "uncreated"
    _symlink(link, target, directory=destination != "audit_json")
    with pytest.raises(ValueError, match="reuse or overwrite"):
        main(_audit_arguments(tmp_path, **{destination: link}))
    assert link.is_symlink()
    assert not target.exists()


def test_driver_rejects_aggregate_dataset_output_collision(tmp_path):
    output = tmp_path / "reports" / f"{next(iter(HUB_DOCUMENTATION_LABELS))}.json"
    with pytest.raises(ValueError, match="conflicts"):
        main(_audit_arguments(tmp_path, audit_json=output))
    assert not output.parent.exists()


def test_driver_rejects_overlapping_work_and_report_destinations(tmp_path):
    with pytest.raises(ValueError, match="must not overlap"):
        main(_audit_arguments(tmp_path, out_dir=tmp_path / "work" / "reports"))
    assert not (tmp_path / "work").exists()


def test_driver_rejects_empty_work_directory_reuse(tmp_path):
    (tmp_path / "work").mkdir()
    with pytest.raises(ValueError, match="reuse or overwrite"):
        main(_audit_arguments(tmp_path))
    assert list((tmp_path / "work").iterdir()) == []


def test_json_output_creation_cannot_overwrite_existing_evidence(tmp_path):
    output = tmp_path / "evidence.json"
    hub.write_new_json(output, {"original": True})
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        hub.write_new_json(output, {"replacement": True})
    assert output.read_bytes() == before


@pytest.fixture
def local_trajectory(tmp_path):
    import pyarrow as pa
    from pyarrow import parquet

    name = next(iter(HUB_DOCUMENTATION_LABELS))
    data = tmp_path / "data" / name
    data.mkdir(parents=True)
    steps = 24
    values = np.sin(np.arange(steps * 3) * .17).reshape(-1, 1).tolist()
    parquet.write_table(pa.table({
        "observation.state": values, "action": values,
        "episode_index": np.repeat(np.arange(3), steps),
        "timestamp": np.tile(np.arange(steps) / 10., 3),
    }), data / "trajectory.parquet")
    meta = {
        "hf_repo": HUB_DOCUMENTATION_LABELS[name]["hf_repo"],
        "revision": "a" * 40, "source_sha256": hub.sha256_path(data), "license": "MIT",
    }
    return name, data, meta


def test_cli_corroboration_writes_fresh_artifacts_and_refuses_reuse(
    tmp_path, local_trajectory, monkeypatch
):
    _, data, _ = local_trajectory
    actual_run = subprocess.run
    calls = []

    def scorer_run(command, **kwargs):
        calls.append(command)
        if command[1] != "infer":
            # Exercise the production Parquet adapter in its own interpreter.
            return actual_run(command, **kwargs)
        output = Path(command[command.index("--output") + 1])
        contracts = [
            json.loads(Path(command[index + 1]).read_text())
            for index, value in enumerate(command) if value == "--contract"
        ]
        hub.write_new_json(output, {"ranked_hypotheses": [
            {"contract": contract, "score": {"heldout_loss": float(index + 1)}}
            for index, contract in enumerate(contracts)
        ]})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(hub.subprocess, "run", scorer_run)
    work = tmp_path / "cli-work"
    report = hub.cli_corroborate(Path("synthetic-scorer"), data, 1, work)
    assert report["min_loss_target"] == "absolute"
    assert len(calls) == 2
    snapshots = {path.name: path.read_bytes() for path in work.iterdir()}
    assert set(snapshots) == {
        "traj.jsonl", "contract_absolute.json", "contract_delta.json",
        "contract_velocity.json", "evidence.json",
    }
    with pytest.raises(ValueError, match="reuse or overwrite"):
        hub.cli_corroborate(Path("synthetic-scorer"), data, 1, work)
    assert len(calls) == 2
    assert snapshots == {path.name: path.read_bytes() for path in work.iterdir()}


def test_direct_cli_rejects_source_destination_before_launch(tmp_path, monkeypatch):
    def unexpected_run(*args, **kwargs):
        pytest.fail("unsafe destination must be rejected before subprocess launch")

    monkeypatch.setattr(hub.subprocess, "run", unexpected_run)
    source = tmp_path / "data"
    with pytest.raises(ValueError, match="source dataset tree"):
        hub.cli_corroborate(Path("unused"), source, 1, source / "cli-work")


@pytest.mark.parametrize("override,message", [
    ({"revision": "main"}, "immutable"),
    ({"revision": "a" * 39}, "immutable"),
    ({"hf_repo": "another/repository"}, "repository"),
    ({"source_sha256": None}, "expected source_sha256"),
    ({"source_sha256": "b" * 64}, "does not match"),
])
def test_harvest_claims_are_validated_before_evaluation(
    tmp_path, local_trajectory, monkeypatch, override, message
):
    name, data, meta = local_trajectory
    monkeypatch.setattr(hub, "evaluate_dataset", lambda *args: pytest.fail("evaluation launched"))
    with pytest.raises(ValueError, match=message):
        hub.audit_dataset(name, data, {**meta, **override}, None, tmp_path / "work")


def test_driver_writes_real_gate_report_without_inventing_license_or_revision_binding(
    tmp_path, local_trajectory
):
    name, _, meta = local_trajectory
    args = _audit_arguments(tmp_path)
    (tmp_path / "harvest.json").write_text(json.dumps({meta["hf_repo"]: meta}), encoding="utf-8")
    assert main(args) == 2  # One synthetic input; the other requested inputs are missing.
    aggregate = json.loads((tmp_path / "report.json").read_text())
    report = json.loads((tmp_path / "reports" / f"{name}.json").read_text())
    assert aggregate["datasets"][name] == report
    assert report["source_sha256"] == meta["source_sha256"]
    assert report["provenance"]["content_validation"]["local_content_matches_harvest"] is True
    assert report["provenance"]["harvest_metadata"]["declared_license"] == "MIT"
    assert report["provenance"]["license_status"] == "UNVERIFIED"
    assert report["provenance"]["revision_status"] == "UNVERIFIED_REMOTE_BINDING"
    before = (tmp_path / "report.json").read_bytes()
    with pytest.raises(ValueError, match="reuse or overwrite"):
        main(args)
    assert (tmp_path / "report.json").read_bytes() == before
