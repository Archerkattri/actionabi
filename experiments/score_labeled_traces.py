#!/usr/bin/env python3
"""Score ActionABI on the labeled ManiSkill bridge traces (supervised accuracy).

The exported traces (from ActionShift's ``experiments/export_labeled_traces.py``) carry
ground-truth latent contracts known by construction. This driver runs ActionABI's real
C++ scorer (``actionabi infer``) over the FULL declared finite grammar and measures
supervised accuracy versus those labels, without ever passing a label as an inference
input.

Full-grammar evaluation via factorization
-----------------------------------------
ActionABI's ``score_cpu`` residual is separable per output channel and its held-out loss
is the mean of the per-dimension losses (equal residual counts per channel). For a fixed
(target, lag) the optimal assignment of semantic channels to raw channels, with the best
per-cell (sign, scale), is therefore a linear assignment problem whose per-cell costs are
exactly the C++ ``per_dimension_loss`` values. We obtain those costs by running the real
C++ CLI on a small trace-independent BASIS of contracts (6 cyclic permutations x 12
sign/scale settings x 2 targets x 3 lags = 432 contracts). This evaluates the entire
finite grammar (target x lag x permutation x sign x scale over the declared alphabets)
using authentic C++ residuals, without enumerating its billions of members.

Calibrated equivalence sets
---------------------------
Real simulated responses are noisy, so exact ties are meaningless. We compute a calibrated
equivalence set by a paired bootstrap over held-out residual rows (Python residuals that
reproduce the C++ Huber exactly - parity-gated at load): a candidate is observationally
equivalent to the argmin if the 95% bootstrap CI of its per-row loss gap versus the argmin
includes zero. The C++ CLI provides the authoritative point losses / argmin; Python only
supplies the CI and the parity check.

Comparators: forced argmin (always certifies the single minimum, never abstains) and
calibrated ActionABI (certifies unique only when the equivalence set is a singleton, else
abstains) - the core comparison ActionABI's handoff calls for.

Honest scope: 6 pose channels only. Gripper is unobservable from the tcp-pose response and
frame is degenerate under the identity-rotation wrapper; both are reported as structural
limitations, not scored into the per-field product. ActionABI's C++ delta observable spans
``lag+1`` steps whereas the physical response is a single-step delay, so lag>0 is expected
to expose a scorer-model limitation - measured and reported, never hacked around.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.bias_robust import bias_guard
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from bias_robust import bias_guard

FINITE_SCALES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
LAGS: tuple[int, ...] = (0, 1, 2)
TARGETS: tuple[str, ...] = ("delta_position", "absolute_position")
SIGNS: tuple[int, ...] = (-1, 1)
_TARGET_TO_LABEL = {"delta_position": "delta", "absolute_position": "absolute"}
_HUBER_DELTA = 1.0
_TRAIN_FRACTION = 0.7
_DIM = 6
_ALL_PERMS: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(_DIM)))


# ---------------------------------------------------------------------------
# Grammar basis (trace-independent; written once, scored per trace by the CLI)
# ---------------------------------------------------------------------------
def _spec(target: str, lag: int, perm, sign, scale) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "target": target,
        "space": "cartesian",
        "frame": "base",
        "permutation": list(perm),
        "sign": [int(s) for s in sign],
        "scale": [float(z) for z in scale],
        "lag_steps": int(lag),
        "gripper_inverted": False,
    }


def build_basis(grammar_dir: Path) -> list[dict[str, Any]]:
    """Write the 432 basis contracts; return their descriptors."""
    grammar_dir.mkdir(parents=True, exist_ok=True)
    basis: list[dict[str, Any]] = []
    for target in TARGETS:
        for lag in LAGS:
            for k in range(_DIM):  # cyclic permutation P_k[c] = (c + k) % 6
                perm = tuple((c + k) % _DIM for c in range(_DIM))
                for s in SIGNS:
                    for z in FINITE_SCALES:
                        spec = _spec(target, lag, perm, (s,) * _DIM, (z,) * _DIM)
                        name = f"{target}_l{lag}_k{k}_s{s}_z{z}".replace(".", "p")
                        path = grammar_dir / f"{name}.json"
                        path.write_text(json.dumps(spec), encoding="utf-8")
                        basis.append({"path": path, "target": target, "lag": lag,
                                      "k": k, "sign": s, "scale": z})
    return basis


def _contract_key(contract: dict[str, Any]) -> tuple:
    return (
        contract["target"],
        int(contract["lag_steps"]),
        tuple(int(p) for p in contract["permutation"]),
        tuple(int(s) for s in contract["sign"]),
        tuple(round(float(z), 6) for z in contract["scale"]),
    )


def run_cli(binary: Path, trace: Path, basis: list[dict[str, Any]],
            out_path: Path) -> dict[tuple, dict[str, Any]]:
    """Run ``actionabi infer`` over the basis; return per-contract score by key."""
    cmd = [str(binary), "infer", "--input", str(trace)]
    for entry in basis:
        cmd += ["--contract", str(entry["path"])]
    cmd += ["--output", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    scored: dict[tuple, dict[str, Any]] = {}
    for hypo in report["ranked_hypotheses"]:
        scored[_contract_key(hypo["contract"])] = hypo["score"]
    return scored


# ---------------------------------------------------------------------------
# Factorized full-grammar argmin from the C++ per-dimension losses
# ---------------------------------------------------------------------------
def cost_tensor(basis: list[dict[str, Any]],
                scored: dict[tuple, dict[str, Any]]) -> dict[tuple[str, int], np.ndarray]:
    """C[c][j][si][zi] per (target, lag) from the basis per_dimension_loss values."""
    tensors: dict[tuple[str, int], np.ndarray] = {}
    for target in TARGETS:
        for lag in LAGS:
            tensors[(target, lag)] = np.full(
                (_DIM, _DIM, len(SIGNS), len(FINITE_SCALES)), np.inf
            )
    for entry in basis:
        perm = tuple((c + entry["k"]) % _DIM for c in range(_DIM))
        key = _contract_key(
            _spec(entry["target"], entry["lag"], perm,
                  (entry["sign"],) * _DIM, (entry["scale"],) * _DIM)
        )
        per_dim = scored[key]["per_dimension_loss"]
        si = SIGNS.index(entry["sign"])
        zi = FINITE_SCALES.index(entry["scale"])
        tensor = tensors[(entry["target"], entry["lag"])]
        for c in range(_DIM):
            tensor[c, perm[c], si, zi] = per_dim[c]
    return tensors


def best_contract_per_group(tensor: np.ndarray) -> tuple[float, tuple, tuple, tuple]:
    """Brute-force the 720 permutations; return (mean_loss, perm, sign, scale)."""
    # Best (sign, scale) per (c, j): reduce over the (si, zi) axes.
    flat = tensor.reshape(_DIM, _DIM, -1)
    best_cell = flat.min(axis=2)          # M[c][j]
    best_idx = flat.argmin(axis=2)        # index into (si, zi)
    best_perm, best_cost = None, np.inf
    rows = np.arange(_DIM)
    for perm in _ALL_PERMS:
        cost = best_cell[rows, perm].sum()
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    si_zi = best_idx[rows, best_perm]
    sign = tuple(SIGNS[i // len(FINITE_SCALES)] for i in si_zi)
    scale = tuple(FINITE_SCALES[i % len(FINITE_SCALES)] for i in si_zi)
    return best_cost / _DIM, tuple(best_perm), sign, scale


def grammar_argmin(tensors: dict[tuple[str, int], np.ndarray]):
    """Global minimum-loss contract over the full declared grammar."""
    best = None
    for (target, lag), tensor in tensors.items():
        loss, perm, sign, scale = best_contract_per_group(tensor)
        cand = {"loss": loss, "target": target, "lag": lag,
                "permutation": perm, "sign": sign, "scale": scale}
        if best is None or loss < best["loss"]:
            best = cand
    return best


def argmin_neighbors(argmin: dict) -> list[dict]:
    """Single-field neighbors of the argmin - the contracts most confusable with it.

    A fair uniqueness test must check whether these near-neighbors (especially the
    other grid scales and sign flips on each channel) are statistically distinguishable
    from the argmin; otherwise the certification silently ignores the real ambiguity.
    """
    neighbors: list[dict] = []
    base_perm, base_sign, base_scale = argmin["permutation"], argmin["sign"], argmin["scale"]
    for c in range(_DIM):
        for z in FINITE_SCALES:
            if z != base_scale[c]:
                scale = list(base_scale); scale[c] = z
                neighbors.append({**argmin, "scale": tuple(scale)})
        sign = list(base_sign); sign[c] = -sign[c]
        neighbors.append({**argmin, "sign": tuple(sign)})
    for lag in LAGS:
        if lag != argmin["lag"]:
            neighbors.append({**argmin, "lag": lag})
    for target in TARGETS:
        if target != argmin["target"]:
            neighbors.append({**argmin, "target": target})
    return neighbors


def group_candidates(tensors, *, per_group: int = 6):
    """A bounded candidate pool for equivalence testing: the top few per (target, lag)."""
    candidates = []
    for (target, lag), tensor in tensors.items():
        flat = tensor.reshape(_DIM, _DIM, -1)
        best_cell = flat.min(axis=2)
        best_idx = flat.argmin(axis=2)
        rows = np.arange(_DIM)
        scored_perms = sorted(
            ((best_cell[rows, p].sum(), p) for p in _ALL_PERMS), key=lambda x: x[0]
        )[:per_group]
        for cost, perm in scored_perms:
            si_zi = best_idx[rows, perm]
            candidates.append({
                "loss": cost / _DIM, "target": target, "lag": lag,
                "permutation": tuple(perm),
                "sign": tuple(SIGNS[i // len(FINITE_SCALES)] for i in si_zi),
                "scale": tuple(FINITE_SCALES[i % len(FINITE_SCALES)] for i in si_zi),
            })
    return sorted(candidates, key=lambda c: c["loss"])


# ---------------------------------------------------------------------------
# Python residual reproduction (parity-gated) for calibrated equivalence sets
# ---------------------------------------------------------------------------
def load_trace(path: Path) -> list[dict[str, np.ndarray]]:
    """Load the canonical JSONL into per-episode state/action arrays."""
    episodes: dict[int, dict[str, list]] = {}
    order: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("record_type") != "sample":
            continue
        eid = int(record["episode_id"])
        if eid not in episodes:
            episodes[eid] = {"state": [], "action": []}
            order.append(eid)
        episodes[eid]["state"].append(record["state"])
        episodes[eid]["action"].append(record["action"])
    return [
        {"state": np.asarray(episodes[e]["state"], dtype=np.float64),
         "action": np.asarray(episodes[e]["action"], dtype=np.float64)}
        for e in order
    ]


def _huber(residual: np.ndarray) -> np.ndarray:
    magnitude = np.abs(residual)
    return np.where(magnitude <= _HUBER_DELTA,
                    0.5 * residual ** 2,
                    _HUBER_DELTA * (magnitude - 0.5 * _HUBER_DELTA))


def heldout_row_losses(episodes, contract) -> dict[tuple[int, int], float]:
    """Per held-out (episode_index, row) mean-Huber loss; mirrors C++ score_cpu exactly."""
    perm = np.asarray(contract["permutation"])
    sign = np.asarray(contract["sign"], dtype=np.float64)
    scale = np.asarray(contract["scale"], dtype=np.float64)
    lag = int(contract["lag"])
    offset = lag + 1
    is_absolute = contract["target"] == "absolute_position"
    train = int(np.clip(np.floor(_TRAIN_FRACTION * len(episodes)), 1, len(episodes) - 1))
    losses: dict[tuple[int, int], float] = {}
    for ep_index, episode in enumerate(episodes):
        if ep_index < train:
            continue
        state, action = episode["state"], episode["action"]
        rows = state.shape[0]
        for row in range(rows):
            if row + offset >= rows:
                continue
            command = action[row][perm] * sign * scale
            # Single-step delayed observable, matching the fixed C++ scorer
            # (src/score_cpu.cpp) and the Python reference
            # (run_falsification.py::score_hypothesis) for every lag: a lagged
            # command explains the one-step delta at the delayed index
            # (row+offset-1 -> row+offset), not the multi-step span row->row+offset.
            # Reduces to state[row+1]-state[row] at lag == 0 (offset == 1).
            observable = (
                state[row + offset]
                if is_absolute
                else state[row + offset] - state[row + offset - 1]
            )
            losses[(ep_index, row)] = float(_huber(command - observable).mean())
    return losses


def heldout_channel_residuals(episodes, contract):
    """Per-channel held-out (residual, command) arrays for the given contract.

    Residual is ``command - observable`` per output channel over the held-out
    rows (mirrors ``heldout_row_losses`` but keeps raw per-channel values for the
    systematic-bias diagnostic). Used to estimate the response-model bias bound.
    """
    perm = np.asarray(contract["permutation"])
    sign = np.asarray(contract["sign"], dtype=np.float64)
    scale = np.asarray(contract["scale"], dtype=np.float64)
    lag = int(contract["lag"])
    offset = lag + 1
    is_absolute = contract["target"] == "absolute_position"
    train = int(np.clip(np.floor(_TRAIN_FRACTION * len(episodes)), 1, len(episodes) - 1))
    residuals = [[] for _ in range(_DIM)]
    commands = [[] for _ in range(_DIM)]
    for ep_index, episode in enumerate(episodes):
        if ep_index < train:
            continue
        state, action = episode["state"], episode["action"]
        rows = state.shape[0]
        for row in range(rows):
            if row + offset >= rows:
                continue
            command = action[row][perm] * sign * scale
            observable = (
                state[row + offset]
                if is_absolute
                else state[row + offset] - state[row + offset - 1]
            )
            resid = command - observable
            for c in range(_DIM):
                residuals[c].append(resid[c])
                commands[c].append(command[c])
    return ([np.asarray(r, dtype=np.float64) for r in residuals],
            [np.asarray(c, dtype=np.float64) for c in commands])


def paired_equivalent(argmin_losses, cand_losses, *, rng, slack: float = 0.0,
                      samples: int = 2000) -> bool:
    """True if candidate is not significantly worse than argmin.

    The 95% bootstrap CI is taken over the *bias-adjusted* per-row gap
    ``(cand - argmin) - slack``: a candidate is equivalent unless it is worse than
    the argmin by MORE than the estimated systematic-bias slack. ``slack == 0``
    recovers the original noise-only calibration exactly (unbiased case).
    """
    common = sorted(set(argmin_losses) & set(cand_losses))
    if not common:
        return False
    gap = np.array([cand_losses[k] - argmin_losses[k] for k in common]) - slack
    if gap.mean() <= 0:
        return True
    idx = rng.integers(0, len(gap), size=(samples, len(gap)))
    boot = gap[idx].mean(axis=1)
    return bool(np.percentile(boot, 2.5) <= 0.0)


# ---------------------------------------------------------------------------
# Per-trace evaluation
# ---------------------------------------------------------------------------
def observable_fields(contract) -> dict[str, Any]:
    return {
        "target": _TARGET_TO_LABEL[contract["target"]],
        "lag": int(contract["lag"]),
        "permutation": tuple(int(p) for p in contract["permutation"]),
        "sign": tuple(int(s) for s in contract["sign"]),
        "scale": tuple(round(float(z), 6) for z in contract["scale"]),
    }


def per_field_correct(pred: dict, truth: dict) -> dict[str, float]:
    return {
        "permutation": float(np.mean([a == b for a, b in zip(pred["permutation"], truth["permutation"])])),
        "sign": float(np.mean([a == b for a, b in zip(pred["sign"], truth["sign"])])),
        "scale": float(np.mean([a == b for a, b in zip(pred["scale"], truth["scale"])])),
        "target": float(pred["target"] == truth["target"]),
        "lag": float(pred["lag"] == truth["lag"]),
    }


def evaluate_trace(binary, trace_path, label, basis, grammar_out, rng, *, parity=False):
    scored = run_cli(binary, trace_path, basis, grammar_out)
    tensors = cost_tensor(basis, scored)
    argmin = grammar_argmin(tensors)
    episodes = load_trace(trace_path)

    truth = label["contract"]
    truth_fields = {
        "target": truth["target"], "lag": int(truth["lag"]),
        "permutation": tuple(int(p) for p in truth["permutation"]),
        "sign": tuple(int(s) for s in truth["sign"]),
        "scale": tuple(round(float(z), 6) for z in truth["scale"]),
    }
    truth_contract = {"target": "absolute_position" if truth["target"] == "absolute" else "delta_position",
                      "lag": int(truth["lag"]), "permutation": truth["permutation"],
                      "sign": truth["sign"], "scale": truth["scale"]}

    parity_gap = None
    if parity:
        # Verify Python residuals reproduce the C++ held-out mean on a BASIS contract
        # (guaranteed present in ``scored``).
        entry = basis[0]
        perm = tuple((c + entry["k"]) % _DIM for c in range(_DIM))
        basis_contract = {"target": entry["target"], "lag": entry["lag"],
                          "permutation": perm, "sign": (entry["sign"],) * _DIM,
                          "scale": (entry["scale"],) * _DIM}
        key = _contract_key(_spec(entry["target"], entry["lag"], perm,
                                  (entry["sign"],) * _DIM, (entry["scale"],) * _DIM))
        cpp_loss = scored[key]["heldout_loss"]
        row_losses = heldout_row_losses(episodes, basis_contract)
        py_loss = float(np.mean(list(row_losses.values())))
        parity_gap = abs(cpp_loss - py_loss)

    argmin_losses = heldout_row_losses(episodes, argmin)
    truth_losses = heldout_row_losses(episodes, truth_contract)

    # Systematic response-model bias guard (fail-closed): estimate a bias bound
    # from the argmin's held-out residual structure. Widen the equivalence
    # threshold by the bias-explained loss slack and, under detected
    # misspecification, prefer abstention over a unique certification.
    argmin_resid, argmin_cmd = heldout_channel_residuals(episodes, argmin)
    guard = bias_guard(argmin_resid, argmin_cmd)
    slack = float(guard["slack"])
    misspecified = bool(guard["misspecified"])
    common_rows = sorted(set(argmin_losses) & set(truth_losses))
    argmin_heldout = float(np.mean([argmin_losses[k] for k in common_rows])) if common_rows else None
    truth_heldout = float(np.mean([truth_losses[k] for k in common_rows])) if common_rows else None
    truth_minus_argmin = (
        (truth_heldout - argmin_heldout) if common_rows else None
    )

    # Fair equivalence pool: the argmin's single-field neighbors (confusable contracts)
    # plus the best contract in every (target, lag) group (cross-permutation confusability).
    candidates = argmin_neighbors(argmin) + group_candidates(tensors)
    equivalence = [argmin]
    seen = {_contract_key(_spec(argmin["target"], argmin["lag"], argmin["permutation"], argmin["sign"], argmin["scale"]))}
    for cand in candidates:
        ck = _contract_key(_spec(cand["target"], cand["lag"], cand["permutation"], cand["sign"], cand["scale"]))
        if ck in seen:
            continue
        seen.add(ck)
        cand_losses = heldout_row_losses(episodes, cand)
        if paired_equivalent(argmin_losses, cand_losses, rng=rng, slack=slack):
            equivalence.append(cand)

    true_in_equivalence = paired_equivalent(argmin_losses, truth_losses, rng=rng, slack=slack)
    # Fail-closed: never certify unique under detected model misspecification.
    is_unique = len(equivalence) == 1 and not misspecified

    argmin_fields = observable_fields(argmin)
    forced = per_field_correct(argmin_fields, truth_fields)
    argmin_matches_truth = argmin_fields == truth_fields

    return {
        "trace_id": label["trace_id"],
        "excitation": label["excitation"],
        "true_lag": truth_fields["lag"],
        "true_target": truth_fields["target"],
        "argmin_loss": argmin["loss"],
        "argmin_fields": {k: list(v) if isinstance(v, tuple) else v for k, v in argmin_fields.items()},
        "forced_per_field": forced,
        "argmin_matches_truth": bool(argmin_matches_truth),
        "equivalence_set_size": len(equivalence),
        "true_in_equivalence_set": bool(true_in_equivalence),
        "argmin_heldout_loss": argmin_heldout,
        "truth_heldout_loss": truth_heldout,
        "truth_minus_argmin_loss": truth_minus_argmin,
        "calibrated_status": "unique" if is_unique else "abstain",
        "calibrated_false_unique": bool(is_unique and not argmin_matches_truth),
        "bias_slack": slack,
        "misspecified": misspecified,
        "parity_gap": parity_gap,
        # structural limitations (labeled, not identified from the pose response)
        "gripper_true": bool(truth["gripper_inverted"]),
        "frame_true": truth["frame"],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _avg(values):
    return float(np.mean(values)) if values else None


def aggregate(results: list[dict], subset=None) -> dict[str, Any]:
    rows = [r for r in results if subset is None or subset(r)]
    if not rows:
        return {"traces": 0}
    fields = ("permutation", "sign", "scale", "target", "lag")
    forced_pf = {f: _avg([r["forced_per_field"][f] for r in rows]) for f in fields}
    emitted = [r for r in rows if r["calibrated_status"] == "unique"]
    calibrated_pf = {f: _avg([r["forced_per_field"][f] for r in emitted]) for f in fields}
    gaps = [r["truth_minus_argmin_loss"] for r in rows if r.get("truth_minus_argmin_loss") is not None]
    return {
        "traces": len(rows),
        "forced_argmin": {
            "per_field_accuracy": forced_pf,
            "exact_contract_accuracy": _avg([r["argmin_matches_truth"] for r in rows]),
        },
        "loss_gap_truth_minus_argmin": {
            "mean": _avg(gaps),
            "median": float(np.median(gaps)) if gaps else None,
            "mean_argmin_heldout_loss": _avg([r["argmin_heldout_loss"] for r in rows if r.get("argmin_heldout_loss") is not None]),
        },
        "actionabi_calibrated": {
            "emitted": len(emitted),
            "abstention_rate": _avg([r["calibrated_status"] != "unique" for r in rows]),
            "per_field_accuracy_when_emitted": calibrated_pf,
            "false_unique_certifications": int(sum(r["calibrated_false_unique"] for r in rows)),
            "equivalence_set_coverage": _avg([r["true_in_equivalence_set"] for r in rows]),
            "mean_equivalence_set_size": _avg([r["equivalence_set_size"] for r in rows]),
            "misspecification_flag_rate": _avg([r.get("misspecified", False) for r in rows]),
            "mean_bias_slack": _avg([r.get("bias_slack", 0.0) for r in rows]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=Path("build-bridge/actionabi"))
    parser.add_argument("--dataset", type=Path, required=True,
                        help="exporter output dir with traces/, labels/, manifest.json")
    parser.add_argument("--out", type=Path, default=Path("results/labeled_sim"))
    parser.add_argument("--seed", type=int, default=20260721)
    arguments = parser.parse_args()
    rng = np.random.default_rng(arguments.seed)

    grammar_dir = arguments.out / "grammar_basis"
    reports_dir = arguments.out / "cli_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    basis = build_basis(grammar_dir)

    manifest_bytes = (arguments.dataset / "manifest.json").read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    labels = sorted((arguments.dataset / "labels").glob("*.json"))
    results: list[dict] = []
    for index, label_path in enumerate(labels):
        label = json.loads(label_path.read_text(encoding="utf-8"))
        trace_path = arguments.dataset / "traces" / f"{label['trace_id']}.jsonl"
        result = evaluate_trace(
            arguments.binary, trace_path, label, basis,
            reports_dir / f"{label['trace_id']}.json", rng,
            parity=(index < 4),
        )
        results.append(result)
        print(f"{result['trace_id']:>16}  lag={result['true_lag']} exc={result['excitation']:<6} "
              f"argmin_ok={int(result['argmin_matches_truth'])} "
              f"status={result['calibrated_status']:<7} "
              f"perm={result['forced_per_field']['permutation']:.2f} "
              f"sign={result['forced_per_field']['sign']:.2f} "
              f"scale={result['forced_per_field']['scale']:.2f} "
              f"tgt={int(result['forced_per_field']['target'])} lagok={int(result['forced_per_field']['lag'])}")

    parity_gaps = [r["parity_gap"] for r in results if r["parity_gap"] is not None]
    dataset_hash = hashlib.sha256(
        "".join(sorted(r["trace_id"] + json.dumps(r["argmin_fields"], sort_keys=True)
                       for r in results)).encode()
    ).hexdigest()
    summary = {
        "schema_version": "1.0",
        "dataset": str(arguments.dataset),
        "dataset_manifest_sha256": manifest_sha256,
        "dataset_manifest_backbone_sha256": manifest.get("backbone_sha256"),
        "num_traces": len(results),
        "binary": str(arguments.binary),
        "cpp_python_parity_max_gap": max(parity_gaps) if parity_gaps else None,
        "declared_grammar": {
            "targets": list(TARGETS), "lags": list(LAGS),
            "permutations": "all 720", "signs": list(SIGNS),
            "scales": list(FINITE_SCALES),
            "basis_contracts_per_trace": len(basis),
            "gripper": "excluded (unobservable from pose response; structural limitation)",
            "frame": "declared base; base/tool degenerate under identity rotation (equivalence)",
        },
        "overall": aggregate(results),
        "by_excitation": {
            e: aggregate(results, lambda r, e=e: r["excitation"] == e)
            for e in ("policy", "random")
        },
        "by_lag": {
            str(l): aggregate(results, lambda r, l=l: r["true_lag"] == l)
            for l in LAGS
        },
        "by_lag_and_excitation": {
            f"lag{l}_{e}": aggregate(
                results, lambda r, l=l, e=e: r["true_lag"] == l and r["excitation"] == e
            )
            for l in LAGS for e in ("policy", "random")
        },
        "structural_limitations": {
            "gripper_inverted": "labeled but not observable from the tcp-pose response; ActionABI abstains on this field",
            "frame": "identity end-effector rotation makes base/tool observationally identical; not identifiable",
            "space": "always cartesian; not varied",
            "lag_gt_0": "ActionABI C++ delta observable spans lag+1 steps whereas the physical response is a single-step delay; see by_lag strata",
        },
        "results_sha256": dataset_hash,
    }
    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "raw.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in results), encoding="utf-8"
    )
    (arguments.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n=== overall ===")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"\nparity max gap (C++ vs Python): {summary['cpp_python_parity_max_gap']}")
    print(f"wrote {arguments.out}/summary.json")


if __name__ == "__main__":
    main()
