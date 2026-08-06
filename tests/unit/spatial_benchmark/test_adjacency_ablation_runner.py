"""Focused contract tests for the grouped adjacency-ablation runner.

These tests deliberately avoid loading the ten-core cohort.  They bind config
validation to the immutable materialized smoke config, exercise the actual
message-passing model on small tensors, and replace only data/training/evaluation
with synthetic results when checking the runner's archive-facing fields.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spatial_benchmark.adjacency_ablation import (
    add_exact_self_adjacency,
    build_five_fold_splits,
    build_seeded_explicit_self_model,
    derive_evaluation_mask_seed,
    mask_realization_sha256,
    ndarray_sha256,
    sample_uniform_mask_numpy,
    state_dict_sha256,
)
from spatial_benchmark.configuration import load_yaml_mapping
from spatial_benchmark.metrics import masked_huber_loss


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/train/run_adjacency_ablation.py"
_SPEC = importlib.util.spec_from_file_location(
    "test_adjacency_ablation_runner_module", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)

_LOCKED_ROOT = (
    _ROOT
    / "scratch/locked_campaigns"
    / _RUNNER.CAMPAIGN_ID
)
_SMOKE_CONFIG = (
    _LOCKED_ROOT / "configs/smoke/fold_0/seed_0/spatial.yaml"
)
_PRIMARY_CONFIG = (
    _LOCKED_ROOT / "configs/primary/fold_0/seed_0/isolated.yaml"
)
_NULL_CONFIG = (
    _LOCKED_ROOT
    / "configs/null/fold_0/seed_0/position_permuted_null.yaml"
)


def _materialized_smoke_config() -> dict[str, Any]:
    if not _SMOKE_CONFIG.is_file():
        pytest.fail(
            "the immutable adjacency-ablation smoke config is missing; "
            "run enqueue_adjacency_ablation.py materialize first"
        )
    return dict(load_yaml_mapping(_SMOKE_CONFIG))


def _materialized_primary_recovery_config() -> dict[str, Any]:
    config = dict(load_yaml_mapping(_PRIMARY_CONFIG))
    config["attempt"] = 2
    return config


def _materialized_null_config() -> dict[str, Any]:
    return dict(load_yaml_mapping(_NULL_CONFIG))


def _recovery_plan() -> dict[str, Any]:
    path = _ROOT / _RUNNER.CPU_RECOVERY_PLAN_RELATIVE
    if not path.is_file():
        pytest.fail(f"signed CPU recovery plan is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_recovery_authority(
    root: Path, *, include_plan: bool = True
) -> None:
    references = [
        _RUNNER.RECOVERY_CONTRACT_RELATIVE,
        _RUNNER.MATERIALIZATION_RELATIVE,
    ]
    if include_plan:
        references.append(_RUNNER.CPU_RECOVERY_PLAN_RELATIVE)
    for reference in references:
        source = _ROOT / reference
        target = root / reference
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _write_null_enqueue_receipt(root: Path) -> dict[str, Any]:
    materialization = json.loads(
        (_ROOT / _RUNNER.MATERIALIZATION_RELATIVE).read_text(encoding="utf-8")
    )
    plan = _recovery_plan()
    plan_by_slot = {
        (item["fold"], item["seed"], item["condition"]): item
        for item in plan["conditional_null_jobs"]
    }
    jobs: list[dict[str, Any]] = []
    for item in materialization["jobs"]:
        if item["stage"] != "null":
            continue
        slot = (item["fold"], item["seed"], item["condition"])
        authorization = plan_by_slot[slot]
        jobs.append(
            {
                "stage": "null",
                "fold": item["fold"],
                "seed": item["seed"],
                "condition": item["condition"],
                "config_sha256": item["config_sha256"],
                "scientific_id": item["scientific_id"],
                "job_id": f"q_null_f{item['fold']}_s{item['seed']}",
                "requested_gpu": authorization["worker_slot"],
                "maximum_attempts": 1,
            }
        )
        reference = Path(item["config_reference"])
        target = root / reference
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / reference, target)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": _RUNNER.NULL_ENQUEUE_KIND,
        "campaign_id": _RUNNER.CAMPAIGN_ID,
        "stage": "null",
        "materialization_checksum": _RUNNER.MATERIALIZATION_CHECKSUM,
        "pilot_gate_checksum": None,
        "null_trigger_checksum": "d" * 64,
        "recovery_plan_checksum": plan["checksum"],
        "recovery_enqueue_checksum": _RUNNER.CPU_RECOVERY_ENQUEUE_CHECKSUM,
        "execution_device": "cpu",
        "maximum_attempts": 1,
        "complete": True,
        "jobs": jobs,
    }
    payload["checksum"] = _RUNNER.canonical_sha256(payload)
    path = root / _RUNNER.NULL_ENQUEUE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _use_recovery_root(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    monkeypatch.setattr(
        _RUNNER,
        "current_paths",
        lambda: SimpleNamespace(project_root=root),
    )
    monkeypatch.setattr(_RUNNER.torch.cuda, "is_available", lambda: False)


def _replace(config: dict[str, Any], path: Sequence[str], value: object) -> None:
    target: dict[str, Any] = config
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value


def test_materialized_smoke_config_is_the_exact_frozen_runner_identity() -> None:
    config = _materialized_smoke_config()

    assert _RUNNER._validate_config(config) == ("smoke", "spatial", 0, 0)
    assert config["dataset"] == {
        **config["dataset"],
        "dataset_id": _RUNNER.EXPECTED_DATASET_ID,
        "version": _RUNNER.EXPECTED_DATASET_VERSION,
        "split_id": _RUNNER.EXPECTED_SPLIT_ID,
        "dataset_fingerprint": _RUNNER.EXPECTED_DATASET_FINGERPRINT,
        "split_fingerprint": _RUNNER.EXPECTED_SPLIT_FINGERPRINT,
        "preprocessing_version": _RUNNER.EXPECTED_PREPROCESSING_FINGERPRINT,
        "prepared_manifest": _RUNNER.EXPECTED_PREPARED_MANIFEST,
        "prepared_manifest_sha256": (
            _RUNNER.EXPECTED_PREPARED_MANIFEST_SHA256
        ),
        "prepared_content_sha256": _RUNNER.EXPECTED_PREPARED_CONTENT_SHA256,
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("campaign", "frozen_contract_sha256"), "0" * 64),
        (("dataset", "dataset_id"), "wrong_adjacent_normal_dataset"),
        (("dataset", "version"), "wrong_version"),
        (("dataset", "split_id"), "wrong_grouped_split"),
        (("dataset", "dataset_fingerprint"), "1" * 64),
        (("dataset", "split_fingerprint"), "2" * 64),
        (("dataset", "prepared_manifest_sha256"), "3" * 64),
        (("dataset", "prepared_content_sha256"), "4" * 64),
        (("preprocessing", "fit_scope"), "all_cores"),
        (("model", "dropout"), 0.2),
        (("graph", "radius_um"), 49.0),
        (("masking", "training_base_seed"), 2026080202),
        (("trainer", "learning_rate"), 0.002),
        (("evaluation", "primary_metric"), "val/masked_mae"),
    ],
)
def test_runner_rejects_each_frozen_identity_departure(
    path: tuple[str, ...], replacement: object
) -> None:
    config = deepcopy(_materialized_smoke_config())
    _replace(config, path, replacement)

    with pytest.raises(Exception) as captured:
        _RUNNER._validate_config(config)

    assert type(captured.value).__name__ in {
        "AdjacencyRunnerError",
        "ConfigurationError",
    }


def test_runner_rejects_null_topology_outside_the_null_stage() -> None:
    config = deepcopy(_materialized_smoke_config())
    config["graph"]["adjacency_condition"] = "position_permuted_null"

    with pytest.raises(
        _RUNNER.AdjacencyRunnerError,
        match="null adjacency is restricted to null stage",
    ):
        _RUNNER._validate_config(config)


def test_primary_attempt_two_is_config_valid_but_recovery_gated() -> None:
    config = _materialized_primary_recovery_config()

    assert _RUNNER._validate_config(config) == ("primary", "isolated", 0, 0)


def test_cpu_recovery_rejects_missing_signed_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_recovery_authority(tmp_path, include_plan=False)
    _use_recovery_root(monkeypatch, tmp_path)
    monkeypatch.setenv("BAGM_JOB_ID", "q_missing_plan_test")
    config = _materialized_primary_recovery_config()

    with pytest.raises(
        _RUNNER.AdjacencyRunnerError, match="signed CPU recovery plan is missing"
    ):
        _RUNNER._resolve_execution(
            config,
            stage="primary",
            condition="isolated",
            fold=0,
            seed=0,
            requested_device=None,
        )


@pytest.mark.parametrize("tampered_authority", ["contract", "plan"])
def test_cpu_recovery_rejects_tampered_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_authority: str,
) -> None:
    _copy_recovery_authority(tmp_path)
    _use_recovery_root(monkeypatch, tmp_path)
    plan = _recovery_plan()
    planned = next(
        item
        for item in plan["primary_jobs"]
        if item["fold"] == 0
        and item["seed"] == 0
        and item["condition"] == "isolated"
    )
    monkeypatch.setenv("BAGM_JOB_ID", planned["retry_job_id"])
    if tampered_authority == "contract":
        path = tmp_path / _RUNNER.RECOVERY_CONTRACT_RELATIVE
        path.write_bytes(path.read_bytes() + b"\n# tampered\n")
        match = "hardware recovery contract checksum changed"
    else:
        path = tmp_path / _RUNNER.CPU_RECOVERY_PLAN_RELATIVE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["execution"]["concurrent_workers"] = 7
        path.write_text(json.dumps(payload), encoding="utf-8")
        match = "plan failed strict verification"

    with pytest.raises(_RUNNER.AdjacencyRunnerError, match=match):
        _RUNNER._resolve_execution(
            _materialized_primary_recovery_config(),
            stage="primary",
            condition="isolated",
            fold=0,
            seed=0,
            requested_device=None,
        )


def test_cpu_recovery_rejects_unauthorized_queue_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_recovery_authority(tmp_path)
    _use_recovery_root(monkeypatch, tmp_path)
    monkeypatch.setenv("BAGM_JOB_ID", "q_not_the_signed_retry")

    with pytest.raises(
        _RUNNER.AdjacencyRunnerError,
        match="queue job is not the retry authorized",
    ):
        _RUNNER._resolve_execution(
            _materialized_primary_recovery_config(),
            stage="primary",
            condition="isolated",
            fold=0,
            seed=0,
            requested_device="cpu",
        )


def test_signed_primary_recovery_forces_cpu_even_if_cuda_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_recovery_authority(tmp_path)
    _use_recovery_root(monkeypatch, tmp_path)
    monkeypatch.setattr(_RUNNER.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        _RUNNER, "_configure_cpu_recovery_threads", lambda: (4, 1)
    )
    plan = _recovery_plan()
    planned = next(
        item
        for item in plan["primary_jobs"]
        if item["fold"] == 0
        and item["seed"] == 0
        and item["condition"] == "isolated"
    )
    monkeypatch.setenv("BAGM_JOB_ID", planned["retry_job_id"])

    execution = _RUNNER._resolve_execution(
        _materialized_primary_recovery_config(),
        stage="primary",
        condition="isolated",
        fold=0,
        seed=0,
        requested_device=None,
    )

    assert execution.device == torch.device("cpu")
    assert execution.mode == "cpu_hardware_recovery"
    assert execution.recovery_plan_checksum == plan["checksum"]
    assert execution.recovery_contract_sha256 == (
        _RUNNER.RECOVERY_CONTRACT_SHA256
    )
    assert execution.recovery_worker_slot == planned["worker_slot"]
    assert execution.recovery_root_job_id == planned["root_job_id"]
    assert execution.torch_intraop_threads == 4
    assert execution.torch_interop_threads == 1

    with pytest.raises(
        _RUNNER.AdjacencyRunnerError, match="uniformly CPU-only"
    ):
        _RUNNER._resolve_execution(
            _materialized_primary_recovery_config(),
            stage="primary",
            condition="isolated",
            fold=0,
            seed=0,
            requested_device="cuda:0",
        )


def test_conditional_null_rejects_missing_enqueue_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_recovery_authority(tmp_path)
    _use_recovery_root(monkeypatch, tmp_path)
    monkeypatch.setenv("BAGM_JOB_ID", "q_null_f0_s0")

    with pytest.raises(
        _RUNNER.AdjacencyRunnerError, match="null enqueue receipt is missing"
    ):
        _RUNNER._resolve_execution(
            _materialized_null_config(),
            stage="null",
            condition="position_permuted_null",
            fold=0,
            seed=0,
            requested_device=None,
        )


@pytest.mark.parametrize("tampered_authority", ["receipt", "config"])
def test_conditional_null_rejects_tampered_enqueue_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_authority: str,
) -> None:
    _copy_recovery_authority(tmp_path)
    receipt = _write_null_enqueue_receipt(tmp_path)
    _use_recovery_root(monkeypatch, tmp_path)
    target = next(
        item
        for item in receipt["jobs"]
        if item["fold"] == 0 and item["seed"] == 0
    )
    monkeypatch.setenv("BAGM_JOB_ID", target["job_id"])
    if tampered_authority == "receipt":
        path = tmp_path / _RUNNER.NULL_ENQUEUE_RELATIVE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["jobs"][0]["job_id"] = "q_tampered_null_job"
        path.write_text(json.dumps(payload), encoding="utf-8")
        match = "failed signed recovery verification"
    else:
        plan = _recovery_plan()
        planned = next(
            item
            for item in plan["conditional_null_jobs"]
            if item["fold"] == 0 and item["seed"] == 0
        )
        path = tmp_path / planned["config_reference"]
        path.write_bytes(path.read_bytes() + b"\n# tampered\n")
        match = "immutable null configuration file changed"

    with pytest.raises(_RUNNER.AdjacencyRunnerError, match=match):
        _RUNNER._resolve_execution(
            _materialized_null_config(),
            stage="null",
            condition="position_permuted_null",
            fold=0,
            seed=0,
            requested_device=None,
        )


def test_conditional_null_rejects_wrong_queue_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_recovery_authority(tmp_path)
    _write_null_enqueue_receipt(tmp_path)
    _use_recovery_root(monkeypatch, tmp_path)
    monkeypatch.setenv("BAGM_JOB_ID", "q_not_the_receipted_null_job")

    with pytest.raises(
        _RUNNER.AdjacencyRunnerError,
        match="not the null job authorized",
    ):
        _RUNNER._resolve_execution(
            _materialized_null_config(),
            stage="null",
            condition="position_permuted_null",
            fold=0,
            seed=0,
            requested_device=None,
        )


def test_signed_conditional_null_slot_binds_enqueue_job_and_cpu_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_recovery_authority(tmp_path)
    receipt = _write_null_enqueue_receipt(tmp_path)
    _use_recovery_root(monkeypatch, tmp_path)
    monkeypatch.setattr(_RUNNER.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        _RUNNER, "_configure_cpu_recovery_threads", lambda: (4, 1)
    )
    target = next(
        item
        for item in receipt["jobs"]
        if item["fold"] == 0 and item["seed"] == 0
    )
    monkeypatch.setenv("BAGM_JOB_ID", target["job_id"])

    execution = _RUNNER._resolve_execution(
        _materialized_null_config(),
        stage="null",
        condition="position_permuted_null",
        fold=0,
        seed=0,
        requested_device=None,
    )

    assert execution.device == torch.device("cpu")
    assert execution.mode == "cpu_hardware_recovery"
    assert execution.recovery_worker_slot == 0
    assert execution.recovery_root_job_id is None
    assert execution.queue_job_id == target["job_id"]
    assert execution.null_enqueue_checksum == receipt["checksum"]
    assert execution.null_enqueue_reference == (
        _RUNNER.NULL_ENQUEUE_RELATIVE.as_posix()
    )
    assert execution.null_trigger_checksum == "d" * 64
    assert execution.recovery_enqueue_checksum == (
        _RUNNER.CPU_RECOVERY_ENQUEUE_CHECKSUM
    )


def test_ordinary_smoke_still_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_RUNNER.torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        _RUNNER.AdjacencyRunnerError,
        match="no signed CPU recovery authorization",
    ):
        _RUNNER._resolve_execution(
            _materialized_smoke_config(),
            stage="smoke",
            condition="spatial",
            fold=0,
            seed=0,
            requested_device=None,
        )


def test_cpu_recovery_thread_settings_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"intraop": 12, "interop": 6}
    monkeypatch.setattr(
        _RUNNER.torch, "get_num_threads", lambda: state["intraop"]
    )
    monkeypatch.setattr(
        _RUNNER.torch, "get_num_interop_threads", lambda: state["interop"]
    )
    monkeypatch.setattr(
        _RUNNER.torch,
        "set_num_threads",
        lambda value: state.__setitem__("intraop", value),
    )
    monkeypatch.setattr(
        _RUNNER.torch,
        "set_num_interop_threads",
        lambda value: state.__setitem__("interop", value),
    )

    assert _RUNNER._configure_cpu_recovery_threads() == (4, 1)
    assert state == {"intraop": 4, "interop": 1}


def test_cpu_seeding_and_model_construction_never_touch_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = build_seeded_explicit_self_model(
        1000,
        seed=19,
        hidden_dim=128,
        ffn_dim=256,
        decoder_dim=256,
        dropout=0.1,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("CPU recovery touched CUDA")

    monkeypatch.setattr(_RUNNER.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(_RUNNER.torch.cuda, "manual_seed_all", forbidden)

    _RUNNER._seed_all(19, device=torch.device("cpu"))
    recovered = _RUNNER._build_runner_model(
        seed=19, device=torch.device("cpu")
    )

    assert state_dict_sha256(recovered) == state_dict_sha256(expected)


def test_fixed_masks_use_canonical_checksum_and_reject_seed_or_count_drift(
    tmp_path: Path,
) -> None:
    alias = "ANC-01"
    n_cells = 4
    realizations = [
        sample_uniform_mask_numpy(
            n_cells,
            _RUNNER.EXPECTED_GENES,
            seed=derive_evaluation_mask_seed(
                core_alias=alias, replicate_index=replicate
            ),
        )
        for replicate in range(3)
    ]
    masks = np.stack([item.mask for item in realizations])
    packed = np.packbits(masks, axis=-1, bitorder="little")
    counts = np.stack(
        [item.masked_gene_counts for item in realizations]
    ).astype(np.uint16)
    seeds = np.asarray([item.seed for item in realizations], dtype=np.uint64)

    def write_bundle(
        path: Path, *, stored_counts: np.ndarray, stored_seeds: np.ndarray
    ) -> None:
        np.savez_compressed(
            path,
            packed_masks=packed,
            masked_counts=stored_counts,
            mask_seeds=stored_seeds,
        )

    nodes = np.arange(n_cells, dtype=np.int64)
    identity = np.stack([nodes, nodes])

    def core(path: Path) -> Any:
        return _RUNNER.CoreData(
            alias=alias,
            standardized_expression=np.zeros(
                (n_cells, _RUNNER.EXPECTED_GENES), dtype=np.float32
            ),
            edge_index=identity,
            true_spatial_edge_index=np.empty((2, 0), dtype=np.int64),
            fov_group=np.zeros(n_cells, dtype=np.int16),
            qc_passed=np.ones(n_cells, dtype=np.bool_),
            evaluation_mask_reference=path,
            data_sha256="a" * 64,
            adjacency_sha256="b" * 64,
            selected_adjacency_sha256=ndarray_sha256(identity),
            masks_sha256="c" * 64,
        )

    valid_path = tmp_path / "valid.npz"
    write_bundle(valid_path, stored_counts=counts, stored_seeds=seeds)
    loaded = _RUNNER._fixed_masks(core(valid_path))
    assert len(loaded) == 3
    for replicate, (mask, loaded_counts, seed, checksum) in enumerate(loaded):
        expected = realizations[replicate]
        np.testing.assert_array_equal(mask, expected.mask)
        np.testing.assert_array_equal(
            loaded_counts, expected.masked_gene_counts
        )
        assert seed == expected.seed
        assert checksum == expected.checksum
        assert checksum == mask_realization_sha256(
            mask, loaded_counts, seed=seed
        )

    seed_drift = seeds.copy()
    seed_drift[1] += 1
    seed_path = tmp_path / "seed-drift.npz"
    write_bundle(seed_path, stored_counts=counts, stored_seeds=seed_drift)
    with pytest.raises(_RUNNER.AdjacencyRunnerError, match="mask seed mismatch"):
        _RUNNER._fixed_masks(core(seed_path))

    count_drift = counts.copy()
    count_drift[1, 0] = np.uint16(int(count_drift[1, 0]) + 1)
    count_path = tmp_path / "count-drift.npz"
    write_bundle(count_path, stored_counts=count_drift, stored_seeds=seeds)
    with pytest.raises(_RUNNER.AdjacencyRunnerError, match="mask counts mismatch"):
        _RUNNER._fixed_masks(core(count_path))


def _small_step(device: torch.device) -> tuple[torch.Tensor, float, list[torch.Tensor]]:
    n_cells, n_genes = 6, 9
    expression = torch.randn(
        n_cells, n_genes, generator=torch.Generator().manual_seed(801)
    ).to(device)
    mask = (
        torch.arange(n_cells * n_genes).reshape(n_cells, n_genes) % 3 == 0
    ).to(device)
    off_diagonal = np.asarray(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
        ],
        dtype=np.int64,
    )
    adjacency = torch.from_numpy(
        add_exact_self_adjacency(off_diagonal, n_nodes=n_cells)
    ).to(device)
    model = build_seeded_explicit_self_model(
        n_genes,
        seed=17,
        hidden_dim=8,
        ffn_dim=12,
        decoder_dim=10,
        dropout=0.0,
    ).to(device)
    prediction = model(expression, mask, adjacency).prediction
    loss = masked_huber_loss(expression, prediction, mask)
    loss.backward()
    gradients = [
        parameter.grad.detach().cpu().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    return prediction.detach().cpu(), float(loss.item()), gradients


def _assert_replayed_step(device: torch.device) -> tuple[torch.Tensor, float]:
    first_prediction, first_loss, first_gradients = _small_step(device)
    second_prediction, second_loss, second_gradients = _small_step(device)
    torch.testing.assert_close(
        first_prediction, second_prediction, rtol=0.0, atol=0.0
    )
    assert first_loss == second_loss
    assert len(first_gradients) == len(second_gradients)
    for first, second in zip(first_gradients, second_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    return first_prediction, first_loss


def test_deterministic_cpu_forward_and_backward_replay() -> None:
    _RUNNER._configure_determinism()

    prediction, loss = _assert_replayed_step(torch.device("cpu"))

    assert torch.isfinite(prediction).all()
    assert np.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_deterministic_cuda_forward_backward_and_cpu_compatibility() -> None:
    _RUNNER._configure_determinism()

    cpu_prediction, cpu_loss = _assert_replayed_step(torch.device("cpu"))
    cuda_prediction, cuda_loss = _assert_replayed_step(torch.device("cuda:0"))

    torch.testing.assert_close(
        cuda_prediction, cpu_prediction, rtol=2e-5, atol=2e-6
    )
    assert cuda_loss == pytest.approx(cpu_loss, rel=2e-6, abs=2e-7)


class _CapturingArchive:
    def __init__(self, root: Path) -> None:
        self.run_id = "r_20260802T000000Z_a30609a2_s0_f0_a1_test"
        self.scratch_path = root
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.documents: dict[str, Any] = {}
        self.metric_events: list[dict[str, Any]] = []
        self.predictions: list[dict[str, Any]] = []
        root.mkdir(parents=True, exist_ok=True)

    def write_table(
        self,
        relative_stem: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        fallback: str,
    ) -> Path:
        del fallback
        self.tables[relative_stem] = [dict(row) for row in rows]
        path = self.scratch_path / f"{relative_stem}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    def append_metric_event(self, event: Mapping[str, Any]) -> Path:
        self.metric_events.append(dict(event))
        return self.scratch_path / "metrics/events.jsonl"

    def write_json(self, relative: str, value: Any) -> Path:
        self.documents[relative] = value
        path = self.scratch_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def write_predictions(
        self,
        split: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        fallback: str,
    ) -> Path:
        del fallback
        self.predictions = [dict(row) for row in rows]
        path = self.scratch_path / f"predictions/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    def write_bytes(self, relative: str, value: bytes) -> Path:
        path = self.scratch_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        self.documents["summary.json"] = dict(summary)
        path = self.scratch_path / "summary.json"
        path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
        return path


def _synthetic_prepared_fold(condition: str) -> Any:
    split = build_five_fold_splits()[0]
    off_diagonal = np.asarray(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64
    )
    spatial = add_exact_self_adjacency(off_diagonal, n_nodes=3)
    nodes = np.arange(3, dtype=np.int64)
    identity = np.stack([nodes, nodes], axis=0)
    selected = spatial if condition == "spatial" else identity
    selected_sha = ndarray_sha256(selected)
    cores = {
        alias: _RUNNER.CoreData(
            alias=alias,
            standardized_expression=np.zeros((3, 1000), dtype=np.float32),
            edge_index=selected,
            true_spatial_edge_index=off_diagonal,
            fov_group=np.zeros(3, dtype=np.int16),
            qc_passed=np.ones(3, dtype=np.bool_),
            evaluation_mask_reference=Path("unused-mask.npz"),
            data_sha256="a" * 64,
            adjacency_sha256="b" * 64,
            selected_adjacency_sha256=selected_sha,
            masks_sha256="c" * 64,
        )
        for alias in _RUNNER.EXPECTED_ALIASES
    }
    return _RUNNER.PreparedFold(
        manifest={},
        manifest_path=_ROOT / _RUNNER.EXPECTED_PREPARED_MANIFEST,
        manifest_sha256=_RUNNER.EXPECTED_PREPARED_MANIFEST_SHA256,
        content_sha256=_RUNNER.EXPECTED_PREPARED_CONTENT_SHA256,
        fold=0,
        train_aliases=split.train_aliases,
        validation_aliases=split.validation_aliases,
        test_aliases=split.test_aliases,
        gene_names=tuple(f"gene_{index}" for index in range(1000)),
        mean=np.zeros(1000, dtype=np.float64),
        scale=np.ones(1000, dtype=np.float64),
        preprocessing_sha256=_RUNNER.EXPECTED_PREPROCESSING_FINGERPRINT,
        cores=cores,
    )


def _evaluation_result(aliases: Sequence[str], split: str) -> Any:
    rows: list[dict[str, Any]] = []
    for alias in aliases:
        for replicate in range(3):
            rows.append(
                {
                    "core_alias": alias,
                    "split": split,
                    "mask_replicate": replicate,
                    "mask_seed": 100 + replicate,
                    "mask_checksum": f"{replicate + 1:064x}",
                    "standardized_huber": 1.0,
                    "log1p_huber": 1.25,
                    "log1p_mae": 1.5,
                    "log1p_rmse": 2.0,
                }
            )
    return _RUNNER.EvaluationResult(
        per_core_rows=rows,
        target_bin_rows=[{} for _ in range(len(aliases) * 3 * 5)],
        neighbor_bin_rows=[{} for _ in range(len(aliases) * 3 * 4)],
        canonical_prediction=(
            {
                "core_alias": aliases[0],
                "y_true": [0.0] * 1000,
                "y_pred": [0.1] * 1000,
                "effective_mask_rate": 0.5,
            }
            if split == "validation"
            else None
        ),
        mask_identity_sha256=("d" if split == "validation" else "e") * 64,
    )


def test_lightweight_run_records_selected_graph_and_pairing_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _materialized_smoke_config()
    data = _synthetic_prepared_fold("spatial")
    archive = _CapturingArchive(tmp_path / "run")

    monkeypatch.setattr(
        _RUNNER,
        "_load_prepared_fold",
        lambda _config, *, condition, fold: data,
    )

    def fake_train(
        model: torch.nn.Module,
        _data: Any,
        _cache: Any,
        _config: Mapping[str, Any],
        **_kwargs: Any,
    ) -> Any:
        state_sha = state_dict_sha256(model)
        return _RUNNER.TrainingResult(
            model=model,
            history=[
                {
                    "epoch": 0,
                    "epoch_number": 1,
                    "training_core_huber_mean": 1.1,
                    "training_core_huber_min": 1.0,
                    "training_core_huber_max": 1.2,
                    "masked_entries": 100,
                    "optimizer_steps_cumulative": 7,
                    "core_order": list(data.train_aliases),
                    "validation_huber": 1.0,
                    "duration_seconds": 0.01,
                }
            ],
            best_epoch=0,
            best_validation_huber=1.0,
            optimizer_steps=7,
            initial_state_sha256=state_sha,
            best_state_sha256=state_sha,
            training_mask_schedule_sha256="f" * 64,
            evaluation_mask_schedule_sha256="1" * 64,
            all_gradients_finite=True,
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(_RUNNER, "_train", fake_train)
    monkeypatch.setattr(
        _RUNNER,
        "_evaluate_split",
        lambda _model, _data, _cache, *, aliases, split, device: (
            _evaluation_result(aliases, split)
        ),
    )
    monkeypatch.setattr(_RUNNER, "_peak_host_bytes", lambda: 1024)
    monkeypatch.setattr(
        _RUNNER,
        "_resolve_execution",
        lambda *_args, **_kwargs: _RUNNER.ExecutionAuthorization(
            device=torch.device("cpu"),
            mode="cpu_hardware_recovery",
            queue_job_id="q_synthetic_test",
            torch_intraop_threads=4,
            torch_interop_threads=1,
            recovery_contract_reference=(
                _RUNNER.RECOVERY_CONTRACT_RELATIVE.as_posix()
            ),
            recovery_contract_sha256=_RUNNER.RECOVERY_CONTRACT_SHA256,
            recovery_plan_reference=(
                _RUNNER.CPU_RECOVERY_PLAN_RELATIVE.as_posix()
            ),
            recovery_plan_checksum="a" * 64,
            recovery_plan_file_sha256="b" * 64,
            recovery_worker_slot=0,
            recovery_root_job_id="q_root_synthetic_test",
        ),
    )

    summary = _RUNNER.run_adjacency_ablation(
        config,
        archive,
        sample_key_salt="test-only-salt-0123456789",
        device="cpu",
    )

    validation_alias = data.validation_aliases[0]
    selected_sha = data.cores[validation_alias].selected_adjacency_sha256
    assert selected_sha == ndarray_sha256(data.cores[validation_alias].edge_index)
    assert selected_sha != data.cores[validation_alias].adjacency_sha256
    assert archive.predictions[0]["graph_id"] == selected_sha

    pairing = archive.documents["diagnostics/pairing_and_leakage_audit.json"]
    assert {
        "fold",
        "model_seed",
        "condition",
        "train_aliases",
        "validation_aliases",
        "test_aliases",
        "split_disjoint",
        "preprocessing_fit_aliases",
        "preprocessing_sha256",
        "initial_state_sha256",
        "training_mask_schedule_sha256",
        "validation_selection_mask_schedule_sha256",
        "validation_evaluation_mask_identity_sha256",
        "test_evaluation_mask_identity_sha256",
        "model_inputs",
        "prohibited_inputs_absent",
        "masked_neighbor_inputs_only",
        "execution_device",
        "execution_mode",
        "queue_job_id",
        "torch_intraop_threads",
        "torch_interop_threads",
        "hardware_recovery_used",
        "recovery_contract_sha256",
        "recovery_plan_checksum",
    }.issubset(pairing)
    assert pairing["split_disjoint"] is True
    assert pairing["preprocessing_fit_aliases"] == list(data.train_aliases)
    assert pairing["masked_neighbor_inputs_only"] is True
    assert pairing["hardware_recovery_used"] is True
    assert pairing["recovery_contract_sha256"] == (
        _RUNNER.RECOVERY_CONTRACT_SHA256
    )
    assert pairing["recovery_plan_checksum"] == "a" * 64

    assert {
        "parameter_count",
        "initial_state_sha256",
        "best_state_sha256",
        "training_mask_schedule_sha256",
        "validation_mask_identity_sha256",
        "test_mask_identity_sha256",
        "optimizer_steps",
        "config_sha256",
        "prepared_content_sha256",
        "preprocessing_sha256",
        "finite_metrics",
        "coverage_complete",
        "update_count_verified",
        "all_gradients_finite",
        "deterministic_algorithms",
        "conclusion_eligible",
        "execution_device",
        "execution_mode",
        "queue_job_id",
        "torch_intraop_threads",
        "torch_interop_threads",
        "hardware_recovery_used",
        "recovery_contract_sha256",
        "recovery_plan_checksum",
    }.issubset(summary)
    assert summary["parameter_count"] == _RUNNER.EXPECTED_PARAMETER_COUNT
    assert summary["optimizer_steps"] == 7
    assert summary["coverage_complete"] is True
    assert summary["update_count_verified"] is True
    assert summary["deterministic_algorithms"] is True
    assert summary["execution_device"] == "cpu"
    assert summary["execution_mode"] == "cpu_hardware_recovery"
    assert summary["hardware_recovery_used"] is True
    assert summary["recovery_contract_sha256"] == (
        _RUNNER.RECOVERY_CONTRACT_SHA256
    )
    assert summary["recovery_plan_checksum"] == "a" * 64

    for relative in (
        "diagnostics/resources.json",
        "provenance/scientific_inputs.json",
    ):
        document = archive.documents[relative]
        assert document["execution_device"] == "cpu"
        assert document["torch_intraop_threads"] == 4
        assert document["torch_interop_threads"] == 1
        assert document["recovery_contract_sha256"] == (
            _RUNNER.RECOVERY_CONTRACT_SHA256
        )
        assert document["recovery_plan_checksum"] == "a" * 64
