from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/analysis/compare_pooled_hybrid_ensemble.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_pooled_hybrid_ensemble", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _signed_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["checksum"] = MODULE.canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_verified_signed_json_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _signed_json(path, {"schema_version": 1, "value": 3})
    assert MODULE._verified_signed_json(path, label="receipt")["value"] == 3
    path.write_text(
        path.read_text(encoding="utf-8").replace('"value": 3', '"value": 4'),
        encoding="utf-8",
    )
    with pytest.raises(MODULE.PooledEnsembleComparisonError, match="does not verify"):
        MODULE._verified_signed_json(path, label="receipt")


def _lineage_fixture() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    enqueued: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    for arm in MODULE.ARMS:
        for seed in MODULE.SEEDS:
            job_id = f"job-{arm}-{seed}"
            run_id = f"run-{arm}-{seed}"
            requested_gpu = tuple(sorted(MODULE.SAFE_GPU_IDS))[seed]
            config = {"test_slot": [arm, seed], "attempt": 1}
            digest = MODULE.canonical_sha256(config)
            queue.append(
                {
                    "job_id": job_id,
                    "canonical_config": config,
                    "status": "completed",
                    "attempt_count": 1,
                    "maximum_attempts": 2,
                    "requested_gpu": requested_gpu,
                    "retry_of": None,
                    "run_id": run_id,
                    "failure_category": None,
                    "last_error": None,
                }
            )
            materialized.append(
                {"arm": arm, "seed": seed, "config_sha256": digest}
            )
            enqueued.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "job_id": job_id,
                    "config_sha256": digest,
                    "requested_gpu": requested_gpu,
                }
            )
            runs[run_id] = {
                "run_id": run_id,
                "campaign_id": MODULE.CAMPAIGN_ID,
                "status": "completed",
                "attempt": 1,
                "retry_of": None,
            }
    return (
        queue,
        {"production_jobs": materialized},
        {"jobs": enqueued},
        runs,
    )


def test_resolve_production_lineages_requires_exact_fourteen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, materialization, enqueue, runs = _lineage_fixture()
    monkeypatch.setattr(
        MODULE,
        "_slot_from_config",
        lambda config, require_production: tuple(config["test_slot"]),
    )
    audit = MODULE.resolve_production_lineages(
        queue_rows=queue,
        materialization=materialization,
        enqueue=enqueue,
        run_lookup=runs.get,
    )
    assert len(audit.selected_jobs) == 14
    assert len(audit.attempt_inventory) == 14
    assert audit.registered_failure_inventory == ()

    with pytest.raises(
        MODULE.PooledEnsembleComparisonError,
        match="lacks its root queue job",
    ):
        MODULE.resolve_production_lineages(
            queue_rows=queue[:-1],
            materialization=materialization,
            enqueue=enqueue,
            run_lookup=runs.get,
        )


def test_resolve_production_lineages_preserves_failed_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, materialization, enqueue, runs = _lineage_fixture()
    monkeypatch.setattr(
        MODULE,
        "_slot_from_config",
        lambda config, require_production: tuple(config["test_slot"]),
    )
    root = queue[0]
    root_run_id = str(root["run_id"])
    root["status"] = "failed"
    root["failure_category"] = "runtime_error"
    runs[root_run_id]["status"] = "failed"
    retry_id = "retry-job"
    retry_run_id = "retry-run"
    queue.append(
        {
            **root,
            "job_id": retry_id,
            "status": "completed",
            "attempt_count": 2,
            "retry_of": root["job_id"],
            "run_id": retry_run_id,
            "failure_category": None,
        }
    )
    runs[retry_run_id] = {
        "run_id": retry_run_id,
        "campaign_id": MODULE.CAMPAIGN_ID,
        "status": "completed",
        "attempt": 2,
        "retry_of": root_run_id,
    }
    audit = MODULE.resolve_production_lineages(
        queue_rows=queue,
        materialization=materialization,
        enqueue=enqueue,
        run_lookup=runs.get,
    )
    assert len(audit.attempt_inventory) == 15
    assert len(audit.registered_failure_inventory) == 1
    slot = tuple(root["canonical_config"]["test_slot"])
    assert audit.selected_jobs[slot]["job_id"] == retry_id


def test_resolve_production_lineages_rejects_orphan_and_unsafe_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, materialization, enqueue, runs = _lineage_fixture()
    monkeypatch.setattr(
        MODULE,
        "_slot_from_config",
        lambda config, require_production: tuple(config["test_slot"]),
    )
    orphan = {
        **queue[0],
        "job_id": "orphan-production-retry",
        "status": "failed",
        "attempt_count": 2,
        "retry_of": "missing-parent",
        "run_id": None,
    }
    with pytest.raises(
        MODULE.PooledEnsembleComparisonError,
        match="unexpected or orphaned",
    ):
        MODULE.resolve_production_lineages(
            queue_rows=[*queue, orphan],
            materialization=materialization,
            enqueue=enqueue,
            run_lookup=runs.get,
        )

    root = queue[0]
    root_run_id = str(root["run_id"])
    root["status"] = "failed"
    runs[root_run_id]["status"] = "failed"
    retry_run_id = "unsafe-retry-run"
    unsafe_retry = {
        **root,
        "job_id": "unsafe-retry",
        "status": "completed",
        "attempt_count": 2,
        "requested_gpu": 4,
        "retry_of": root["job_id"],
        "run_id": retry_run_id,
    }
    runs[retry_run_id] = {
        "run_id": retry_run_id,
        "campaign_id": MODULE.CAMPAIGN_ID,
        "status": "completed",
        "attempt": 2,
        "retry_of": root_run_id,
    }
    with pytest.raises(
        MODULE.PooledEnsembleComparisonError,
        match="unsafe GPU",
    ):
        MODULE.resolve_production_lineages(
            queue_rows=[*queue, unsafe_retry],
            materialization=materialization,
            enqueue=enqueue,
            run_lookup=runs.get,
        )


def _pilot_lineage_fixture() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    queue: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    enqueued: list[dict[str, Any]] = []
    gated: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    for index, arm in enumerate(MODULE.ARMS):
        slot = [arm, 0]
        config = {"test_slot": slot, "attempt": 1}
        digest = MODULE.canonical_sha256(config)
        root_job_id = f"pilot-root-{index}"
        retry_job_id = f"pilot-retry-{index}"
        failed_run_id = f"pilot-failed-run-{index}"
        completed_run_id = f"pilot-completed-run-{index}"
        requested_gpu = index
        failed = {
            "attempt": 1,
            "failure_category": "invalid_configuration",
            "is_original_enqueue_job": True,
            "job_id": root_job_id,
            "maximum_attempts": 2,
            "retry_of": None,
            "run_id": failed_run_id,
            "selected_completed_attempt": False,
            "status": "failed",
        }
        completed = {
            "attempt": 2,
            "failure_category": None,
            "is_original_enqueue_job": False,
            "job_id": retry_job_id,
            "maximum_attempts": 2,
            "retry_of": root_job_id,
            "run_id": completed_run_id,
            "selected_completed_attempt": True,
            "status": "completed",
        }
        queue.extend(
            [
                {
                    "job_id": root_job_id,
                    "canonical_config": config,
                    "status": "failed",
                    "attempt_count": 1,
                    "maximum_attempts": 2,
                    "requested_gpu": requested_gpu,
                    "retry_of": None,
                    "run_id": failed_run_id,
                    "failure_category": "invalid_configuration",
                    "last_error": "stale validator",
                },
                {
                    "job_id": retry_job_id,
                    "canonical_config": {**config, "attempt": 2},
                    "status": "completed",
                    "attempt_count": 2,
                    "maximum_attempts": 2,
                    "requested_gpu": requested_gpu,
                    "retry_of": root_job_id,
                    "run_id": completed_run_id,
                    "failure_category": None,
                    "last_error": None,
                },
            ]
        )
        materialized.append(
            {
                "arm": arm,
                "seed": 0,
                "config_sha256": digest,
                "requested_gpu": requested_gpu,
            }
        )
        enqueued.append(
            {
                "arm": arm,
                "seed": 0,
                "job_id": root_job_id,
                "config_sha256": digest,
                "requested_gpu": requested_gpu,
            }
        )
        gated.append(
            {
                "arm": arm,
                "seed": 0,
                "job_id": root_job_id,
                "original_enqueue_job_id": root_job_id,
                "completed_job_id": retry_job_id,
                "run_id": completed_run_id,
                "config_sha256": digest,
                "attempt_count": 2,
                "completed_attempt": 2,
                "attempts": [failed, completed],
                "failed_attempts": [failed],
            }
        )
        runs[failed_run_id] = {
            "run_id": failed_run_id,
            "campaign_id": MODULE.CAMPAIGN_ID,
            "status": "failed",
            "attempt": 1,
            "retry_of": None,
        }
        runs[completed_run_id] = {
            "run_id": completed_run_id,
            "campaign_id": MODULE.CAMPAIGN_ID,
            "status": "completed",
            "attempt": 2,
            "retry_of": failed_run_id,
        }
    return (
        queue,
        {"pilot_jobs": materialized},
        {"jobs": enqueued},
        {"jobs": gated},
        runs,
    )


def test_resolve_pilot_lineages_preserves_both_operational_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, materialization, enqueue, gate, runs = _pilot_lineage_fixture()
    monkeypatch.setattr(
        MODULE,
        "_slot_from_config",
        lambda config, require_production: tuple(config["test_slot"]),
    )
    attempts, failures = MODULE.resolve_pilot_lineages(
        queue_rows=queue,
        materialization=materialization,
        pilot_enqueue=enqueue,
        pilot_gate=gate,
        run_lookup=runs.get,
    )
    assert len(attempts) == 4
    assert len(failures) == 2
    assert {row["stage"] for row in attempts} == {"pilot"}
    assert {row["failure_category"] for row in failures} == {
        "invalid_configuration"
    }
    assert sum(row["selected_completed_attempt"] for row in attempts) == 2


def test_resolve_pilot_lineages_rejects_changed_retry_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, materialization, enqueue, gate, runs = _pilot_lineage_fixture()
    monkeypatch.setattr(
        MODULE,
        "_slot_from_config",
        lambda config, require_production: tuple(config["test_slot"]),
    )
    queue[1]["retry_of"] = "changed-parent"
    with pytest.raises(
        MODULE.PooledEnsembleComparisonError,
        match="signed gate",
    ):
        MODULE.resolve_pilot_lineages(
            queue_rows=queue,
            materialization=materialization,
            pilot_enqueue=enqueue,
            pilot_gate=gate,
            run_lookup=runs.get,
        )


def test_paired_member_initialization_requires_matching_arm_digests() -> None:
    evidence = {
        (arm, seed): SimpleNamespace(
            encoder_initial_state_sha256=f"encoder-{seed}",
            decoder_initial_state_sha256=f"decoder-{seed}",
        )
        for arm in MODULE.ARMS
        for seed in MODULE.SEEDS
    }
    MODULE.verify_paired_member_initialization(evidence)
    evidence[(MODULE.SELF_ARM, 3)] = SimpleNamespace(
        encoder_initial_state_sha256="changed",
        decoder_initial_state_sha256="decoder-3",
    )
    with pytest.raises(
        MODULE.PooledEnsembleComparisonError,
        match="initialization differs",
    ):
        MODULE.verify_paired_member_initialization(evidence)


def _replicate_rows(*, ensemble: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = (None,) if ensemble else MODULE.SEEDS
    for arm in MODULE.ARMS:
        for seed in seeds:
            for alias in MODULE.ALIASES:
                for mode in MODULE.MASK_MODES:
                    for replicate in MODULE.REPLICATES:
                        row = {
                            "arm": arm,
                            "core_alias": alias,
                            "mask_mode": mode,
                            "mask_replicate": replicate,
                            "hybrid_loss": 1.0 + replicate,
                            "detection_precision": None,
                            "state8_recall_0": 0.5 + 0.1 * replicate,
                        }
                        if seed is not None:
                            row["seed"] = seed
                        rows.append(row)
    return rows


@pytest.mark.parametrize(
    ("ensemble", "expected"),
    [(False, 2 * 7 * 10 * 3), (True, 2 * 10 * 3)],
)
def test_collapse_rich_replicates_has_frozen_coverage(
    ensemble: bool, expected: int
) -> None:
    rows = _replicate_rows(ensemble=ensemble)
    collapsed = MODULE.collapse_rich_replicates(rows, ensemble=ensemble)
    assert len(collapsed) == expected
    assert collapsed[0]["hybrid_loss"] == pytest.approx(2.0)
    assert collapsed[0]["hybrid_loss_defined_replicates"] == 3
    assert collapsed[0]["detection_precision"] is None
    with pytest.raises(
        MODULE.PooledEnsembleComparisonError, match="replicates|coverage"
    ):
        MODULE.collapse_rich_replicates(rows[:-1], ensemble=ensemble)


def test_checkpoint_metadata_binds_all_graphs_and_masks() -> None:
    state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    graph_sources = {
        alias: {"graph_sha256": f"{index:064x}"}
        for index, alias in enumerate(MODULE.ALIASES, start=1)
    }
    mask_sources = {
        alias: {"bundle_checksum": f"{index + 100:064x}"}
        for index, alias in enumerate(MODULE.ALIASES, start=1)
    }
    materialization = {
        "checksum": "a" * 64,
        "cohort": {
            "dataset_fingerprint": "b" * 64,
            "cores": {
                alias: {"pooled_preprocessing_sha256": f"{index + 200:064x}"}
                for index, alias in enumerate(MODULE.ALIASES)
            },
        },
        "graph_sources": graph_sources,
        "evaluation_mask_sources": mask_sources,
    }
    payload = {
        "schema_version": 1,
        "checkpoint_role": "last",
        "checkpoint_policy": "final_global_epoch_no_validation_selection",
        "public_variant": MODULE.GAT_ARM,
        "model_name": "hybrid-count-gat",
        "task_family": MODULE.EXPECTED_TASK_FAMILY,
        "model_seed": 0,
        "aliases": list(MODULE.ALIASES),
        "epoch": 199,
        "completed_global_epochs": 200,
        "optimizer_steps_completed": 2000,
        "fixed_epoch_budget": 200,
        "pooled_cohort_fingerprint_sha256": "b" * 64,
        "per_core_preprocessing_sha256": {
            alias: materialization["cohort"]["cores"][alias][
                "pooled_preprocessing_sha256"
            ]
            for alias in MODULE.ALIASES
        },
        "per_core_graph_sha256": {
            alias: graph_sources[alias]["graph_sha256"] for alias in MODULE.ALIASES
        },
        "per_core_evaluation_mask_bundle_sha256": {
            alias: mask_sources[alias]["bundle_checksum"] for alias in MODULE.ALIASES
        },
        "materialization_checksum": "a" * 64,
        "config_sha256": "c" * 64,
        "count_representation_schema": MODULE.EXPECTED_REPRESENTATION_SCHEMA,
        "objective": MODULE.EXPECTED_OBJECTIVE,
        "effective_amp": True,
        "selection_policy": "last_epoch_without_validation_selection",
        "monitored_metric": None,
        "model_state_dict": state,
        "state_dict_sha256": MODULE._state_dict_sha256(state),
    }
    state_sha, loaded = MODULE._validate_checkpoint_metadata(
        payload,
        arm=MODULE.GAT_ARM,
        seed=0,
        materialization=materialization,
        expected_config_sha256="c" * 64,
    )
    assert state_sha == payload["state_dict_sha256"]
    assert torch.equal(loaded["weight"], state["weight"])
    payload["per_core_graph_sha256"] = dict(payload["per_core_graph_sha256"])
    payload["per_core_graph_sha256"]["ANC-10"] = "d" * 64
    with pytest.raises(MODULE.PooledEnsembleComparisonError, match="metadata changed"):
        MODULE._validate_checkpoint_metadata(
            payload,
            arm=MODULE.GAT_ARM,
            seed=0,
            materialization=materialization,
            expected_config_sha256="c" * 64,
        )


def test_compact_equal_core_reference_matches_public_fitter() -> None:
    counts = {
        alias: np.asarray(
            [
                [0, 1, 2],
                [2 + index, 0, 8],
                [1, 4, 0],
            ],
            dtype=np.int64,
        )
        for index, alias in enumerate(MODULE.ALIASES)
    }
    mean, scale = MODULE._pooled_full_core.fit_equal_core_log1p_statistics(
        list(counts.values())
    )
    per_core = {
        alias: MODULE.fit_hybrid_count_references(
            values,
            expression_mean=mean,
            expression_scale=scale,
        )
        for alias, values in counts.items()
    }
    compact = MODULE._combine_equal_core_references(
        per_core,
        expression_mean=mean,
        expression_scale=scale,
    )
    from spatial_benchmark.pooled_references import (
        fit_equal_core_hybrid_count_references,
    )

    expected = fit_equal_core_hybrid_count_references(
        counts,
        expression_mean=mean,
        expression_scale=scale,
        expected_aliases=MODULE.ALIASES,
    )
    assert compact.audit == expected.audit
    for name in (
        "detection_probability",
        "positive_ordinal_probability",
        "positive_continuous_standardized",
        "detected_state",
        "positive_state",
        "count_state",
    ):
        assert np.array_equal(getattr(compact, name), getattr(expected, name))


def test_required_metric_schema_uses_runner_equal_core_prefix() -> None:
    row = {
        field: 0.5
        for field in (
            *MODULE.REQUIRED_MODEL_SCALAR_METRICS,
            *MODULE.REQUIRED_EQUAL_CORE_REFERENCE_METRICS,
        )
    }
    row["detection_precision"] = None
    MODULE._validate_required_metric_fields(
        row,
        label="test row",
        require_equal_core_reference=True,
    )
    assert (
        "reference_equal_core_detection_balanced_accuracy"
        in MODULE.ENSEMBLE_GATE_METRICS
    )
    assert not any(
        field.startswith("reference_pooled_")
        for field in MODULE.ENSEMBLE_GATE_METRICS
    )
    del row["ordinal_bce"]
    with pytest.raises(
        MODULE.PooledEnsembleComparisonError,
        match="ordinal_bce",
    ):
        MODULE._validate_required_metric_fields(
            row,
            label="test row",
            require_equal_core_reference=True,
        )


def test_comparison_tables_consume_equal_core_reference_name() -> None:
    member_rows = [
        {
            "arm": arm,
            "seed": seed,
            "core_alias": alias,
            "mask_mode": "whole_node",
            "hybrid_loss": 0.4 if arm == MODULE.GAT_ARM else 0.5,
            "detection_bce": 0.3,
            "positive_ordinal_mae": 0.6,
            "positive_continuous_huber": 0.4,
            "reconstructed_count_log1p_mae": 0.5,
        }
        for arm in MODULE.ARMS
        for seed in MODULE.SEEDS
        for alias in MODULE.ALIASES
    ]
    ensemble_rows = [
        {
            "arm": arm,
            "core_alias": alias,
            "mask_mode": "whole_node",
            "hybrid_loss": 0.4 if arm == MODULE.GAT_ARM else 0.5,
            "detection_balanced_accuracy": 0.6,
            "positive_ordinal_mae": 0.5,
            "positive_continuous_huber": 0.3,
            "reference_per_gene_positive_ordinal_mae": 0.7,
            "reference_per_gene_positive_continuous_huber": 0.4,
            "reference_per_gene_detection_balanced_accuracy": 0.55,
            "reference_equal_core_detection_balanced_accuracy": 0.54,
        }
        for arm in MODULE.ARMS
        for alias in MODULE.ALIASES
    ]
    prior_rows = [
        {
            "core_alias": alias,
            "mask_mode": "whole_node",
            "detection_bce": 0.4,
            "positive_ordinal_mae": 0.7,
            "reconstructed_count_log1p_mae": 0.6,
        }
        for alias in MODULE.ALIASES
    ]
    tables = MODULE.build_comparison_tables(
        member_core_rows=member_rows,
        ensemble_core_rows=ensemble_rows,
        prior_core_rows=prior_rows,
    )
    assert tables["representation_core_comparison"][0][
        "equal_core_reference_detection_balanced_accuracy"
    ] == pytest.approx(0.54)


def test_streaming_cohort_retains_no_core_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts = tuple(
        MODULE._pooled_full_core.PreparedCoreReceipt(
            alias=alias,
            n_nodes=2,
            manifest_sha256=MODULE.canonical_sha256(
                {"manifest": alias}
            ),
            prepared_data_sha256=MODULE.canonical_sha256({"data": alias}),
        )
        for alias in MODULE.ALIASES
    )
    calls: list[str] = []

    def fake_load(
        alias: str,
        path: Path,
        *,
        receipt: Any,
        epsilon: float,
    ) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
        del path, epsilon
        calls.append(alias)
        index = MODULE.ALIASES.index(alias)
        counts = np.asarray(
            [[0, 1 + index], [2 + index, 0]], dtype=np.int64
        )
        source = SimpleNamespace(
            receipt=receipt,
            expression_counts=counts,
            node_covariates=np.asarray([[1.0], [2.0]], dtype=np.float32),
            coordinates_um=np.asarray(
                [[0.0, 0.0], [1.0, 1.0]], dtype=np.float64
            ),
            macroblock_ids=np.asarray(
                [f"{alias}-BLOCK-0000", f"{alias}-BLOCK-0001"]
            ),
            gene_names=("G1", "G2"),
            metadata_names=("M1",),
            metadata_median=np.asarray([0.0]),
            metadata_mean=np.asarray([0.0]),
            metadata_scale=np.asarray([1.0]),
            metadata_missing_indicator_indices=np.asarray([], dtype=np.int64),
            source_checksums=SimpleNamespace(
                preprocessing_sha256=MODULE.canonical_sha256(
                    {"source": alias}
                ),
                expression_counts_sha256=MODULE.canonical_sha256(
                    {"counts": alias}
                ),
                node_covariates_sha256=MODULE.canonical_sha256(
                    {"covariates": alias}
                ),
                coordinates_um_sha256=MODULE.canonical_sha256(
                    {"coordinates": alias}
                ),
            ),
        )
        return source, source.gene_names, source.metadata_names

    monkeypatch.setattr(
        MODULE._pooled_full_core,
        "EXPECTED_PREPARED_CORES",
        receipts,
    )
    monkeypatch.setattr(
        MODULE._pooled_full_core,
        "_load_verified_core",
        fake_load,
    )
    prepared = {
        alias: tmp_path / alias.lower() for alias in MODULE.ALIASES
    }
    cohort = MODULE.load_streaming_pooled_cohort(prepared)
    assert len(calls) == 20
    assert not hasattr(cohort, "cores")
    assert cohort.total_nodes == 20
    assert tuple(cohort.core_preprocessing_sha256) == MODULE.ALIASES

    iterator = iter(cohort.iter_cores())
    first = next(iterator)
    assert first.alias == "ANC-01"
    assert len(calls) == 21
    del first
    remaining = list(iterator)
    assert [core.alias for core in remaining] == list(MODULE.ALIASES[1:])
    assert len(calls) == 30


def test_categorical_comparison_is_checksum_bound_and_explicitly_descriptive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants: dict[str, Any] = {}
    for variant in MODULE.PRIOR_CATEGORICAL_VARIANTS:
        variants[variant] = {
            "model_seeds": [0, 1, 2],
            "metrics": {
                "exact_percent": {"mean": 91.0, "n": 3},
                "balanced_percent": {"mean": 28.0, "n": 3},
                "nonzero_percent": {"mean": 2.0, "n": 3},
            },
            "baselines": {
                "baseline_always_zero_accuracy_percent": {
                    "mean": 90.0,
                    "n": 3,
                }
            },
            "token_recalls_percent": {
                str(state): {"mean": 25.0, "n": 3}
                for state in range(4)
            },
        }
    path = tmp_path / "categorical.json"
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "g2_token_multiseed_comparison",
                "campaign_id": MODULE.PRIOR_CATEGORICAL_CAMPAIGN,
                "status": "complete",
                "variant_aggregates": variants,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        MODULE,
        "PRIOR_CATEGORICAL_COMPARISON_SHA256",
        MODULE._sha256_file(path),
    )
    current = [
        {
            "arm": MODULE.GAT_ARM,
            "core_alias": alias,
            "mask_mode": "whole_node",
            "collapsed4_exact_accuracy": 0.6,
            "collapsed4_balanced_accuracy": 0.4,
            "collapsed4_positive_exact_accuracy": 0.3,
            "reference_all_zero_collapsed4_exact_accuracy": 0.9,
            **{f"collapsed4_recall_{state}": 0.25 for state in range(4)},
        }
        for alias in MODULE.ALIASES
    ]
    rows = MODULE.load_categorical_comparison(
        path=path, gat_ensemble_core_rows=current
    )
    assert len(rows) == 3
    assert all(row["comparison_is_descriptive_only"] for row in rows)
    assert rows[0]["collapsed4_exact_accuracy_percent"] == pytest.approx(60.0)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"model_seeds": [0, 1, 2]',
            '"model_seeds": [0, 1, 3]',
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        MODULE,
        "PRIOR_CATEGORICAL_COMPARISON_SHA256",
        MODULE._sha256_file(path),
    )
    with pytest.raises(
        MODULE.PooledEnsembleComparisonError,
        match="seed coverage",
    ):
        MODULE.load_categorical_comparison(
            path=path,
            gat_ensemble_core_rows=current,
        )


def test_atomic_report_is_portable_and_checksum_manifested(
    tmp_path: Path,
) -> None:
    output = tmp_path / "comparison"
    result = MODULE._publish_report_directory(
        output_dir=output,
        analysis={"status": "complete"},
        tables={"small": [{"core_alias": "ANC-01", "value": 1.0}]},
        markdown="# Report\n",
        html_report=(
            "<!doctype html><style>body{color:black}</style>"
            "<main>portable</main>"
        ),
        provenance={"protected_identifiers_emitted": False},
    )
    assert output.is_dir()
    assert result["manifest_sha256"] == MODULE._sha256_file(
        output / "manifest.json"
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "report.html" in manifest["files"]
    document = (output / "report.html").read_text(encoding="utf-8")
    assert "<link" not in document
    assert "src=\"http" not in document
    with pytest.raises(MODULE.PooledEnsembleComparisonError, match="will not be overwritten"):
        MODULE._publish_report_directory(
            output_dir=output,
            analysis={},
            tables={},
            markdown="",
            html_report="",
            provenance={},
        )
    with pytest.raises(
        MODULE.PooledEnsembleComparisonError,
        match="external asset",
    ):
        MODULE._publish_report_directory(
            output_dir=tmp_path / "external",
            analysis={},
            tables={},
            markdown="",
            html_report=(
                "<!doctype html><style>body{color:black}</style>"
                "<main><img src='https://example.invalid/a.png'></main>"
            ),
            provenance={},
        )
    with pytest.raises(
        MODULE.PooledEnsembleComparisonError,
        match="prohibited identifier field",
    ):
        MODULE._publish_report_directory(
            output_dir=tmp_path / "unsafe",
            analysis={},
            tables={"unsafe": [{"patient_id": "not-allowed"}]},
            markdown="",
            html_report=(
                "<!doctype html><style>body{color:black}</style>"
                "<main>portable</main>"
            ),
            provenance={},
        )


def test_reports_render_gate_details_without_external_assets() -> None:
    gates = {
        "pooled_data_gate": {
            "passed": False,
            "metrics": {
                metric: {
                    "mean_relative_improvement": 0.01,
                    "favoring_core_count": 7,
                    "passed": False,
                }
                for metric in (
                    "detection_bce",
                    "positive_ordinal_mae",
                    "reconstructed_count_log1p_mae",
                )
            },
        },
        "graph_gate": {
            "passed": True,
            "mean_relative_hybrid_loss_improvement": 0.03,
            "favoring_core_count": 8,
            "favoring_seed_pair_count": 5,
            "positive_metric_noninferiority": {
                "positive_ordinal_mae": True,
                "positive_continuous_huber": True,
            },
        },
        "representation_gate": {
            "passed": True,
            "positive_metrics": {
                metric: {
                    "mean_relative_improvement": 0.03,
                    "favoring_core_count": 8,
                    "passed": True,
                }
                for metric in (
                    "positive_ordinal_mae",
                    "positive_continuous_huber",
                )
            },
            "detection_balanced_accuracy": {
                "model": 0.6,
                "per_core_reference": 0.5,
                "pooled_reference": 0.51,
                "passed": True,
            },
        },
        "ensemble_gate": {
            "passed": True,
            "metrics": {
                metric: {
                    "ensemble_equal_core_mean": 0.4,
                    "mean_individual_equal_core_metric": 0.41,
                    "passed": True,
                }
                for metric in (
                    "hybrid_loss",
                    "positive_ordinal_mae",
                    "positive_continuous_huber",
                )
            },
        },
    }
    analysis = {
        "frozen_gates": gates,
        "failure_attempt_count": 2,
        "pilot_failure_attempt_count": 2,
        "production_failure_attempt_count": 0,
        "operational_failures": {
            "pilot_invalid_configuration_attempts": 2,
            "production_failed_attempts": 0,
            "diagnosis": "stale validator",
        },
        "maximum_defensible_conclusion": "held-in evidence only",
    }
    graph = [
        {
            "core_alias": "ANC-01",
            "gat_hybrid_loss": 0.4,
            "matched_self_hybrid_loss": 0.42,
            "gat_relative_hybrid_loss_improvement": 0.04,
            "gat_positive_ordinal_mae": 0.5,
            "self_positive_ordinal_mae": 0.52,
            "gat_positive_continuous_huber": 0.3,
            "self_positive_continuous_huber": 0.31,
        }
    ]
    seeds = [
        {
            "seed": 0,
            "gat_equal_core_hybrid_loss": 0.4,
            "self_equal_core_hybrid_loss": 0.42,
            "gat_relative_improvement": 0.04,
        }
    ]
    categorical = [
        {
            "source": "current",
            "variant": MODULE.GAT_ARM,
            "collapsed4_exact_accuracy_percent": 60.0,
            "collapsed4_balanced_accuracy_percent": 40.0,
            "collapsed4_positive_exact_accuracy_percent": 30.0,
            "all_zero_exact_accuracy_percent": 90.0,
        }
    ]
    resources = [
        {
            "arm": MODULE.GAT_ARM,
            "seed": 0,
            "attempt": 1,
            "requested_gpu": 0,
            "cuda_device_name": "test GPU",
            "duration_seconds": 3600.0,
            "duration_hours": 1.0,
            "peak_vram_gib": 10.0,
            "failed_attempt_count": 0,
        }
    ]
    markdown = MODULE._markdown_report(
        analysis=analysis,
        graph_rows=graph,
        seed_rows=seeds,
        categorical_rows=categorical,
        resource_rows=resources,
    )
    document = MODULE._html_report(
        analysis=analysis,
        graph_rows=graph,
        seed_rows=seeds,
        categorical_rows=categorical,
        resource_rows=resources,
    )
    assert "Failed gates" in markdown
    assert "2 pilot and 0 production" in markdown
    assert "legacy true-Normal" in markdown
    assert "core-level inference" in document
    assert "2 failed pilot attempts" in document
    assert "legacy true-Normal" in document
    assert "<style>" in document
    assert "<caption>" in document
    assert "<link" not in document
    MODULE._assert_portable_html(document)
