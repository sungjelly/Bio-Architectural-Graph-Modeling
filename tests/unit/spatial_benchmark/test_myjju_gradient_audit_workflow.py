from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.analysis import audit_myjju_genemae_gradients as workflow


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Cohort:
    def __init__(self, cores: dict[str, Any]) -> None:
        self._cores = cores

    def core(self, alias: str) -> Any:
        return self._cores[alias]


class _Tiled:
    def __init__(self, tiles: list[Any]) -> None:
        self.tiles = tuple(tiles)

    def for_alias(self, alias: str) -> tuple[Any, ...]:
        return tuple(tile for tile in self.tiles if tile.alias == alias)


def _synthetic_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> workflow.AuditExecutionContext:
    nodes_per_core = 32
    total_nodes = nodes_per_core * len(workflow.ALIASES)
    monkeypatch.setattr(workflow, "EXPECTED_TOTAL_NODES", total_nodes)
    shapes = dict(workflow.REQUIRED_ARRAYS)
    for name in (
        "selected_truth",
        "selected_mask",
        "selected_prediction_observed",
        "selected_prediction_permuted",
    ):
        shape = shapes[name]
        shapes[name] = (total_nodes, *shape[1:])
    monkeypatch.setattr(workflow, "REQUIRED_ARRAYS", shapes)

    marker_indices = tuple(range(39))
    target_indices = tuple(
        workflow.FROZEN_MARKER_GENES.index(gene)
        for gene in workflow.FROZEN_TARGET_GENES
    )
    cores: dict[str, Any] = {}
    normalized: dict[str, np.ndarray] = {}
    masks: dict[str, Any] = {}
    tiles: list[Any] = []
    for core_index, alias in enumerate(workflow.ALIASES):
        values = np.zeros((nodes_per_core, 1000), dtype=np.float32)
        base = np.arange(nodes_per_core, dtype=np.float32)[:, None]
        values[:, :39] = (
            base + np.arange(39, dtype=np.float32)[None, :] * 0.1 + core_index
        )
        normalized[alias] = values
        cores[alias] = SimpleNamespace(alias=alias, n_nodes=nodes_per_core)
        common = []
        for replicate in workflow.MASK_REPLICATES:
            mask = np.zeros((nodes_per_core, 1000), dtype=np.bool_)
            mask[:, target_indices] = True
            common.append(
                SimpleNamespace(
                    label="common_20",
                    replicate=replicate,
                    checksum=_sha(f"{alias}-mask-{replicate}"),
                    mask=mask,
                )
            )
        masks[alias] = SimpleNamespace(common=tuple(common))
        tile_count = workflow.TILE_COUNTS[alias]
        partitions = np.array_split(np.arange(nodes_per_core), tile_count)
        for tile_index, indices in enumerate(partitions):
            tiles.append(
                SimpleNamespace(
                    alias=alias,
                    tile_index=tile_index,
                    node_indices=np.asarray(indices, dtype=np.int64),
                    n_nodes=len(indices),
                )
            )
    evidence = tuple(
        SimpleNamespace(
            seed=seed,
            run_id=f"run-seed-{seed}",
            checkpoint_sha256=_sha(f"checkpoint-{seed}"),
            state_dict_sha256=_sha(f"state-{seed}"),
        )
        for seed in workflow.SEEDS
    )
    return workflow.AuditExecutionContext(
        paths=SimpleNamespace(
            project_root=tmp_path,
            scratch_root=tmp_path / "scratch",
        ),
        database_path=tmp_path / "registry.sqlite3",
        evidence=evidence,
        config={},
        cohort=_Cohort(cores),
        tiled=_Tiled(tiles),
        normalized_expression=normalized,
        masks_by_alias=masks,
        marker_gene_indices=marker_indices,
        target_gene_indices=target_indices,
        analysis_input_sha256=_sha("analysis-input"),
        input_provenance={"fixture": "synthetic"},
    )


def _receiver_rows(context: workflow.AuditExecutionContext) -> list[dict[str, Any]]:
    rows = []
    for tile in context.tiled.tiles:
        entry = workflow._common_masks(
            context.masks_by_alias[tile.alias]
        )[0]
        for target, gene_index in zip(
            workflow.FROZEN_TARGET_GENES,
            context.target_gene_indices,
            strict=True,
        ):
            population = int(
                np.count_nonzero(entry.mask[tile.node_indices, gene_index])
            )
            candidates = np.flatnonzero(
                entry.mask[tile.node_indices, gene_index]
            ).astype(np.int64)
            _, sample_sha256 = workflow._expected_receiver_sample(
                alias=tile.alias,
                tile_index=tile.tile_index,
                target_gene=target,
                mask_checksum=entry.checksum,
                candidates=candidates,
            )
            rows.append(
                {
                    "core_alias": tile.alias,
                    "target_gene": target,
                    "tile_index": tile.tile_index,
                    "population_count": population,
                    "sample_count": 8,
                    "sample_identity_sha256": sample_sha256,
                }
            )
    return rows


def _arrays(
    context: workflow.AuditExecutionContext,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, shape in workflow.REQUIRED_ARRAYS.items():
        if name == "selected_mask":
            arrays[name] = np.zeros(shape, dtype=np.bool_)
        elif name in {"core_offsets", "tile_decomposition_population_count"}:
            arrays[name] = np.zeros(shape, dtype=np.int64)
        else:
            arrays[name] = np.zeros(shape, dtype=np.float32)
    offsets = np.cumsum(
        [0]
        + [
            context.cohort.core(alias).n_nodes
            for alias in workflow.ALIASES
        ],
        dtype=np.int64,
    )
    arrays["core_offsets"][:] = offsets
    target_positions = tuple(
        workflow.FROZEN_MARKER_GENES.index(gene)
        for gene in workflow.FROZEN_TARGET_GENES
    )
    for core_index, alias in enumerate(workflow.ALIASES):
        values = context.normalized_expression[alias]
        selection = slice(offsets[core_index], offsets[core_index + 1])
        truth = values[:, context.target_gene_indices]
        arrays["selected_truth"][selection] = truth
        arrays["per_core_gene_mean"][core_index] = truth.mean(axis=0)
        for replicate in workflow.MASK_REPLICATES:
            mask = workflow._common_masks(
                context.masks_by_alias[alias]
            )[replicate].mask[:, context.target_gene_indices]
            arrays["selected_mask"][selection, replicate] = mask
            arrays["selected_prediction_observed"][
                selection, replicate
            ] = truth[:, None, :][:, 0, :] + np.float32(seed * 0.001)
            arrays["selected_prediction_permuted"][
                selection, replicate
            ] = truth[:, None, :][:, 0, :] + np.float32(0.2 + seed * 0.001)
        marker = values[:, context.marker_gene_indices]
        arrays["source_nonzero_prevalence"][core_index] = np.mean(
            marker > 0, axis=0
        )
        arrays["source_mean_expression"][core_index] = marker.mean(axis=0)
        arrays["source_population_sd"][core_index] = marker.std(axis=0)
        quantiles = np.quantile(
            marker, (0.01, 0.99), axis=0, method="linear"
        )
        arrays["source_q01"][core_index] = quantiles[0]
        arrays["source_q99"][core_index] = quantiles[1]
        for target_index, gene_index in enumerate(
            context.target_gene_indices
        ):
            for source_index in range(39):
                arrays["abs_target_source_raw_pearson"][
                    core_index, target_index, source_index
                ] = abs(
                    workflow._pearson_or_zero(
                        values[:, gene_index], marker[:, source_index]
                    )
                )
        matrix = (
            np.arange(39 * 39, dtype=np.float32).reshape(39, 39)
            + 1.0
            + seed * 0.01
            + core_index * 0.001
        )
        arrays["source_unmasked_signed"][core_index] = matrix
        arrays["masked_rep0_observed_signed"][core_index] = matrix
        arrays["masked_rep0_permuted_signed"][core_index] = matrix * 0.5
        arrays["masked_locked_observed_signed"][
            core_index, 0
        ] = matrix[list(target_positions)]
        arrays["masked_locked_observed_signed"][
            core_index, 1
        ] = matrix[list(target_positions)] * 1.01
        arrays["masked_locked_observed_signed"][
            core_index, 2
        ] = matrix[list(target_positions)] * 0.99
        arrays["faithfulness_predicted"][core_index] = 0.1
        arrays["faithfulness_actual"][core_index] = 0.1

    for tile_index, tile in enumerate(context.tiled.tiles):
        entry = workflow._common_masks(
            context.masks_by_alias[tile.alias]
        )[0]
        populations = [
            int(np.count_nonzero(entry.mask[tile.node_indices, gene_index]))
            for gene_index in context.target_gene_indices
        ]
        arrays["tile_decomposition_population_count"][tile_index] = populations
        same = np.full((5, 39), 1.0 + seed * 0.01, dtype=np.float32)
        other = np.full((5, 39), 2.0, dtype=np.float32)
        arrays["tile_decomposition_same_signed"][tile_index] = same
        arrays["tile_decomposition_other_signed"][tile_index] = other
        arrays["tile_decomposition_total_signed"][tile_index] = same + other
        arrays["tile_decomposition_same_l1"][tile_index] = np.abs(same)
        arrays["tile_decomposition_other_l1"][tile_index] = np.abs(other)
        arrays["tile_decomposition_total_l1"][tile_index] = (
            np.abs(same) + np.abs(other)
        )
    for core_index, alias in enumerate(workflow.ALIASES):
        positions = [
            index
            for index, tile in enumerate(context.tiled.tiles)
            if tile.alias == alias
        ]
        population = arrays["tile_decomposition_population_count"][positions]
        denominator = population.sum(axis=0)
        for short_name in (
            "same_signed",
            "other_signed",
            "total_signed",
            "same_l1",
            "other_l1",
            "total_l1",
        ):
            tile_values = arrays[f"tile_decomposition_{short_name}"][positions]
            arrays[f"decomposition_{short_name}"][core_index] = (
                (tile_values * population[:, :, None]).sum(axis=0)
                / denominator[:, None]
            )
    arrays["randomized_rep0_observed_signed"][:] = 0.25
    return arrays


def _resources() -> dict[str, Any]:
    return {
        "device": "cpu",
        "gpu_name": None,
        "cuda_version": None,
        "torch_version": "synthetic",
        "python_version": "synthetic",
        "runtime_seconds": 1.0,
        "peak_allocated_vram_gib": 0.0,
    }


class _Backend:
    def run_pilot(
        self,
        *,
        context: workflow.AuditExecutionContext,
        device: str,
    ) -> dict[str, Any]:
        del context, device
        return {
            "diagnostics": {
                "finite_outputs": True,
                "finite_gradients": True,
                "exact_gradient_decomposition": True,
                "masked_input_zero_gradient": True,
                "outside_receptive_field_zero_gradient": True,
                "cross_tile_zero_gradient": True,
                "deterministic_replay": True,
                "analytical_tiny_graph_control": True,
                "autograd_centered_finite_difference": True,
                "identical_checkpoint_reload": True,
                "sufficient_disk": True,
                "pilot_receiver_inventory_complete": True,
            },
            "resources": {
                **_resources(),
                "projected_runtime_hours_per_seed": 0.1,
            },
        }

    def run_seed_shard(
        self,
        *,
        context: workflow.AuditExecutionContext,
        seed: int,
        device: str,
    ) -> dict[str, Any]:
        del device
        return {
            "arrays": _arrays(context, seed=seed),
            "receiver_samples": _receiver_rows(context),
            "diagnostics": {
                "finite_outputs": True,
                "finite_gradients": True,
                "exact_gradient_decomposition": True,
                "masked_input_zero_gradient": True,
                "outside_receptive_field_zero_gradient": True,
                "cross_tile_zero_gradient": True,
                "model_eval_mode": True,
                "model_parameter_gradients_disabled": True,
                "population_weighted_tile_aggregation": True,
                "full_core_perturbation_bounds": True,
            },
            "resources": _resources(),
        }


def _pilot(
    context: workflow.AuditExecutionContext,
    work: Path,
) -> str:
    workflow.run_pilot(
        context=context,
        backend=_Backend(),
        work_root=work,
        device="cpu",
    )
    return workflow._sha256_file(workflow._pilot_path(work))


def test_pilot_and_seed_shard_publish_strict_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _synthetic_context(tmp_path, monkeypatch)
    work = tmp_path / "active"
    pilot_sha = _pilot(context, work)
    metadata = workflow.run_seed_shard(
        context=context,
        backend=_Backend(),
        work_root=work,
        seed=0,
        device="cpu",
        reviewed_pilot_sha256=pilot_sha,
    )

    metadata_path, arrays_path = workflow._shard_paths(work, 0)
    assert metadata_path.is_file()
    assert arrays_path.is_file()
    assert workflow._verify_sidecar(metadata_path)["path"] == metadata_path.name
    assert workflow._verify_sidecar(arrays_path)["path"] == arrays_path.name
    assert metadata["arrays"]["retention"] == workflow.ARRAY_RETENTION
    assert metadata["row_identifiers_persisted"] is False
    assert len(metadata["receiver_samples"]) == 26 * 5


def test_seed_shard_rejects_wrong_pilot_and_population_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _synthetic_context(tmp_path, monkeypatch)
    work = tmp_path / "active"
    pilot_sha = _pilot(context, work)
    with pytest.raises(
        workflow.GradientAuditWorkflowError, match="reviewed pilot checksum"
    ):
        workflow.run_seed_shard(
            context=context,
            backend=_Backend(),
            work_root=work,
            seed=0,
            device="cpu",
            reviewed_pilot_sha256=_sha("wrong"),
        )

    result = _Backend().run_seed_shard(
        context=context, seed=0, device="cpu"
    )
    result["receiver_samples"][0]["population_count"] += 1
    with pytest.raises(
        workflow.GradientAuditWorkflowError,
        match="sampling population/count/identity",
    ):
        workflow._validate_seed_result(result, context=context, seed=0)
    result = _Backend().run_seed_shard(
        context=context, seed=0, device="cpu"
    )
    result["receiver_samples"][0]["sample_identity_sha256"] = _sha(
        "different-valid-hash"
    )
    with pytest.raises(
        workflow.GradientAuditWorkflowError,
        match="sampling population/count/identity",
    ):
        workflow._validate_seed_result(result, context=context, seed=0)
    assert workflow._sha256_file(workflow._pilot_path(work)) == pilot_sha


def test_seed_schema_rejects_false_decomposition_and_wrong_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _synthetic_context(tmp_path, monkeypatch)
    result = _Backend().run_seed_shard(
        context=context, seed=0, device="cpu"
    )
    result["arrays"]["decomposition_total_signed"][0, 0, 0] += 1
    with pytest.raises(
        workflow.GradientAuditWorkflowError, match="signed decomposition"
    ):
        workflow._validate_seed_result(result, context=context, seed=0)

    result = _Backend().run_seed_shard(
        context=context, seed=0, device="cpu"
    )
    result["arrays"]["masked_rep0_observed_signed"] = result["arrays"][
        "masked_rep0_observed_signed"
    ].astype(np.float64)
    with pytest.raises(
        workflow.GradientAuditWorkflowError, match="must use float32"
    ):
        workflow._validate_seed_result(result, context=context, seed=0)


def test_full_core_bounds_are_sliced_not_reestimated_per_tile() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 100.0, 101.0, 102.0], dtype=np.float32)
    lower, upper = np.quantile(values, (0.01, 0.99), method="linear")
    population_sd = float(np.std(values, ddof=0))
    plus, minus = workflow._full_core_directions(
        values,
        np.ones(values.shape, dtype=bool),
        scale=0.10,
        population_sd=population_sd,
        lower=float(lower),
        upper=float(upper),
    )

    # Value 2 is at the top of the first tile [0,1,2] but well inside the
    # full-core bounds.  A per-tile quantile would incorrectly suppress it.
    assert plus[2] > 0
    assert minus[3] < 0
    assert plus[0] == minus[0] == 0
    assert plus[-1] == minus[-1] == 0


def test_target_receiver_count_selection_uses_vector_fancy_indexing() -> None:
    counts = np.arange(39, dtype=np.int64)
    positions = tuple(
        workflow.FROZEN_MARKER_GENES.index(gene)
        for gene in workflow.FROZEN_TARGET_GENES
    )

    selected = workflow._selected_receiver_counts(counts, positions)

    assert selected.shape == (5,)
    assert np.array_equal(selected, counts[list(positions)])


def test_runtime_projection_uses_larger_node_or_edge_ratio_and_overheads() -> None:
    projected, details = workflow._project_full_seed_runtime_seconds(
        trained_pilot_runtime_seconds=100.0,
        random_reference_runtime_seconds=20.0,
        context_load_runtime_seconds=10.0,
        pilot_nodes=100,
        total_nodes=800,
        pilot_edges=1_000,
        total_edges=9_000,
        safety_factor=1.25,
    )

    assert details["pilot_to_full_node_ratio"] == 8.0
    assert details["pilot_to_full_edge_ratio"] == 9.0
    assert details["projection_workload_ratio"] == 9.0
    assert details["projection_safety_factor"] == 1.25
    assert projected == pytest.approx(1.25 * (100.0 * 9.0 + 20.0 + 10.0))


def test_conclusion_code_and_external_source_are_bound(
    tmp_path: Path,
) -> None:
    runner_path = tmp_path / "runner.py"
    runner_path.write_text("# synthetic runner\n", encoding="utf-8")
    hashes = workflow._implementation_file_hashes(
        SimpleNamespace(__file__=str(runner_path))
    )
    assert set(hashes) == {
        "workflow_script",
        "gradient_core",
        "gradient_reducer",
        "upstream_runner",
    }
    assert all(len(value) == 64 for value in hashes.values())
    assert hashes["gradient_reducer"] == workflow._sha256_file(
        Path(workflow.reduce_gradient_audit.__code__.co_filename)
    )

    source = workflow._audited_external_source_identity(
        SimpleNamespace(project_root=workflow.PROJECT_ROOT)
    )
    assert source["repository_commit"] == workflow.EXTERNAL_SOURCE_COMMIT
    assert (
        source["script_sha256"]
        == workflow.EXTERNAL_SOURCE_SCRIPT_SHA256
    )
    assert (
        source["source_statistic_audit_sha256"]
        == workflow.SOURCE_STATISTIC_AUDIT_SHA256
    )


def test_aggregate_requires_exact_seven_shards_and_binds_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _synthetic_context(tmp_path, monkeypatch)
    work = tmp_path / "active"
    pilot_sha = _pilot(context, work)
    for seed in workflow.SEEDS:
        workflow.run_seed_shard(
            context=context,
            backend=_Backend(),
            work_root=work,
            seed=seed,
            device="cpu",
            reviewed_pilot_sha256=pilot_sha,
        )
    output = tmp_path / "aggregate"
    result = workflow.aggregate_verified_shards(
        context=context,
        work_root=work,
        output_dir=output,
        reviewed_pilot_sha256=pilot_sha,
        execution_commands=[
            ["audit", "pilot"],
            *[["audit", "seed-shard", str(seed)] for seed in workflow.SEEDS],
            ["audit", "aggregate"],
        ],
    )

    assert result["analysis_status"] == "complete"
    assert result["claim_verdict"] == "biological_mechanism_not_validated"
    assert result["mechanism_validation_available"] is False
    assert result["mechanism_claim_supported"] is False
    assert isinstance(
        result["candidate_set_computational_precursors_supported"], bool
    )
    assert len(result["gate_rows"]) == 40
    assert len(
        {
            (row["gate"], row["scope"])
            for row in result["gate_rows"]
        }
    ) == 40
    expected_fraction = sum(
        int(row["pass"]) for row in result["gate_rows"]
    ) / 40
    assert result["final_metrics"] == {
        workflow.FINAL_METRIC_NAME: expected_fraction
    }
    assert result["final_metric_roles"] == {
        workflow.FINAL_METRIC_NAME: (
            "registry_bookkeeping_only_not_an_evidence_score_or_claim"
        )
    }
    for reducer_key in (
        "eligibility",
        "seed_rank_stability",
        "mask_rank_stability",
        "signed_pair_stability",
        "bounded_faithfulness",
        "graph_gradient_null",
        "parameter_randomization",
        "matched_pair_null",
        "source_style_reproduction",
        "sampled_receiver_decomposition",
    ):
        assert reducer_key in result
    with (output / "gate_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
    assert reader.fieldnames == [
        "gate",
        "scope",
        "threshold",
        "observed",
        "pass",
    ]
    assert len(csv_rows) == 40
    report_rows = {
        (row["gate"], row["scope"]): row for row in result["gate_rows"]
    }
    for row in csv_rows:
        reported = report_rows[(row["gate"], row["scope"])]
        assert json.loads(row["threshold"]) == reported["threshold"]
        assert json.loads(row["observed"]) == reported["observed"]
        assert json.loads(row["pass"]) is reported["pass"]
    markdown = (output / "report.md").read_text(encoding="utf-8")
    portable_html = (output / "report.html").read_text(encoding="utf-8")
    assert "biological mechanism not validated" in markdown.lower()
    assert "`mechanism_claim_supported`: `false`" in markdown
    assert "Biological mechanism not validated." in portable_html
    assert "<code>false</code>" in portable_html
    assert "http://" not in portable_html and "https://" not in portable_html

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    checksum = manifest.pop("checksum")
    assert checksum == workflow.canonical_sha256(manifest)
    assert result["analysis_input_sha256"] == manifest["analysis_input_sha256"]
    assert result["scientific_input_sha256"] == context.analysis_input_sha256
    assert [row["seed"] for row in manifest["shards"]] == list(workflow.SEEDS)
    assert {
        row["role"] for row in manifest["files"]
    } == {
        "audit_report_json",
        "gate_results_csv",
        "audit_report_markdown",
        "audit_report_html",
        "canonical_fit_predictions_jsonl",
    }
    assert len(manifest["execution_commands"]) == 9
    assert result["execution_resources"]["failure_count"] == 0
    assert result["execution_resources"]["failures"] == []
    assert [
        row["seed"] for row in result["execution_resources"]["seed_shards"]
    ] == list(workflow.SEEDS)
    assert result["execution_resources"]["locked_parallel_gpu_map"] == {
        str(seed): device
        for seed, device in workflow.PRODUCTION_GPU_MAP.items()
    }
    assert not any(
        key in (output / "report.json").read_text(encoding="utf-8")
        for key in ('"cell_id"', '"row_id"', '"patient_id"')
    )
