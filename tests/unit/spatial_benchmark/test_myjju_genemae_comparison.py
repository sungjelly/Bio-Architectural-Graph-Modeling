from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pytest
import torch
import yaml

from spatial_benchmark import myjju_genemae_comparison as comparison
from spatial_benchmark import myjju_genemae_provider as builtin_provider
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.myjju_genemae import (
    SOURCE_COMMIT,
    SOURCE_FILE_SHA256,
)
from spatial_benchmark.paths import current_paths
from scripts.train.run_myjju_genemae_pooled import state_dict_sha256


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mask(rate: float, replicate: int, cells: int = 4) -> np.ndarray:
    width = int(round(rate * comparison.EXPECTED_GENE_COUNT))
    result = np.zeros(
        (cells, comparison.EXPECTED_GENE_COUNT), dtype=np.bool_
    )
    for cell in range(cells):
        start = (
            replicate * 131 + cell * 47
        ) % comparison.EXPECTED_GENE_COUNT
        indices = (
            np.arange(start, start + width)
            % comparison.EXPECTED_GENE_COUNT
        )
        result[cell, indices] = True
    return result


def _target(alias: str, cells: int = 4) -> np.ndarray:
    alias_index = comparison.ALIASES.index(alias)
    genes = np.linspace(
        0.05,
        4.0,
        comparison.EXPECTED_GENE_COUNT,
        dtype=np.float32,
    )
    cell_offsets = np.arange(cells, dtype=np.float32)[:, None] * 0.65
    return genes[None, :] + cell_offsets + alias_index * 0.01


def _member_metrics(
    target: np.ndarray,
    mask: np.ndarray,
    base_prediction: np.ndarray,
) -> dict[int, Mapping[str, Any]]:
    output: dict[int, Mapping[str, Any]] = {}
    gene_pattern = np.linspace(
        -1.0, 1.0, comparison.EXPECTED_GENE_COUNT, dtype=np.float32
    )[None, :]
    for seed in comparison.SEEDS:
        prediction = base_prediction + gene_pattern * (seed - 3) * 0.002
        output[seed] = comparison.masked_regression_metrics(
            target, prediction, mask
        )
    return output


class SyntheticProvider:
    def __init__(
        self,
        tmp_path: Path,
        *,
        invalid_mask: bool = False,
        weak_genemae: bool = False,
    ) -> None:
        members: dict[
            str, tuple[comparison.RegisteredRunEvidence, ...]
        ] = {}
        for model_key in comparison.MODEL_KEYS:
            model_members = []
            for seed in comparison.SEEDS:
                root = tmp_path / f"{model_key}-{seed}"
                checkpoint = root / "checkpoints" / "last.ckpt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(f"{model_key}-{seed}".encode("utf-8"))
                model_members.append(
                    comparison.RegisteredRunEvidence(
                        model_key=model_key,
                        seed=seed,
                        run_id=f"run-{model_key}-{seed}",
                        attempt=1,
                        artifact_root=root,
                        checkpoint_path=checkpoint,
                        checkpoint_sha256=comparison.sha256_file(checkpoint),
                        state_dict_sha256=_checksum(
                            f"state-{model_key}-{seed}"
                        ),
                        config_sha256=_checksum(
                            f"config-{model_key}-{seed}"
                        ),
                        bundle_verified=True,
                        registry_artifacts_verified=True,
                        checkpoint_catalog_verified=True,
                        parameter_count=(
                            comparison.EXPECTED_GENEMAE_PARAMETERS
                            if model_key == comparison.GENEMAE
                            else 11_674_880
                        ),
                        completed_epochs=200,
                        final_epoch=199,
                        duration_seconds=100.0 + seed,
                        peak_vram_gib=4.0,
                        peak_host_memory_gib=2.0,
                        convergence={"finite": True},
                        resources={"device": "synthetic"},
                    )
                )
            members[model_key] = tuple(model_members)
        self._audit = comparison.ComparisonAudit(
            members=members,
            attempt_inventory=tuple(
                {
                    "model_key": model_key,
                    "seed": seed,
                    "run_id": f"run-{model_key}-{seed}",
                    "attempt": 1,
                    "status": "completed",
                    "selected_completed_attempt": True,
                }
                for model_key in comparison.MODEL_KEYS
                for seed in comparison.SEEDS
            ),
            failure_inventory=(),
            provenance={
                "fixture": "synthetic",
                "current_bagm_whole_node_context": {
                    **comparison.EXPECTED_CURRENT_BAGM_CONTEXT,
                    "estimand": (
                        "held_in_whole_node_hybrid_count_reconstruction"
                    ),
                    "context_only_not_ranked_against_genemae": True,
                    "graph_gate_passed": True,
                    "overall_campaign_outcome": "negative",
                    "failed_gates": [
                        "pooled_data_gate",
                        "representation_gate",
                    ],
                },
                "source_report_historical_context": {
                    "context_only_not_comparable": True,
                    "historical_weights_available": False,
                    "metrics": comparison.EXPECTED_SOURCE_HISTORICAL_METRICS,
                    "checkpoint_selection_leakage": True,
                    "source_audit_sha256": _checksum("source-audit"),
                },
            },
        )
        self.invalid_mask = invalid_mask
        self.weak_genemae = weak_genemae

    def audit(self) -> comparison.ComparisonAudit:
        return self._audit

    def provenance(self) -> Mapping[str, Any]:
        return {"adapter": "synthetic", "row_level_outputs": False}

    def iter_evaluation_batches(
        self,
    ) -> Iterable[comparison.EvaluationBatch]:
        for alias in comparison.ALIASES:
            target = _target(alias)
            ordered_gene_sha = _checksum("ordered-1000-genes")
            for replicate in comparison.REPLICATES:
                for rate in (
                    comparison.COMMON_MASK_RATE,
                    comparison.NATIVE_MASK_RATE,
                ):
                    mask = _mask(rate, replicate)
                    mask_checksum = _checksum(
                        f"{alias}-{rate}-{replicate}"
                    )
                    for condition in (
                        "observed",
                        "node_label_permuted",
                    ):
                        error = (
                            0.95
                            if condition == "node_label_permuted"
                            else (
                                0.75
                                if self.weak_genemae
                                and rate == comparison.COMMON_MASK_RATE
                                else 0.05
                            )
                        )
                        prediction = target + error
                        yield comparison.EvaluationBatch(
                            model_key=comparison.GENEMAE,
                            core_alias=alias,
                            mask_rate=rate,
                            replicate=replicate,
                            graph_condition=condition,
                            target=target,
                            mask=mask,
                            ensemble_prediction=prediction,
                            member_metrics=_member_metrics(
                                target, mask, prediction
                            ),
                            mask_seed=10_000 + replicate,
                            expected_mask_checksum=mask_checksum,
                            regenerated_mask_checksum=(
                                _checksum("wrong")
                                if self.invalid_mask
                                and alias == "ANC-01"
                                and replicate == 0
                                and rate == comparison.COMMON_MASK_RATE
                                and condition == "observed"
                                else mask_checksum
                            ),
                            ordered_gene_sha256=ordered_gene_sha,
                            ensemble_rule=comparison.GENEMAE_ENSEMBLE_RULE,
                            ensemble_rule_verified=True,
                            prediction_scale=comparison.TARGET_SCALE,
                            target_preprocessing_uses_full_cell_library=True,
                            oracle_true_library_size_used=False,
                            graph_null_verified=(
                                condition == "node_label_permuted"
                            ),
                            degree_sequence_preserved=(
                                condition == "node_label_permuted"
                            ),
                            topology_preserved=(
                                condition == "node_label_permuted"
                            ),
                            permutation_seed=(
                                20_000 + replicate
                                if condition == "node_label_permuted"
                                else None
                            ),
                        )
                common_mask = _mask(comparison.COMMON_MASK_RATE, replicate)
                common_checksum = _checksum(
                    f"{alias}-{comparison.COMMON_MASK_RATE}-{replicate}"
                )
                for model_key, error in (
                    (comparison.BAGM_GAT, 0.45),
                    (comparison.BAGM_SELF, 0.6),
                ):
                    prediction = target + error
                    yield comparison.EvaluationBatch(
                        model_key=model_key,
                        core_alias=alias,
                        mask_rate=comparison.COMMON_MASK_RATE,
                        replicate=replicate,
                        graph_condition="observed",
                        target=target,
                        mask=common_mask,
                        ensemble_prediction=prediction,
                        member_metrics=_member_metrics(
                            target, common_mask, prediction
                        ),
                        mask_seed=10_000 + replicate,
                        expected_mask_checksum=common_checksum,
                        regenerated_mask_checksum=common_checksum,
                        ordered_gene_sha256=ordered_gene_sha,
                        ensemble_rule=comparison.BAGM_ENSEMBLE_RULE,
                        ensemble_rule_verified=True,
                        prediction_scale=comparison.TARGET_SCALE,
                        target_preprocessing_uses_full_cell_library=True,
                        oracle_true_library_size_used=True,
                    )


def test_masked_regression_metrics_reconcile_simple_values() -> None:
    target = np.zeros((2, comparison.EXPECTED_GENE_COUNT), dtype=np.float32)
    prediction = np.zeros_like(target)
    mask = np.zeros_like(target, dtype=np.bool_)
    mask[:, :2] = True
    target[:, :2] = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    prediction[:, :2] = np.asarray([[0.0, 2.0], [2.0, 5.0]])

    metrics = comparison.masked_regression_metrics(
        target, prediction, mask
    )

    assert metrics["n_masked"] == 4
    assert metrics["masked_mae"] == pytest.approx(0.75)
    assert metrics["masked_mse"] == pytest.approx(1.25)
    assert metrics["masked_huber"] == pytest.approx(0.5)
    assert metrics["pooled_pearson"] is not None
    assert metrics["defined_gene_correlations"] == 2
    assert metrics["defined_cell_correlations"] == 2


def test_favoring_core_means_strictly_lower_not_two_percent_gain() -> None:
    core_rows: list[dict[str, Any]] = []
    for index, alias in enumerate(comparison.ALIASES):
        genemae_huber = 0.99 if index < 8 else 0.80
        for model_key, condition, huber in (
            (comparison.GENEMAE, "observed", genemae_huber),
            (comparison.BAGM_GAT, "observed", 1.0),
            (
                comparison.GENEMAE,
                "node_label_permuted",
                1.0,
            ),
            ("per_core_gene_mean", "observed", 1.0),
        ):
            core_rows.append(
                {
                    "model_key": model_key,
                    "core_alias": alias,
                    "mask_rate": comparison.COMMON_MASK_RATE,
                    "graph_condition": condition,
                    "masked_huber": huber,
                }
            )
    equal_core_rows = [
        {
            "model_key": model_key,
            "mask_rate": comparison.COMMON_MASK_RATE,
            "graph_condition": condition,
            "masked_huber": huber,
            "masked_mae": mae,
            "gene_pearson_mean": gene_pearson,
        }
        for model_key, condition, huber, mae, gene_pearson in (
            (comparison.GENEMAE, "observed", 0.95, 0.80, 0.30),
            (comparison.BAGM_GAT, "observed", 1.00, 0.90, 0.20),
            (
                comparison.GENEMAE,
                "node_label_permuted",
                1.00,
                0.90,
                0.20,
            ),
            ("per_core_gene_mean", "observed", 1.00, 0.90, 0.20),
        )
    ]
    member_equal_core_rows = [
        {
            "model_key": model_key,
            "seed": seed,
            "mask_rate": comparison.COMMON_MASK_RATE,
            "graph_condition": "observed",
            "masked_huber": huber,
        }
        for seed in comparison.SEEDS
        for model_key, huber in (
            (comparison.GENEMAE, 0.95),
            (comparison.BAGM_GAT, 1.00),
        )
    ]

    gates, rows = comparison.evaluate_frozen_gates(
        core_rows=core_rows,
        equal_core_rows=equal_core_rows,
        member_equal_core_rows=member_equal_core_rows,
    )

    by_gate = {row["gate"]: row for row in gates}
    assert by_gate["primary_comparison"]["passed"] is True
    assert by_gate["primary_comparison"]["observed"][
        "genemae_favoring_cores"
    ] == 10
    assert by_gate["baseline"]["observed"]["genemae_favoring_cores"] == 10
    assert by_gate["graph_use"]["observed"]["unpermuted_favoring_cores"] == 10
    first_eight = [
        row
        for row in rows
        if row.get("core_alias") in comparison.ALIASES[:8]
    ]
    assert all(
        0.0 < row["genemae_vs_bagm_gat_relative_improvement"] < 0.02
        for row in first_eight
    )
    assert all(row["genemae_favored_vs_bagm_gat"] for row in first_eight)


def test_maximum_conclusion_names_non_primary_failures() -> None:
    conclusion = comparison._maximum_conclusion(
        primary_passed=True,
        failed_gates=("baseline", "graph_use"),
    )

    assert "baseline, graph_use" in conclusion
    assert "target-derived per-core gene-mean oracle" in conclusion
    assert "does not support graph-specific predictive gain" in conclusion
    assert "end-to-end trained systems" in conclusion


def test_analyze_provider_requires_exact_coverage_and_passes_frozen_gates(
    tmp_path: Path,
) -> None:
    provider = SyntheticProvider(tmp_path)

    analysis, tables, provenance = comparison.analyze_provider(provider)

    assert analysis["status"] == "complete"
    assert analysis["outcome"] == "supported"
    assert analysis["coverage"]["observed_batches"] == 180
    assert all(gate["passed"] for gate in analysis["frozen_gates"])
    assert len(tables["run_audit"]) == 21
    assert len(tables["ensemble_replicate_metrics"]) == 180
    assert len(tables["baseline_replicate_metrics"]) == 120
    assert len(tables["member_replicate_metrics"]) == 180 * 7
    assert len(tables["core_effect_distributions"]) == 3
    assert len(tables["control_contrasts"]) == 2
    assert all(
        row["n_cores"] == 10
        and row["population_confidence_interval"] is None
        and row["bootstrap_used"] is False
        for row in tables["core_effect_distributions"]
    )
    assert analysis["uncertainty"]["population_confidence_interval"] is None
    assert analysis["uncertainty"]["bootstrap_used"] is False
    assert analysis["comparison_scope"].startswith("end_to_end")
    assert analysis["per_core_gene_mean_reference"]["target_derived"] is True
    assert provenance["protected_identifiers_emitted"] is False


def test_control_contrasts_surface_metric_and_estimand_dependence(
    tmp_path: Path,
) -> None:
    equal_rows = [
        {
            "model_key": model_key,
            "mask_rate": comparison.COMMON_MASK_RATE,
            "graph_condition": "observed",
            "masked_huber": huber,
            "masked_mae": mae,
        }
        for model_key, huber, mae in (
            (comparison.GENEMAE, 0.3482101, 0.5470899),
            ("all_zero", 0.3588236, 0.4065959),
            (comparison.BAGM_GAT, 1.286428, 1.481547),
            (comparison.BAGM_SELF, 1.128644, 1.303384),
        )
    ]
    controls = comparison._control_contrast_rows(
        equal_rows,
        current_bagm_context={
            "graph_gate_passed": True,
            "estimand": "held_in_whole_node_hybrid_count_reconstruction",
        },
    )
    by_name = {row["contrast"]: row for row in controls}
    zero = by_name["genemae_vs_all_zero_common_partial_gene"]
    bagm = by_name[
        "bagm_gat_vs_matched_self_common_partial_gene"
    ]

    assert zero["candidate_huber_relative_improvement"] == pytest.approx(
        0.029578600738635913
    )
    assert zero["candidate_mae_relative_improvement"] == pytest.approx(
        -0.3455371783138984
    )
    assert zero["metric_rank_agreement"] is False
    assert zero["metric_dependent_conclusion"] is True
    assert bagm["candidate_huber_relative_loss_increase"] == pytest.approx(
        0.13979961794861792
    )
    assert bagm["huber_favors_candidate"] is False
    assert bagm["separate_whole_node_graph_gate_passed"] is True
    assert bagm["estimands_are_distinct"] is True
    assert bagm["architecture_level_conclusion_permitted"] is False

    findings = comparison._control_negative_results(controls)
    assert any("2.96% better" in finding for finding in findings)
    assert any("34.55% worse" in finding for finding in findings)
    assert any("13.98% higher" in finding for finding in findings)
    assert any("not an architecture-level result" in finding for finding in findings)

    provider = SyntheticProvider(tmp_path / "members")
    analysis, tables, _ = comparison.analyze_provider(provider)
    markdown = comparison.markdown_report(
        analysis, {**tables, "control_contrasts": controls}
    )
    assert "## Control contrasts and metric dependence" in markdown
    assert "2.96% better than all-zero by Huber" in markdown
    assert "13.98% higher than matched self" in markdown
    assert "undefined or numerically degenerate" in markdown


def test_analyze_provider_reports_valid_negative_primary_gate(
    tmp_path: Path,
) -> None:
    provider = SyntheticProvider(tmp_path, weak_genemae=True)

    analysis, _, _ = comparison.analyze_provider(provider)

    assert analysis["status"] == "complete"
    assert analysis["outcome"] == "negative"
    gates = {row["gate"]: row for row in analysis["frozen_gates"]}
    assert gates["primary_comparison"]["passed"] is False


def test_analyze_provider_rejects_mask_checksum_drift(
    tmp_path: Path,
) -> None:
    provider = SyntheticProvider(tmp_path, invalid_mask=True)

    with pytest.raises(
        comparison.GeneMAEComparisonError,
        match="regenerated mask checksum",
    ):
        comparison.analyze_provider(provider)


def test_validate_comparison_audit_rejects_missing_seed(
    tmp_path: Path,
) -> None:
    provider = SyntheticProvider(tmp_path)
    audit = provider.audit()
    incomplete = comparison.ComparisonAudit(
        members={
            **audit.members,
            comparison.GENEMAE: audit.members[comparison.GENEMAE][:-1],
        },
        attempt_inventory=audit.attempt_inventory,
        failure_inventory=audit.failure_inventory,
    )

    with pytest.raises(
        comparison.GeneMAEComparisonError,
        match="exactly seven",
    ):
        comparison.validate_comparison_audit(incomplete)


def test_publish_report_is_portable_atomic_and_checksum_bound(
    tmp_path: Path,
) -> None:
    provider = SyntheticProvider(tmp_path / "members")
    analysis, tables, provenance = comparison.analyze_provider(provider)
    output = tmp_path / "report"

    manifest = comparison.publish_report(
        output_dir=output,
        analysis=analysis,
        tables=tables,
        provenance=provenance,
    )

    assert manifest["portable_single_file_html"] is True
    assert (output / "report.md").is_file()
    document = (output / "report.html").read_text(encoding="utf-8")
    assert "<style>" in document
    assert "http://" not in document
    assert "https://" not in document
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "| Gate | Result | Observed | Frozen threshold |" in markdown
    assert "## Descriptive across-core effects" in markdown
    assert "end-to-end trained-system comparison" in markdown
    assert "target-derived oracle" in markdown
    assert "constructor was minimally repaired" in markdown
    assert "Spearman" in markdown
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for relative, record in saved["files"].items():
        assert comparison.sha256_file(output / relative) == record["sha256"]
    with pytest.raises(
        comparison.GeneMAEComparisonError, match="will not be overwritten"
    ):
        comparison.publish_report(
            output_dir=output,
            analysis=analysis,
            tables=tables,
            provenance=provenance,
        )


def test_publish_report_rejects_row_level_identifier(
    tmp_path: Path,
) -> None:
    provider = SyntheticProvider(tmp_path / "members")
    analysis, tables, provenance = comparison.analyze_provider(provider)
    unsafe_tables = {
        **tables,
        "unsafe": [{"cell_id": "not-allowed", "value": 1}],
    }

    with pytest.raises(
        comparison.GeneMAEComparisonError,
        match="prohibited identifier",
    ):
        comparison.publish_report(
            output_dir=tmp_path / "unsafe",
            analysis=analysis,
            tables=unsafe_tables,
            provenance=provenance,
        )


def test_batch_rejects_unverified_canonical_ensemble(
    tmp_path: Path,
) -> None:
    provider = SyntheticProvider(tmp_path)
    batches = iter(provider.iter_evaluation_batches())
    first = next(batches)
    invalid = replace(first, ensemble_rule_verified=False)

    with pytest.raises(
        comparison.GeneMAEComparisonError,
        match="canonical prediction ensemble",
    ):
        comparison._validate_batch(invalid)


class _AuditRegistry:
    def __init__(self, catalog: Mapping[str, Mapping[str, Any]]) -> None:
        self.catalog = catalog

    def verify_artifacts(self, *, run_id: str) -> list[dict[str, Any]]:
        return []

    def list_checkpoint_catalog(
        self, *, run_id: str, role: str, limit: int | None
    ) -> list[dict[str, Any]]:
        assert role == "last"
        assert limit is None
        return [dict(self.catalog[run_id])]


def test_discover_registered_production_accepts_real_runner_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = current_paths()
    source_audit_path = (
        paths.project_root
        / "experiments/campaigns"
        / comparison.CAMPAIGN_ID
        / "external_source_audit.yaml"
    )
    source_audit_sha = comparison.sha256_file(source_audit_path)
    rows: list[dict[str, Any]] = []
    catalog: dict[str, dict[str, Any]] = {}
    for seed in comparison.SEEDS:
        run_id = f"r-schema-{seed}"
        root = tmp_path / run_id
        for relative in (
            "checkpoints",
            "diagnostics",
            "provenance",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)
        config = {
            "seed": seed,
            "campaign": {"campaign_id": comparison.CAMPAIGN_ID},
            "model": {"name": comparison.GENEMAE},
            "metadata": {"execution_role": "production"},
        }
        config_path = root / "config.resolved.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
        )
        state = {"weight": torch.tensor([seed + 0.25], dtype=torch.float32)}
        state_sha = state_dict_sha256(state)
        graph_identity = {
            "graph_bundle_sha256": _checksum(f"graph-{seed}")
        }
        mask_identities = {
            alias: {"identity": f"{alias}-{seed}"}
            for alias in comparison.ALIASES
        }
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "model_name": comparison.GENEMAE,
            "seed": seed,
            "final_epoch": comparison.EXPECTED_GENEMAE_FINAL_EPOCH,
            "parameter_count": comparison.EXPECTED_GENEMAE_PARAMETERS,
            "model_state_dict": state,
            "state_dict_sha256": state_sha,
            "external_source_commit": SOURCE_COMMIT,
            "external_source_file_sha256": dict(SOURCE_FILE_SHA256),
            "source_audit_sha256": source_audit_sha,
            "source_fidelity_repair": (
                "assign_self_hidden_constructor_attribute"
            ),
            "historical_weights_used": False,
            "target_scale": comparison.TARGET_SCALE,
            "training_mask_rate": comparison.NATIVE_MASK_RATE,
            "runtime_config_sha256": canonical_sha256(config),
            "graph_bundle_sha256": graph_identity[
                "graph_bundle_sha256"
            ],
            "evaluation_mask_identities": mask_identities,
        }
        checkpoint = root / "checkpoints/last.ckpt"
        torch.save(payload, checkpoint)
        checkpoint_sha = comparison.sha256_file(checkpoint)
        summary = {
            "run_id": run_id,
            "status": "success",
            "campaign_id": comparison.CAMPAIGN_ID,
            "model_name": comparison.GENEMAE,
            "model_seed": seed,
            "parameter_count": comparison.EXPECTED_GENEMAE_PARAMETERS,
            "completed_global_epochs": comparison.EXPECTED_GENEMAE_EPOCHS,
            "final_epoch": comparison.EXPECTED_GENEMAE_FINAL_EPOCH,
            "checkpoint_role": "last",
            "primary_metric_name": (
                "fit/partial_gene/log1p_cp10k_masked_huber"
            ),
            "generalization_estimate": False,
            "checkpoint_path": "checkpoints/last.ckpt",
            "checkpoint_sha256": checkpoint_sha,
            "state_dict_sha256": state_sha,
            "duration_seconds": 123.0,
        }
        (root / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        convergence = {
            "completed_epochs": comparison.EXPECTED_GENEMAE_EPOCHS,
            "expected_epochs": comparison.EXPECTED_GENEMAE_EPOCHS,
            "every_tile_once_each_epoch": True,
            "all_losses_finite": True,
            "all_gradients_finite": True,
            "all_parameters_finite": True,
            "all_global_epochs_completed": True,
            "all_losses_and_gradients_finite": True,
            "checkpoint_is_final_epoch_only": True,
        }
        (root / "diagnostics/convergence.json").write_text(
            json.dumps(convergence), encoding="utf-8"
        )
        (root / "diagnostics/resource.json").write_text(
            json.dumps(
                {
                    "peak_allocated_vram_gib": 8.0,
                    "peak_host_memory_gib": 4.0,
                }
            ),
            encoding="utf-8",
        )
        (root / "provenance/external_source_audit.json").write_text(
            json.dumps(
                {
                    "contract_sha256": comparison.FROZEN_CONTRACT_SHA256,
                    "source_audit_sha256": source_audit_sha,
                    "external_source_commit": SOURCE_COMMIT,
                    "external_source_file_sha256": dict(
                        SOURCE_FILE_SHA256
                    ),
                }
            ),
            encoding="utf-8",
        )
        (root / "provenance/tiled_graphs.json").write_text(
            json.dumps(graph_identity), encoding="utf-8"
        )
        (root / "provenance/fixed_evaluation_masks.json").write_text(
            json.dumps(
                {
                    "masks": mask_identities,
                    "common_mask_rate": comparison.COMMON_MASK_RATE,
                    "native_mask_rate": comparison.NATIVE_MASK_RATE,
                    "replicates_per_rate": len(comparison.REPLICATES),
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "run_id": run_id,
                "campaign_id": comparison.CAMPAIGN_ID,
                "status": "completed",
                "seed": seed,
                "attempt": 1,
                "retry_of": None,
                "artifact_path": root.as_posix(),
                "parameter_count": comparison.EXPECTED_GENEMAE_PARAMETERS,
                "duration_seconds": 123.0,
                "peak_vram_gb": 8.0,
                "failure_category": None,
                "config": config,
                "created_at": f"2026-07-30T00:00:0{seed}Z",
            }
        )
        catalog[run_id] = {
            "path": checkpoint.as_posix(),
            "sha256": checkpoint_sha,
            "verification_status": "verified",
            "artifact_status": "present",
            "role": "last",
            "best_epoch": comparison.EXPECTED_GENEMAE_FINAL_EPOCH,
        }
    pilot_statuses = {
        "r-pilot-failed": "failed",
        "r-pilot-pruned": "pruned",
        "r-pilot-cancelled": "cancelled",
        "r-pilot-completed": "completed",
    }
    for index, (run_id, status) in enumerate(pilot_statuses.items()):
        rows.append(
            {
                "run_id": run_id,
                "campaign_id": comparison.CAMPAIGN_ID,
                "status": status,
                "seed": 0,
                "attempt": 1,
                "retry_of": None,
                "artifact_path": None,
                "parameter_count": None,
                "duration_seconds": None,
                "peak_vram_gb": None,
                "failure_category": (
                    None if status == "completed" else "resource_failure"
                ),
                "config": {
                    "seed": 0,
                    "campaign": {
                        "campaign_id": comparison.CAMPAIGN_ID
                    },
                    "model": {"name": comparison.GENEMAE},
                    "metadata": {"execution_role": "resource_pilot"},
                },
                "created_at": f"2026-07-29T00:00:0{index}Z",
            }
        )
    monkeypatch.setattr(
        comparison, "_registry_campaign_runs", lambda registry, campaign: rows
    )
    registry = _AuditRegistry(catalog)

    evidence, inventory, failures, pilots = (
        comparison.discover_registered_genemae_production(
            registry=registry,  # type: ignore[arg-type]
            paths=paths,
            bundle_verifier=lambda root: {
                "valid": True,
                "status": "success",
            },
            checkpoint_loader=torch.load,
        )
    )

    assert [member.seed for member in evidence] == list(comparison.SEEDS)
    assert len(inventory) == 11
    assert {
        row["run_id"] for row in inventory if row["stage"] == "production"
    } == {f"r-schema-{seed}" for seed in comparison.SEEDS}
    assert {
        row["run_id"] for row in inventory if row["stage"] == "pilot"
    } == set(pilot_statuses)
    assert {row["run_id"] for row in pilots} == set(pilot_statuses)
    assert {row["run_id"] for row in failures} == {
        "r-pilot-failed",
        "r-pilot-pruned",
        "r-pilot-cancelled",
    }
    assert sum(
        row["selected_completed_attempt"]
        for row in inventory
        if row["stage"] == "production"
    ) == 7
    assert not any(
        row["selected_completed_attempt"] for row in pilots
    )
    safe_fields = {
        "stage",
        "model_key",
        "seed",
        "run_id",
        "attempt",
        "status",
        "retry_of",
        "failure_category",
        "selected_completed_attempt",
    }
    assert all(set(row) == safe_fields for row in inventory)
    comparison._assert_alias_safe_payload(
        {
            "attempt_inventory": inventory,
            "failure_inventory": failures,
            "pilot_inventory": pilots,
        },
        label="test inventories",
    )


def test_builtin_provider_combines_genemae_and_bagm_pilot_inventories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = SyntheticProvider(tmp_path / "members")
    members = synthetic.audit().members
    gene_failed = {
        "stage": "pilot",
        "model_key": comparison.GENEMAE,
        "seed": 0,
        "run_id": "r-gene-pilot-failed",
        "attempt": 1,
        "status": "failed",
        "retry_of": None,
        "failure_category": "resource_failure",
        "selected_completed_attempt": False,
    }
    gene_completed = {
        **gene_failed,
        "run_id": "r-gene-pilot-completed",
        "status": "completed",
        "failure_category": None,
    }
    bagm_pilot = {
        **gene_failed,
        "model_key": comparison.BAGM_GAT,
        "run_id": "r-bagm-pilot-failed",
    }
    provider = builtin_provider.RegisteredCheckpointComparisonProvider(
        paths=current_paths(),
        database_path=tmp_path / "registry.sqlite3",
        device_name="cpu",
    )
    monkeypatch.setattr(
        builtin_provider,
        "discover_registered_genemae_production",
        lambda **kwargs: (
            members[comparison.GENEMAE],
            (gene_failed, gene_completed),
            (gene_failed,),
            (gene_failed, gene_completed),
        ),
    )
    monkeypatch.setattr(
        provider,
        "_audit_current_bagm",
        lambda: (
            {
                comparison.BAGM_GAT: members[comparison.BAGM_GAT],
                comparison.BAGM_SELF: members[comparison.BAGM_SELF],
            },
            (bagm_pilot,),
            (bagm_pilot,),
            (bagm_pilot,),
        ),
    )

    audit = provider.audit()

    assert {row["run_id"] for row in audit.pilot_inventory} == {
        "r-gene-pilot-failed",
        "r-gene-pilot-completed",
        "r-bagm-pilot-failed",
    }
    assert {row["run_id"] for row in audit.failure_inventory} == {
        "r-gene-pilot-failed",
        "r-bagm-pilot-failed",
    }
    assert {row["run_id"] for row in audit.attempt_inventory} == {
        "r-gene-pilot-failed",
        "r-gene-pilot-completed",
        "r-bagm-pilot-failed",
    }
    assert all(
        row["failed_attempts_before_completion"] == 0
        for row in comparison._resource_rows(audit)
    )


def test_builtin_provider_smoke_combines_all_replay_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = SyntheticProvider(tmp_path / "members")
    expected_batches = list(synthetic.iter_evaluation_batches())
    provider = builtin_provider.RegisteredCheckpointComparisonProvider(
        paths=current_paths(),
        database_path=tmp_path / "registry.sqlite3",
        device_name="cpu",
    )
    monkeypatch.setattr(provider, "audit", synthetic.audit)
    monkeypatch.setattr(provider, "_load_shared_cohort", lambda: object())
    monkeypatch.setattr(
        provider,
        "_iter_genemae",
        lambda cohort: (
            batch
            for batch in expected_batches
            if batch.model_key == comparison.GENEMAE
        ),
    )
    monkeypatch.setattr(
        provider,
        "_iter_bagm",
        lambda cohort: (
            batch
            for batch in expected_batches
            if batch.model_key
            in {comparison.BAGM_GAT, comparison.BAGM_SELF}
        ),
    )

    observed = list(provider.iter_evaluation_batches())

    assert len(observed) == 180
    assert sum(
        batch.model_key == comparison.GENEMAE for batch in observed
    ) == 120
    assert sum(
        batch.model_key == comparison.BAGM_GAT for batch in observed
    ) == 30
    assert sum(
        batch.model_key == comparison.BAGM_SELF for batch in observed
    ) == 30
    assert all(batch.ensemble_rule_verified for batch in observed)
    common = [
        batch
        for batch in observed
        if batch.core_alias == "ANC-01"
        and batch.mask_rate == comparison.COMMON_MASK_RATE
        and batch.replicate == 0
    ]
    assert len({batch.expected_mask_checksum for batch in common}) == 1


def test_builtin_bagm_oracle_scale_conversion() -> None:
    reconstructed = np.asarray([[1.0, 3.0], [0.0, 4.0]], dtype=np.float32)
    library = np.asarray([[4.0], [8.0]], dtype=np.float32)

    observed = builtin_provider._raw_count_to_oracle_log_cp10k(
        reconstructed, library
    )

    expected = np.log1p(
        reconstructed / library * np.float32(10_000.0)
    )
    assert np.allclose(observed, expected)


def test_builtin_bagm_graph_gain_uses_mean_of_core_relative_gains() -> None:
    rows = []
    for index, alias in enumerate(comparison.ALIASES):
        self_loss = 1.0 if index < 5 else 10.0
        gain = 0.10 if index < 5 else 0.02
        gat_loss = self_loss * (1.0 - gain)
        rows.append(
            {
                "core_alias": alias,
                "gat_hybrid_loss": gat_loss,
                "matched_self_hybrid_loss": self_loss,
                "gat_relative_hybrid_loss_improvement": gain,
            }
        )

    observed = builtin_provider._mean_core_relative_graph_improvement(rows)
    ratio_of_equal_core_means = (
        np.mean([row["matched_self_hybrid_loss"] for row in rows])
        - np.mean([row["gat_hybrid_loss"] for row in rows])
    ) / np.mean([row["matched_self_hybrid_loss"] for row in rows])

    assert observed == pytest.approx(0.06)
    assert ratio_of_equal_core_means != pytest.approx(observed)
