"""Built-in checkpoint provider for the final GeneMAE/BAGM comparison.

This adapter owns model-specific reconstruction.  It reuses the current pooled
BAGM campaign's checksum-bound audit and canonical hybrid-head ensemble, and
the source-fidelity GeneMAE runner's graph, mask, and checkpoint helpers.  It
never persists row-level predictions; one core is evaluated and released at a
time.
"""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
import platform
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from .hybrid_count_metrics import (
    evaluate_hybrid_count_output,
    fit_hybrid_count_references,
)
from .identifiers import canonical_sha256
from .myjju_genemae import log1p_cp10k
from .myjju_genemae_comparison import (
    ALIASES,
    BAGM_ENSEMBLE_RULE,
    BAGM_GAT,
    BAGM_SELF,
    COMMON_MASK_RATE,
    ComparisonAudit,
    EvaluationBatch,
    EXPECTED_CURRENT_BAGM_CONTEXT,
    GENEMAE,
    GENEMAE_ENSEMBLE_RULE,
    GeneMAEComparisonError,
    NATIVE_MASK_RATE,
    RegisteredRunEvidence,
    SEEDS,
    TARGET_SCALE,
    discover_registered_genemae_production,
    masked_regression_metrics,
    sha256_file,
)
from .paths import ProjectPaths
from .registry import Registry
from .run_archive import verify_run_bundle


CURRENT_LOCKED_CAMPAIGN = (
    "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
)
CURRENT_LOCKED_FILES = {
    "materialization": "locked_config_materialization.json",
    "pilot_enqueue": "pilot_enqueue_receipt.json",
    "pilot_gate": "pilot_gate_receipt.json",
    "production_enqueue": "production_enqueue_receipt.json",
}


def _load_pooled_module() -> Any:
    return importlib.import_module(
        "scripts.analysis.compare_pooled_hybrid_ensemble"
    )


def _load_genemae_runner() -> Any:
    return importlib.import_module("scripts.train.run_myjju_genemae_pooled")


def _bagm_model_key(arm: str, pooled: Any) -> str:
    if arm == pooled.GAT_ARM:
        return BAGM_GAT
    if arm == pooled.SELF_ARM:
        return BAGM_SELF
    raise GeneMAEComparisonError(f"unexpected current BAGM arm {arm!r}")


def _normalise_attempt_row(row: Mapping[str, Any], pooled: Any) -> dict[str, Any]:
    result = dict(row)
    arm = result.pop("arm", None)
    if arm is not None:
        result["model_key"] = _bagm_model_key(str(arm), pooled)
    return result


def _raw_count_to_oracle_log_cp10k(
    reconstructed_count: torch.Tensor | np.ndarray,
    true_library_size: np.ndarray,
) -> np.ndarray:
    prediction = (
        reconstructed_count.detach().cpu().numpy()
        if isinstance(reconstructed_count, torch.Tensor)
        else np.asarray(reconstructed_count)
    )
    values = np.asarray(prediction, dtype=np.float32)
    library = np.asarray(true_library_size, dtype=np.float32)
    if (
        values.ndim != 2
        or library.shape != (values.shape[0], 1)
        or not np.isfinite(values).all()
        or not np.isfinite(library).all()
        or np.any(values < 0)
        or np.any(library < 0)
    ):
        raise GeneMAEComparisonError(
            "decoded BAGM count or true library size is invalid"
        )
    safe_library = np.where(library > 0, library, np.float32(1.0))
    return np.log1p(
        values / safe_library * np.float32(10_000.0)
    ).astype(np.float32, copy=False)


def _null_seed_summary(tiles: Iterable[Any]) -> int:
    seeds = [int(tile.graph_null_seed) for tile in tiles]
    if not seeds:
        raise GeneMAEComparisonError("GeneMAE core has no graph-null seeds")
    return int(canonical_sha256(seeds)[:15], 16)


def _verify_null_tiles(tiles: Iterable[Any]) -> None:
    for tile in tiles:
        observed = np.asarray(tile.edge_index)
        permuted = np.asarray(tile.permuted_edge_index)
        if observed.shape != permuted.shape or observed.shape[0] != 2:
            raise GeneMAEComparisonError(
                "GeneMAE node-label null changed graph dimensions"
            )
        node_count = int(tile.n_nodes)
        observed_out = np.bincount(observed[0], minlength=node_count)
        observed_in = np.bincount(observed[1], minlength=node_count)
        permuted_out = np.bincount(permuted[0], minlength=node_count)
        permuted_in = np.bincount(permuted[1], minlength=node_count)
        if not (
            np.array_equal(np.sort(observed_out), np.sort(permuted_out))
            and np.array_equal(np.sort(observed_in), np.sort(permuted_in))
            and observed.shape[1] == permuted.shape[1]
        ):
            raise GeneMAEComparisonError(
                "GeneMAE node-label null did not preserve topology/degree"
            )


def _mean_core_relative_graph_improvement(rows: Any) -> float:
    """Recompute the current BAGM graph gain using its frozen aggregation rule."""

    if not isinstance(rows, list):
        raise GeneMAEComparisonError(
            "current BAGM graph-core comparison is malformed"
        )
    by_alias: dict[str, float] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise GeneMAEComparisonError(
                "current BAGM graph-core comparison is malformed"
            )
        alias = str(raw.get("core_alias", ""))
        if alias not in ALIASES or alias in by_alias:
            raise GeneMAEComparisonError(
                "current BAGM graph-core comparison lacks unique alias coverage"
            )
        try:
            gat_loss = float(raw["gat_hybrid_loss"])
            self_loss = float(raw["matched_self_hybrid_loss"])
            declared_gain = float(
                raw["gat_relative_hybrid_loss_improvement"]
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise GeneMAEComparisonError(
                f"current BAGM graph-core result is incomplete for {alias}"
            ) from exc
        if (
            not np.isfinite(gat_loss)
            or not np.isfinite(self_loss)
            or not np.isfinite(declared_gain)
            or self_loss <= 0
        ):
            raise GeneMAEComparisonError(
                f"current BAGM graph-core result is invalid for {alias}"
            )
        recomputed_gain = (self_loss - gat_loss) / self_loss
        if not np.isclose(
            declared_gain,
            recomputed_gain,
            rtol=0.0,
            atol=1e-12,
        ):
            raise GeneMAEComparisonError(
                f"current BAGM graph-core gain changed for {alias}"
            )
        by_alias[alias] = recomputed_gain
    if set(by_alias) != set(ALIASES):
        raise GeneMAEComparisonError(
            "current BAGM graph-core comparison lacks exact ten-core coverage"
        )
    return float(
        np.mean([by_alias[alias] for alias in ALIASES], dtype=np.float64)
    )


class RegisteredCheckpointComparisonProvider:
    """Working built-in provider for all 21 final production checkpoints."""

    def __init__(
        self,
        *,
        paths: ProjectPaths,
        database_path: Path,
        device_name: str,
    ) -> None:
        self.paths = paths
        self.database_path = database_path.resolve(strict=False)
        self.device_name = str(device_name)
        self._registry: Registry | None = None
        self._audit: ComparisonAudit | None = None
        self._gene_evidence: tuple[RegisteredRunEvidence, ...] = ()
        self._bagm_evidence: dict[tuple[str, int], Any] = {}
        self._bagm_materialization: Mapping[str, Any] | None = None
        self._bagm_pilot: Mapping[str, Any] | None = None
        self._input_sha256: dict[str, str] = {}
        self._current_bagm_context: dict[str, Any] = {}
        self._source_historical_context: dict[str, Any] = {}

    @property
    def registry(self) -> Registry:
        if self._registry is None:
            self._registry = Registry(self.database_path)
        return self._registry

    def _current_receipt_paths(self) -> dict[str, Path]:
        root = (
            self.paths.scratch_root
            / "locked_campaigns"
            / CURRENT_LOCKED_CAMPAIGN
        )
        result = {
            key: root / filename
            for key, filename in CURRENT_LOCKED_FILES.items()
        }
        if any(not path.is_file() for path in result.values()):
            raise GeneMAEComparisonError(
                "current BAGM locked campaign receipts are incomplete"
            )
        return result

    def _audit_current_bagm(
        self,
    ) -> tuple[
        Mapping[str, tuple[RegisteredRunEvidence, ...]],
        tuple[Mapping[str, Any], ...],
        tuple[Mapping[str, Any], ...],
        tuple[Mapping[str, Any], ...],
    ]:
        pooled = _load_pooled_module()
        receipts = self._current_receipt_paths()
        materialization, pilot_enqueue, pilot, enqueue = (
            pooled.validate_campaign_receipts(
                paths=self.paths,
                materialization_path=receipts["materialization"],
                pilot_enqueue_path=receipts["pilot_enqueue"],
                pilot_gate_path=receipts["pilot_gate"],
                production_enqueue_path=receipts["production_enqueue"],
            )
        )
        if self.registry.get_campaign(pooled.CAMPAIGN_ID) is None:
            raise GeneMAEComparisonError(
                "current pooled BAGM campaign is absent from the registry"
            )
        queue_rows = pooled._campaign_queue_rows(self.registry)
        lineage = pooled.resolve_production_lineages(
            queue_rows=queue_rows,
            materialization=materialization,
            enqueue=enqueue,
            run_lookup=self.registry.get_run,
        )
        pilot_attempts, pilot_failures = pooled.resolve_pilot_lineages(
            queue_rows=queue_rows,
            materialization=materialization,
            pilot_enqueue=pilot_enqueue,
            pilot_gate=pilot,
            run_lookup=self.registry.get_run,
        )
        pooled_audit = pooled.CampaignAudit(
            selected_jobs=lineage.selected_jobs,
            attempt_inventory=lineage.attempt_inventory,
            registered_failure_inventory=(
                lineage.registered_failure_inventory
            ),
            pilot_attempt_inventory=pilot_attempts,
            pilot_failure_inventory=pilot_failures,
        )
        pooled.validate_registered_production_membership(
            run_rows=pooled._campaign_run_rows(self.registry),
            audit=pooled_audit,
        )
        planned = {
            (str(job["arm"]), int(job["seed"])): job
            for job in materialization["production_jobs"]
        }
        failed_by_slot = {slot: 0 for slot in planned}
        for row in pooled_audit.registered_failure_inventory:
            failed_by_slot[(str(row["arm"]), int(row["seed"]))] += 1
        evidence: dict[tuple[str, int], Any] = {}
        converted: dict[str, list[RegisteredRunEvidence]] = {
            BAGM_GAT: [],
            BAGM_SELF: [],
        }
        for slot in sorted(planned):
            item = pooled.audit_production_run(
                registry=self.registry,
                completed_job=pooled_audit.selected_jobs[slot],
                slot=slot,
                planned_job=planned[slot],
                materialization=materialization,
                failed_attempt_count=failed_by_slot[slot],
                paths=self.paths,
                bundle_verifier=verify_run_bundle,
                checkpoint_loader=torch.load,
            )
            evidence[slot] = item
            model_key = _bagm_model_key(item.arm, pooled)
            converted[model_key].append(
                RegisteredRunEvidence(
                    model_key=model_key,
                    seed=item.seed,
                    run_id=item.run_id,
                    attempt=item.attempt,
                    artifact_root=item.root,
                    checkpoint_path=item.checkpoint_path,
                    checkpoint_sha256=item.checkpoint_file_sha256,
                    state_dict_sha256=item.state_dict_sha256,
                    config_sha256=pooled._normalized_root_config_sha256(
                        item.config
                    ),
                    bundle_verified=True,
                    registry_artifacts_verified=True,
                    checkpoint_catalog_verified=True,
                    parameter_count=pooled.EXPECTED_PARAMETER_COUNT,
                    completed_epochs=pooled.EXPECTED_GLOBAL_EPOCHS,
                    final_epoch=pooled.EXPECTED_CHECKPOINT_EPOCH,
                    duration_seconds=item.duration_seconds,
                    peak_vram_gib=item.peak_vram_gib,
                    peak_host_memory_gib=(
                        item.peak_host_memory_bytes / 1024**3
                    ),
                    convergence=item.convergence,
                    resources=item.resource_usage,
                )
            )
        pooled.verify_paired_member_initialization(evidence)
        if len(evidence) != 14:
            raise GeneMAEComparisonError(
                "current BAGM audit did not resolve fourteen members"
            )
        self._bagm_evidence = evidence
        self._bagm_materialization = materialization
        self._bagm_pilot = pilot
        self._input_sha256.update(
            {
                f"current_bagm_{name}": sha256_file(path)
                for name, path in receipts.items()
            }
        )
        report_root = (
            self.paths.report_root
            / "analyses"
            / "adjacent_normal_10core_pooled_hybrid_ensemble"
            / "comparison"
        )
        report_manifest_path = report_root / "manifest.json"
        report_comparison_path = report_root / "comparison.json"
        report_equal_path = report_root / "ensemble_equal_core_aggregates.json"
        report_graph_core_path = report_root / "graph_core_comparison.json"
        if any(
            not path.is_file()
            for path in (
                report_manifest_path,
                report_comparison_path,
                report_equal_path,
                report_graph_core_path,
            )
        ):
            raise GeneMAEComparisonError(
                "immutable current BAGM comparison context is incomplete"
            )
        report_manifest = json.loads(
            report_manifest_path.read_text(encoding="utf-8")
        )
        report_files = report_manifest.get("files")
        if not isinstance(report_files, Mapping):
            raise GeneMAEComparisonError(
                "current BAGM report manifest is malformed"
            )
        for path in (
            report_comparison_path,
            report_equal_path,
            report_graph_core_path,
        ):
            record = report_files.get(path.name)
            if (
                not isinstance(record, Mapping)
                or record.get("sha256") != sha256_file(path)
            ):
                raise GeneMAEComparisonError(
                    f"current BAGM report checksum changed for {path.name}"
                )
        report_comparison = json.loads(
            report_comparison_path.read_text(encoding="utf-8")
        )
        equal_rows = json.loads(report_equal_path.read_text(encoding="utf-8"))
        graph_core_rows = json.loads(
            report_graph_core_path.read_text(encoding="utf-8")
        )
        if not isinstance(equal_rows, list):
            raise GeneMAEComparisonError(
                "current BAGM equal-core context is malformed"
            )
        whole = {
            str(row.get("arm")): row
            for row in equal_rows
            if isinstance(row, Mapping)
            and row.get("mask_mode") == "whole_node"
        }
        graph_gate = (
            report_comparison.get("frozen_gates", {}).get("graph_gate", {})
            if isinstance(report_comparison, Mapping)
            else {}
        )
        try:
            gat_loss = float(whole[pooled.GAT_ARM]["hybrid_loss"])
            self_loss = float(whole[pooled.SELF_ARM]["hybrid_loss"])
            graph_gain = float(
                graph_gate["mean_relative_hybrid_loss_improvement"]
            )
            recomputed_graph_gain = (
                _mean_core_relative_graph_improvement(graph_core_rows)
            )
            core_count = int(graph_gate["favoring_core_count"])
            seed_count = int(graph_gate["favoring_seed_pair_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GeneMAEComparisonError(
                "current BAGM whole-node context is incomplete"
            ) from exc
        if (
            report_comparison.get("outcome") != "negative"
            or set(report_comparison.get("failed_gates", ()))
            != {"pooled_data_gate", "representation_gate"}
            or graph_gate.get("passed") is not True
            or not np.isclose(
                graph_gain,
                recomputed_graph_gain,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                graph_gain,
                EXPECTED_CURRENT_BAGM_CONTEXT[
                    "gat_relative_graph_improvement"
                ],
                rtol=0.0,
                atol=1e-12,
            )
            or core_count != 10
            or seed_count != 7
        ):
            raise GeneMAEComparisonError(
                "current BAGM contextual result changed"
            )
        self._current_bagm_context = {
            "estimand": "held_in_whole_node_hybrid_count_reconstruction",
            "context_only_not_ranked_against_genemae": True,
            "gat_equal_core_hybrid_loss": gat_loss,
            "matched_self_equal_core_hybrid_loss": self_loss,
            "gat_relative_graph_improvement": graph_gain,
            "gat_favoring_core_count": core_count,
            "gat_favoring_seed_pair_count": seed_count,
            "graph_gate_passed": True,
            "overall_campaign_outcome": "negative",
            "failed_gates": ["pooled_data_gate", "representation_gate"],
            "source_comparison_sha256": sha256_file(
                report_comparison_path
            ),
            "source_equal_core_table_sha256": sha256_file(
                report_equal_path
            ),
            "source_graph_core_table_sha256": sha256_file(
                report_graph_core_path
            ),
            "source_manifest_sha256": sha256_file(report_manifest_path),
        }
        self._input_sha256.update(
            {
                "current_bagm_comparison": sha256_file(
                    report_comparison_path
                ),
                "current_bagm_equal_core_context": sha256_file(
                    report_equal_path
                ),
                "current_bagm_graph_core_context": sha256_file(
                    report_graph_core_path
                ),
                "current_bagm_report_manifest": sha256_file(
                    report_manifest_path
                ),
            }
        )
        attempt_rows = tuple(
            _normalise_attempt_row(row, pooled)
            for row in (
                *pooled_audit.pilot_attempt_inventory,
                *pooled_audit.attempt_inventory,
            )
        )
        failure_rows = tuple(
            _normalise_attempt_row(row, pooled)
            for row in (
                *pooled_audit.pilot_failure_inventory,
                *pooled_audit.registered_failure_inventory,
            )
        )
        pilot_rows = tuple(
            _normalise_attempt_row(row, pooled)
            for row in pooled_audit.pilot_attempt_inventory
        )
        return (
            {
                BAGM_GAT: tuple(
                    sorted(converted[BAGM_GAT], key=lambda item: item.seed)
                ),
                BAGM_SELF: tuple(
                    sorted(converted[BAGM_SELF], key=lambda item: item.seed)
                ),
            },
            attempt_rows,
            failure_rows,
            pilot_rows,
        )

    def audit(self) -> ComparisonAudit:
        if self._audit is not None:
            return self._audit
        gene, gene_attempts, gene_failures, gene_pilot_rows = (
            discover_registered_genemae_production(
                registry=self.registry,
                paths=self.paths,
                bundle_verifier=verify_run_bundle,
                checkpoint_loader=torch.load,
            )
        )
        bagm, bagm_attempts, bagm_failures, pilot_rows = (
            self._audit_current_bagm()
        )
        self._gene_evidence = gene
        contract = (
            self.paths.project_root
            / "experiments"
            / "campaigns"
            / "cmp_20260730_myjju_genemae_10core_comparison"
            / "frozen_task_contract.yaml"
        )
        source_audit = contract.with_name("external_source_audit.yaml")
        self._input_sha256.update(
            {
                "genemae_frozen_contract": sha256_file(contract),
                "genemae_external_source_audit": sha256_file(source_audit),
            }
        )
        try:
            source_document = yaml.safe_load(
                source_audit.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise GeneMAEComparisonError(
                "frozen external source audit is unreadable"
            ) from exc
        if not isinstance(source_document, Mapping):
            raise GeneMAEComparisonError(
                "frozen external source audit is malformed"
            )
        reported = source_document.get("reported_context")
        limitations = source_document.get("known_provenance_limitations")
        if (
            not isinstance(reported, Mapping)
            or not isinstance(limitations, list)
            or source_document.get("checkpoint_inventory", {}).get(
                "local_checkpoint_count"
            )
            != 0
        ):
            raise GeneMAEComparisonError(
                "frozen external historical context is incomplete"
            )
        self._source_historical_context = {
            "context_only_not_comparable": True,
            "historical_weights_available": False,
            "metrics": {
                "so1_sb50": dict(reported["so1_sb50"]),
                "so2_sb50": dict(reported["so2_sb50"]),
            },
            "checkpoint_selection_leakage": True,
            "selection_leakage_basis": [
                item
                for item in limitations
                if item
                in {
                    "held_out_donor_was_used_for_early_stopping_and_scoring",
                    "lodo_test_donor_was_used_for_checkpoint_selection_and_scoring",
                }
            ],
            "source_audit_sha256": sha256_file(source_audit),
        }
        self._audit = ComparisonAudit(
            members={GENEMAE: gene, **bagm},
            attempt_inventory=tuple(gene_attempts) + tuple(bagm_attempts),
            failure_inventory=tuple(gene_failures) + tuple(bagm_failures),
            pilot_inventory=tuple(gene_pilot_rows) + tuple(pilot_rows),
            provenance={
                "gene_campaign_discovery": "authoritative_registry",
                "current_bagm_discovery": (
                    "signed_materialization_enqueue_and_registry_lineages"
                ),
                "bundle_verifier": "spatial_benchmark.run_archive.verify_run_bundle",
                "checkpoint_catalog_required": True,
                "selected_best_seed": False,
                "current_bagm_whole_node_context": dict(
                    self._current_bagm_context
                ),
                "source_report_historical_context": dict(
                    self._source_historical_context
                ),
            },
        )
        return self._audit

    def _gene_config(self) -> Mapping[str, Any]:
        if not self._gene_evidence:
            self.audit()
        runner = _load_genemae_runner()
        configs = [
            runner.load_yaml_mapping(member.artifact_root / "config.resolved.yaml")
            for member in self._gene_evidence
        ]
        common_sections = ("dataset", "graph", "evaluation", "model")
        identity = canonical_sha256(
            {name: configs[0].get(name) for name in common_sections}
        )
        if any(
            canonical_sha256(
                {name: config.get(name) for name in common_sections}
            )
            != identity
            for config in configs[1:]
        ):
            raise GeneMAEComparisonError(
                "GeneMAE production members do not share data/graph/mask/model"
            )
        return configs[0]

    def _load_shared_cohort(self) -> Any:
        runner = _load_genemae_runner()
        config = self._gene_config()
        cohort = runner.load_verified_ten_core_cohort(config)
        if (
            tuple(cohort.aliases) != ALIASES
            or cohort.n_genes != 1000
            or cohort.total_nodes != 117_386
        ):
            raise GeneMAEComparisonError(
                "GeneMAE provider loaded an unexpected ten-core cohort"
            )
        materialization = self._bagm_materialization
        if materialization is None:
            raise GeneMAEComparisonError("current BAGM audit was not initialized")
        bagm_cohort = materialization.get("cohort")
        if not isinstance(bagm_cohort, Mapping):
            raise GeneMAEComparisonError(
                "current BAGM materialization has no cohort identity"
            )
        if (
            cohort.fingerprint_sha256 != bagm_cohort.get("dataset_fingerprint")
            or cohort.checksums.to_dict()
            != dict(bagm_cohort.get("cohort_checksums", {}))
        ):
            raise GeneMAEComparisonError(
                "GeneMAE and current BAGM cohorts are not identical"
            )
        return cohort

    def _iter_genemae(self, cohort: Any) -> Iterable[EvaluationBatch]:
        runner = _load_genemae_runner()
        config = self._gene_config()
        graph_config = config.get("graph")
        evaluation = config.get("evaluation")
        if not isinstance(graph_config, Mapping) or not isinstance(
            evaluation, Mapping
        ):
            raise GeneMAEComparisonError(
                "GeneMAE config lacks graph or evaluation sections"
            )
        tiled = runner.prepare_source_tiles(
            cohort,
            k=int(graph_config["neighbor_k"]),
            max_nodes=int(graph_config["maximum_tile_nodes"]),
        )
        graph_identity = tiled.identity()
        if graph_identity != graph_config.get("expected_tiled_graphs"):
            raise GeneMAEComparisonError(
                "GeneMAE tiled graphs differ from production"
            )
        sources = evaluation.get("prior_mask_sources")
        if not isinstance(sources, Mapping) or tuple(sources) != ALIASES:
            raise GeneMAEComparisonError(
                "GeneMAE common/native mask sources are incomplete"
            )
        gene_checksum = cohort.checksums.ordered_gene_schema_sha256
        payload_metadata: dict[int, Mapping[str, Any]] = {}
        evidence_by_seed = {
            member.seed: member for member in self._gene_evidence
        }
        if set(evidence_by_seed) != set(SEEDS):
            raise GeneMAEComparisonError(
                "GeneMAE replay lacks exact seed coverage"
            )
        for member in self._gene_evidence:
            payload = torch.load(
                member.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if (
                payload.get("cohort_fingerprint_sha256")
                != cohort.fingerprint_sha256
                or payload.get("cohort_checksums") != cohort.checksums.to_dict()
                or payload.get("ordered_gene_schema_sha256") != gene_checksum
                or payload.get("graph_bundle_sha256")
                != graph_identity["graph_bundle_sha256"]
            ):
                raise GeneMAEComparisonError(
                    f"GeneMAE seed {member.seed} checkpoint data identity changed"
                )
            payload_metadata[member.seed] = {
                "evaluation_mask_identities": payload.get(
                    "evaluation_mask_identities"
                ),
                "per_core_graph_bundle_sha256": payload.get(
                    "per_core_graph_bundle_sha256"
                ),
            }
            del payload

        for core in cohort.cores:
            configured = sources[core.alias]
            if not isinstance(configured, Mapping):
                raise GeneMAEComparisonError(
                    f"{core.alias} GeneMAE mask identity is malformed"
                )
            common = configured.get("common")
            native = configured.get("native")
            if not isinstance(common, Mapping) or not isinstance(native, Mapping):
                raise GeneMAEComparisonError(
                    f"{core.alias} GeneMAE mask sources are incomplete"
                )
            masks = runner.regenerate_evaluation_masks(
                core,
                comparator_source=common,
                native_base_seed=int(native["base_seed"]),
            )
            if masks.identity() != configured:
                raise GeneMAEComparisonError(
                    f"{core.alias} GeneMAE masks differ from production"
                )
            for seed in SEEDS:
                checkpoint_masks = payload_metadata[seed].get(
                    "evaluation_mask_identities"
                )
                if (
                    not isinstance(checkpoint_masks, Mapping)
                    or checkpoint_masks.get(core.alias) != masks.identity()
                    or not isinstance(
                        payload_metadata[seed].get(
                            "per_core_graph_bundle_sha256"
                        ),
                        Mapping,
                    )
                    or payload_metadata[seed][
                        "per_core_graph_bundle_sha256"
                    ].get(core.alias)
                    != graph_identity["cores"][core.alias][
                        "graph_bundle_sha256"
                    ]
                ):
                    raise GeneMAEComparisonError(
                        f"GeneMAE seed {seed} mask identity changed for {core.alias}"
                    )
            core_tiles = tiled.for_alias(core.alias)
            _verify_null_tiles(core_tiles)
            null_seed = _null_seed_summary(core_tiles)
            target = np.asarray(
                log1p_cp10k(core.expression_counts), dtype=np.float32
            )
            sums: dict[tuple[float, int, str], np.ndarray] = {}
            member_metrics: dict[
                tuple[float, int, str], dict[int, Mapping[str, Any]]
            ] = {}
            entries: dict[tuple[float, int], Any] = {
                (float(entry.mask_rate), int(entry.replicate)): entry
                for entry in masks.all_masks
            }
            if set(entries) != {
                (rate, replicate)
                for rate in (COMMON_MASK_RATE, NATIVE_MASK_RATE)
                for replicate in range(3)
            }:
                raise GeneMAEComparisonError(
                    f"{core.alias} GeneMAE masks lack exact 20/50 coverage"
                )
            for seed in SEEDS:
                model = runner.reconstruct_model_from_checkpoint(
                    evidence_by_seed[seed].checkpoint_path,
                    device=self.device_name,
                )
                for (rate, replicate), entry in sorted(entries.items()):
                    for condition in ("observed", "node_label_permuted"):
                        identity = (rate, replicate, condition)
                        prediction = runner.predict_core_mask(
                            model,
                            core,
                            core_tiles,
                            entry.mask,
                            device=self.device_name,
                            graph_condition=condition,
                            normalized_expression=target,
                        )
                        member_metrics.setdefault(identity, {})[seed] = (
                            masked_regression_metrics(
                                target, prediction, entry.mask
                            )
                        )
                        accumulator = sums.get(identity)
                        if accumulator is None:
                            accumulator = np.zeros_like(
                                prediction, dtype=np.float32
                            )
                            sums[identity] = accumulator
                        np.add(
                            accumulator,
                            prediction / np.float32(len(SEEDS)),
                            out=accumulator,
                            casting="unsafe",
                        )
                        del prediction
                model.to("cpu")
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for (rate, replicate, condition), prediction in sorted(
                sums.items()
            ):
                entry = entries[(rate, replicate)]
                yield EvaluationBatch(
                    model_key=GENEMAE,
                    core_alias=core.alias,
                    mask_rate=rate,
                    replicate=replicate,
                    graph_condition=condition,
                    target=target,
                    mask=np.asarray(entry.mask, dtype=np.bool_),
                    ensemble_prediction=prediction,
                    member_metrics=member_metrics[
                        (rate, replicate, condition)
                    ],
                    mask_seed=int(entry.seed),
                    expected_mask_checksum=str(entry.checksum),
                    regenerated_mask_checksum=str(entry.checksum),
                    ordered_gene_sha256=str(gene_checksum),
                    ensemble_rule=GENEMAE_ENSEMBLE_RULE,
                    ensemble_rule_verified=True,
                    prediction_scale=TARGET_SCALE,
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
                        null_seed
                        if condition == "node_label_permuted"
                        else None
                    ),
                )
            del sums, member_metrics, target, masks
            gc.collect()

    def _iter_bagm(self, cohort: Any) -> Iterable[EvaluationBatch]:
        pooled = _load_pooled_module()
        materialization = self._bagm_materialization
        if materialization is None or len(self._bagm_evidence) != 14:
            raise GeneMAEComparisonError(
                "current BAGM production audit was not initialized"
            )
        device = torch.device(self.device_name)
        if device.type == "cuda":
            index = 0 if device.index is None else int(device.index)
            if index not in pooled.SAFE_GPU_IDS:
                raise GeneMAEComparisonError(
                    f"CUDA device {index} is excluded from current BAGM evaluation"
                )
            if not torch.cuda.is_available():
                raise GeneMAEComparisonError(
                    "CUDA BAGM evaluation requested but CUDA is unavailable"
                )
        base_config = self._bagm_evidence[(pooled.GAT_ARM, 0)].config
        gene_checksum = cohort.checksums.ordered_gene_schema_sha256
        for core in cohort.cores:
            alias = core.alias
            graph_config = pooled._mapping(
                base_config.get("graph"), "pooled graph config"
            )
            per_core_graph_config = pooled._graph_config_for_alias(
                graph_config, alias
            )
            graph = pooled._build_graph(core, per_core_graph_config)
            pooled._validate_graph(graph, core, per_core_graph_config)
            expression = pooled.validate_raw_counts(
                core.expression_counts,
                name=f"{alias} expression_counts",
            )
            views = {
                pooled.GAT_ARM: pooled._fit_view(
                    core=core,
                    graph=graph,
                    uses_graph=True,
                    expression=expression,
                ),
                pooled.SELF_ARM: pooled._fit_view(
                    core=core,
                    graph=graph,
                    uses_graph=False,
                    expression=expression,
                ),
            }
            training = pooled._training_config(base_config)
            mask_bundle = pooled._mask_bundle_for_core(
                config=base_config,
                core=core,
                training=training,
            )
            entries = tuple(
                entry
                for entry in mask_bundle.manifest["entries"]
                if entry["spec"]["label"] == "partial_gene"
            )
            if len(entries) != 3:
                raise GeneMAEComparisonError(
                    f"{alias} current BAGM lacks three partial-gene masks"
                )
            expected_masks = pooled._expected_mask_entries(
                materialization, alias
            )
            target = np.asarray(
                log1p_cp10k(core.expression_counts), dtype=np.float32
            )
            true_library = np.asarray(
                core.expression_counts.sum(
                    axis=1, keepdims=True, dtype=np.float32
                ),
                dtype=np.float32,
            )
            per_core_references = fit_hybrid_count_references(
                core.expression_counts,
                expression_mean=cohort.expression_mean,
                expression_scale=cohort.expression_scale,
            )
            for arm in (pooled.GAT_ARM, pooled.SELF_ARM):
                model_key = _bagm_model_key(arm, pooled)
                config = self._bagm_evidence[(arm, 0)].config
                trainer = pooled._mapping(
                    config.get("trainer"), "current BAGM trainer"
                )
                device_view = pooled._to_device_view(
                    views[arm], device=device, dtype=torch.float32
                )
                accumulators = {
                    str(entry["entry_id"]): pooled.HybridCountEnsembleAccumulator(
                        expected_member_count=len(SEEDS),
                        accumulation_device="cpu",
                    )
                    for entry in entries
                }
                targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                per_entry_member_metrics: dict[
                    str, dict[int, Mapping[str, Any]]
                ] = {str(entry["entry_id"]): {} for entry in entries}
                for seed in SEEDS:
                    member = self._bagm_evidence[(arm, seed)]
                    payload = pooled._checkpoint_payload(
                        member.checkpoint_path,
                        checkpoint_loader=torch.load,
                    )
                    _, state = pooled._validate_checkpoint_metadata(
                        payload,
                        arm=arm,
                        seed=seed,
                        materialization=materialization,
                        expected_config_sha256=(
                            pooled._normalized_root_config_sha256(
                                member.config
                            )
                        ),
                    )
                    model, _, parameter_audit = pooled._paired_models(
                        core=core,
                        graph=SimpleNamespace(
                            edge_attribute_names=pooled.EDGE_ATTRIBUTE_NAMES
                        ),
                        model_config=pooled._mapping(
                            member.config.get("model"),
                            "current BAGM model config",
                        ),
                        selected_model_name=pooled.ARM_TO_MODEL[arm],
                        seed=seed,
                    )
                    if (
                        parameter_audit.get(
                            "trainable_parameter_count_graph"
                        )
                        != pooled.EXPECTED_PARAMETER_COUNT
                        or parameter_audit.get(
                            "trainable_parameter_count_self"
                        )
                        != pooled.EXPECTED_PARAMETER_COUNT
                        or parameter_audit.get(
                            "exact_trainable_parameter_match"
                        )
                        is not True
                    ):
                        raise GeneMAEComparisonError(
                            f"current BAGM architecture changed for {(arm, seed)}"
                        )
                    model.load_state_dict(state, strict=True)
                    del payload, state
                    pooled._clear_graph_layout_caches(model)
                    for (
                        entry_id,
                        raw_prediction,
                        raw_target,
                        target_mask,
                    ) in pooled._predict_checkpoint_masks(
                        model=model,
                        device_view=device_view,
                        mask_entries=entries,
                        mask_bundle=mask_bundle,
                        device=device,
                        amp=bool(trainer.get("amp")),
                        amp_dtype=str(
                            trainer.get("amp_dtype", "auto")
                        ).lower(),
                    ):
                        accumulators[entry_id].update(raw_prediction)
                        evaluation = evaluate_hybrid_count_output(
                            raw_prediction,
                            raw_target,
                            target_mask,
                            expression_mean=cohort.expression_mean,
                            expression_scale=cohort.expression_scale,
                            references=per_core_references,
                        )
                        prediction_log = _raw_count_to_oracle_log_cp10k(
                            evaluation.reconstructed_count, true_library
                        )
                        mask_array = target_mask.detach().cpu().numpy()
                        per_entry_member_metrics[entry_id][seed] = (
                            masked_regression_metrics(
                                target, prediction_log, mask_array
                            )
                        )
                        previous = targets.get(entry_id)
                        cpu_target = raw_target.detach().float().cpu()
                        cpu_mask = target_mask.detach().cpu()
                        if previous is None:
                            targets[entry_id] = (cpu_target, cpu_mask)
                        elif not (
                            torch.equal(previous[0], cpu_target)
                            and torch.equal(previous[1], cpu_mask)
                        ):
                            raise GeneMAEComparisonError(
                                f"current BAGM targets changed for {alias}/{entry_id}"
                            )
                        del (
                            evaluation,
                            prediction_log,
                            raw_prediction,
                            raw_target,
                            target_mask,
                            cpu_target,
                            cpu_mask,
                        )
                    model.to("cpu")
                    del model
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                for entry in entries:
                    entry_id = str(entry["entry_id"])
                    replicate = int(entry["replicate"])
                    frozen = expected_masks[("partial_gene", replicate)]
                    if (
                        entry_id != frozen.get("entry_id")
                        or entry.get("mask_checksum")
                        != frozen.get("mask_checksum")
                        or entry.get("seed") != frozen.get("seed")
                    ):
                        raise GeneMAEComparisonError(
                            f"{alias} current BAGM partial mask identity changed"
                        )
                    raw_ensemble = accumulators[entry_id].finalize()
                    raw_target, target_mask = targets[entry_id]
                    evaluation = evaluate_hybrid_count_output(
                        raw_ensemble,
                        raw_target,
                        target_mask,
                        expression_mean=cohort.expression_mean,
                        expression_scale=cohort.expression_scale,
                        references=per_core_references,
                    )
                    prediction_log = _raw_count_to_oracle_log_cp10k(
                        evaluation.reconstructed_count, true_library
                    )
                    yield EvaluationBatch(
                        model_key=model_key,
                        core_alias=alias,
                        mask_rate=COMMON_MASK_RATE,
                        replicate=replicate,
                        graph_condition="observed",
                        target=target,
                        mask=target_mask.detach().cpu().numpy(),
                        ensemble_prediction=prediction_log,
                        member_metrics=per_entry_member_metrics[entry_id],
                        mask_seed=int(entry["seed"]),
                        expected_mask_checksum=str(
                            frozen["mask_checksum"]
                        ),
                        regenerated_mask_checksum=str(
                            entry["mask_checksum"]
                        ),
                        ordered_gene_sha256=str(gene_checksum),
                        ensemble_rule=BAGM_ENSEMBLE_RULE,
                        ensemble_rule_verified=True,
                        prediction_scale=TARGET_SCALE,
                        target_preprocessing_uses_full_cell_library=True,
                        oracle_true_library_size_used=True,
                    )
                    del raw_ensemble, evaluation, prediction_log
                del accumulators, targets, device_view
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del (
                views,
                graph,
                mask_bundle,
                per_core_references,
                target,
                true_library,
                core,
            )
            gc.collect()

    def iter_evaluation_batches(self) -> Iterable[EvaluationBatch]:
        self.audit()
        cohort = self._load_shared_cohort()
        yield from self._iter_genemae(cohort)
        yield from self._iter_bagm(cohort)

    def provenance(self) -> Mapping[str, Any]:
        self.audit()
        return {
            "provider": (
                "spatial_benchmark.myjju_genemae_provider."
                "RegisteredCheckpointComparisonProvider"
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "evaluation_device": self.device_name,
            "input_file_sha256": dict(sorted(self._input_sha256.items())),
            "gene_ensemble_rule": GENEMAE_ENSEMBLE_RULE,
            "bagm_ensemble_rule": BAGM_ENSEMBLE_RULE,
            "bagm_decoding_precedes_oracle_cp10k_transform": True,
            "row_level_predictions_persisted": False,
        }


__all__ = ["RegisteredCheckpointComparisonProvider"]
