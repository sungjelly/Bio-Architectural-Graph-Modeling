from __future__ import annotations

from dataclasses import dataclass
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYZER_PATH = PROJECT_ROOT / "scripts/analysis/analyze_same_gene_robustness.py"
CONTRACT_PATH = (
    PROJECT_ROOT
    / "experiments/campaigns/cmp_20260810_same_gene_robustness_multiverse_v1/frozen_task_contract.yaml"
)
_SPEC = importlib.util.spec_from_file_location(
    "same_gene_robustness_analyzer_tests", ANALYZER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analysis
_SPEC.loader.exec_module(analysis)


@dataclass
class _Slot:
    variant_id: str
    model_seed: int
    fold: int


@dataclass
class _AttemptSlot(_Slot):
    attempt: int


def test_exact_coverage_accepts_only_complete_unique_cartesian_product() -> None:
    entries = [
        _Slot(variant, seed, fold)
        for variant in ("V0", "V1")
        for seed in (10, 11)
        for fold in (0, 1)
    ]
    selected = analysis._validate_coverage(
        entries, variants=("V0", "V1"), seeds=(10, 11), folds=(0, 1)
    )
    assert len(selected) == 8
    with pytest.raises(analysis.RobustnessAnalysisError, match="missing"):
        analysis._validate_coverage(
            entries[:-1], variants=("V0", "V1"), seeds=(10, 11), folds=(0, 1)
        )
    with pytest.raises(analysis.RobustnessAnalysisError, match="duplicate"):
        analysis._validate_coverage(
            entries + [entries[0]],
            variants=("V0", "V1"),
            seeds=(10, 11),
            folds=(0, 1),
        )


def test_coverage_selects_latest_contiguous_declared_attempt() -> None:
    entries = [
        _AttemptSlot("V0", 10, 0, 1),
        _AttemptSlot("V0", 10, 0, 2),
    ]
    selected = analysis._validate_coverage(
        entries, variants=("V0",), seeds=(10,), folds=(0,)
    )
    assert len(selected) == 1 and selected[0].attempt == 2
    with pytest.raises(analysis.RobustnessAnalysisError, match="noncontiguous"):
        analysis._validate_coverage(
            [_AttemptSlot("V0", 10, 0, 1), _AttemptSlot("V0", 10, 0, 3)],
            variants=("V0",),
            seeds=(10,),
            folds=(0,),
        )


def test_registry_inventory_rejects_an_undeclared_full_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = analysis.Registry(tmp_path / "registry.sqlite3")
    registry.create_campaign(
        analysis.CAMPAIGN_ID,
        name="synthetic robustness inventory",
    )
    configuration = {
        "evaluation": {"protocol": "held_out_geometry_masked_reconstruction"},
        "robustness_variant": {"variant_id": "V0"},
    }
    identifier = analysis.scientific_id(configuration)
    registry.register_variant(
        identifier,
        campaign_id=analysis.CAMPAIGN_ID,
        configuration=configuration,
    )
    registry.create_run(
        "run-a1",
        campaign_id=analysis.CAMPAIGN_ID,
        scientific_id=identifier,
        repro_id="repro-a1",
        seed=260810,
        fold=0,
        attempt=1,
        configuration=configuration,
    )
    declared = [_AttemptSlot("V0", 20260810, 0, 1)]
    monkeypatch.setattr(
        analysis,
        "_declared_attempt_authority",
        lambda _row, *, project_root: {
            "slot": ("V0", 260810, 0, 1),
            "scientific_id": identifier,
            "configuration": configuration,
        },
    )
    analysis._verify_registry_attempt_inventory(
        declared,
        registry=registry,
        project_root=tmp_path,
    )

    registry.create_run(
        "run-a2-undeclared",
        campaign_id=analysis.CAMPAIGN_ID,
        scientific_id=identifier,
        repro_id="repro-a2",
        seed=260810,
        fold=0,
        attempt=2,
        configuration=configuration,
    )
    with pytest.raises(analysis.RobustnessAnalysisError, match="absent"):
        analysis._verify_registry_attempt_inventory(
            declared,
            registry=registry,
            project_root=tmp_path,
        )


def test_jacobian_reconstruction_is_exact_and_tamper_is_detected() -> None:
    state = {
        "neighbor_linear.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "neighbor_in.weight": torch.tensor([[0.5, 1.0], [-1.0, 2.0]]),
        "neighbor_out.weight": torch.tensor([[2.0, 0.25], [1.5, -2.0]]),
    }
    hidden = np.asarray([0.2, 0.8], dtype=np.float64)
    parts = analysis._reconstruct_jacobian_parts(state, hidden)
    expected_nonlinear = (
        state["neighbor_out.weight"].double().numpy() * hidden[None, :]
    ) @ state["neighbor_in.weight"].double().numpy()
    assert np.array_equal(parts["linear"], state["neighbor_linear.weight"].double().numpy())
    assert np.allclose(parts["nonlinear"], expected_nonlinear, rtol=0, atol=0)
    assert np.allclose(parts["total"], parts["linear"] + expected_nonlinear, rtol=0, atol=0)
    tampered = parts["total"].copy()
    tampered[0, 0] += 1e-3
    with pytest.raises(analysis.RobustnessAnalysisError, match="reconstruction differs"):
        analysis._assert_close(parts["total"], tampered, label="synthetic")


def test_phase_transform_requires_train_only_roles_and_matching_checkpoint() -> None:
    metadata = {
        "kind": "library",
        "tuning_fit_mask": "tuning_train",
        "final_fit_mask": "final_train",
        "tuning": {
            "kind": "library",
            "training_row_count": 20,
            "training_component_count": 2,
            "library_slope": [0.0] * analysis.EXPECTED_GENES,
            "intercept": [0.0] * analysis.EXPECTED_GENES,
        },
        "final": {
            "kind": "library",
            "training_row_count": 30,
            "training_component_count": 3,
            "library_slope": [0.0] * analysis.EXPECTED_GENES,
            "intercept": [0.0] * analysis.EXPECTED_GENES,
        },
    }
    result = {
        "arms": {
            arm: {"phase_transform": {**copy.deepcopy(metadata), "arm": arm}}
            for arm in analysis.ARMS
        }
    }
    checkpoint = copy.deepcopy(result)
    analysis._verify_phase_transform("V4", result=result, checkpoint=checkpoint)
    broken = json.loads(json.dumps(result))
    broken["arms"]["observed_near"]["phase_transform"]["final_fit_mask"] = "test"
    with pytest.raises(analysis.RobustnessAnalysisError, match="metadata differs"):
        analysis._verify_phase_transform("V4", result=broken, checkpoint=checkpoint)


def test_v6_phase_transform_requires_all_12_frozen_training_levels() -> None:
    levels = [f"type_{index}" for index in range(12)]

    def fit(masses: list[float]) -> dict[str, object]:
        return {
            "kind": "cell_type_library",
            "training_row_count": 100,
            "training_component_count": 4,
            "library_slope": [0.0] * analysis.EXPECTED_GENES,
            "levels": levels,
            "type_weight_mass": masses,
            "type_intercepts": [
                [0.0] * analysis.EXPECTED_GENES for _ in levels
            ],
            "global_intercept": [0.0] * analysis.EXPECTED_GENES,
        }

    def payload(masses: list[float]) -> dict[str, object]:
        return {
            "kind": "cell_type_library",
            "tuning_fit_mask": "tuning_train",
            "final_fit_mask": "final_train",
            "tuning": fit(masses),
            "final": fit(masses),
        }

    valid = {
        "arms": {
            arm: {"phase_transform": {**payload([1.0] * 12), "arm": arm}}
            for arm in analysis.ARMS
        }
    }
    analysis._verify_phase_transform("V6", result=valid, checkpoint=copy.deepcopy(valid))
    invalid = copy.deepcopy(valid)
    invalid["arms"]["observed_near"]["phase_transform"]["final"][
        "type_weight_mass"
    ][4] = 0.0
    with pytest.raises(analysis.RobustnessAnalysisError, match="training coverage"):
        analysis._verify_phase_transform(
            "V6", result=invalid, checkpoint=copy.deepcopy(invalid)
        )


def test_frozen_contract_exactly_authorizes_analyzer_semantics() -> None:
    contract = analysis.train_wrapper._strict_yaml(CONTRACT_PATH, label="test contract")
    contract["status"] = "frozen_preoutcome"
    contract["launch_authorized"] = True
    analysis._verify_analysis_contract(contract)
    drifted = copy.deepcopy(contract)
    drifted["post_core_secondary_analyses"]["gene_label_null"][
        "draws_per_family"
    ] = 9999
    with pytest.raises(analysis.RobustnessAnalysisError, match="draws_per_family"):
        analysis._verify_analysis_contract(drifted)
    drifted = copy.deepcopy(contract)
    drifted["arms_core"][-1] = "renamed_permutation"
    with pytest.raises(analysis.RobustnessAnalysisError, match="arms_core"):
        analysis._verify_analysis_contract(drifted)


def test_run_technical_controls_are_all_fail_closed() -> None:
    controls = {
        "analytical_nonlinear_jacobian": {
            "passed": True,
            "maximum_autograd_error": 0.0,
            "maximum_finite_difference_error": 0.0,
        },
        "all_outputs_finite": True,
        "train_validation_test_component_overlap": False,
        "receiver_expression_input": False,
        "receiver_rna_or_derived_covariate_model_input": False,
        "identity_oracle_actually_executed": True,
        "identity_oracle_row_top1_fraction": 1.0,
        "graph_specific_invariants": True,
        "source_config_data_hashes_verified": True,
        "checkpoint_replay_device_type": "cuda",
        "canonical_production_split_label": "test",
        "outer_test_untouched": True,
        "peak_vram_gate_passed": True,
        "environment_lock_verified": True,
        "environment_visibility_mode": "job",
        "environment_lock_sha256": "a" * 64,
        "environment_verification_sha256": "b" * 64,
        "split_overlap_control": {"passed": True},
        "checkpoint_gpu_replay_max_abs_metric_error": 0.0,
        "checkpoint_gpu_replay_max_abs_prediction_error": 0.0,
        "peak_vram_gb": 1.0,
    }
    analysis._verify_run_technical_controls(controls, run_id="synthetic")
    broken = copy.deepcopy(controls)
    broken["source_config_data_hashes_verified"] = False
    with pytest.raises(analysis.RobustnessAnalysisError, match="source_config"):
        analysis._verify_run_technical_controls(broken, run_id="synthetic")
    broken = copy.deepcopy(controls)
    broken["checkpoint_gpu_replay_max_abs_prediction_error"] = 2e-7
    with pytest.raises(analysis.RobustnessAnalysisError, match="prediction_error"):
        analysis._verify_run_technical_controls(broken, run_id="synthetic")


def test_slide_stratified_bootstrap_is_deterministic_and_paired() -> None:
    groups = np.asarray([101, 102, 103, 201, 202, 203])
    baseline = np.asarray([2.0, 2.1, 1.9, 2.3, 2.2, 2.4])
    candidate = baseline * 0.9
    first = analysis._paired_slide_bootstrap(
        baseline, candidate, groups, draws=250, seed=17, return_draws=True
    )
    second = analysis._paired_slide_bootstrap(
        baseline, candidate, groups, draws=250, seed=17, return_draws=True
    )
    assert first["point"] == pytest.approx(0.1)
    assert np.array_equal(first["samples"], second["samples"])
    assert np.allclose(first["samples"], 0.1)


def test_gene_label_null_is_deterministic_for_full_and_matched_families() -> None:
    genes = 40
    matrix = np.full((genes, genes), 0.01, dtype=np.float64)
    np.fill_diagonal(matrix, np.linspace(1.0, 2.0, genes))
    eligible = np.ones(genes, dtype=bool)
    # Repeated covariate blocks ensure every synthetic matched stratum can derange.
    prevalence = np.repeat(np.linspace(0.1, 0.9, 10), 4)
    target_std = np.tile(np.repeat([1.0, 2.0], 2), 10)
    first = analysis._gene_label_null(
        matrix, eligible, prevalence, target_std, draws=80, seed=99
    )
    second = analysis._gene_label_null(
        matrix, eligible, prevalence, target_std, draws=80, seed=99
    )
    assert set(first["summary"]) == {"full", "prevalence_sd_decile_matched"}
    for family in first["distributions"]:
        for statistic in first["distributions"][family]:
            assert np.array_equal(
                first["distributions"][family][statistic],
                second["distributions"][family][statistic],
            )
    assert first["summary"]["full"]["statistics"]["row_top1_fraction"]["upper_tail_p"] == pytest.approx(1 / 81)


def test_seed_classification_uses_frozen_four_of_five_rule() -> None:
    assert analysis._three_way_classification(True, 4) == "robust_pass"
    assert analysis._three_way_classification(False, 1) == "robust_gate_failure"
    assert analysis._three_way_classification(True, 3) == "seed_sensitive"
    assert analysis._three_way_classification(False, 2) == "seed_sensitive"


def test_budget_fold_support_compares_selected_near_with_anchor_near(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "MODEL_SEEDS", (10, 11, 12, 13, 14))
    monkeypatch.setattr(analysis, "FOLDS", (0, 1, 2, 3))
    monkeypatch.setattr(analysis, "EXPECTED_COMPONENTS", 4)

    def rows(
        *, anchor_mse: float
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        selected: list[dict[str, object]] = []
        anchor: list[dict[str, object]] = []
        for seed_index, seed in enumerate(analysis.MODEL_SEEDS):
            passing_folds = 3 if seed_index < 4 else 2
            for fold in analysis.FOLDS:
                common = {
                    "variant_id": "V0",
                    "model_seed": seed,
                    "fold": fold,
                    "arm": "observed_near",
                    "geometry_group": 101 + fold,
                    "cell_count": 10,
                }
                selected.append(
                    {
                        **common,
                        "mse": 0.9 if fold < passing_folds else 1.1,
                    }
                )
                anchor.append({**common, "mse": anchor_mse})
        return selected, anchor

    selected, anchor = rows(anchor_mse=1.0)
    counts, passed = analysis._selected_near_anchor_fold_support(selected, anchor)
    assert counts == {"10": 3, "11": 3, "12": 3, "13": 3, "14": 2}
    assert passed is True

    # Moving only the anchor below the selected loss must reverse the support;
    # morphology is intentionally absent so it cannot satisfy this predicate.
    _, lower_anchor = rows(anchor_mse=0.8)
    counts, passed = analysis._selected_near_anchor_fold_support(
        selected, lower_anchor
    )
    assert counts == {str(seed): 0 for seed in analysis.MODEL_SEEDS}
    assert passed is False


def test_cross_variant_classification_distinguishes_agreement_from_direction() -> None:
    variants = {
        variant: {
            "gate_classification": {
                "gate": {"classification": "robust_gate_failure"}
            }
        }
        for variant in analysis.CORE_VARIANTS
    }
    result = analysis._cross_variant_classification(variants)["gate"]
    assert result["classification"] == "robust_across_preprocessing"
    assert result["shared_gate_classification"] == "robust_gate_failure"
    variants["V5"]["gate_classification"]["gate"]["classification"] = "robust_pass"
    result = analysis._cross_variant_classification(variants)["gate"]
    assert result["classification"] == "preprocessing_sensitive"
    assert result["shared_gate_classification"] is None


def test_csv_writer_emits_each_declared_row_once(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    analysis._write_csv(path, [{"value": 1}, {"value": 2}], ("value",))
    assert path.read_text(encoding="utf-8").splitlines() == ["value", "1", "2"]
    with pytest.raises(analysis.RobustnessAnalysisError, match="CSV row schema"):
        analysis._csv_text([{}], ("value",))
    with pytest.raises(analysis.RobustnessAnalysisError, match="CSV row schema"):
        analysis._csv_text([{"value": 1, "extra": 2}], ("value",))


def test_verify_only_rebuilds_publication_from_current_cli_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = SimpleNamespace(
        contract_sha256="a" * 64,
        sha256="b" * 64,
    )
    payload = {
        "contract_sha256": launch.contract_sha256,
        "launch_manifest_sha256": launch.sha256,
    }
    arrays = {"array": np.asarray([1.0], dtype=np.float64)}
    checks = [{"check": True}]
    component_rows = [{"component": True}]
    gene_rows = [{"gene": True}]
    provenance = {"authority": True}
    declared = [object()]
    planned = [object()]
    verified_run = object()

    monkeypatch.setattr(
        analysis,
        "_verify_campaign_inputs",
        lambda *_args, **_kwargs: ({"contract": True}, launch, {"environment": True}),
    )
    monkeypatch.setattr(
        analysis, "_collect_plan_entries", lambda *_args, **_kwargs: declared
    )
    monkeypatch.setattr(analysis, "_validate_coverage", lambda value: planned)
    monkeypatch.setattr(
        analysis, "current_paths", lambda: SimpleNamespace(state_root=tmp_path)
    )
    monkeypatch.setattr(analysis, "Registry", lambda _path: object())
    monkeypatch.setattr(
        analysis, "_verify_registry_attempt_inventory", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        analysis, "_verify_superseded_attempts", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        analysis, "_verify_one_run", lambda *_args, **_kwargs: verified_run
    )
    monkeypatch.setattr(
        analysis, "_verify_common_scientific_id", lambda _runs: {"V0": "scientific"}
    )
    monkeypatch.setattr(
        analysis,
        "_build_payload",
        lambda *_args, **_kwargs: (
            payload,
            arrays,
            checks,
            component_rows,
            gene_rows,
        ),
    )
    monkeypatch.setattr(
        analysis, "_source_provenance", lambda **_kwargs: provenance
    )
    monkeypatch.setattr(analysis, "_report_markdown", lambda _payload: "report\n")

    def verify(root: Path, *, expected: analysis.PublicationExpectation) -> dict[str, object]:
        assert root == tmp_path / "published"
        assert expected.payload is payload
        assert expected.arrays is arrays
        assert expected.checks is checks
        assert expected.component_rows is component_rows
        assert expected.gene_rows is gene_rows
        assert expected.provenance is provenance
        assert expected.report_markdown == "report\n"
        return {"verified": True}

    monkeypatch.setattr(analysis, "_verify_analysis_output", verify)
    arguments = SimpleNamespace(
        contract=tmp_path / "contract.yaml",
        launch_manifest=tmp_path / "launch.json",
        plan=[tmp_path / "plan.json"],
        output=tmp_path / "published",
        verify_only=True,
    )
    assert analysis.run(arguments, project_root=tmp_path) == {"verified": True}


def test_published_bundle_recomputes_json_npz_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis, "CORE_VARIANTS", ("V0",))
    monkeypatch.setattr(analysis, "MATRIX_ARMS", ("observed_near",))
    monkeypatch.setattr(analysis, "EXPECTED_GENES", 3)
    monkeypatch.setattr(analysis, "GENE_NULL_DRAWS", 4)
    monkeypatch.setattr(analysis, "BOOTSTRAP_DRAWS", 4)
    matrix = np.full((3, 3), 0.1, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    summary = analysis.diagonal_summary(matrix, np.ones(3, dtype=bool)).as_dict()
    arrays = {
        f"V0_observed_near_{label}_{part}": matrix.copy()
        for label in ("selected", "anchor12")
        for part in analysis.MATRIX_PARTS
    }
    null_summary: dict[str, object] = {}
    for family_index, family in enumerate(analysis.GENE_NULL_FAMILIES):
        statistics = {}
        for statistic_index, statistic in enumerate(analysis.GENE_NULL_STATISTICS):
            distribution = np.asarray(
                [0.1, 0.2, 0.3, 0.4], dtype=np.float64
            ) + family_index + statistic_index
            observed = 0.25 + family_index + statistic_index
            arrays[f"gene_label_null_{family}_{statistic}"] = distribution
            statistics[statistic] = {
                "observed": observed,
                "null_mean": float(np.mean(distribution)),
                "null_95": [
                    float(np.quantile(distribution, 0.025)),
                    float(np.quantile(distribution, 0.975)),
                ],
                "upper_tail_p": float(
                    (1 + np.sum(distribution >= observed)) / 5
                ),
            }
        null_summary[family] = {
            "draws": 4,
            "base_seed": analysis.GENE_NULL_SEED,
            "eligible_gene_count": 3,
            "statistics": statistics,
        }
    budget_draws = np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=np.float64)
    arrays["V0_budget_selected_minus_anchor12_bootstrap"] = budget_draws
    payload = {
        "campaign_id": analysis.CAMPAIGN_ID,
        "contract_sha256": "a" * 64,
        "launch_manifest_sha256": "b" * 64,
        "coverage": {
            "verified_jobs": 140,
            "expected_jobs": 140,
            "eligible_gene_count": 3,
            "declared_attempts": 140,
            "superseded_unsuccessful_attempts": 0,
        },
        "attempt_history": [
            {
                "variant_id": variant,
                "model_seed": seed,
                "fold": fold,
                "attempt": 1,
                "selected": True,
                "registry_status": "completed",
                "artifact_status": "success",
                "run_id": f"{variant}-{seed}-{fold}-a1",
            }
            for variant in tuple(f"V{index}" for index in range(7))
            for seed in analysis.MODEL_SEEDS
            for fold in analysis.FOLDS
        ],
        "exact_row_counts": {
            "run_verification_csv": 1,
            "component_metrics_csv": 1,
            "eligible_gene_summary_csv": 3,
        },
        "variants": {
            "V0": {
                "jacobian": {
                    "consensus": {
                        "observed_near": {"selected": {"total": summary}}
                    }
                },
                "consensus_gates": {
                    "same_name_diagonal_enrichment": {
                        "observed": summary["diagonal_offdiagonal_ratio"]
                    },
                    "strict_row_selectivity": {
                        "row_top1_fraction": summary["row_top1_fraction"]
                    },
                },
            }
        },
        "gene_label_null": null_summary,
        "fixed_randomization": {
            "bootstrap": {"draws": 4, "seed": analysis.BOOTSTRAP_SEED},
            "gene_label_null": {"draws": 4, "seed": analysis.GENE_NULL_SEED},
        },
        "budget_attribution": {
            "selected_minus_anchor12_gain_bootstrap": {
                "lower_95": float(np.quantile(budget_draws, 0.025)),
                "upper_95": float(np.quantile(budget_draws, 0.975)),
                "positive_draw_fraction": float(np.mean(budget_draws > 0)),
            }
        },
    }
    environment_verification = {
        "schema_version": 1,
        "verified": True,
        "visibility_mode": "analysis",
        "environment_lock_sha256": analysis.sha256_file(
            PROJECT_ROOT / analysis.ENVIRONMENT_LOCK_RELATIVE_PATH
        ),
        "observation": {"synthetic": True},
    }
    environment_verification["verification_sha256"] = analysis.canonical_sha256(
        environment_verification
    )
    provenance = {
        "float_policy": {"aggregate_dtype": "float64", "storage_dtype": "float64"},
        "environment_verification": environment_verification,
        "sources": [
            {
                "path": ANALYZER_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": ANALYZER_PATH.stat().st_size,
                "sha256": analysis.sha256_file(ANALYZER_PATH),
            }
        ],
    }
    check = {field: None for field in analysis.RUN_VERIFICATION_FIELDS}
    check["run_id"] = "x"
    checks = [check]
    component_row = {field: None for field in analysis.COMPONENT_METRIC_FIELDS}
    component_row.update({"arm": "near", "mse": 1.0})
    component_rows = [component_row]
    gene_rows = analysis._eligible_gene_rows(
        ("A", "B", "C"), matrix, np.ones(3, dtype=bool)
    )
    report_markdown = "synthetic\n"
    expectation = analysis.PublicationExpectation(
        payload=payload,
        arrays=arrays,
        checks=checks,
        component_rows=component_rows,
        gene_rows=gene_rows,
        provenance=provenance,
        report_markdown=report_markdown,
    )
    output = tmp_path / "analysis"
    output.mkdir()
    analysis._write_json(output / "aggregate_results.json", payload)
    np.savez_compressed(output / "aggregate_jacobians.npz", **arrays)
    analysis._write_csv(
        output / "run_verification.csv",
        checks,
        analysis.RUN_VERIFICATION_FIELDS,
    )
    analysis._write_csv(
        output / "component_metrics.csv",
        component_rows,
        analysis.COMPONENT_METRIC_FIELDS,
    )
    analysis._write_csv(
        output / "eligible_gene_summary.csv",
        gene_rows,
        analysis.ELIGIBLE_GENE_FIELDS,
    )
    analysis._write_json(output / "analysis_provenance.json", provenance)
    (output / "report.md").write_text(report_markdown, encoding="utf-8")
    analysis._write_analysis_manifest(output)
    assert analysis._verify_analysis_output(output, expected=expectation)["verified"] is True

    # A self-consistent manifest cannot make another contract/launch, a changed
    # selected run ID, an attempt gap, or a changed scientific value pass the
    # independently rebuilt current-CLI authority.
    for mutate in (
        lambda value: value.update(contract_sha256="c" * 64),
        lambda value: value.update(launch_manifest_sha256="c" * 64),
        lambda value: value["attempt_history"][0].update(run_id="different"),
        lambda value: value["attempt_history"][0].update(attempt=2),
        lambda value: value["variants"]["V0"]["consensus_gates"]
        ["same_name_diagonal_enrichment"].update(observed=999.0),
        lambda value: value["gene_label_null"][analysis.GENE_NULL_FAMILIES[0]]
        ["statistics"][analysis.GENE_NULL_STATISTICS[0]].update(observed=999.0),
        lambda value: value["budget_attribution"]
        ["selected_minus_anchor12_gain_bootstrap"].update(lower_95=999.0),
    ):
        drifted_payload = copy.deepcopy(payload)
        mutate(drifted_payload)
        analysis._write_json(output / "aggregate_results.json", drifted_payload)
        analysis._write_analysis_manifest(output)
        with pytest.raises(analysis.RobustnessAnalysisError, match="aggregate JSON"):
            analysis._verify_analysis_output(output, expected=expectation)
    analysis._write_json(output / "aggregate_results.json", payload)

    # Every NPZ array is checked, including arrays not used by a headline gate.
    drifted_arrays = {name: value.copy() for name, value in arrays.items()}
    drifted_arrays["V0_observed_near_anchor12_linear"][0, 0] += 1e-9
    np.savez_compressed(output / "aggregate_jacobians.npz", **drifted_arrays)
    analysis._write_analysis_manifest(output)
    with pytest.raises(analysis.RobustnessAnalysisError, match="aggregate NPZ"):
        analysis._verify_analysis_output(output, expected=expectation)
    np.savez_compressed(output / "aggregate_jacobians.npz", **arrays)

    csv_tamper_cases = (
        (
            "run_verification.csv",
            [{**check, "run_id": "different"}],
            analysis.RUN_VERIFICATION_FIELDS,
        ),
        (
            "component_metrics.csv",
            [{**component_row, "mse": 2.0}],
            analysis.COMPONENT_METRIC_FIELDS,
        ),
        (
            "eligible_gene_summary.csv",
            [{**row, "signed_diagonal": 2.0} for row in gene_rows],
            analysis.ELIGIBLE_GENE_FIELDS,
        ),
    )
    expected_csv_rows = (checks, component_rows, gene_rows)
    for (filename, drifted_rows, fields), original_rows in zip(
        csv_tamper_cases, expected_csv_rows, strict=True
    ):
        analysis._write_csv(output / filename, drifted_rows, fields)
        analysis._write_analysis_manifest(output)
        with pytest.raises(analysis.RobustnessAnalysisError, match=filename):
            analysis._verify_analysis_output(output, expected=expectation)
        analysis._write_csv(output / filename, original_rows, fields)

    component_csv = output / "component_metrics.csv"
    component_text = component_csv.read_text(encoding="utf-8")
    component_csv.write_text(
        component_text.replace("variant_id", "variant", 1), encoding="utf-8"
    )
    analysis._write_analysis_manifest(output)
    with pytest.raises(
        analysis.RobustnessAnalysisError, match="component_metrics.csv"
    ):
        analysis._verify_analysis_output(output, expected=expectation)
    analysis._write_csv(
        component_csv, component_rows, analysis.COMPONENT_METRIC_FIELDS
    )

    analysis._write_analysis_manifest(output)
    assert analysis._verify_analysis_output(output, expected=expectation)["verified"] is True
    (output / "report.md").write_text("tampered\n", encoding="utf-8")
    analysis._write_analysis_manifest(output)
    with pytest.raises(analysis.RobustnessAnalysisError, match="report differs"):
        analysis._verify_analysis_output(output, expected=expectation)
