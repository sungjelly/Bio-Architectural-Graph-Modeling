from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import spatial_benchmark.so2_nb_data as so2_nb_data

from spatial_benchmark.so2_nb_data import (
    EXPECTED_DONOR_GROUP_PAIRS,
    SELECTED_VALIDATION_PAIR_DIGEST,
    SELECTION_NAMESPACE,
    SO2NBCoreBatch,
    SO2NBDataBundle,
    SO2NBDataContractError,
    SO2_NB_TEST_ALIASES,
    SO2_NB_TRAINING_ALIASES,
    SO2_NB_VALIDATION_ALIASES,
    VALIDATION_MASK_CHUNK_CELLS,
    derive_so2_nb_validation_mask_seed,
    fit_so2_nb_preprocessing,
    make_so2_nb_validation_mask,
    recenter_so2_nb_covariates,
    standardize_so2_nb_expression_input,
    validation_pair_digest,
    verify_protected_so2_grouping,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _synthetic_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    counts: dict[str, np.ndarray] = {}
    covariates: dict[str, np.ndarray] = {}
    for index, alias in enumerate(
        SO2_NB_TRAINING_ALIASES + SO2_NB_VALIDATION_ALIASES
    ):
        rows = 2 + index % 3
        base = np.arange(rows * 4, dtype=np.int32).reshape(rows, 4)
        counts[alias] = base + np.int32(index)
        covariate_base = np.arange(rows * 22, dtype=np.float32).reshape(rows, 22)
        covariates[alias] = covariate_base / np.float32(17.0) + np.float32(index)
    return counts, covariates


def _fit_statistics():
    counts, covariates = _synthetic_arrays()
    statistics = fit_so2_nb_preprocessing(
        counts,
        covariates,
        source_global_covariate_mean=np.linspace(-2.0, 2.0, 22),
        source_global_covariate_scale=np.linspace(0.5, 3.0, 22),
    )
    return counts, covariates, statistics


def test_frozen_split_is_pair_disjoint_and_has_no_test() -> None:
    assert SO2_NB_TRAINING_ALIASES == tuple(
        f"SO2-C{core:02d}" for core in range(15, 27)
    )
    assert SO2_NB_VALIDATION_ALIASES == ("SO2-C27", "SO2-C28")
    assert SO2_NB_TEST_ALIASES == ()
    assert set(SO2_NB_TRAINING_ALIASES).isdisjoint(SO2_NB_VALIDATION_ALIASES)
    assert set(SO2_NB_TRAINING_ALIASES + SO2_NB_VALIDATION_ALIASES) == {
        alias for pair in EXPECTED_DONOR_GROUP_PAIRS for alias in pair
    }


def test_validation_pair_digest_uses_frozen_payload_and_is_minimum() -> None:
    literal = (
        f"{SELECTION_NAMESPACE}|SO2-C27+SO2-C28".encode("utf-8")
    )
    assert hashlib.sha256(literal).hexdigest() == SELECTED_VALIDATION_PAIR_DIGEST
    assert (
        validation_pair_digest(tuple(reversed(SO2_NB_VALIDATION_ALIASES)))
        == SELECTED_VALIDATION_PAIR_DIGEST
    )
    digests = [validation_pair_digest(pair) for pair in EXPECTED_DONOR_GROUP_PAIRS]
    assert min(digests) == SELECTED_VALIDATION_PAIR_DIGEST


def test_validation_changes_cannot_change_training_statistics() -> None:
    counts, covariates = _synthetic_arrays()
    first = fit_so2_nb_preprocessing(counts, covariates)

    changed_counts = dict(counts)
    changed_covariates = dict(covariates)
    for alias in SO2_NB_VALIDATION_ALIASES:
        changed_counts[alias] = np.full_like(counts[alias], 2_000_000_000)
        changed_covariates[alias] = np.full_like(covariates[alias], -1_000_000.0)
    second = fit_so2_nb_preprocessing(changed_counts, changed_covariates)

    assert first.fingerprint == second.fingerprint
    for name in first.arrays():
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_expression_statistics_are_equal_core_not_equal_cell() -> None:
    counts, covariates = _synthetic_arrays()
    statistics = fit_so2_nb_preprocessing(counts, covariates)
    per_core_means = [
        np.log1p(counts[alias].astype(np.float64)).mean(axis=0)
        for alias in SO2_NB_TRAINING_ALIASES
    ]
    expected_mean = np.stack(per_core_means).mean(axis=0)
    pooled_cell_mean = np.concatenate(
        [counts[alias] for alias in SO2_NB_TRAINING_ALIASES], axis=0
    )
    pooled_cell_mean = np.log1p(pooled_cell_mean.astype(np.float64)).mean(axis=0)
    np.testing.assert_allclose(statistics.expression_log1p_mean, expected_mean)
    assert not np.allclose(statistics.expression_log1p_mean, pooled_cell_mean)


def test_covariate_affine_cancellation_and_train_equal_core_scaling() -> None:
    _, covariates, statistics = _fit_statistics()
    assert statistics.affine_cancellation_max_abs_error <= 1e-10
    transformed = {
        alias: recenter_so2_nb_covariates(covariates[alias], statistics)
        for alias in SO2_NB_TRAINING_ALIASES
    }
    mean = np.stack(
        [values.astype(np.float64).mean(axis=0) for values in transformed.values()]
    ).mean(axis=0)
    second = np.stack(
        [np.square(values.astype(np.float64)).mean(axis=0) for values in transformed.values()]
    ).mean(axis=0)
    np.testing.assert_allclose(mean, 0.0, atol=2e-7)
    np.testing.assert_allclose(second, 1.0, atol=3e-7)


def test_raw_int32_target_is_aligned_and_does_not_alias_input() -> None:
    counts, covariates, statistics = _fit_statistics()
    alias = SO2_NB_TRAINING_ALIASES[0]
    target = np.array(counts[alias], dtype=np.int32, copy=True)
    expression = standardize_so2_nb_expression_input(target, statistics)
    node_covariates = recenter_so2_nb_covariates(covariates[alias], statistics)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    geometry = torch.zeros((2, 70), dtype=torch.float32)
    batch = SO2NBCoreBatch(
        alias=alias,
        role="train",
        input_expression=torch.from_numpy(expression),
        raw_count_target=torch.from_numpy(target),
        node_covariates=torch.from_numpy(node_covariates),
        edge_index=edges,
        relative_geometry=geometry,
    )

    before = batch.input_expression.clone()
    assert batch.raw_count_target.dtype == torch.int32
    assert batch.input_expression.dtype == torch.float32
    assert batch.raw_count_target.shape == batch.input_expression.shape
    assert batch.raw_count_target.untyped_storage().data_ptr() != (
        batch.input_expression.untyped_storage().data_ptr()
    )
    batch.raw_count_target[0, 0] += 100
    torch.testing.assert_close(batch.input_expression, before)


def test_raw_count_and_graph_alignment_fail_closed() -> None:
    counts, covariates, statistics = _fit_statistics()
    alias = SO2_NB_TRAINING_ALIASES[0]
    with pytest.raises(SO2NBDataContractError, match="int32"):
        standardize_so2_nb_expression_input(counts[alias].astype(np.int64), statistics)

    expression = standardize_so2_nb_expression_input(counts[alias], statistics)
    node_covariates = recenter_so2_nb_covariates(covariates[alias], statistics)
    with pytest.raises(SO2NBDataContractError, match="out-of-range"):
        SO2NBCoreBatch(
            alias=alias,
            role="train",
            input_expression=torch.from_numpy(expression),
            raw_count_target=torch.from_numpy(counts[alias].copy()),
            node_covariates=torch.from_numpy(node_covariates),
            edge_index=torch.tensor([[0], [len(expression)]], dtype=torch.long),
            relative_geometry=torch.zeros((1, 70), dtype=torch.float32),
        )


def test_fixed_validation_masks_are_stable_exact_and_view_specific() -> None:
    first = make_so2_nb_validation_mask(
        512, 17, alias="SO2-C27", view_index=0
    )
    repeated = make_so2_nb_validation_mask(
        512, 17, alias="so2-c27", view_index=0
    )
    other = make_so2_nb_validation_mask(
        512, 17, alias="SO2-C27", view_index=1
    )

    assert first.seed == 3_529_373_580_991_043_030
    assert first.seed == derive_so2_nb_validation_mask_seed("SO2-C27", 0)
    assert first.checksum_sha256 == repeated.checksum_sha256
    assert first.receipt_sha256 == repeated.receipt_sha256
    np.testing.assert_array_equal(first.mask, repeated.mask)
    np.testing.assert_array_equal(
        first.mask.sum(axis=1, dtype=np.int64), first.masked_gene_counts
    )
    assert first.masked_gene_counts.min() == 0
    assert first.masked_gene_counts.max() == 17
    assert first.n_masked_entries == int(first.mask.sum())
    assert other.checksum_sha256 != first.checksum_sha256


def test_validation_mask_domain_and_generation_chunk_are_frozen() -> None:
    with pytest.raises(SO2NBDataContractError, match="validation alias"):
        derive_so2_nb_validation_mask_seed("SO2-C26", 0)
    with pytest.raises(SO2NBDataContractError, match="0 through 9"):
        derive_so2_nb_validation_mask_seed("SO2-C27", 10)
    with pytest.raises(SO2NBDataContractError, match="frozen"):
        make_so2_nb_validation_mask(
            8,
            4,
            alias="SO2-C27",
            view_index=0,
            chunk_cells=VALIDATION_MASK_CHUNK_CELLS // 2,
        )


def _write_grouping_workbook(path: Path, group_values: list[str]) -> str:
    rows: list[list[object]] = [["No.", "protected"]]
    for core, value in zip(range(15, 29), group_values, strict=True):
        rows.append([core, value])
    pd.DataFrame(rows).to_excel(path, index=False, header=False)
    return _sha256(path)


def test_protected_grouping_verification_never_serializes_values(tmp_path: Path) -> None:
    protected_values = [
        value
        for index in range(7)
        for value in (f"restricted-person-{index}", f"restricted-person-{index}")
    ]
    path = tmp_path / "protected.xlsx"
    checksum = _write_grouping_workbook(path, protected_values)
    result = verify_protected_so2_grouping(
        path, expected_source_sha256=checksum
    )
    serialized = json.dumps(result.to_receipt(), sort_keys=True)

    assert result.core_pairs == EXPECTED_DONOR_GROUP_PAIRS
    assert result.selected_validation_pair_digest == SELECTED_VALIDATION_PAIR_DIGEST
    assert result.donor_group_count == 7
    assert result.cores_per_group == 2
    assert all(value not in serialized for value in set(protected_values))
    assert "restricted-person" not in repr(result)


def test_protected_grouping_failure_does_not_echo_values(tmp_path: Path) -> None:
    protected_values = [
        value
        for index in range(7)
        for value in (f"private-{index}", f"private-{index}")
    ]
    protected_values[3] = protected_values[0]
    path = tmp_path / "bad-protected.xlsx"
    checksum = _write_grouping_workbook(path, protected_values)
    with pytest.raises(SO2NBDataContractError) as captured:
        verify_protected_so2_grouping(path, expected_source_sha256=checksum)
    assert all(value not in str(captured.value) for value in set(protected_values))


def test_statistics_repr_and_mapping_do_not_contain_raw_arrays() -> None:
    _, _, statistics = _fit_statistics()
    rendered = repr(statistics)
    assert "expression_log1p_mean=array" not in rendered
    assert "covariate_recenter_mean=array" not in rendered
    assert asdict(statistics)["training_aliases"] == SO2_NB_TRAINING_ALIASES


def test_loader_exposes_manifest_file_and_content_hashes_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "overlay"
    root.mkdir(mode=0o700)
    manifest_path = root / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest_path.chmod(0o600)
    split = so2_nb_data._split_payload()
    split_fingerprint = so2_nb_data._canonical_sha256(split)
    manifest: dict[str, object] = {
        "artifact_kind": so2_nb_data.OVERLAY_SCHEMA,
        "campaign_id": so2_nb_data.CAMPAIGN_ID,
        "split": {**split, "split_fingerprint": split_fingerprint},
        "source_artifacts": {
            "cohort": {
                "reference": "cohort",
                "manifest_file_sha256": "c" * 64,
                "manifest_content_sha256": "d" * 64,
            },
            "graph": {
                "reference": "graph",
                "manifest_file_sha256": "e" * 64,
                "manifest_content_sha256": "f" * 64,
            },
        },
        "validation_masks": {"receipts": {}},
    }
    content_sha256 = so2_nb_data._canonical_sha256(manifest)
    manifest["manifest_content_sha256"] = content_sha256
    statistics = SimpleNamespace(
        training_aliases=SO2_NB_TRAINING_ALIASES,
        fingerprint="b" * 64,
    )

    monkeypatch.setattr(so2_nb_data, "_strict_json", lambda path: manifest)
    monkeypatch.setattr(
        so2_nb_data, "_validate_overlay_semantics", lambda root, value: None
    )
    monkeypatch.setattr(
        so2_nb_data,
        "sha256_file",
        lambda path: "a" * 64
        if Path(path) == manifest_path
        else (
            "c" * 64
            if Path(path).parent.name == "cohort"
            else "e" * 64
        ),
    )
    monkeypatch.setattr(
        so2_nb_data,
        "_verify_so2_cohort_manifest",
        lambda root: {"manifest_content_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        so2_nb_data,
        "_verify_so2_graph_collection",
        lambda root, *, cohort_manifest_path: {
            "manifest_content_sha256": "f" * 64
        },
    )
    monkeypatch.setattr(
        so2_nb_data, "_statistics_from_overlay", lambda root, value: statistics
    )
    monkeypatch.setattr(
        so2_nb_data,
        "_load_core_batch",
        lambda alias, **kwargs: SimpleNamespace(alias=alias),
    )

    bundle = so2_nb_data.load_so2_nb_data(root)

    assert isinstance(bundle, SO2NBDataBundle)
    assert bundle.manifest_sha256 == "a" * 64
    assert bundle.manifest_content_sha256 == content_sha256
