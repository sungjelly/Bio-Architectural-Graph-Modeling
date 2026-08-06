from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import spatial_benchmark.pooled_full_core as pooled
from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS
from spatial_benchmark.full_core import (
    FullCorePreprocessingChecksums,
    FullCorePreprocessingQC,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _install_synthetic_receipts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sizes: tuple[int, ...] = tuple(range(1, 11)),
) -> tuple[pooled.PreparedCoreReceipt, ...]:
    receipts = tuple(
        pooled.PreparedCoreReceipt(
            alias=alias,
            n_nodes=size,
            manifest_sha256=_sha(f"manifest-{alias}"),
            prepared_data_sha256=_sha(f"data-{alias}"),
        )
        for alias, size in zip(pooled.ANC_ALIASES, sizes, strict=True)
    )
    monkeypatch.setattr(pooled, "EXPECTED_PREPARED_CORES", receipts)
    monkeypatch.setattr(pooled, "EXPECTED_TOTAL_NODES", sum(sizes))
    monkeypatch.setattr(pooled, "EXPECTED_N_GENES", 2)
    return receipts


def _manifest(
    receipt: pooled.PreparedCoreReceipt,
    *,
    gene_names: tuple[str, ...] = ("GeneA", "GeneB"),
) -> dict[str, object]:
    return {
        "artifact_kind": "adjacent_normal_tissue_spatial_benchmark_preparation",
        "artifact_id": receipt.manifest_sha256[:16],
        "manifest_content_sha256": receipt.manifest_sha256,
        "files": {"prepared_data.npz": receipt.prepared_data_sha256},
        "selection": {
            "opaque_alias": receipt.alias,
            "restricted_identifiers_emitted": False,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "n_cells": receipt.n_nodes,
        },
        "features": {
            "gene_names": list(gene_names),
            "measured_metadata_names": list(ALLOWED_METADATA_COLUMNS),
            "model_covariate_names": list(ALLOWED_METADATA_COLUMNS),
            "n_biological_probes": len(gene_names),
            "technical_control_prefixes_excluded": [
                "Negative",
                "SystemControl",
            ],
            "coordinates_are_model_covariates": False,
            "routing_keys_are_model_covariates": False,
            "qc_indicator_is_model_covariate": False,
        },
    }


def _source_core(
    receipt: pooled.PreparedCoreReceipt,
    *,
    counts: np.ndarray | None = None,
) -> SimpleNamespace:
    alias_index = int(receipt.alias[-2:])
    if counts is None:
        counts = np.column_stack(
            [
                np.full(receipt.n_nodes, alias_index - 1, dtype=np.int32),
                np.arange(receipt.n_nodes, dtype=np.int32) % 4,
            ]
        )
    covariates = np.full(
        (receipt.n_nodes, len(ALLOWED_METADATA_COLUMNS)),
        alias_index / 10.0,
        dtype=np.float32,
    )
    coordinates = np.column_stack(
        [
            np.arange(receipt.n_nodes, dtype=np.float64),
            np.arange(receipt.n_nodes, dtype=np.float64) + alias_index,
        ]
    )
    macroblocks = np.asarray(
        [
            f"fov={alias_index * 100}|x={index % 2}|y=0"
            for index in range(receipt.n_nodes)
        ],
        dtype="U32",
    )
    checksums = FullCorePreprocessingChecksums(
        source_artifact_id=receipt.manifest_sha256[:16],
        source_prepared_data_sha256=receipt.prepared_data_sha256,
        expression_counts_sha256=pooled._full_core._array_sha256(  # noqa: SLF001
            "expression_counts",
            counts,
        ),
        target_expression_sha256=_sha(f"obsolete-target-{receipt.alias}"),
        node_covariates_sha256=pooled._full_core._array_sha256(  # noqa: SLF001
            "node_covariates",
            covariates,
        ),
        coordinates_um_sha256=pooled._full_core._array_sha256(  # noqa: SLF001
            "coordinates_um",
            coordinates,
        ),
        macroblock_ids_sha256=pooled._full_core._array_sha256(  # noqa: SLF001
            "macroblock_ids",
            macroblocks,
        ),
        preprocessing_sha256=_sha(f"full-core-{receipt.alias}"),
    )
    qc = FullCorePreprocessingQC(
        n_nodes=receipt.n_nodes,
        n_genes=2,
        n_measured_metadata_features=len(ALLOWED_METADATA_COLUMNS),
        n_model_covariates=len(ALLOWED_METADATA_COLUMNS),
        n_metadata_missing_values=0,
        n_constant_expression_features=0,
        n_constant_metadata_features=0,
        expression_max_abs_fitted_mean=0.0,
        metadata_max_abs_fitted_mean=0.0,
        source_metadata_roundtrip_max_abs_error=0.0,
        fit_scope="all nodes (transductive)",
        metadata_reconstruction="synthetic",
        metadata_source_precision="float32",
        protected_identifier_arrays_returned=False,
        all_outputs_finite=True,
    )
    return SimpleNamespace(
        n_nodes=receipt.n_nodes,
        n_genes=2,
        expression_counts=counts,
        target_expression=np.zeros_like(counts, dtype=np.float32),
        node_covariates=covariates,
        coordinates_um=coordinates,
        macroblock_ids=macroblocks,
        gene_names=("GeneA", "GeneB"),
        metadata_names=ALLOWED_METADATA_COLUMNS,
        metadata_median=np.zeros(len(ALLOWED_METADATA_COLUMNS)),
        metadata_mean=np.full(
            len(ALLOWED_METADATA_COLUMNS),
            alias_index,
            dtype=np.float64,
        ),
        metadata_scale=np.ones(len(ALLOWED_METADATA_COLUMNS)),
        metadata_missing_indicator_indices=np.empty(0, dtype=np.int64),
        preprocessing_qc=qc,
        checksums=checksums,
    )


def _install_fake_loaders(
    monkeypatch: pytest.MonkeyPatch,
    receipts: tuple[pooled.PreparedCoreReceipt, ...],
    *,
    manifest_mutator: object | None = None,
    core_mutator: object | None = None,
) -> tuple[list[tuple[str, bool]], list[str]]:
    by_alias = {receipt.alias: receipt for receipt in receipts}
    manifest_calls: list[tuple[str, bool]] = []
    full_core_calls: list[str] = []

    def alias_from_path(path: str | Path) -> str:
        value = Path(path).name
        if value == "prepared_v1":
            value = Path(path).parent.name.upper()
        return value

    def fake_artifact_loader(
        path: str | Path,
        *,
        load_arrays: bool,
    ) -> tuple[dict[str, object], None, dict[str, object]]:
        alias = alias_from_path(path)
        manifest_calls.append((alias, load_arrays))
        value = _manifest(by_alias[alias])
        if callable(manifest_mutator):
            manifest_mutator(alias, value)
        return value, None, {}

    def fake_full_core_loader(
        path: str | Path,
        *,
        epsilon: float,
    ) -> SimpleNamespace:
        del epsilon
        alias = alias_from_path(path)
        full_core_calls.append(alias)
        value = _source_core(by_alias[alias])
        if callable(core_mutator):
            return core_mutator(alias, value)
        return value

    monkeypatch.setattr(pooled, "load_prepared_artifact", fake_artifact_loader)
    monkeypatch.setattr(
        pooled._full_core,
        "load_and_refit_full_core",
        fake_full_core_loader,
    )
    return manifest_calls, full_core_calls


def test_equal_core_statistics_use_core_not_cell_weighting() -> None:
    first = np.asarray([[0], [0]], dtype=np.int32)
    second = np.full((8, 1), 8, dtype=np.int32)
    mean, scale = pooled.fit_equal_core_log1p_statistics([first, second])

    first_log = np.log1p(first.astype(np.float64))
    second_log = np.log1p(second.astype(np.float64))
    expected_mean = (first_log.mean(axis=0) + second_log.mean(axis=0)) / 2
    expected_second = (
        np.square(first_log).mean(axis=0)
        + np.square(second_log).mean(axis=0)
    ) / 2
    expected_scale = np.sqrt(expected_second - np.square(expected_mean))
    np.testing.assert_allclose(mean, expected_mean, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scale, expected_scale, rtol=0.0, atol=0.0)
    assert mean[0] != np.log1p(np.concatenate([first, second])).mean()
    assert mean.dtype == np.float64
    assert scale.dtype == np.float64
    assert not mean.flags.writeable
    assert not scale.flags.writeable


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([[np.nan]]),
        np.asarray([[-1]], dtype=np.int32),
        np.asarray([[1.5]], dtype=np.float64),
        np.asarray([["1"]]),
    ],
)
def test_equal_core_statistics_reject_invalid_counts(
    invalid: np.ndarray,
) -> None:
    with pytest.raises(
        pooled.PooledFullCoreContractError,
        match="counts",
    ):
        pooled.fit_equal_core_log1p_statistics([invalid])


def test_pooled_loader_binds_sources_computes_shared_transform_and_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts = _install_synthetic_receipts(monkeypatch)
    manifest_calls, full_core_calls = _install_fake_loaders(
        monkeypatch,
        receipts,
    )
    inputs = [(alias, alias) for alias in pooled.ANC_ALIASES]

    result = pooled.load_pooled_full_core_cohort(inputs)
    repeated = pooled.load_pooled_full_core(inputs)

    expected_mean, expected_scale = (
        pooled.fit_equal_core_log1p_statistics(
            [_source_core(receipt).expression_counts for receipt in receipts]
        )
    )
    np.testing.assert_array_equal(result.expression_mean, expected_mean)
    np.testing.assert_array_equal(result.expression_scale, expected_scale)
    assert result.aliases == pooled.ANC_ALIASES
    assert result.total_nodes == sum(range(1, 11))
    assert result.n_cores == 10
    assert result.n_genes == 2
    assert result.fingerprint_sha256 == repeated.fingerprint_sha256
    assert len(result.fingerprint_sha256) == 64
    assert result.core("ANC-04") is result.cores[3]

    for core, receipt in zip(result.cores, receipts, strict=True):
        expected_target = (
            np.log1p(core.expression_counts.astype(np.float64))
            - expected_mean
        ) / expected_scale
        np.testing.assert_allclose(
            core.target_expression,
            expected_target,
            rtol=2e-7,
            atol=2e-7,
        )
        np.testing.assert_array_equal(
            core.node_covariates,
            _source_core(receipt).node_covariates,
        )
        assert all(
            value.startswith(f"{core.alias}-BLOCK-")
            for value in core.macroblock_ids.tolist()
        )
        assert not any("fov=" in value for value in core.macroblock_ids.tolist())
        source_blocks = _source_core(receipt).macroblock_ids
        np.testing.assert_array_equal(
            source_blocks[:, None] == source_blocks[None, :],
            core.macroblock_ids[:, None] == core.macroblock_ids[None, :],
        )
        assert core.expression_mean is result.expression_mean
        assert core.expression_scale is result.expression_scale
        assert core.preprocessing_qc.protected_identifier_arrays_returned is False
        for item in fields(core):
            assert item.name not in {
                "cell_ID",
                "fov",
                "slide",
                "donor",
                "patient",
                "core_id",
            }
        for array in (
            core.expression_counts,
            core.target_expression,
            core.node_covariates,
            core.coordinates_um,
            core.macroblock_ids,
            core.metadata_mean,
        ):
            assert not array.flags.writeable

    assert manifest_calls == [
        (alias, False) for alias in pooled.ANC_ALIASES
    ] * 2
    assert full_core_calls == list(pooled.ANC_ALIASES) * 2
    with pytest.raises(FrozenInstanceError):
        result.total_nodes = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        result._cores_by_alias["ANC-01"] = result.cores[0]  # type: ignore[index]
    with pytest.raises(pooled.PooledFullCoreContractError):
        result.core("not-an-opaque-alias")
    assert "array(" not in repr(result.cores[0])


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (
            [(alias, alias) for alias in pooled.ANC_ALIASES[:-1]],
            "every frozen ANC alias",
        ),
        (
            [
                *[(alias, alias) for alias in pooled.ANC_ALIASES[:-1]],
                ("ANC-09", "duplicate"),
            ],
            "duplicate",
        ),
        (
            [
                ("ANC-02", "ANC-02"),
                ("ANC-01", "ANC-01"),
                *[(alias, alias) for alias in pooled.ANC_ALIASES[2:]],
            ],
            "frozen order",
        ),
    ],
)
def test_pooled_loader_rejects_missing_duplicate_and_wrong_order_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    inputs: list[tuple[str, str]],
    message: str,
) -> None:
    receipts = _install_synthetic_receipts(monkeypatch)
    manifest_calls, full_core_calls = _install_fake_loaders(
        monkeypatch,
        receipts,
    )
    with pytest.raises(pooled.PooledFullCoreContractError, match=message):
        pooled.load_pooled_full_core(inputs)
    assert manifest_calls == []
    assert full_core_calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda alias, manifest: (
                manifest.__setitem__("manifest_content_sha256", "0" * 64)
                if alias == "ANC-03"
                else None
            ),
            "frozen manifest checksum",
        ),
        (
            lambda alias, manifest: (
                manifest["selection"].__setitem__(  # type: ignore[union-attr]
                    "opaque_alias",
                    "restricted-value",
                )
                if alias == "ANC-03"
                else None
            ),
            "selection metadata",
        ),
        (
            lambda alias, manifest: (
                manifest["features"].__setitem__(  # type: ignore[union-attr]
                    "gene_names",
                    ["GeneB", "GeneA"],
                )
                if alias == "ANC-03"
                else None
            ),
            "ordered schema",
        ),
    ],
)
def test_pooled_loader_rejects_checksum_alias_and_ordered_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    message: str,
) -> None:
    receipts = _install_synthetic_receipts(monkeypatch)
    _install_fake_loaders(
        monkeypatch,
        receipts,
        manifest_mutator=mutation,
    )
    with pytest.raises(pooled.PooledFullCoreContractError, match=message) as error:
        pooled.load_pooled_full_core(
            [(alias, alias) for alias in pooled.ANC_ALIASES]
        )
    assert "restricted-value" not in str(error.value)


def test_pooled_loader_rejects_loaded_count_and_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts = _install_synthetic_receipts(monkeypatch)

    def mutate_core(alias: str, source: SimpleNamespace) -> SimpleNamespace:
        if alias == "ANC-02":
            source.expression_counts = source.expression_counts.copy()
            source.expression_counts[0, 0] = -1
        return source

    _install_fake_loaders(
        monkeypatch,
        receipts,
        core_mutator=mutate_core,
    )
    with pytest.raises(
        pooled.PooledFullCoreContractError,
        match="finite and nonnegative",
    ):
        pooled.load_pooled_full_core(
            [(alias, alias) for alias in pooled.ANC_ALIASES]
        )


def test_default_paths_are_data_root_relative_and_alias_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipts = _install_synthetic_receipts(monkeypatch)
    manifest_calls, full_core_calls = _install_fake_loaders(
        monkeypatch,
        receipts,
    )
    result = pooled.load_pooled_full_core(data_root=tmp_path)
    assert result.aliases == pooled.ANC_ALIASES
    assert manifest_calls == [
        (alias, False) for alias in pooled.ANC_ALIASES
    ]
    assert full_core_calls == list(pooled.ANC_ALIASES)
