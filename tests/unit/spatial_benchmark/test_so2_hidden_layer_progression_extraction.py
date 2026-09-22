from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
import spatial_benchmark.so2_hidden_layer_progression_extraction as extraction


def _small_model() -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    torch.manual_seed(20260903)
    return ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=6,
        node_covariate_dim=3,
        hidden_dim=16,
        attention_heads=4,
        attention_head_dim=4,
        graph_layers=4,
        ffn_dim=32,
        decoder_dim=24,
        positional_bias_hidden_dim=12,
        dropout=0.25,
        attention_dropout=0.2,
        receiver_chunk_size=2,
        max_edges_per_chunk=4,
        activation_checkpointing=False,
    ).eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(37)
    expression = torch.randn(5, 6, generator=generator)
    mask = torch.zeros_like(expression, dtype=torch.bool)
    covariates = torch.randn(5, 3, generator=generator)
    edges = torch.tensor(
        [[1, 2, 0, 2, 3, 0, 4], [0, 0, 1, 1, 1, 2, 3]],
        dtype=torch.long,
    )
    geometry = torch.randn(edges.shape[1], 70, generator=generator)
    return expression, mask, covariates, edges, geometry


def test_one_forward_captures_exact_full_node_h0_through_h4() -> None:
    expression, mask, covariates, edges, geometry = _inputs()
    before = tuple(
        value.clone() for value in (expression, mask, covariates, edges, geometry)
    )
    model = _small_model()
    encoder_calls = 0
    decoder_inputs: list[tuple[int, ...]] = []

    def count_encoder(
        _module: torch.nn.Module,
        _arguments: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    encoder_hook = model.encoder.register_forward_hook(count_encoder)
    decoder_hook = model.decoder.register_forward_pre_hook(
        lambda _module, arguments: decoder_inputs.append(tuple(arguments[0].shape))
    )
    states = extraction.extract_full_hidden_layer_progression(
        model,
        input_expression=expression,
        gene_mask=mask,
        edge_index=edges,
        relative_geometry=geometry,
        node_covariates=covariates,
    )
    encoder_hook.remove()
    decoder_hook.remove()

    assert encoder_calls == 1
    assert decoder_inputs == [(0, model.hidden_dim)]
    assert len(states) == 5
    assert all(state.shape == (5, model.hidden_dim) for state in states)
    assert all(state.device.type == "cpu" for state in states)
    assert all(torch.isfinite(state).all() for state in states)
    for observed, original in zip(
        (expression, mask, covariates, edges, geometry), before, strict=True
    ):
        assert torch.equal(observed, original)

    with torch.inference_mode():
        expected_h0 = model.encoder(expression, mask, covariates)
        expected = model(
            expression,
            mask,
            edge_index=edges,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_nodes=torch.empty((0,), dtype=torch.long),
            return_graph_step_embeddings=True,
        )
    assert expected.graph_step_embeddings is not None
    assert torch.equal(states[0], expected_h0)
    for observed, expected_step in zip(
        states[1:], expected.graph_step_embeddings, strict=True
    ):
        assert torch.equal(observed, expected_step)
    assert torch.equal(states[-1], expected.full_node_embedding)


def test_full_progression_requires_eval_zero_mask_and_four_blocks() -> None:
    expression, mask, covariates, edges, geometry = _inputs()
    arguments = {
        "input_expression": expression,
        "gene_mask": mask,
        "edge_index": edges,
        "relative_geometry": geometry,
        "node_covariates": covariates,
    }
    with pytest.raises(
        extraction.SO2HiddenLayerProgressionExtractionError, match="eval"
    ):
        extraction.extract_full_hidden_layer_progression(
            _small_model().train(), **arguments
        )

    nonzero = mask.clone()
    nonzero[0, 0] = True
    with pytest.raises(
        extraction.SO2HiddenLayerProgressionExtractionError, match="all-zero"
    ):
        extraction.extract_full_hidden_layer_progression(
            _small_model(), **{**arguments, "gene_mask": nonzero}
        )

    two_blocks = _small_model()
    two_blocks.graph_layers = 2
    with pytest.raises(
        extraction.SO2HiddenLayerProgressionExtractionError, match="four-block"
    ):
        extraction.extract_full_hidden_layer_progression(two_blocks, **arguments)


def test_h4_reference_control_records_exact_and_tolerant_replay_separately() -> None:
    generator = np.random.default_rng(9)
    h4 = generator.normal(size=(5, 16)).astype(np.float32)
    exact = extraction._compare_captured_h4_to_verified_hl(h4, h4.copy())
    assert exact["captured_h4_bitwise_equal"] is True
    assert exact["captured_h4_allclose"] is True
    assert exact["maximum_absolute_difference"] == 0.0

    replay = h4.copy()
    replay[0, 0] = np.nextafter(replay[0, 0], np.float32(np.inf))
    tolerant = extraction._compare_captured_h4_to_verified_hl(replay, h4)
    assert tolerant["captured_h4_bitwise_equal"] is False
    assert tolerant["captured_h4_allclose"] is True
    assert 0.0 < tolerant["maximum_absolute_difference"] <= 2e-6
    assert tolerant["maximum_scaled_tolerance_ratio"] <= 1.0

    divergent = h4.copy()
    divergent[0, 0] += np.float32(1e-3)
    with pytest.raises(
        extraction.SO2HiddenLayerProgressionExtractionError,
        match="beyond tolerance",
    ):
        extraction._compare_captured_h4_to_verified_hl(divergent, h4)


def test_hidden_layer_npz_loader_has_exact_schema_and_deterministic_bytes(
    tmp_path: Path,
) -> None:
    generator = np.random.default_rng(3)
    arrays = {
        "cell_index": np.arange(5, dtype=np.int64),
        "core_number": np.asarray(15, dtype=np.int16),
        "coordinates_um": generator.normal(size=(5, 2)).astype(np.float64),
        **{
            name: generator.normal(size=(5, 16)).astype(np.float32)
            for name in extraction.STORED_LAYER_NAMES
        },
    }
    path = tmp_path / "core_15_hidden_layers.npz"
    extraction._write_deterministic_npz(path, arrays)
    first_sha = sha256_file(path)
    extraction._write_deterministic_npz(path, arrays)
    assert sha256_file(path) == first_sha

    core = extraction.load_hidden_layer_core(
        path,
        alias="SO2-C15",
        core_number=15,
        expected_cells=5,
        hidden_dim=16,
    )
    assert core.n_cells == 5
    for name in extraction.STORED_LAYER_NAMES:
        assert np.array_equal(core.layer(name), arrays[name])
    with pytest.raises(
        extraction.SO2HiddenLayerProgressionExtractionError, match="one of"
    ):
        core.layer("h4")

    invalid = dict(arrays)
    del invalid["h2"]
    extraction._write_deterministic_npz(path, invalid)
    with pytest.raises(
        extraction.SO2HiddenLayerProgressionExtractionError, match="schema"
    ):
        extraction.load_hidden_layer_core(
            path,
            alias="SO2-C15",
            core_number=15,
            expected_cells=5,
            hidden_dim=16,
        )


def test_per_core_extraction_is_resumable_and_checksum_validating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alias = "SO2-C15"
    core_number = 15
    expression, _mask, covariates, edges, geometry = _inputs()
    coordinates = np.arange(10, dtype=np.float64).reshape(5, 2)
    model = _small_model()
    source_states = extraction.extract_full_hidden_layer_progression(
        model,
        input_expression=expression,
        gene_mask=torch.zeros_like(expression, dtype=torch.bool),
        edge_index=edges,
        relative_geometry=geometry,
        node_covariates=covariates,
    )
    source_hl = source_states[-1].numpy().copy()
    source_hl[0, 0] = np.nextafter(source_hl[0, 0], np.float32(np.inf))

    cohort_dir = tmp_path / "cohort"
    cohort_dir.joinpath("cores").mkdir(parents=True)
    np.savez(
        cohort_dir / "cores" / f"{alias}.npz",
        coordinates_um=coordinates,
    )
    hl_root = tmp_path / "verified_hl"
    hl_file = hl_root / "embeddings" / "core_15_hL.npz"
    extraction._write_deterministic_npz(
        hl_file,
        {
            "cell_index": np.arange(5, dtype=np.int64),
            "core_number": np.asarray(core_number, dtype=np.int16),
            "coordinates_um": coordinates,
            "hL": source_hl,
        },
    )
    hl_record = {
        "alias": alias,
        "core_number": core_number,
        "cell_count": 5,
        "embedding_file": "embeddings/core_15_hL.npz",
        "hL_array_sha256": extraction._array_sha256("hL", source_hl),
    }
    hl_extraction = {"cpu_threads": 1, "cores": [hl_record]}
    hl_manifest_sha = "f" * 64
    expected = {
        "alias": alias,
        "core_number": core_number,
        "cell_count": 5,
        "prepared_core_artifact_sha256": "a" * 64,
        "prepared_component_checksums": {},
        "graph_record_sha256": "b" * 64,
        "graph_logical_sha256": "c" * 64,
        "graph_file_checksums": {},
    }
    batch = SimpleNamespace(
        alias=alias,
        n_nodes=5,
        target_expression=expression,
        node_covariates=covariates,
        edge_index=edges,
        relative_geometry=geometry,
    )
    inputs = SimpleNamespace(
        run_id="synthetic_so2_four_block",
        checkpoint_sha256="d" * 64,
        checkpoint_payload={
            "model_construction": {
                "class": "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
                "graph_layers": 4,
                "hidden_dim": 16,
            }
        },
        cohort_dir=cohort_dir,
        graph_dir=tmp_path / "graphs",
        project_root=tmp_path,
        provenance={"fixture": "synthetic"},
    )
    monkeypatch.setattr(extraction, "SO2_CORE_NUMBERS", (core_number,))
    monkeypatch.setattr(extraction, "SO2_ALIASES", (alias,))
    monkeypatch.setattr(extraction, "EXPECTED_TOTAL_CELLS", 5)
    monkeypatch.setattr(
        extraction, "_expected_core_records", lambda _inputs: (expected,)
    )
    monkeypatch.setattr(
        extraction,
        "_load_verified_hl_extraction",
        lambda **_kwargs: (hl_root, hl_extraction, hl_manifest_sha),
    )
    monkeypatch.setattr(
        extraction, "load_so2_relative_qkv_batches", lambda **_kwargs: (batch,)
    )
    model_loads = 0

    def load_model(
        _inputs: object,
    ) -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
        nonlocal model_loads
        model_loads += 1
        return model

    monkeypatch.setattr(extraction, "load_so2_checkpoint_model", load_model)
    output_root = tmp_path / "progression"
    first = extraction.extract_hidden_layer_progression(
        inputs=inputs, output_root=output_root, cpu_threads=1
    )
    assert first["status"] == "complete"
    assert model_loads == 1
    assert first["cores"][0]["inference"]["model_forward_count"] == 1
    comparison = first["cores"][0]["reference_hL"]
    assert comparison["captured_h4_bitwise_equal"] is False
    assert comparison["captured_h4_allclose"] is True
    stored_path = output_root / "embeddings" / "core_15_hidden_layers.npz"
    with np.load(stored_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "cell_index",
            "core_number",
            "coordinates_um",
            "h0",
            "h1",
            "h2",
            "h3",
        }

    second = extraction.extract_hidden_layer_progression(
        inputs=inputs, output_root=output_root, cpu_threads=1
    )
    assert second == first
    assert model_loads == 1

    with np.load(stored_path, allow_pickle=False) as archive:
        corrupted = {name: np.asarray(archive[name]) for name in archive.files}
    corrupted["h1"] = corrupted["h1"].copy()
    corrupted["h1"][0, 0] += np.float32(1.0)
    extraction._write_deterministic_npz(stored_path, corrupted)
    with pytest.raises(
        extraction.SO2HiddenLayerProgressionExtractionError, match="checksum"
    ):
        extraction.extract_hidden_layer_progression(
            inputs=inputs, output_root=output_root, cpu_threads=1
        )


def test_per_core_extraction_rejects_reference_coordinate_misalignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise the control before any derived hidden-layer file can be accepted.
    alias = "SO2-C15"
    expression, _mask, covariates, edges, geometry = _inputs()
    model = _small_model()
    source_hl = extraction.extract_full_hidden_layer_progression(
        model,
        input_expression=expression,
        gene_mask=torch.zeros_like(expression, dtype=torch.bool),
        edge_index=edges,
        relative_geometry=geometry,
        node_covariates=covariates,
    )[-1].numpy()
    cohort_coordinates = np.arange(10, dtype=np.float64).reshape(5, 2)
    source_coordinates = cohort_coordinates.copy()
    source_coordinates[0, 0] += 1.0
    cohort_dir = tmp_path / "cohort"
    cohort_dir.joinpath("cores").mkdir(parents=True)
    np.savez(
        cohort_dir / "cores" / f"{alias}.npz",
        coordinates_um=cohort_coordinates,
    )
    hl_root = tmp_path / "verified_hl"
    hl_file = hl_root / "embeddings" / "core_15_hL.npz"
    extraction._write_deterministic_npz(
        hl_file,
        {
            "cell_index": np.arange(5, dtype=np.int64),
            "core_number": np.asarray(15, dtype=np.int16),
            "coordinates_um": source_coordinates,
            "hL": source_hl,
        },
    )
    source_record = {
        "alias": alias,
        "core_number": 15,
        "cell_count": 5,
        "embedding_file": "embeddings/core_15_hL.npz",
        "hL_array_sha256": extraction._array_sha256("hL", source_hl),
    }
    expected = {
        "alias": alias,
        "core_number": 15,
        "cell_count": 5,
        "prepared_core_artifact_sha256": "a" * 64,
        "prepared_component_checksums": {},
        "graph_record_sha256": "b" * 64,
        "graph_logical_sha256": "c" * 64,
        "graph_file_checksums": {},
    }
    inputs = SimpleNamespace(
        run_id="synthetic",
        checkpoint_sha256="d" * 64,
        checkpoint_payload={
            "model_construction": {
                "class": "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
                "graph_layers": 4,
                "hidden_dim": 16,
            }
        },
        cohort_dir=cohort_dir,
        graph_dir=tmp_path / "graphs",
        project_root=tmp_path,
        provenance={},
    )
    batch = SimpleNamespace(
        alias=alias,
        n_nodes=5,
        target_expression=expression,
        node_covariates=covariates,
        edge_index=edges,
        relative_geometry=geometry,
    )
    monkeypatch.setattr(extraction, "SO2_CORE_NUMBERS", (15,))
    monkeypatch.setattr(extraction, "SO2_ALIASES", (alias,))
    monkeypatch.setattr(extraction, "EXPECTED_TOTAL_CELLS", 5)
    monkeypatch.setattr(
        extraction, "_expected_core_records", lambda _inputs: (expected,)
    )
    monkeypatch.setattr(
        extraction,
        "_load_verified_hl_extraction",
        lambda **_kwargs: (
            hl_root,
            {"cpu_threads": 1, "cores": [source_record]},
            "f" * 64,
        ),
    )
    monkeypatch.setattr(
        extraction, "load_so2_relative_qkv_batches", lambda **_kwargs: (batch,)
    )
    monkeypatch.setattr(extraction, "load_so2_checkpoint_model", lambda _inputs: model)

    with pytest.raises(
        extraction.SO2HiddenLayerProgressionExtractionError,
        match="rows or coordinates",
    ):
        extraction.extract_hidden_layer_progression(
            inputs=inputs,
            output_root=tmp_path / "progression",
            cpu_threads=1,
        )
    assert not (
        tmp_path
        / "progression"
        / "embeddings"
        / "core_15_hidden_layers.npz"
    ).exists()
