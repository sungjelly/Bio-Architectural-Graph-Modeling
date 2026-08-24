from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.pooled_relative_qkv_training import (
    CORE_ORDER_SEED,
    MASK_BASE_SEED,
    MASK_VIEWS_PER_CORE_STEP,
    MODEL_STEP_RNG_DERIVATION,
    PooledRelativeQKVCoreBatch,
    PooledRelativeQKVCoreStepRecord,
    PooledRelativeQKVGlobalEpochRecord,
    PooledRelativeQKVMaskViewRecord,
    _history_checksum,
    _payload_sha256,
    _tree_sha256,
    relative_qkv_core_order,
    relative_qkv_mask_seed,
    relative_qkv_model_step_seed,
    single_seed_plateau_decision,
)
from spatial_benchmark.relative_qkv_checkpoint_verification import (
    ALLOWED_MODEL_SEEDS,
    VERIFICATION_SCHEMA,
    RelativeQKVCheckpointVerificationError,
    verify_relative_qkv_checkpoint,
    write_verification_receipt,
)
from spatial_benchmark.relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from spatial_benchmark.relative_qkv_post_training import (
    CAMPAIGN_ID,
    CHECKPOINT_SCHEMA,
)


def _batches() -> tuple[PooledRelativeQKVCoreBatch, ...]:
    batches: list[PooledRelativeQKVCoreBatch] = []
    pairs = [
        (source, receiver)
        for receiver in range(3)
        for source in range(3)
        if source != receiver
    ]
    edges = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    for index, alias in enumerate(CANCER_ALIASES):
        generator = torch.Generator().manual_seed(100 + index)
        batches.append(
            PooledRelativeQKVCoreBatch(
                alias=alias,
                target_expression=torch.randn(3, 5, generator=generator),
                edge_index=edges.clone(),
                relative_geometry=torch.randn(
                    len(pairs), 70, generator=generator
                ),
                node_covariates=torch.randn(3, 2, generator=generator),
            )
        )
    return tuple(batches)


def _model() -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    torch.manual_seed(711)
    return ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=5,
        node_covariate_dim=2,
        hidden_dim=12,
        attention_heads=3,
        attention_head_dim=4,
        graph_layers=2,
        ffn_dim=20,
        decoder_dim=9,
        positional_bias_hidden_dim=7,
        dropout=0.0,
        attention_dropout=0.0,
        relative_geometry_dim=70,
        receiver_chunk_size=2,
        max_edges_per_chunk=6,
        activation_checkpointing=False,
    ).eval()


def _construction() -> dict[str, object]:
    return {
        "class": "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
        "num_genes": 5,
        "node_covariate_dim": 2,
        "hidden_dim": 12,
        "attention_heads": 3,
        "attention_head_dim": 4,
        "graph_layers": 2,
        "ffn_dim": 20,
        "decoder_dim": 9,
        "positional_bias_hidden_dim": 7,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "relative_geometry_dim": 70,
        "receiver_chunk_size": 2,
        "max_edges_per_chunk": 6,
        "activation_checkpointing": False,
    }


def _write_manifests(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str, str]:
    cohort = {
        "format_version": 1,
        "cohort": {
            "aliases": list(CANCER_ALIASES),
            "validation_or_test_partition_present": False,
        },
        "features": {
            "gene_names": [f"gene-{index}" for index in range(5)],
            "model_covariate_names": ["area", "intensity"],
            "coordinates_are_model_covariates": False,
        },
        "manifest_content_sha256": "1" * 64,
    }
    cohort_path = tmp_path / "cohort_manifest.json"
    cohort_path.write_text(json.dumps(cohort, sort_keys=True), encoding="utf-8")
    cohort_sha = sha256_file(cohort_path)
    graph = {
        "format_version": 1,
        "aliases": list(CANCER_ALIASES),
        "cohort_manifest_sha256": cohort_sha,
        "manifest_content_sha256": "2" * 64,
    }
    graph_path = tmp_path / "graph_manifest.json"
    graph_path.write_text(json.dumps(graph, sort_keys=True), encoding="utf-8")
    graph_sha = sha256_file(graph_path)
    amendment_path = tmp_path / "task_contract_amendment_test.yaml"
    amendment_path.write_text("schema: synthetic-verifier-test\n", encoding="utf-8")
    return cohort_path, graph_path, amendment_path, cohort_sha, graph_sha


def _histories(
    batches: tuple[PooledRelativeQKVCoreBatch, ...],
    *,
    model_seed: int,
    completed: int = 175,
) -> tuple[
    tuple[PooledRelativeQKVCoreStepRecord, ...],
    tuple[PooledRelativeQKVGlobalEpochRecord, ...],
]:
    by_alias = {batch.alias: batch for batch in batches}
    core_records: list[PooledRelativeQKVCoreStepRecord] = []
    global_records: list[PooledRelativeQKVGlobalEpochRecord] = []
    for epoch in range(completed):
        order = relative_qkv_core_order(epoch)
        for step, alias in enumerate(order):
            batch = by_alias[alias]
            views: list[PooledRelativeQKVMaskViewRecord] = []
            for view_index in range(MASK_VIEWS_PER_CORE_STEP):
                seed = relative_qkv_mask_seed(
                    alias, epoch, view_index=view_index
                )
                checksum = hashlib.sha256(
                    f"{alias}:{epoch}:{view_index}".encode("utf-8")
                ).hexdigest()
                views.append(
                    PooledRelativeQKVMaskViewRecord(
                        view_index=view_index,
                        initial_mask_seed=seed,
                        effective_mask_seed=seed,
                        zero_total_mask_resamples=0,
                        mask_checksum_sha256=checksum,
                        n_masked_entries=batch.n_nodes * 2,
                        masked_count_min=2,
                        masked_count_mean=2.0,
                        masked_count_median=2.0,
                        masked_count_max=2,
                        zero_mask_cells=0,
                        full_mask_cells=0,
                        masked_huber_loss=1.0,
                    )
                )
            core_records.append(
                PooledRelativeQKVCoreStepRecord(
                    global_epoch=epoch,
                    completed_global_epoch=epoch + 1,
                    step_in_epoch=step,
                    optimizer_step=epoch * len(CANCER_ALIASES) + step + 1,
                    alias=alias,
                    n_nodes=batch.n_nodes,
                    n_edges=batch.n_edges,
                    mask_views=tuple(views),
                    n_mask_views=MASK_VIEWS_PER_CORE_STEP,
                    n_masked_entries_across_views=(
                        MASK_VIEWS_PER_CORE_STEP * batch.n_nodes * 2
                    ),
                    model_step_seed=relative_qkv_model_step_seed(
                        model_seed, epoch, alias
                    ),
                    masked_huber_loss=1.0,
                    gradient_norm=0.5,
                )
            )
        global_records.append(
            PooledRelativeQKVGlobalEpochRecord(
                global_epoch=epoch,
                completed_global_epochs=epoch + 1,
                ordered_aliases=order,
                optimizer_steps_this_epoch=len(CANCER_ALIASES),
                cumulative_optimizer_steps=(epoch + 1) * len(CANCER_ALIASES),
                equal_core_mean_masked_huber=1.0,
            )
        )
    return tuple(core_records), tuple(global_records)


def _checkpoint_fixture(
    tmp_path: Path,
    *,
    model_seed: int,
) -> tuple[
    Path,
    tuple[PooledRelativeQKVCoreBatch, ...],
    Path,
    Path,
    Path,
]:
    batches = _batches()
    model = _model()
    cohort_path, graph_path, amendment_path, cohort_sha, graph_sha = (
        _write_manifests(tmp_path)
    )
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer_state = optimizer.state_dict()
    scaler_state: dict[str, object] = {}
    core_history, global_history = _histories(
        batches, model_seed=model_seed
    )
    model_checksum = _tree_sha256(state)
    optimizer_checksum = _tree_sha256(optimizer_state)
    scaler_checksum = _tree_sha256(scaler_state)
    history_checksum = _history_checksum(core_history, global_history)
    completed = len(global_history)
    resume_checksum = _payload_sha256(
        {
            "schema": "pooled_relative_qkv_epoch_boundary_resume_v1",
            "completed_global_epochs": completed,
            "optimizer_steps_completed": completed * len(CANCER_ALIASES),
            "model_seed": model_seed,
            "mask_base_seed": MASK_BASE_SEED,
            "core_order_seed": CORE_ORDER_SEED,
            "mask_views_per_core_step": MASK_VIEWS_PER_CORE_STEP,
            "model_state_checksum": model_checksum,
            "optimizer_state_checksum": optimizer_checksum,
            "scaler_state_checksum": scaler_checksum,
            "history_checksum": history_checksum,
            "model_step_rng_derivation": MODEL_STEP_RNG_DERIVATION,
        }
    )
    plateau = {
        **asdict(
            single_seed_plateau_decision(
                [1.0] * completed,
                model_seed=model_seed,
                completed_global_epochs=completed,
            )
        ),
        "validation_or_test_metric": False,
        "checkpoint_selection_metric": False,
    }
    config = {
        "campaign": {"campaign_id": CAMPAIGN_ID},
        "seed": model_seed,
        "dataset": {
            "cohort_manifest_file_sha256": cohort_sha,
            "graph_manifest_file_sha256": graph_sha,
            "core_aliases": list(CANCER_ALIASES),
            "validation_or_test_partition_present": False,
            "generalization_claim_supported": False,
        },
        "trainer": {
            "minimum_global_epochs": 150,
            "continuation_block_global_epochs": 25,
            "mask_views_per_core_step": 10,
            "optimizer_steps_per_core_step": 1,
            "steps_per_global_epoch": 6,
            "early_stopping": False,
            "restore_best": False,
            "amp": False,
            "amp_dtype": "auto",
            "stage_complete_core_graph_on_device": False,
            "staged_relative_geometry_dtype": "float32",
        },
        "masking": {
            "mask_base_seed": MASK_BASE_SEED,
            "independent_views_per_core_epoch": 10,
            "model_seed_in_mask_derivation": False,
        },
    }
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "run_id": f"r_synthetic_seed_{model_seed}",
        "campaign_id": CAMPAIGN_ID,
        "model_seed": model_seed,
        "completed_global_epochs": completed,
        "optimizer_steps_completed": completed * len(CANCER_ALIASES),
        "mask_base_seed": MASK_BASE_SEED,
        "core_order_seed": CORE_ORDER_SEED,
        "mask_views_per_core_step": MASK_VIEWS_PER_CORE_STEP,
        "model_construction": _construction(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_state_dict": state,
        "model_state_checksum": model_checksum,
        "optimizer_state_dict": optimizer_state,
        "optimizer_state_checksum": optimizer_checksum,
        "amp_scaler_state_dict": scaler_state,
        "amp_scaler_state_checksum": scaler_checksum,
        "core_history": [asdict(record) for record in core_history],
        "global_history": [asdict(record) for record in global_history],
        "history_checksum": history_checksum,
        "resume_checksum": resume_checksum,
        "resolved_config": config,
        "active_amendment_sha256": sha256_file(amendment_path),
        "plateau": plateau,
    }
    checkpoint = tmp_path / f"seed_{model_seed}_last.ckpt"
    torch.save(payload, checkpoint)
    return checkpoint, batches, cohort_path, graph_path, amendment_path


@pytest.mark.parametrize("model_seed", ALLOWED_MODEL_SEEDS)
def test_verifier_reloads_and_replays_each_active_seed(
    tmp_path: Path,
    model_seed: int,
) -> None:
    checkpoint, batches, cohort, graph, _amendment = _checkpoint_fixture(
        tmp_path, model_seed=model_seed
    )
    receipt = verify_relative_qkv_checkpoint(
        checkpoint,
        batches,
        cohort_manifest_path=cohort,
        graph_manifest_path=graph,
        amendment_path=tmp_path,
        device="cpu",
        attention_receivers_per_core=2,
        enforce_production_contract=False,
    )
    assert receipt["schema"] == VERIFICATION_SCHEMA
    assert receipt["status"] == "passed"
    assert receipt["model_seed"] == model_seed
    assert receipt["plateau_verification"]["recomputed_decision"]["should_stop"]
    assert receipt["checkpoint"]["completed_global_epochs"] == 175
    assert len(receipt["held_in_fit_replay"]["per_core"]) == 6
    for row in receipt["held_in_fit_replay"]["per_core"]:
        assert row["prediction_replay"]["byte_identical"] is True
        attention = row["selected_attention_replay"]
        assert len(attention["selected_receivers"]) == 2
        assert attention["channels"]["attention"]["byte_identical"] is True
        assert attention["maximum_receiver_head_normalization_error"] < 1e-6

    output = tmp_path / f"seed_{model_seed}_verification.json"
    assert write_verification_receipt(receipt, output) == output.resolve()
    stored = json.loads(output.read_text(encoding="utf-8"))
    checksum = stored.pop("receipt_content_sha256")
    canonical = json.dumps(
        stored,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert checksum == hashlib.sha256(canonical).hexdigest()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_verification_receipt(receipt, output)


def test_verifier_rejects_tampered_plateau_and_unapproved_seed(
    tmp_path: Path,
) -> None:
    checkpoint, batches, cohort, graph, amendment = _checkpoint_fixture(
        tmp_path, model_seed=0
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["plateau"]["should_stop"] = False
    tampered = tmp_path / "tampered_plateau.ckpt"
    torch.save(payload, tampered)
    with pytest.raises(
        RelativeQKVCheckpointVerificationError,
        match="independent recomputation",
    ):
        verify_relative_qkv_checkpoint(
            tampered,
            batches,
            cohort_manifest_path=cohort,
            graph_manifest_path=graph,
            amendment_path=amendment,
            enforce_production_contract=False,
        )

    payload["plateau"]["should_stop"] = True
    payload["model_seed"] = 4
    disallowed = tmp_path / "disallowed_seed.ckpt"
    torch.save(payload, disallowed)
    with pytest.raises(
        RelativeQKVCheckpointVerificationError,
        match="must be one of",
    ):
        verify_relative_qkv_checkpoint(
            disallowed,
            batches,
            cohort_manifest_path=cohort,
            graph_manifest_path=graph,
            amendment_path=amendment,
            enforce_production_contract=False,
        )
