#!/usr/bin/env python3
"""Train the active seed-0 six-core relative-QKV model to loss plateau."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.train.run_full_core_capacity import _worker_archive_and_config  # noqa: E402
from spatial_benchmark.adjacency_ablation import (  # noqa: E402
    sample_uniform_mask_numpy,
)
from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES  # noqa: E402
from spatial_benchmark.cancer_relative_graphs import (  # noqa: E402
    load_cancer_relative_qkv_batches,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.masking import derive_mask_seed  # noqa: E402
from spatial_benchmark.pooled_relative_qkv_training import (  # noqa: E402
    MASK_BASE_SEED,
    PooledRelativeQKVEpochBoundaryResume,
    PooledRelativeQKVTrainingConfig,
    fit_pooled_relative_qkv_segment,
    single_seed_plateau_decision,
)
from spatial_benchmark.relative_qkv_graph_transformer import (  # noqa: E402
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    deidentify_prediction_rows,
)
from spatial_benchmark.training import _autocast_context  # noqa: E402


CAMPAIGN_ID = "cmp_20260824_cancer_6core_relative_qkv_multiseed"
ACTIVE_PROTOCOL = "held_in_pooled_6core_relative_qkv_seed_plateau"
ACTIVE_SEED = 0
ACTIVE_AMENDMENT_SHA256 = (
    "0ad3c6373edc45b1c61649217f043f486cfe3cfcf925bad042a2ed635032d174"
)
ACTIVE_AMENDMENT = (
    Path("experiments/campaigns")
    / CAMPAIGN_ID
    / "task_contract_amendment_004_seed0_first.yaml"
)
HELD_IN_MASK_BASE_SEED = 2026082491
HARDWARE_PREFLIGHT_RECEIPT = Path(
    "state/preflight/cancer_6core_relative_qkv_seed0.json"
)
HARDWARE_PREFLIGHT_SCHEMA = "cancer_6core_relative_qkv_hardware_preflight_v1"


class RelativeQKVRunnerError(RuntimeError):
    """Raised before the active seed-0 run can violate its contract."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise RelativeQKVRunnerError(f"Resolved config requires {name!r} mapping.")
    return value


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise RelativeQKVRunnerError(
            f"{field} must be {expected!r}; got {actual!r}."
        )


def _validate_active_contract(config: Mapping[str, Any]) -> None:
    _require_equal(config.get("seed"), ACTIVE_SEED, field="seed")
    campaign = _section(config, "campaign")
    _require_equal(campaign.get("campaign_id"), CAMPAIGN_ID, field="campaign_id")
    evaluation = _section(config, "evaluation")
    _require_equal(
        evaluation.get("protocol"), ACTIVE_PROTOCOL, field="evaluation.protocol"
    )
    model = _section(config, "model")
    locked_model = {
        "name": "relative-qkv-gat",
        "hidden_dim": 256,
        "graph_layers": 4,
        "attention_heads": 8,
        "attention_head_dim": 32,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "relative_geometry_dim": 70,
        "attention_dropout": 0.0,
        "activation_checkpointing": True,
        "uses_edge_inputs": False,
    }
    for field, expected in locked_model.items():
        _require_equal(model.get(field), expected, field=f"model.{field}")
    masking = _section(config, "masking")
    _require_equal(
        masking.get("independent_views_per_core_epoch"),
        10,
        field="masking.independent_views_per_core_epoch",
    )
    _require_equal(
        masking.get("model_seed_in_mask_derivation"),
        False,
        field="masking.model_seed_in_mask_derivation",
    )
    trainer = _section(config, "trainer")
    locked_trainer = {
        "minimum_global_epochs": 150,
        "continuation_block_global_epochs": 25,
        "mask_views_per_core_step": 10,
        "optimizer_steps_per_core_step": 1,
        "stage_complete_core_graph_on_device": True,
        "staged_relative_geometry_dtype": "float16",
        "early_stopping": False,
        "restore_best": False,
    }
    for field, expected in locked_trainer.items():
        _require_equal(trainer.get(field), expected, field=f"trainer.{field}")
    amendment = _PROJECT_ROOT / ACTIVE_AMENDMENT
    if not amendment.is_file() or sha256_file(amendment) != ACTIVE_AMENDMENT_SHA256:
        raise RelativeQKVRunnerError("Active seed-0 task amendment checksum mismatch.")


def _validate_hardware_preflight(
    config: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Require a checksum-valid preflight bound to current data and execution limits."""

    path = (
        (_PROJECT_ROOT / HARDWARE_PREFLIGHT_RECEIPT).resolve()
        if receipt_path is None
        else receipt_path.resolve()
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RelativeQKVRunnerError(
            f"A passing hardware preflight receipt is required: {path}."
        ) from exc
    if not isinstance(value, dict):
        raise RelativeQKVRunnerError("Hardware preflight receipt must be a mapping.")
    checksum = value.get("receipt_content_sha256")
    content = dict(value)
    content.pop("receipt_content_sha256", None)
    observed = hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if checksum != observed:
        raise RelativeQKVRunnerError("Hardware preflight receipt checksum mismatch.")
    if (
        value.get("schema") != HARDWARE_PREFLIGHT_SCHEMA
        or value.get("status") != "passed"
        or value.get("all_required_gates_passed") is not True
        or value.get("completed_experiment") is not False
    ):
        raise RelativeQKVRunnerError("Hardware preflight did not pass every gate.")
    resolved_config_sha256 = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if value.get("resolved_config_sha256") != resolved_config_sha256:
        raise RelativeQKVRunnerError("Resolved configuration changed after preflight.")
    dataset = _section(config, "dataset")
    cohort_dir = (_PROJECT_ROOT / str(dataset["prepared_artifact"])).resolve()
    graph_dir = (_PROJECT_ROOT / str(dataset["prepared_graph_artifact"])).resolve()
    if value.get("cohort_manifest_sha256") != sha256_file(
        cohort_dir / "manifest.json"
    ):
        raise RelativeQKVRunnerError("Preflight cohort manifest has changed.")
    if value.get("graph_manifest_sha256") != sha256_file(graph_dir / "manifest.json"):
        raise RelativeQKVRunnerError("Preflight graph manifest has changed.")
    model = _section(config, "model")
    receipt_model = value.get("model")
    if not isinstance(receipt_model, Mapping):
        raise RelativeQKVRunnerError("Preflight receipt lacks model execution settings.")
    for field in (
        "receiver_chunk_size",
        "max_edges_per_chunk",
        "activation_checkpointing",
    ):
        if receipt_model.get(field) != model.get(field):
            raise RelativeQKVRunnerError(
                f"Preflight execution setting {field} no longer matches the run."
            )
    if receipt_model.get("stage_complete_core_graph_on_device") != _section(
        config, "trainer"
    ).get("stage_complete_core_graph_on_device"):
        raise RelativeQKVRunnerError(
            "Preflight graph-staging strategy no longer matches the run."
        )
    if receipt_model.get("staged_relative_geometry_dtype") != _section(
        config, "trainer"
    ).get("staged_relative_geometry_dtype"):
        raise RelativeQKVRunnerError(
            "Preflight staged geometry dtype no longer matches the run."
        )
    return value


def _model_from_config(
    config: Mapping[str, Any],
    *,
    num_genes: int,
    node_covariate_dim: int,
) -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    model = _section(config, "model")
    return ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=num_genes,
        node_covariate_dim=node_covariate_dim,
        hidden_dim=int(model["hidden_dim"]),
        attention_heads=int(model["attention_heads"]),
        attention_head_dim=int(model["attention_head_dim"]),
        graph_layers=int(model["graph_layers"]),
        ffn_dim=int(model["ffn_dim"]),
        decoder_dim=int(model["decoder_dim"]),
        positional_bias_hidden_dim=int(model["positional_bias_hidden_dim"]),
        dropout=float(model["dropout"]),
        attention_dropout=float(model["attention_dropout"]),
        relative_geometry_dim=int(model["relative_geometry_dim"]),
        receiver_chunk_size=int(model["receiver_chunk_size"]),
        max_edges_per_chunk=int(model["max_edges_per_chunk"]),
        activation_checkpointing=bool(model["activation_checkpointing"]),
    )


def _training_config(
    config: Mapping[str, Any],
    *,
    start_epoch: int,
    end_epoch: int,
) -> PooledRelativeQKVTrainingConfig:
    trainer = _section(config, "trainer")
    masking = _section(config, "masking")
    return PooledRelativeQKVTrainingConfig(
        model_seed=ACTIVE_SEED,
        segment_start_global_epoch=start_epoch,
        segment_end_global_epoch=end_epoch,
        learning_rate=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
        gradient_clip_norm=float(trainer["gradient_clip_norm"]),
        huber_delta=float(trainer["huber_delta"]),
        mask_base_seed=int(masking["mask_base_seed"]),
        core_order_seed=int(trainer["core_order_seed"]),
        amp=bool(trainer["amp"]),
        amp_dtype=str(trainer["amp_dtype"]),
        deterministic=bool(trainer["deterministic"]),
        deterministic_warn_only=bool(trainer["deterministic_warn_only"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        stage_complete_core_graph_on_device=bool(
            trainer["stage_complete_core_graph_on_device"]
        ),
        staged_relative_geometry_dtype=str(
            trainer["staged_relative_geometry_dtype"]
        ),
        checkpoint_interval_global_epochs=int(
            trainer["checkpoint_every_global_epochs"]
        ),
    )


def _checkpoint_payload(
    *,
    archive: RunArchive,
    config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    parameter_count: int,
    resume: PooledRelativeQKVEpochBoundaryResume,
    plateau: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema": "cancer_6core_relative_qkv_resume_v1",
        "run_id": archive.run_id,
        "campaign_id": CAMPAIGN_ID,
        "model_seed": ACTIVE_SEED,
        "completed_global_epochs": resume.completed_global_epochs,
        "optimizer_steps_completed": resume.optimizer_steps_completed,
        "mask_base_seed": resume.mask_base_seed,
        "core_order_seed": resume.core_order_seed,
        "mask_views_per_core_step": resume.mask_views_per_core_step,
        "model_construction": dict(model_construction),
        "parameter_count": int(parameter_count),
        "model_state_dict": resume.model_state_dict,
        "model_state_checksum": resume.model_state_checksum,
        "optimizer_state_dict": resume.optimizer_state_dict,
        "optimizer_state_checksum": resume.optimizer_state_checksum,
        "amp_scaler_state_dict": resume.scaler_state_dict,
        "amp_scaler_state_checksum": resume.scaler_state_checksum,
        "core_history": [asdict(record) for record in resume.core_history],
        "global_history": [asdict(record) for record in resume.global_history],
        "history_checksum": resume.history_checksum,
        "resume_checksum": resume.resume_checksum,
        "resolved_config": dict(config),
        "active_amendment_sha256": ACTIVE_AMENDMENT_SHA256,
        "plateau": None if plateau is None else dict(plateau),
    }


def _checkpoint_bytes(payload: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    return stream.getvalue()


def _seed_plateau_decision(losses: Sequence[float]) -> dict[str, Any]:
    decision = single_seed_plateau_decision(
        losses,
        model_seed=ACTIVE_SEED,
        completed_global_epochs=len(losses),
    )
    return {
        **asdict(decision),
        "validation_or_test_metric": False,
        "checkpoint_selection_metric": False,
    }


def _held_in_diagnostics(
    model: torch.nn.Module,
    batches: Sequence[Any],
    *,
    amp: bool,
    amp_dtype: str,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    model.eval()
    core_metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in batches:
            seed = derive_mask_seed(
                HELD_IN_MASK_BASE_SEED,
                "relative-qkv-held-in-fit-diagnostic",
                batch.alias,
            )
            realization = sample_uniform_mask_numpy(
                batch.n_nodes,
                batch.n_genes,
                seed=seed,
            )
            retry = 0
            while int(realization.masked_gene_counts.sum()) == 0:
                retry += 1
                seed = derive_mask_seed(seed, "nonzero-held-in", retry)
                realization = sample_uniform_mask_numpy(
                    batch.n_nodes,
                    batch.n_genes,
                    seed=seed,
                )
            target = batch.target_expression.to(device=device)
            covariates = batch.node_covariates.to(device=device)
            mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(
                device=device, dtype=torch.bool
            )
            with _autocast_context(
                enabled=amp,
                device=device,
                dtype_name=amp_dtype,
            ):
                output = model(
                    input_expression=target.masked_fill(mask, 0.0),
                    gene_mask=mask,
                    edge_index=batch.edge_index,
                    relative_geometry=batch.relative_geometry,
                    node_covariates=covariates,
                )
            prediction = output.prediction.float()
            truth = target.float()
            selected_prediction = prediction[mask]
            selected_truth = truth[mask]
            residual = selected_prediction - selected_truth
            huber = F.huber_loss(
                selected_prediction,
                selected_truth,
                delta=1.0,
                reduction="mean",
            )
            mae = residual.abs().mean()
            mse = residual.square().mean()
            centered = selected_truth - selected_truth.mean()
            denominator = centered.square().sum()
            r2 = 1.0 - residual.square().sum() / denominator.clamp_min(
                torch.finfo(torch.float32).tiny
            )
            values = {
                "alias": batch.alias,
                "mask_seed": int(seed),
                "mask_checksum": realization.checksum,
                "n_masked_entries": int(mask.sum().item()),
                "masked_huber": float(huber.cpu()),
                "masked_mae": float(mae.cpu()),
                "masked_mse": float(mse.cpu()),
                "masked_r2": float(r2.cpu()),
                "mean_y_true": float(selected_truth.mean().cpu()),
                "mean_y_pred": float(selected_prediction.mean().cpu()),
            }
            if not all(
                math.isfinite(values[name])
                for name in ("masked_huber", "masked_mae", "masked_mse", "masked_r2")
            ):
                raise FloatingPointError("Held-in fit diagnostics are non-finite.")
            core_metrics.append(values)
            prediction_rows.append(
                {
                    "core_alias": batch.alias,
                    "dataset_id": "cosmx_cancer_6core_pooled_fit_v1",
                    "split": "fit",
                    "y_true": values["mean_y_true"],
                    "y_pred": values["mean_y_pred"],
                    "sample_loss": values["masked_huber"],
                    "node_count": batch.n_nodes,
                    "edge_count": batch.n_edges,
                }
            )
            del target, covariates, mask, output, prediction, truth
            if device.type == "cuda":
                torch.cuda.empty_cache()
    metrics = {
        "fit/uniform_per_cell/masked_huber": float(
            np.mean([row["masked_huber"] for row in core_metrics])
        ),
        "fit/uniform_per_cell/masked_mae": float(
            np.mean([row["masked_mae"] for row in core_metrics])
        ),
        "fit/uniform_per_cell/masked_mse": float(
            np.mean([row["masked_mse"] for row in core_metrics])
        ),
        "fit/uniform_per_cell/masked_r2": float(
            np.mean([row["masked_r2"] for row in core_metrics])
        ),
    }
    return metrics, core_metrics, prediction_rows


def run_seed0_to_plateau(
    config: Mapping[str, Any],
    archive: RunArchive,
) -> dict[str, Any]:
    """Train seed 0 in deterministic segments until two plateau audits pass."""

    started = time.monotonic()
    _validate_active_contract(config)
    preflight = _validate_hardware_preflight(config)
    dataset = _section(config, "dataset")
    cohort_dir = (_PROJECT_ROOT / str(dataset["prepared_artifact"])).resolve()
    graph_dir = (_PROJECT_ROOT / str(dataset["prepared_graph_artifact"])).resolve()
    batches = load_cancer_relative_qkv_batches(
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    if tuple(batch.alias for batch in batches) != CANCER_ALIASES:
        raise RelativeQKVRunnerError("Loaded batches do not match the six-core order.")
    archive.write_json("diagnostics/hardware_preflight.json", preflight)
    model = _model_from_config(
        config,
        num_genes=batches[0].n_genes,
        node_covariate_dim=int(batches[0].node_covariates.shape[1]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_config = _section(config, "model")
    model_construction = {
        "class": type(model).__name__,
        "num_genes": batches[0].n_genes,
        "node_covariate_dim": int(batches[0].node_covariates.shape[1]),
        **dict(model_config),
    }

    written_epochs: set[int] = set()

    def save_periodic(resume: PooledRelativeQKVEpochBoundaryResume) -> None:
        epoch = resume.completed_global_epochs
        if epoch in written_epochs:
            return
        archive.write_bytes(
            f"checkpoints/epoch_{epoch:04d}.ckpt",
            _checkpoint_bytes(
                _checkpoint_payload(
                    archive=archive,
                    config=config,
                    model_construction=model_construction,
                    parameter_count=parameter_count,
                    resume=resume,
                )
            ),
        )
        written_epochs.add(epoch)

    resume = None
    start_epoch = 0
    end_epoch = int(_section(config, "trainer")["minimum_global_epochs"])
    plateau: dict[str, Any] | None = None
    final_result = None
    while True:
        final_result = fit_pooled_relative_qkv_segment(
            model,
            batches,
            _training_config(
                config,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
            ),
            resume=resume,
            checkpoint_callback=save_periodic,
        )
        losses = [
            record.equal_core_mean_masked_huber
            for record in final_result.global_history
        ]
        plateau = _seed_plateau_decision(losses)
        archive.write_json(
            f"diagnostics/plateau_audit_epoch_{end_epoch:04d}.json",
            plateau,
        )
        if plateau["should_stop"]:
            break
        resume = final_result.resume
        start_epoch = end_epoch
        end_epoch += int(
            _section(config, "trainer")["continuation_block_global_epochs"]
        )

    assert final_result is not None and plateau is not None
    final_payload = _checkpoint_payload(
        archive=archive,
        config=config,
        model_construction=model_construction,
        parameter_count=parameter_count,
        resume=final_result.resume,
        plateau=plateau,
    )
    checkpoint_path = archive.write_bytes(
        "checkpoints/last.ckpt", _checkpoint_bytes(final_payload)
    )

    evaluation = _section(config, "evaluation")
    held_in, core_diagnostics, prediction_rows = _held_in_diagnostics(
        model,
        batches,
        amp=bool(_section(config, "trainer")["amp"]),
        amp_dtype=str(_section(config, "trainer")["amp_dtype"]),
    )
    salt = os.environ.get("BAGM_SAMPLE_KEY_SALT", "").strip()
    if len(salt.encode("utf-8")) < 16:
        salt = hashlib.sha256(
            f"{archive.run_id}:{ACTIVE_AMENDMENT_SHA256}".encode("utf-8")
        ).hexdigest()
    deidentified = deidentify_prediction_rows(
        prediction_rows,
        identifier_fields=["core_alias"],
        salt=salt,
        namespace="cancer-6core-held-in-alias",
    )
    for row in deidentified:
        row["run_id"] = archive.run_id
    archive.write_predictions("fit", deidentified)

    global_rows = [
        {
            "run_id": archive.run_id,
            "split": "fit",
            **asdict(record),
        }
        for record in final_result.global_history
    ]
    archive.write_table("metrics/history", global_rows)
    archive.write_table(
        "metrics/core_steps",
        [
            {
                "run_id": archive.run_id,
                "split": "fit",
                **{
                    key: value
                    for key, value in asdict(record).items()
                    if key != "mask_views"
                },
                "mask_view_checksums": [
                    view.mask_checksum_sha256 for view in record.mask_views
                ],
            }
            for record in final_result.core_history
        ],
    )
    archive.write_json("diagnostics/held_in_fit_metrics_by_core.json", core_diagnostics)
    final_metrics = {
        **held_in,
        "fit/training/final_equal_core_masked_huber": float(
            final_result.final_train_loss
        ),
        "fit/training/final_global_epoch": float(
            final_result.completed_global_epochs
        ),
    }
    archive.write_json("metrics/final.json", final_metrics)
    for name, value in final_metrics.items():
        archive.append_metric_event(
            {
                "name": name,
                "value": value,
                "step": final_result.completed_global_epochs,
            }
        )
    archive.write_json(
        "provenance/relative_qkv_training.json",
        {
            "active_amendment": str(ACTIVE_AMENDMENT),
            "active_amendment_sha256": ACTIVE_AMENDMENT_SHA256,
            "model_seed": ACTIVE_SEED,
            "deferred_model_seeds": [1, 2, 3, 4],
            "five_seed_campaign_complete": False,
            "mask_base_seed": MASK_BASE_SEED,
            "held_in_mask_base_seed": HELD_IN_MASK_BASE_SEED,
            "parameter_count": parameter_count,
            "model_construction": model_construction,
            "completed_global_epochs": final_result.completed_global_epochs,
            "optimizer_steps_completed": final_result.optimizer_steps_completed,
            "mask_views_per_core_step": final_result.mask_views_per_core_step,
            "state_dict_sha256": final_result.final_state_checksum,
            "history_sha256": final_result.history_checksum,
            "plateau": plateau,
            "cohort_manifest_sha256": sha256_file(cohort_dir / "manifest.json"),
            "graph_manifest_sha256": sha256_file(graph_dir / "manifest.json"),
            "hardware_preflight_receipt_sha256": preflight[
                "receipt_content_sha256"
            ],
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    )
    peak_vram_gib = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    summary = {
        "run_id": archive.run_id,
        "status": "success",
        "campaign_id": CAMPAIGN_ID,
        "model_name": "relative-qkv-gat",
        "model_seed": ACTIVE_SEED,
        "final_epoch": final_result.completed_global_epochs,
        "optimizer_steps": final_result.optimizer_steps_completed,
        "parameter_count": parameter_count,
        "primary_metric_name": evaluation["primary_metric"],
        "primary_metric_value": held_in[evaluation["primary_metric"]],
        "peak_vram_gib": peak_vram_gib,
        "duration_seconds": time.monotonic() - started,
        "checkpoint": "checkpoints/last.ckpt",
        "plateau_confirmed": True,
        "five_seed_campaign_complete": False,
        "deferred_model_seeds": [1, 2, 3, 4],
        "generalization_estimate": False,
    }
    archive.write_summary(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the active seed-0 six-core relative-QKV model to plateau."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    summary = run_seed0_to_plateau(config, archive)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
