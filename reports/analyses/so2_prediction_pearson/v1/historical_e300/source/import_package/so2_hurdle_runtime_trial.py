"""Narrow, checkpoint-free contract for the registered SO2 runtime pilot."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .fingerprints import sha256_file
from .identifiers import canonical_sha256

CAMPAIGN = "cmp_20261001_so2_hurdle_gpu_scaling"
PROTOCOL = "so2_hurdle_single_gpu_runtime_trial_v1"
SOURCE_RUN = "r_20260928T021117Z_9ea46874_s000_f00_a01_41b54097"
SOURCE_CONFIG_SHA = "07d79245f5d36082975444b87edf2f13ba20be79155c94dbe80cf34d22135482"
SOURCE_SUCCESS_SHA = "709a571ce6441684578448aabd85edd5bcbe9f00609c87261afc44374f1f90c9"
PRIMARY_METRIC = "runtime/epoch_seconds"
GIB = 1024**3


def is_runtime_trial(config: Mapping[str, Any]) -> bool:
    evaluation = config.get("evaluation", {})
    return isinstance(evaluation, Mapping) and evaluation.get("protocol") == PROTOCOL


def build_configuration(source: Mapping[str, Any], source_bundle: Mapping[str, Any],
                        *, gpu: str, epochs: int = 2) -> dict[str, Any]:
    config = deepcopy(dict(source))
    config.update({
        "campaign": {"campaign_id": CAMPAIGN, "display_name": "SO2 hurdle GPU runtime trial"},
        "classification": {
            "schema_version": 1, "lifecycle_stage": "diagnostic",
            "study_axis": "runtime_resource_equivalence",
            "scientific_variant": "so2_hurdle_single_gpu_two_epoch_trial",
            "retention_class": "retain_diagnostic_runtime_evidence",
            "classification_confidence": "high",
        },
        "experiment": {
            "estimand": "runtime_and_numerical_correctness_of_partial_gene_count_reconstruction",
            "permitted_claim": "single_gpu_runtime_feasibility_only",
            "variant_label": "so2_hurdle_single_gpu_two_epoch_trial",
        },
        "evaluation": {
            "protocol": PROTOCOL, "artifact_contract": "runtime_diagnostic",
            "canonical_prediction_split": "runtime_diagnostic",
            "primary_metric": PRIMARY_METRIC, "primary_direction": "minimize",
            "generalization_estimate": False,
        },
        "launcher": {"requested_gpu": str(gpu), "process_count": 1,
                     "elastic_max_restarts": 0},
        "runtime": {"gpu": str(gpu), "epochs": epochs, "cpu_threads": 4,
                    "effective_world_size": 1, "min_free_gib": 20},
        "metadata": {"runtime_diagnostic_only": True, "fresh_initialization": True,
                     "no_scientific_checkpoint_selection": True},
        "source_configuration": deepcopy(dict(source)),
        "source_bundle": deepcopy(dict(source_bundle)),
    })
    return config


def validate_configuration(config: Mapping[str, Any]) -> None:
    """Reject scientific changes and exception reuse outside this one pilot."""
    source = config.get("source_configuration")
    bundle = config.get("source_bundle")
    runtime = config.get("runtime")
    if not isinstance(source, Mapping) or canonical_sha256(source) != SOURCE_CONFIG_SHA:
        raise ValueError("Runtime trial requires the exact frozen hurdle source configuration.")
    if not isinstance(bundle, Mapping) or bundle.get("run_id") != SOURCE_RUN:
        raise ValueError("Runtime trial requires its immutable source run.")
    if bundle.get("success_marker_sha256") != SOURCE_SUCCESS_SHA:
        raise ValueError("Runtime trial source completion marker differs from the frozen source.")
    for key in ("config_file_sha256", "manifest_sha256", "checksum_manifest_sha256",
                "data_receipt_sha256", "split_receipt_sha256"):
        value = bundle.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Runtime trial requires source {key}.")
    if not isinstance(runtime, Mapping):
        raise ValueError("Runtime trial requires a runtime mapping.")
    gpu = runtime.get("gpu")
    if gpu not in {"0", "1", "2", "3"} or runtime.get("epochs") != 2:
        raise ValueError("Runtime trial supports one explicit GPU 0..3 and exactly two epochs.")
    expected = build_configuration(source, bundle, gpu=gpu, epochs=2)
    # Attempts are execution metadata, never scientific changes.
    attempt = config.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("Runtime trial requires a positive attempt.")
    expected["attempt"] = attempt
    if dict(config) != expected:
        raise ValueError("Runtime trial configuration differs from its bounded inherited contract.")


def validate_outputs(root: Path, run_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    validate_configuration(config)
    required = ("diagnostics/runtime_trial_result.json", "diagnostics/runtime_environment.json",
                "diagnostics/runtime_resources.jsonl", "metrics/runtime_epoch_metrics.jsonl",
                "metrics/runtime_progress.jsonl", "provenance/source_bundle.json")
    for relative in required:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Runtime trial is missing {relative}.")
    result = json.loads((root / required[0]).read_text())
    if (result.get("schema") != "so2_hurdle_gpu_runtime_trial_result_v1"
            or result.get("run_id") != run_id or result.get("status") != "passed"
            or result.get("epochs_requested") != 2 or result.get("epochs_completed") != 2
            or result.get("effective_world_size") != 1
            or result.get("correctness_gate_passed") is not True
            or result.get("resource_gate_passed") is not True):
        raise ValueError("Runtime trial result did not pass the bounded correctness/resource gates.")
    epochs = result.get("epochs")
    history = [json.loads(line) for line in (root / required[3]).read_text().splitlines() if line.strip()]
    if not isinstance(epochs, list) or len(epochs) != 2 or epochs != history:
        raise ValueError("Runtime trial terminal epochs disagree with append-only epoch metrics.")
    for expected_epoch, epoch in enumerate(epochs, 1):
        if (epoch.get("epoch") != expected_epoch or epoch.get("optimizer_updates") != 6
                or epoch.get("train_target_cells") != 208696
                or epoch.get("validation_target_cells") != 37367
                or epoch.get("finite_gradients") is not True
                or epoch.get("resource_gate_passed") is not True):
            raise ValueError("Runtime trial epoch coverage, update count or gradient gate is invalid.")
        for key in ("duration_seconds", "training_duration_seconds", "validation_duration_seconds",
                    "train_loss", "validation_loss", "train_masked_entries", "validation_masked_entries"):
            value = epoch.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Runtime trial epoch requires finite positive {key}.")
        receipt = f"diagnostics/coverage_epoch_{expected_epoch:03d}.json"
        if epoch.get("coverage_file") != receipt or sha256_file(root / receipt) != epoch.get("coverage_sha256"):
            raise ValueError("Runtime trial coverage receipt checksum does not match.")
    for values in [result, *epochs]:
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "total_device_bytes", "headroom_bytes"):
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Runtime trial requires nonnegative integer {key}.")
        if (values["peak_allocated_bytes"] > 22 * GIB
                or values["peak_reserved_bytes"] > 22 * GIB
                or values["headroom_bytes"] < 2 * GIB
                or values["headroom_bytes"] > values["total_device_bytes"] - values["peak_reserved_bytes"]):
            raise ValueError("Runtime trial exceeds the frozen VRAM/headroom gate.")
    if not epochs[0].get("validation_mask_receipts_sha256") or epochs[0]["validation_mask_receipts_sha256"] != epochs[1].get("validation_mask_receipts_sha256"):
        raise ValueError("Runtime trial fixed validation mask receipts disagree across epochs.")
    if json.loads((root / "provenance/source_bundle.json").read_text()) != config["source_bundle"]:
        raise ValueError("Runtime trial source provenance disagrees with resolved configuration.")
    if any((root / "checkpoints").rglob("*")) or any((root / "predictions").rglob("*")):
        raise ValueError("Runtime-only trial may not contain predictions or checkpoints.")
    return result


def adapt_outputs(archive: Any, configuration: Mapping[str, Any]) -> None:
    """Write canonical scalar evidence only after checking worker receipts."""
    result = validate_outputs(archive.scratch_path, archive.run_id, configuration)
    rows = [{"epoch": item["epoch"], PRIMARY_METRIC: item["duration_seconds"],
             "runtime/train_hurdle_nll": item["train_loss"],
             "runtime/validation_hurdle_nll": item["validation_loss"],
             "runtime/peak_allocated_gib": item["peak_allocated_bytes"] / GIB,
             "runtime/peak_reserved_gib": item["peak_reserved_bytes"] / GIB}
            for item in result["epochs"]]
    for row in rows:
        for name, value in row.items():
            if name != "epoch":
                archive.append_metric_event({"name": name, "value": value, "step": row["epoch"]})
    archive.write_table("metrics/history", rows)
    final = dict(rows[-1])
    final.pop("epoch")
    final[PRIMARY_METRIC] = sum(row[PRIMARY_METRIC] for row in rows) / len(rows)
    archive.write_json("metrics/final.json", final)
    archive.write_summary({"status": "success", "runtime_diagnostic_only": True,
                           "primary_metric_name": PRIMARY_METRIC,
                           "primary_metric_value": final[PRIMARY_METRIC],
                           "peak_vram_gib": result["peak_allocated_bytes"] / GIB,
                           "parameter_count": 6160088, "epochs_completed": 2,
                           "source_run_id": SOURCE_RUN,
                           "permitted_claim": "single_gpu_runtime_feasibility_only"})
