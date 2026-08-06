#!/usr/bin/env python3
"""Run the prespecified signed-program models on the locked Stage-0 fixture.

This is intentionally a standalone diagnostic.  It neither registers nor
enqueues work and writes one compact identifier-free JSON receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.artifacts import sha256_file  # noqa: E402
from spatial_benchmark.configuration import load_yaml_mapping  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.multiscale_synthetic import (  # noqa: E402
    SYNTHETIC_GEOMETRY_ALIAS,
    SyntheticRecoveryConfig,
    assert_alias_safe_configuration,
    build_multiscale_synthetic_fixture,
    load_alias_safe_observed_geometry,
)
from spatial_benchmark.signed_program_synthetic import (  # noqa: E402
    SIGNED_PROGRAM_VARIANTS,
    run_signed_program_variant,
)


CAMPAIGN_ID = "cmp_20260729_signed_program_stage0_pilot"
SOURCE_CONFIG_RELATIVE = Path(
    "experiments/campaigns/"
    "cmp_20260729_multiscale_hurdle_count_pilot/"
    "stage0_synthetic_recovery_config.yaml"
)
SOURCE_CONFIG_SHA256 = (
    "61a2fcf80f9f7e0ae4bae58448682fe6c286c12689f1fda526bbd9268eba7e00"
)
TASK_CONTRACT_RELATIVE = Path(
    "experiments/campaigns/"
    "cmp_20260729_signed_program_stage0_pilot/"
    "frozen_task_contract.yaml"
)
TASK_CONTRACT_SHA256 = (
    "9a2220e8e60565ac97ce65c9eb8b3f1445242e9ad9de7696574f96381085dd1e"
)
EXPECTED_FIXTURE_SHA256 = (
    "f53076d07e81842b4e73a4bb6ec66b795d2899d207a620c35a136377b2aa1cd7"
)
RECEIPT_SCHEMA = "signed_program_stage0_diagnostic_v1"


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _safe_output_path(value: str | Path) -> Path:
    requested = Path(value)
    path = (
        requested
        if requested.is_absolute()
        else _PROJECT_ROOT / requested
    ).resolve()
    allowed = (_PROJECT_ROOT / "scratch/diagnostics").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError("output must remain under scratch/diagnostics")
    return path


def _load_locked_inputs(
    *,
    device: str,
) -> tuple[Any, SyntheticRecoveryConfig]:
    source_path = _PROJECT_ROOT / SOURCE_CONFIG_RELATIVE
    contract_path = _PROJECT_ROOT / TASK_CONTRACT_RELATIVE
    if sha256_file(source_path) != SOURCE_CONFIG_SHA256:
        raise RuntimeError("locked source configuration checksum mismatch")
    if sha256_file(contract_path) != TASK_CONTRACT_SHA256:
        raise RuntimeError("signed-program task contract checksum mismatch")
    source = load_yaml_mapping(source_path)
    assert_alias_safe_configuration(source)
    dataset = _mapping(source.get("dataset"), name="config.dataset")
    metadata = _mapping(source.get("metadata"), name="config.metadata")
    settings_mapping = _mapping(
        metadata.get("synthetic_recovery"),
        name="config.metadata.synthetic_recovery",
    )
    settings = SyntheticRecoveryConfig.from_mapping(
        settings_mapping,
        seed=int(source.get("seed", 0)),
        trainer=_mapping(source.get("trainer"), name="config.trainer"),
    )
    # Device is operational only.  No scientific or optimization field is
    # changed from the checksum-bound source configuration.
    settings = replace(settings, device=device)
    alias = dataset.get("biological_unit_alias")
    if alias != SYNTHETIC_GEOMETRY_ALIAS:
        raise RuntimeError("locked fixture alias mismatch")
    prepared_sha = dataset.get("prepared_data_sha256")
    if not isinstance(prepared_sha, str):
        raise RuntimeError("locked dataset lacks prepared-data checksum")
    geometry = load_alias_safe_observed_geometry(
        str(dataset["prepared_artifact_reference"]),
        project_root=_PROJECT_ROOT,
        biological_unit_alias=alias,
        expected_prepared_data_sha256=prepared_sha,
        selected_node_count=settings.selected_node_count,
    )
    return geometry, settings


def _variant_receipt(result: Any, *, runtime_seconds: float) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    peak_memory = 0
    for arm_name, outcome in result.arms.items():
        arm_peak = max(
            (
                int(record.peak_cuda_memory_bytes)
                for record in outcome.training.history
            ),
            default=0,
        )
        peak_memory = max(peak_memory, arm_peak)
        arms[arm_name] = {
            "regional_routing": outcome.regional_routing,
            "local_routing": outcome.local_routing,
            "parameter_count": outcome.parameter_count,
            "parameter_structure_sha256": (
                outcome.parameter_structure_sha256
            ),
            "initial_parameter_sha256": outcome.initial_parameter_sha256,
            "final_state_sha256": outcome.training.final_state_checksum,
            "fixed_epoch_budget": outcome.training.fixed_epoch_budget,
            "final_train_hurdle_loss": outcome.training.final_train_loss,
            "evaluation_whole_node_hurdle_loss": outcome.whole_node_loss,
            "peak_cuda_memory_bytes": arm_peak,
        }
    return {
        "fixture_sha256": result.fixture.fixture_checksum,
        "parameter_match_verified": result.parameter_match_verified,
        "arms": arms,
        "planted_deletion_diagnostic": asdict(result.planted_diagnostic),
        "null_deletion_diagnostic": asdict(result.null_diagnostic),
        "gate": asdict(result.gate),
        "final_metrics": result.final_metrics(),
        "runtime_seconds": runtime_seconds,
        "peak_cuda_memory_bytes": peak_memory,
    }


def run_diagnostic(*, device: str) -> dict[str, Any]:
    geometry, settings = _load_locked_inputs(device=device)
    fixture = build_multiscale_synthetic_fixture(geometry, settings)
    if fixture.fixture_checksum != EXPECTED_FIXTURE_SHA256:
        raise RuntimeError(
            "locked fixture checksum mismatch: expected "
            f"{EXPECTED_FIXTURE_SHA256}, got {fixture.fixture_checksum}"
        )
    results: dict[str, Any] = {}
    started = time.monotonic()
    for variant in SIGNED_PROGRAM_VARIANTS:
        variant_started = time.monotonic()
        result = run_signed_program_variant(
            fixture,
            settings,
            variant=variant,
        )
        results[variant] = _variant_receipt(
            result,
            runtime_seconds=time.monotonic() - variant_started,
        )
    elapsed = time.monotonic() - started
    passing = [
        variant
        for variant in SIGNED_PROGRAM_VARIANTS
        if bool(results[variant]["gate"]["gate_passed"])
    ]
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "completed",
        "outcome": "supported" if passing else "negative",
        "exploratory": True,
        "registered_run": False,
        "production_authorized": False,
        "task_contract": {
            "path": TASK_CONTRACT_RELATIVE.as_posix(),
            "sha256": TASK_CONTRACT_SHA256,
        },
        "source_configuration": {
            "path": SOURCE_CONFIG_RELATIVE.as_posix(),
            "sha256": SOURCE_CONFIG_SHA256,
        },
        "fixture": {
            "biological_unit_alias": SYNTHETIC_GEOMETRY_ALIAS,
            "selected_node_count": settings.selected_node_count,
            "fixture_sha256": fixture.fixture_checksum,
            "observed_geometry": fixture.audit["observed_geometry"],
            "graph_bundle_sha256": fixture.graph_audit["bundle_checksums"][
                "bundle_sha256"
            ],
            "sender_state_permutation_receipt_checksum": (
                fixture.sender_state_permutation_audit["checksum"]
            ),
            "direct_identifiers_emitted": False,
        },
        "frozen_scientific_settings": {
            key: value
            for key, value in asdict(settings).items()
            if key != "device"
        },
        "operational_device": device,
        "prespecified_variants": list(SIGNED_PROGRAM_VARIANTS),
        "passing_variants": passing,
        "all_variants_reported": set(results) == set(SIGNED_PROGRAM_VARIANTS),
        "results": results,
        "decision": (
            "separate_contract_required_before_any_real_data_pilot"
            if passing
            else "stop_architecture_branch_negative"
        ),
        "maximum_claim": (
            "At least one constrained signed-program architecture recovered "
            "the generated local dependency under the single locked Stage-0 "
            "fixture; this is synthetic architecture validation only."
            if passing
            else
            "None of the prespecified constrained signed-program "
            "architectures passed the single locked Stage-0 recovery gate."
        ),
        "runtime": {
            "total_seconds": elapsed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            "--device",
            device,
        ],
        "working_directory": str(_PROJECT_ROOT),
        "process_id_recorded": False,
    }
    receipt["checksum"] = canonical_sha256(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        help="Operational torch device; scientific settings remain frozen.",
    )
    parser.add_argument(
        "--output",
        default="scratch/diagnostics/signed_program_stage0_result.json",
    )
    args = parser.parse_args()
    output = _safe_output_path(args.output)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite diagnostic receipt: {output}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    receipt = run_diagnostic(device=str(args.device))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.relative_to(_PROJECT_ROOT)),
                "outcome": receipt["outcome"],
                "passing_variants": receipt["passing_variants"],
                "checksum": receipt["checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
