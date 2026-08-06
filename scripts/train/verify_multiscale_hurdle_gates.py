#!/usr/bin/env python3
"""Build checksum-bound resource or representation gate receipts.

The verifier reads only registered, completed, canonically verified run
bundles.  A scientific/resource threshold failure is written as a negative
receipt and returns a nonzero status; it never silently authorizes the next
stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402

from scripts.train.enqueue_multiscale_hurdle_campaign import (  # noqa: E402
    CAMPAIGN_ID,
    FROZEN_CONTRACT_SHA256,
    MATERIALIZATION_KIND,
    REPRESENTATION_GATE_KIND,
    REPRESENTATION_THRESHOLDS,
    RESOURCE_GATE_KIND,
    RESOURCE_THRESHOLDS,
)


STAGE1_ALIASES = ("ANC-03", "ANC-05")
MATERIALIZATION_RELATIVE = (
    Path("scratch/locked_campaigns")
    / CAMPAIGN_ID
    / "locked_config_materialization.json"
)
RESOURCE_GATE_RELATIVE = MATERIALIZATION_RELATIVE.with_name(
    "resource_gate_receipt.json"
)
REPRESENTATION_GATE_RELATIVE = MATERIALIZATION_RELATIVE.with_name(
    "representation_gate_receipt.json"
)


class MultiscaleHurdleGateError(RuntimeError):
    """Raised when prerequisite evidence is missing or inconsistent."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MultiscaleHurdleGateError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MultiscaleHurdleGateError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                MultiscaleHurdleGateError(
                    f"{label} contains non-finite constant {token}"
                )
            ),
            object_pairs_hook=unique_object,
        )
    except MultiscaleHurdleGateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MultiscaleHurdleGateError(
            f"{label} is not strict JSON"
        ) from exc
    return dict(_mapping(value, label))


def _verified_payload(path: Path, *, label: str) -> dict[str, Any]:
    payload = _strict_json(path, label=label)
    checksum = payload.get("checksum")
    core = dict(payload)
    core.pop("checksum", None)
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or canonical_sha256(core) != checksum
    ):
        raise MultiscaleHurdleGateError(
            f"{label} checksum does not verify"
        )
    return payload


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiscaleHurdleGateError(
            f"{label} must be a finite JSON number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise MultiscaleHurdleGateError(
            f"{label} must be a finite JSON number"
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise MultiscaleHurdleGateError(f"{label} is unreadable") from exc
    return dict(_mapping(value, label))


def _load_materialization(project_root: Path) -> dict[str, Any]:
    materialization = _verified_payload(
        project_root / MATERIALIZATION_RELATIVE,
        label="locked materialization",
    )
    frozen = _mapping(
        materialization.get("frozen_contract"),
        "materialization frozen contract",
    )
    if (
        materialization.get("schema_version") != 1
        or materialization.get("receipt_kind") != MATERIALIZATION_KIND
        or materialization.get("campaign_id") != CAMPAIGN_ID
        or frozen.get("sha256") != FROZEN_CONTRACT_SHA256
    ):
        raise MultiscaleHurdleGateError(
            "locked materialization identity is invalid"
        )
    return materialization


def _expected_jobs(
    materialization: Mapping[str, Any],
    *,
    resource: bool,
) -> dict[str, Mapping[str, Any]]:
    field = "pilot_jobs" if resource else "science_jobs"
    jobs = materialization.get(field)
    if not isinstance(jobs, list):
        raise MultiscaleHurdleGateError(
            f"materialization {field} is missing"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for alias in STAGE1_ALIASES:
        matches = [
            _mapping(item, f"{field} item")
            for item in jobs
            if isinstance(item, Mapping)
            and item.get("alias") == alias
            and item.get("arm") == "self"
        ]
        if len(matches) != 1:
            raise MultiscaleHurdleGateError(
                f"materialization lacks unique {alias} self prerequisite"
            )
        result[alias] = matches[0]
    return result


def _completed_runs_by_config(
    registry: Registry,
) -> dict[str, dict[str, Any]]:
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT q.job_id, q.status AS queue_status, q.run_id,
                   q.canonical_config_json, r.status AS run_status,
                   r.artifact_path
            FROM queue_jobs q
            LEFT JOIN runs r ON r.run_id = q.run_id
            WHERE q.campaign_id = ? AND q.status = 'completed'
            ORDER BY q.created_at, q.job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        try:
            config = json.loads(str(item["canonical_config_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MultiscaleHurdleGateError(
                "completed queue config is invalid JSON"
            ) from exc
        digest = canonical_sha256(_mapping(config, "completed queue config"))
        if digest in result:
            raise MultiscaleHurdleGateError(
                "multiple completed runs share one locked config"
            )
        item["config"] = dict(config)
        result[digest] = item
    return result


def _resolve_bundle(
    *,
    project_root: Path,
    run: Mapping[str, Any],
    expected_config_sha256: str,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    run_id = run.get("run_id")
    artifact = run.get("artifact_path")
    if (
        not isinstance(run_id, str)
        or not run_id.startswith("r_")
        or run.get("run_status") != "completed"
        or not isinstance(artifact, str)
    ):
        raise MultiscaleHurdleGateError(
            "completed prerequisite lacks a completed registered run"
        )
    bundle = Path(artifact)
    if not bundle.is_absolute():
        bundle = project_root / bundle
    bundle = bundle.resolve()
    try:
        bundle.relative_to(project_root)
    except ValueError as exc:
        raise MultiscaleHurdleGateError(
            "registered bundle escapes project root"
        ) from exc
    verification = verify_run_bundle(bundle)
    if bundle.name != run_id:
        raise MultiscaleHurdleGateError(
            "registered bundle does not match run ID"
        )
    config = _load_yaml(
        bundle / "config.resolved.yaml",
        label="resolved prerequisite config",
    )
    if canonical_sha256(config) != expected_config_sha256:
        raise MultiscaleHurdleGateError(
            "resolved prerequisite config checksum changed"
        )
    summary = _strict_json(bundle / "summary.json", label="run summary")
    final = _strict_json(
        bundle / "metrics/final.json",
        label="final metrics",
    )
    resource = _strict_json(
        bundle / "diagnostics/resource_usage.json",
        label="resource diagnostics",
    )
    return bundle, dict(verification), summary, final, resource


def _success_binding(
    *,
    project_root: Path,
    bundle: Path,
    run_id: str,
) -> dict[str, Any]:
    success_path = bundle / "_SUCCESS"
    success = _strict_json(success_path, label="success marker")
    marker = success.get("content_sha256")
    if (
        success.get("run_id") != run_id
        or success.get("status") != "success"
        or not isinstance(marker, str)
        or len(marker) != 64
    ):
        raise MultiscaleHurdleGateError(
            "success marker identity is invalid"
        )
    return {
        "run_id": run_id,
        "bundle_reference": bundle.relative_to(project_root).as_posix(),
        "success_marker_content_sha256": marker,
        "success_marker_file_sha256": _sha256_file(success_path),
        "verified_bundle": True,
    }


def _prerequisite_evidence(
    *,
    project_root: Path,
    registry: Registry,
    expected: Mapping[str, Mapping[str, Any]],
) -> dict[
    str,
    tuple[
        Mapping[str, Any],
        Path,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
]:
    completed = _completed_runs_by_config(registry)
    result: dict[
        str,
        tuple[
            Mapping[str, Any],
            Path,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ],
    ] = {}
    for alias, job in expected.items():
        config_sha = str(job.get("config_sha256"))
        run = completed.get(config_sha)
        if run is None:
            raise MultiscaleHurdleGateError(
                f"{alias} prerequisite has no completed registered run"
            )
        bundle, _, summary, final, resource = _resolve_bundle(
            project_root=project_root,
            run=run,
            expected_config_sha256=config_sha,
        )
        result[alias] = (run, bundle, summary, final, resource)
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def verify_resource_gate(
    *,
    project_root: Path,
    registry: Registry,
    output_path: Path,
) -> dict[str, Any]:
    materialization = _load_materialization(project_root)
    expected = _expected_jobs(materialization, resource=True)
    evidence = _prerequisite_evidence(
        project_root=project_root,
        registry=registry,
        expected=expected,
    )
    jobs: list[dict[str, Any]] = []
    failures: list[str] = []
    for alias in STAGE1_ALIASES:
        run, bundle, _, _, resource = evidence[alias]
        checks = _mapping(
            resource.get("checks"), f"{alias} resource checks"
        )
        if (
            resource.get("schema")
            != "multiscale_hurdle_resource_diagnostic_v1"
            or resource.get("resource_pilot") is not True
            or resource.get("arm") != "self"
            or resource.get("biological_unit_alias") != alias
            or resource.get("thresholds") != RESOURCE_THRESHOLDS
        ):
            raise MultiscaleHurdleGateError(
                f"{alias} resource diagnostics do not match the gate contract"
            )
        row = {
            "alias": alias,
            "arm": "self",
            "config_sha256": expected[alias]["config_sha256"],
            **_success_binding(
                project_root=project_root,
                bundle=bundle,
                run_id=str(run["run_id"]),
            ),
            "finite_losses_and_gradients": resource.get(
                "finite_losses_and_gradients"
            ),
            "parameter_count": resource.get("parameter_count"),
            "parameter_match": resource.get("parameter_match"),
            "fp32_amp_absolute_loss_discrepancy": resource.get(
                "fp32_amp_absolute_loss_discrepancy"
            ),
            "precision_equivalence_passed": checks.get(
                "precision_equivalence_passed"
            ),
            "peak_allocated_vram_gib": resource.get(
                "peak_allocated_vram_gib"
            ),
            "peak_vram_passed": checks.get("peak_vram_passed"),
            "projected_gpu_hours_per_200_epochs": resource.get(
                "projected_gpu_hours_per_200_epochs"
            ),
            "projected_runtime_passed": checks.get(
                "projected_runtime_passed"
            ),
            "filesystem_used_decimal_gb": resource.get(
                "filesystem_used_decimal_gb"
            ),
            "disk_safety_passed": checks.get("disk_safety_passed"),
            "runner_pilot_gate_passed": resource.get(
                "runner_pilot_gate_passed"
            ),
        }
        checks = (
            row["finite_losses_and_gradients"] is True,
            row["parameter_count"] == 7_559_184,
            row["parameter_match"] is True,
            row["precision_equivalence_passed"] is True,
            _finite(
                row["fp32_amp_absolute_loss_discrepancy"],
                label=f"{alias} precision discrepancy",
            )
            >= 0.0,
            _finite(
                row["fp32_amp_absolute_loss_discrepancy"],
                label=f"{alias} precision discrepancy",
            )
            <= RESOURCE_THRESHOLDS[
                "fp32_amp_absolute_loss_discrepancy_maximum"
            ],
            row["peak_vram_passed"] is True,
            _finite(
                row["peak_allocated_vram_gib"],
                label=f"{alias} peak VRAM",
            )
            >= 0.0,
            _finite(
                row["peak_allocated_vram_gib"],
                label=f"{alias} peak VRAM",
            )
            <= RESOURCE_THRESHOLDS[
                "stage1_peak_allocated_vram_gib_maximum"
            ],
            row["projected_runtime_passed"] is True,
            _finite(
                row["projected_gpu_hours_per_200_epochs"],
                label=f"{alias} projected runtime",
            )
            >= 0.0,
            _finite(
                row["projected_gpu_hours_per_200_epochs"],
                label=f"{alias} projected runtime",
            )
            <= RESOURCE_THRESHOLDS[
                "stage1_projected_gpu_hours_per_200_epoch_core_maximum"
            ],
            row["disk_safety_passed"] is True,
            _finite(
                row["filesystem_used_decimal_gb"],
                label=f"{alias} filesystem use",
            )
            >= 0.0,
            _finite(
                row["filesystem_used_decimal_gb"],
                label=f"{alias} filesystem use",
            )
            < RESOURCE_THRESHOLDS[
                "filesystem_used_decimal_gb_hard_stop"
            ],
            row["runner_pilot_gate_passed"] is True,
        )
        if not all(checks):
            failures.append(f"{alias} failed one or more resource gates")
        jobs.append(row)
    passed = not failures
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": RESOURCE_GATE_KIND,
        "campaign_id": CAMPAIGN_ID,
        "stage": "resource",
        "materialization_checksum": materialization["checksum"],
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "thresholds": dict(RESOURCE_THRESHOLDS),
        "complete": True,
        "gate_passed": passed,
        "production_authorized": passed,
        "failure_reasons": failures,
        "jobs": jobs,
    }
    payload["checksum"] = canonical_sha256(payload)
    _atomic_json(output_path, payload)
    return payload


def _validate_passing_resource_gate(
    *,
    project_root: Path,
    registry: Registry,
    materialization: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> None:
    expected_top = {
        "schema_version": 1,
        "receipt_kind": RESOURCE_GATE_KIND,
        "campaign_id": CAMPAIGN_ID,
        "stage": "resource",
        "materialization_checksum": materialization.get("checksum"),
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "thresholds": RESOURCE_THRESHOLDS,
        "complete": True,
        "gate_passed": True,
        "production_authorized": True,
        "failure_reasons": [],
    }
    if any(resource.get(key) != value for key, value in expected_top.items()):
        raise MultiscaleHurdleGateError(
            "representation gate requires the exact passing resource schema"
        )
    expected = _expected_jobs(materialization, resource=True)
    evidence = _prerequisite_evidence(
        project_root=project_root,
        registry=registry,
        expected=expected,
    )
    jobs = resource.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(STAGE1_ALIASES):
        raise MultiscaleHurdleGateError(
            "resource receipt does not bind exactly two jobs"
        )
    by_alias = {
        str(item.get("alias")): _mapping(item, "resource gate job")
        for item in jobs
        if isinstance(item, Mapping)
    }
    if set(by_alias) != set(STAGE1_ALIASES):
        raise MultiscaleHurdleGateError(
            "resource receipt job aliases are not prespecified"
        )
    for alias in STAGE1_ALIASES:
        run, bundle, _, _, diagnostics = evidence[alias]
        expected_binding = _success_binding(
            project_root=project_root,
            bundle=bundle,
            run_id=str(run["run_id"]),
        )
        job = by_alias[alias]
        for field, value in {
            "alias": alias,
            "arm": "self",
            "config_sha256": expected[alias]["config_sha256"],
            **expected_binding,
        }.items():
            if job.get(field) != value:
                raise MultiscaleHurdleGateError(
                    f"{alias} resource receipt binding changed: {field}"
                )
        checks = _mapping(
            diagnostics.get("checks"), f"{alias} resource checks"
        )
        evidence_fields = {
            "finite_losses_and_gradients": diagnostics.get(
                "finite_losses_and_gradients"
            ),
            "parameter_count": diagnostics.get("parameter_count"),
            "parameter_match": diagnostics.get("parameter_match"),
            "fp32_amp_absolute_loss_discrepancy": diagnostics.get(
                "fp32_amp_absolute_loss_discrepancy"
            ),
            "precision_equivalence_passed": checks.get(
                "precision_equivalence_passed"
            ),
            "peak_allocated_vram_gib": diagnostics.get(
                "peak_allocated_vram_gib"
            ),
            "peak_vram_passed": checks.get("peak_vram_passed"),
            "projected_gpu_hours_per_200_epochs": diagnostics.get(
                "projected_gpu_hours_per_200_epochs"
            ),
            "projected_runtime_passed": checks.get(
                "projected_runtime_passed"
            ),
            "filesystem_used_decimal_gb": diagnostics.get(
                "filesystem_used_decimal_gb"
            ),
            "disk_safety_passed": checks.get("disk_safety_passed"),
            "runner_pilot_gate_passed": diagnostics.get(
                "runner_pilot_gate_passed"
            ),
        }
        for field, value in evidence_fields.items():
            if job.get(field) != value:
                raise MultiscaleHurdleGateError(
                    f"{alias} resource evidence changed: {field}"
                )
        if (
            diagnostics.get("schema")
            != "multiscale_hurdle_resource_diagnostic_v1"
            or diagnostics.get("resource_pilot") is not True
            or diagnostics.get("arm") != "self"
            or diagnostics.get("biological_unit_alias") != alias
            or diagnostics.get("thresholds") != RESOURCE_THRESHOLDS
        ):
            raise MultiscaleHurdleGateError(
                f"{alias} resource diagnostics no longer match the gate contract"
            )
        discrepancy = _finite(
            diagnostics.get("fp32_amp_absolute_loss_discrepancy"),
            label=f"{alias} precision discrepancy",
        )
        peak_vram = _finite(
            diagnostics.get("peak_allocated_vram_gib"),
            label=f"{alias} peak VRAM",
        )
        projected = _finite(
            diagnostics.get("projected_gpu_hours_per_200_epochs"),
            label=f"{alias} projected runtime",
        )
        filesystem_used = _finite(
            diagnostics.get("filesystem_used_decimal_gb"),
            label=f"{alias} filesystem use",
        )
        criteria = (
            diagnostics.get("finite_losses_and_gradients") is True,
            diagnostics.get("parameter_count") == 7_559_184,
            diagnostics.get("parameter_match") is True,
            checks.get("precision_equivalence_passed") is True,
            0.0
            <= discrepancy
            <= RESOURCE_THRESHOLDS[
                "fp32_amp_absolute_loss_discrepancy_maximum"
            ],
            checks.get("peak_vram_passed") is True,
            0.0
            <= peak_vram
            <= RESOURCE_THRESHOLDS[
                "stage1_peak_allocated_vram_gib_maximum"
            ],
            checks.get("projected_runtime_passed") is True,
            0.0
            <= projected
            <= RESOURCE_THRESHOLDS[
                "stage1_projected_gpu_hours_per_200_epoch_core_maximum"
            ],
            checks.get("disk_safety_passed") is True,
            0.0
            <= filesystem_used
            < RESOURCE_THRESHOLDS[
                "filesystem_used_decimal_gb_hard_stop"
            ],
            diagnostics.get("runner_pilot_gate_passed") is True,
        )
        if not all(criteria):
            raise MultiscaleHurdleGateError(
                f"{alias} resource evidence does not pass every resource gate"
            )


def verify_representation_gate(
    *,
    project_root: Path,
    registry: Registry,
    output_path: Path,
) -> dict[str, Any]:
    materialization = _load_materialization(project_root)
    resource = _verified_payload(
        project_root / RESOURCE_GATE_RELATIVE,
        label="resource gate receipt",
    )
    _validate_passing_resource_gate(
        project_root=project_root,
        registry=registry,
        materialization=materialization,
        resource=resource,
    )
    expected = _expected_jobs(materialization, resource=False)
    evidence = _prerequisite_evidence(
        project_root=project_root,
        registry=registry,
        expected=expected,
    )
    jobs: list[dict[str, Any]] = []
    failures: list[str] = []
    for alias in STAGE1_ALIASES:
        run, bundle, _, final, _ = evidence[alias]
        detection = _finite(
            final.get("fit/whole_node/detection_balanced_accuracy"),
            label=f"{alias} detection balanced accuracy",
        )
        prevalence = _finite(
            final.get(
                "fit/whole_node/"
                "reference_per_gene_detection_balanced_accuracy"
            ),
            label=f"{alias} detection reference",
        )
        huber = _finite(
            final.get(
                "fit/whole_node/"
                "positive_continuous_huber_relative_improvement_"
                "over_per_gene_reference"
            ),
            label=f"{alias} positive Huber improvement",
        )
        state = _finite(
            final.get(
                "fit/whole_node/"
                "positive_count_state_mae_relative_improvement_"
                "over_per_gene_reference"
            ),
            label=f"{alias} positive count-state improvement",
        )
        if not (0.0 <= detection <= 1.0):
            raise MultiscaleHurdleGateError(
                f"{alias} detection balanced accuracy is outside [0, 1]"
            )
        if not (0.0 <= prevalence <= 1.0):
            raise MultiscaleHurdleGateError(
                f"{alias} detection reference is outside [0, 1]"
            )
        detection_pass = detection > prevalence
        passed = (
            detection_pass
            and huber
            >= REPRESENTATION_THRESHOLDS["minimum_relative_improvement"]
            and state
            >= REPRESENTATION_THRESHOLDS["minimum_relative_improvement"]
        )
        if not passed:
            failures.append(f"{alias} failed one or more H1 criteria")
        jobs.append(
            {
                "alias": alias,
                "arm": "self",
                "config_sha256": expected[alias]["config_sha256"],
                **_success_binding(
                    project_root=project_root,
                    bundle=bundle,
                    run_id=str(run["run_id"]),
                ),
                "detection_balanced_accuracy": detection,
                "prevalence_reference_balanced_accuracy": prevalence,
                "detection_balanced_accuracy_above_prevalence_reference": (
                    detection_pass
                ),
                "positive_continuous_huber_relative_improvement": huber,
                "positive_count_state_mae_relative_improvement": state,
                "h1_gate_passed": passed,
            }
        )
    passed = not failures
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": REPRESENTATION_GATE_KIND,
        "campaign_id": CAMPAIGN_ID,
        "stage": "stage1",
        "materialization_checksum": materialization["checksum"],
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "resource_gate_checksum": resource["checksum"],
        "thresholds": dict(REPRESENTATION_THRESHOLDS),
        "complete": True,
        "gate_passed": passed,
        "both_h1_gates_passed": passed,
        "stage2_authorized": passed,
        "failure_reasons": failures,
        "jobs": jobs,
    }
    payload["checksum"] = canonical_sha256(payload)
    _atomic_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("resource", "representation"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=paths.project_root,
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking/bagm.sqlite3",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    registry = Registry(args.database)
    if args.stage == "resource":
        output = (
            project_root / RESOURCE_GATE_RELATIVE
            if args.output is None
            else args.output.resolve()
        )
        receipt = verify_resource_gate(
            project_root=project_root,
            registry=registry,
            output_path=output,
        )
    else:
        output = (
            project_root / REPRESENTATION_GATE_RELATIVE
            if args.output is None
            else args.output.resolve()
        )
        receipt = verify_representation_gate(
            project_root=project_root,
            registry=registry,
            output_path=output,
        )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "gate_passed": receipt["gate_passed"],
                "failure_reasons": receipt["failure_reasons"],
                "output": str(output),
                "checksum": receipt["checksum"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
