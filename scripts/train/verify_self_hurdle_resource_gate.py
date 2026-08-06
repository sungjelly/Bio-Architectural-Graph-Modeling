#!/usr/bin/env python3
"""Verify both completed self-hurdle resource pilots and write one gate."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


CAMPAIGN_ID = "cmp_20260729_self_hurdle_full_core_capacity"
MATERIALIZATION_KIND = "self_hurdle_locked_config_materialization_v1"
GATE_KIND = "self_hurdle_resource_gate_v1"
ALIASES = ("ANC-03", "ANC-05")
EXPECTED_PARAMETER_COUNT = 16_917_200
PEAK_MAX = 12.0
DISCREPANCY_MAX = 1e-3
PER_RUN_HOURS_MAX = 6.0
AGGREGATE_HOURS_MAX = 12.0
DISK_MAX = 55.0


class SelfHurdleGateError(RuntimeError):
    pass


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelfHurdleGateError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelfHurdleGateError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                SelfHurdleGateError(
                    f"{label} contains non-finite constant {token}"
                )
            ),
            object_pairs_hook=unique,
        )
    except SelfHurdleGateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelfHurdleGateError(f"{label} is invalid") from exc
    return dict(_mapping(value, label))


def _verified(path: Path, label: str) -> dict[str, Any]:
    value = _strict_json(path, label)
    checksum = value.get("checksum")
    core = dict(value)
    core.pop("checksum", None)
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or canonical_sha256(core) != checksum
    ):
        raise SelfHurdleGateError(f"{label} checksum does not verify")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelfHurdleGateError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SelfHurdleGateError(f"{label} must be a finite number")
    return result


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed(registry: Registry) -> dict[str, dict[str, Any]]:
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT q.run_id, q.canonical_config_json,
                   r.status AS run_status, r.artifact_path
            FROM queue_jobs q
            LEFT JOIN runs r ON r.run_id = q.run_id
            WHERE q.campaign_id = ? AND q.status = 'completed'
            ORDER BY q.created_at, q.job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        item = dict(raw)
        try:
            config = json.loads(str(item["canonical_config_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SelfHurdleGateError(
                "completed config is invalid JSON"
            ) from exc
        digest = canonical_sha256(_mapping(config, "completed config"))
        if digest in result:
            raise SelfHurdleGateError(
                "multiple completed runs share one locked config"
            )
        result[digest] = item
    return result


def _expected(materialization: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    jobs = materialization.get("resource_jobs")
    if not isinstance(jobs, list):
        raise SelfHurdleGateError("resource job list is missing")
    result: dict[str, Mapping[str, Any]] = {}
    for alias in ALIASES:
        matches = [
            _mapping(item, "resource job")
            for item in jobs
            if isinstance(item, Mapping) and item.get("alias") == alias
        ]
        if len(matches) != 1:
            raise SelfHurdleGateError(
                f"resource job is not unique for {alias}"
            )
        result[alias] = matches[0]
    return result


def _bundle(
    *,
    project_root: Path,
    run: Mapping[str, Any],
    config_sha256: str,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    run_id = run.get("run_id")
    artifact = run.get("artifact_path")
    if (
        not isinstance(run_id, str)
        or run.get("run_status") != "completed"
        or not isinstance(artifact, str)
    ):
        raise SelfHurdleGateError("registered run is not completed")
    path = Path(artifact)
    path = (project_root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise SelfHurdleGateError("run bundle escapes project root") from exc
    verify_run_bundle(path)
    if path.name != run_id:
        raise SelfHurdleGateError("run bundle name differs from run ID")
    try:
        config = yaml.safe_load(
            (path / "config.resolved.yaml").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SelfHurdleGateError("resolved run config is invalid") from exc
    if canonical_sha256(_mapping(config, "resolved config")) != config_sha256:
        raise SelfHurdleGateError("resolved run config checksum changed")
    summary = _strict_json(path / "summary.json", "run summary")
    diagnostic = _strict_json(
        path / "diagnostics/resource_usage.json",
        "resource diagnostics",
    )
    if (
        summary.get("run_id") != run_id
        or summary.get("status") != "success"
        or summary.get("resource_pilot") is not True
    ):
        raise SelfHurdleGateError("run summary is not a successful pilot")
    return path, summary, diagnostic


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            dict(value), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def verify_resource_gate(
    *,
    project_root: Path,
    registry: Registry,
    materialization_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    materialization = _verified(
        materialization_path, "locked materialization"
    )
    if (
        materialization.get("receipt_kind") != MATERIALIZATION_KIND
        or materialization.get("campaign_id") != CAMPAIGN_ID
    ):
        raise SelfHurdleGateError("materialization identity is invalid")
    expected = _expected(materialization)
    completed = _completed(registry)
    jobs: list[dict[str, Any]] = []
    failures: list[str] = []
    projected_total = 0.0
    for alias in ALIASES:
        config_sha = str(expected[alias].get("config_sha256"))
        run = completed.get(config_sha)
        if run is None:
            raise SelfHurdleGateError(
                f"{alias} has no completed registered resource run"
            )
        bundle, _, diagnostic = _bundle(
            project_root=project_root,
            run=run,
            config_sha256=config_sha,
        )
        peak = _finite(
            diagnostic.get("peak_allocated_vram_gib"),
            f"{alias} peak VRAM",
        )
        discrepancy = _finite(
            diagnostic.get("fp32_amp_absolute_loss_discrepancy"),
            f"{alias} AMP discrepancy",
        )
        projected = _finite(
            diagnostic.get("projected_gpu_hours_per_200_epochs"),
            f"{alias} projected hours",
        )
        disk = _finite(
            diagnostic.get("filesystem_used_decimal_gb"),
            f"{alias} disk use",
        )
        passed = all(
            (
                diagnostic.get("receipt_kind")
                == "self_hurdle_resource_usage_v1",
                diagnostic.get("resource_pilot") is True,
                diagnostic.get("biological_unit_alias") == alias,
                diagnostic.get("parameter_count")
                == EXPECTED_PARAMETER_COUNT,
                diagnostic.get("all_losses_and_gradients_finite") is True,
                diagnostic.get("all_epochs_completed") is True,
                diagnostic.get("runner_pilot_gate_passed") is True,
                int(diagnostic.get("graph_construction_count", -1)) == 0,
                int(diagnostic.get("graph_input_tensor_count", -1)) == 0,
                int(diagnostic.get("edge_input_tensor_count", -1)) == 0,
                0 <= peak <= PEAK_MAX,
                0 <= discrepancy <= DISCREPANCY_MAX,
                0 <= projected <= PER_RUN_HOURS_MAX,
                0 <= disk < DISK_MAX,
            )
        )
        if not passed:
            failures.append(f"{alias} failed one or more resource gates")
        projected_total += projected
        success = bundle / "_SUCCESS"
        jobs.append(
            {
                "alias": alias,
                "run_id": str(run["run_id"]),
                "config_sha256": config_sha,
                "artifact_bundle": bundle.relative_to(
                    project_root.resolve()
                ).as_posix(),
                "success_marker_sha256": _file_sha(success),
                "verified_bundle": True,
                "peak_allocated_vram_gib": peak,
                "fp32_amp_absolute_loss_discrepancy": discrepancy,
                "projected_gpu_hours_per_200_epochs": projected,
                "filesystem_used_decimal_gb": disk,
                "graph_construction_count": 0,
                "graph_input_tensor_count": 0,
                "edge_input_tensor_count": 0,
                "passed": passed,
            }
        )
    if projected_total > AGGREGATE_HOURS_MAX:
        failures.append(
            "projected two-run aggregate exceeds 12 GPU-hours"
        )
    passed = not failures
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": GATE_KIND,
        "campaign_id": CAMPAIGN_ID,
        "materialization_checksum": materialization["checksum"],
        "thresholds": {
            "pilot_peak_vram_gib_maximum": PEAK_MAX,
            "fp32_amp_loss_discrepancy_maximum": DISCREPANCY_MAX,
            "projected_gpu_hours_per_science_run_maximum": (
                PER_RUN_HOURS_MAX
            ),
            "projected_aggregate_science_gpu_hours_maximum": (
                AGGREGATE_HOURS_MAX
            ),
            "filesystem_used_decimal_gb_hard_stop": DISK_MAX,
        },
        "complete": True,
        "passed": passed,
        "science_authorized": passed,
        "projected_aggregate_science_gpu_hours": projected_total,
        "failure_reasons": failures,
        "jobs": jobs,
    }
    payload["checksum"] = canonical_sha256(payload)
    _atomic(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    locked = paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--materialization",
        type=Path,
        default=locked / "locked_config_materialization.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=locked / "resource_gate_receipt.json",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking/bagm.sqlite3",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_resource_gate(
        project_root=PROJECT_ROOT,
        registry=Registry(args.database.resolve()),
        materialization_path=args.materialization.resolve(),
        output_path=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "science_authorized": result["science_authorized"],
                "projected_aggregate_science_gpu_hours": result[
                    "projected_aggregate_science_gpu_hours"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

