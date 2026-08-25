#!/usr/bin/env python3
"""Keep Relative-QKV finals and remove exact registered intermediate payloads.

This entry point implements the user-authorized 2026-08-25 retention decision:
retain the four verified ``last.ckpt`` model checkpoints, tombstone every
registered periodic ``epoch_*.ckpt`` for those runs, and remove four exact
oversized edge tables from terminal failed attention-niche attempts while
preserving their compact outputs, receipts, config, logs, metrics, provenance,
summaries, and failure markers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spatial_benchmark.artifact_retention import (  # noqa: E402
    DELETED_BY_RETENTION,
    RetentionCandidate,
    _assert_registry_plan_current,
    _mark_deleted,
    _mark_pending,
    _sqlite_backup,
    _utc_now,
    _write_immutable,
    assert_no_live_experiments,
    verify_candidates,
    verify_retention_result,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


DECISION_ID = "cleanup_20260825_relative_qkv_final_only_v2"
RELATIVE_QKV_CAMPAIGN = "cmp_20260824_cancer_6core_relative_qkv_multiseed"
FAILED_ANALYSIS_CAMPAIGN = "cmp_20260825_six_core_attention_routing_niches"
EXPECTED_PERIODIC_COUNT = 32
EXPECTED_PERIODIC_BYTES = 1_974_247_968
EXPECTED_FAILED_ARTIFACTS = {
    187: (
        "r_20260825T053454Z_0da9fbf2_s000_f00_a01_69a07db0",
        809_067_253,
        "abc8a5c7c5fbd4947bc0db0c45011d5c2a2f3e1afae022daedf4895cda357c3c",
        "diagnostics/attention_niche_core_work/core_13/directed_attention_edges.parquet",
    ),
    188: (
        "r_20260825T053454Z_0da9fbf2_s000_f00_a01_69a07db0",
        51_011_833,
        "61c43daa7a5d0fe9b223ae5e4d6ce833245cf87f20ad243801eeb5941467c16c",
        "diagnostics/attention_niche_core_work/core_13/mutual_attention_edges.parquet",
    ),
    241: (
        "r_20260825T055100Z_0da9fbf2_s000_f00_a01_ead1d2c2",
        21_224_967_486,
        "e37c7d2cba7e08ff2a7dec1259721409c008491484c9c4cf54c024f2a212fb06",
        "directed_attention_edges.parquet",
    ),
    246: (
        "r_20260825T055100Z_0da9fbf2_s000_f00_a01_ead1d2c2",
        1_555_129_334,
        "fa9fb29f252e7bb9b7218661bf44d16dcb88356294d6242cd307197dc5df0798",
        "mutual_attention_edges.parquet",
    ),
}
EXPECTED_SELECTED_COUNT = 36
EXPECTED_SELECTED_BYTES = 25_614_423_874
EXPECTED_FINALS = {
    "r_20260824T121803Z_16144620_s000_f00_a02_62498796": (
        62,
        62_884_847,
        "c5b7fdd6e3192c3146d415f1c77474f9e4483bba090c4fc9c2ac7d79ab554c0c",
    ),
    "r_20260824T124852Z_95591978_s001_f00_a01_266e6db4": (
        135,
        63_227_567,
        "3ce4a0e7983a14cba34a32e182800c62ed0184377247443da6bbec72531d0d3c",
    ),
    "r_20260824T124855Z_95591978_s002_f00_a01_3d4cbf7d": (
        27,
        62_542_191,
        "91f3f8bfe03ee1624c6ac954a612c65fd6cabc4bf0cce03dc2debf1c44de6411",
    ),
    "r_20260824T125052Z_95591978_s003_f00_a01_c02388f5": (
        98,
        62_884_847,
        "b8943dcb21bac2ea7b3077f1c51e7b6b24b31effe3d23ce471aa05201a3f36d4",
    ),
}
_PERIODIC_NAME = re.compile(r"epoch_[0-9]{4}\.ckpt")
def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _candidate(row: Any) -> RetentionCandidate:
    return RetentionCandidate(
        artifact_id=int(row["artifact_id"]),
        run_id=str(row["run_id"]),
        campaign_id=str(row["campaign_id"]),
        run_status=str(row["run_status"]),
        lifecycle_stage=str(row["lifecycle_stage"]),
        retention_class=str(row["retention_class"]),
        kind=str(row["kind"]),
        path=str(row["path"]),
        size_bytes=int(row["size_bytes"]),
        sha256=str(row["sha256"]).lower(),
        artifact_root=str(row["artifact_root"]),
    )


def _relative_payload(row: Any) -> Path:
    payload = Path(str(row["path"])).resolve(strict=False)
    run_root = Path(str(row["artifact_root"])).resolve(strict=False)
    try:
        return payload.relative_to(run_root)
    except ValueError as error:
        raise RuntimeError(f"Artifact escapes its run root: {payload}") from error


def _select(registry: Registry) -> tuple[list[RetentionCandidate], list[RetentionCandidate]]:
    failed_run_ids = tuple(
        sorted({contract[0] for contract in EXPECTED_FAILED_ARTIFACTS.values()})
    )
    run_ids = (*EXPECTED_FINALS, *failed_run_ids)
    placeholders = ",".join("?" for _ in run_ids)
    with registry.connect() as connection:
        rows = connection.execute(
            f"""
            SELECT a.artifact_id, a.run_id, a.kind, a.path, a.size_bytes,
                   a.sha256, r.campaign_id, r.status AS run_status,
                   r.artifact_path AS artifact_root, rc.lifecycle_stage,
                   rc.retention_class, c.role AS checkpoint_role,
                   c.verification_status
            FROM artifacts a
            JOIN runs r ON r.run_id = a.run_id
            JOIN run_categories rc ON rc.run_id = r.run_id
            LEFT JOIN checkpoint_catalog c ON c.artifact_id = a.artifact_id
            WHERE a.status = 'present'
              AND a.run_id IN ({placeholders})
            ORDER BY a.artifact_id
            """,
            run_ids,
        ).fetchall()

    selected: list[RetentionCandidate] = []
    finals: list[RetentionCandidate] = []
    for row in rows:
        if str(row["campaign_id"]) == RELATIVE_QKV_CAMPAIGN:
            name = Path(str(row["path"])).name
            if str(row["kind"]) != "checkpoints":
                continue
            if name == "last.ckpt":
                if (
                    row["checkpoint_role"] != "last"
                    or row["verification_status"] != "verified"
                ):
                    raise RuntimeError(
                        f"Final checkpoint is not role=last/verified: {row['path']}"
                    )
                finals.append(_candidate(row))
            elif _PERIODIC_NAME.fullmatch(name):
                if row["checkpoint_role"] != "checkpoint":
                    raise RuntimeError(
                        f"Periodic checkpoint has unexpected role: {row['path']}"
                    )
                selected.append(_candidate(row))
            else:
                raise RuntimeError(f"Unexpected checkpoint payload: {row['path']}")
            continue

        expected_failed = EXPECTED_FAILED_ARTIFACTS.get(int(row["artifact_id"]))
        if expected_failed is None:
            continue
        if (
            str(row["run_id"]) != expected_failed[0]
            or str(row["campaign_id"]) != FAILED_ANALYSIS_CAMPAIGN
            or str(row["run_status"]) != "failed"
        ):
            raise RuntimeError(
                "Failed-analysis retention contract drifted for "
                f"artifact {row['artifact_id']}"
            )
        relative = _relative_payload(row)
        candidate = _candidate(row)
        observed_failed = (
            candidate.run_id,
            candidate.size_bytes,
            candidate.sha256,
            relative.as_posix(),
        )
        if observed_failed != expected_failed:
            raise RuntimeError(
                f"Failed artifact {candidate.artifact_id} drifted: {observed_failed}"
            )
        selected.append(candidate)

    periodic = [item for item in selected if item.kind == "checkpoints"]
    if len(periodic) != EXPECTED_PERIODIC_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PERIODIC_COUNT} periodic checkpoints, got {len(periodic)}"
        )
    periodic_bytes = sum(item.size_bytes for item in periodic)
    if periodic_bytes != EXPECTED_PERIODIC_BYTES:
        raise RuntimeError(
            f"Periodic checkpoint bytes drifted: {periodic_bytes}"
        )

    observed_finals = {
        item.run_id: (item.artifact_id, item.size_bytes, item.sha256)
        for item in finals
    }
    if observed_finals != EXPECTED_FINALS:
        raise RuntimeError(
            f"Final checkpoint contract drifted: {observed_finals}"
        )
    observed_failed_ids = {
        item.artifact_id for item in selected if item.kind != "checkpoints"
    }
    if observed_failed_ids != set(EXPECTED_FAILED_ARTIFACTS):
        raise RuntimeError(
            f"Failed-analysis artifact IDs drifted: {observed_failed_ids}"
        )
    if (
        len(selected) != EXPECTED_SELECTED_COUNT
        or sum(item.size_bytes for item in selected) != EXPECTED_SELECTED_BYTES
    ):
        raise RuntimeError(
            "Overall deletion contract drifted: "
            f"count={len(selected)}, bytes={sum(item.size_bytes for item in selected)}"
        )
    return selected, finals


def _verify_affected_bundles(
    registry: Registry,
    candidates: list[RetentionCandidate],
) -> list[dict[str, Any]]:
    """Run complete bundle verification with registry-backed tombstones."""

    run_ids = sorted({item.run_id for item in candidates})
    placeholders = ",".join("?" for _ in run_ids)
    with registry.connect() as connection:
        runs = connection.execute(
            f"""
            SELECT run_id, status, artifact_path
            FROM runs
            WHERE run_id IN ({placeholders})
            ORDER BY run_id
            """,
            run_ids,
        ).fetchall()
        tombstone_rows = connection.execute(
            f"""
            SELECT run_id, path, size_bytes, sha256
            FROM artifacts
            WHERE run_id IN ({placeholders})
              AND status = ?
            ORDER BY run_id, artifact_id
            """,
            (*run_ids, DELETED_BY_RETENTION),
        ).fetchall()
    if len(runs) != len(run_ids):
        raise RuntimeError("Affected run set changed during bundle verification")

    by_run: dict[str, dict[str, dict[str, Any]]] = {run_id: {} for run_id in run_ids}
    roots = {str(row["run_id"]): Path(str(row["artifact_path"])) for row in runs}
    for row in tombstone_rows:
        run_id = str(row["run_id"])
        relative = Path(str(row["path"])).resolve(strict=False).relative_to(
            roots[run_id].resolve(strict=False)
        ).as_posix()
        by_run[run_id][relative] = {
            "type": "file",
            "size": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }

    results: list[dict[str, Any]] = []
    for row in runs:
        run_id = str(row["run_id"])
        result = verify_run_bundle(
            row["artifact_path"],
            require_success_contract=str(row["status"]) == "completed",
            tombstoned_artifacts=by_run[run_id],
        )
        results.append(
            {
                "run_id": run_id,
                "status": result["status"],
                "present_file_count": result["present_file_count"],
                "tombstoned_file_count": result["tombstoned_file_count"],
            }
        )
    return results


def _plan_records(candidates: list[RetentionCandidate]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in candidates:
        record = item.record()
        record["retention_reason"] = (
            "redundant_periodic_resume_checkpoint"
            if item.kind == "checkpoints"
            else "derived_payload_from_terminal_failed_analysis"
        )
        records.append(record)
    return records


def _write_plan(
    candidates: list[RetentionCandidate],
    finals: list[RetentionCandidate],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    records = _plan_records(candidates)
    payload = (
        "\n".join(_canonical_json(record) for record in records) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    _write_immutable(output_dir / "deletion_plan.jsonl", payload)
    _write_immutable(
        output_dir / "deletion_plan.sha256",
        f"{digest}  deletion_plan.jsonl\n".encode("utf-8"),
    )
    summary = {
        "schema_version": 1,
        "decision_id": DECISION_ID,
        "authorization": "user_requested_keep_only_final_checkpoints_and_remove_intermediates",
        "artifact_count": len(candidates),
        "run_count": len({item.run_id for item in candidates}),
        "total_bytes": sum(item.size_bytes for item in candidates),
        "plan_sha256": digest,
        "by_kind": dict(sorted(Counter(item.kind for item in candidates).items())),
        "by_reason": dict(
            sorted(Counter(record["retention_reason"] for record in records).items())
        ),
        "protected_final_checkpoints": [item.record() for item in finals],
        "protected_evidence": [
            "config.resolved.yaml",
            "logs",
            "metrics",
            "provenance",
            "manifest.yaml",
            "summary.json",
            "completion_markers",
            "per_core_analysis_receipts",
        ],
    }
    _write_immutable(
        output_dir / "deletion_plan_summary.json",
        (_canonical_json(summary) + "\n").encode("utf-8"),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database", default="state/tracking/bagm.sqlite3"
    )
    parser.add_argument(
        "--output-dir",
        default="reports/retention/cleanup_20260825_relative_qkv_final_only_v2",
    )
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args(argv)

    paths = current_paths(anchor=PROJECT_ROOT)
    paths.validate()
    database = Path(arguments.database)
    if not database.is_absolute():
        database = paths.project_root / database
    output_dir = Path(arguments.output_dir)
    if not output_dir.is_absolute():
        output_dir = paths.project_root / output_dir
    if arguments.apply:
        required_plan_files = (
            output_dir / "deletion_plan.jsonl",
            output_dir / "deletion_plan.sha256",
            output_dir / "deletion_plan_summary.json",
        )
        missing = [str(path) for path in required_plan_files if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Refusing --apply without a previously generated dry-run plan: "
                + ", ".join(missing)
            )
    registry = Registry(database)
    registry.initialize()
    assert_no_live_experiments(registry)
    candidates, finals = _select(registry)
    verify_candidates(candidates, paths=paths, verify_sha256=False)
    verify_candidates(finals, paths=paths, verify_sha256=False)
    summary = _write_plan(candidates, finals, output_dir=output_dir)
    if not arguments.apply:
        print(json.dumps({"applied": False, **summary}, indent=2, sort_keys=True))
        return 0

    _assert_registry_plan_current(registry, candidates)
    verified = verify_candidates(candidates, paths=paths, verify_sha256=True)
    verify_candidates(finals, paths=paths, verify_sha256=True)
    backup = paths.state_root / "backups" / f"bagm_pre_{DECISION_ID}.sqlite3"
    _sqlite_backup(registry, path=backup)
    _mark_pending(registry, candidates)
    for index, (candidate, payload) in enumerate(
        zip(candidates, verified, strict=True), start=1
    ):
        payload.unlink()
        print(
            f"deleted {index}/{len(candidates)}: {candidate.run_id}/{payload.name} "
            f"({candidate.size_bytes} bytes)",
            flush=True,
        )
    _mark_deleted(registry, candidates, verified)
    verify_retention_result(registry, candidates, paths=paths)
    verify_candidates(finals, paths=paths, verify_sha256=True)
    bundle_verification = _verify_affected_bundles(registry, candidates)

    receipt = {
        "schema_version": 1,
        "decision_id": DECISION_ID,
        "completed_at": _utc_now(),
        "artifact_count": len(candidates),
        "run_count": len({item.run_id for item in candidates}),
        "deleted_bytes": sum(item.size_bytes for item in candidates),
        "plan_sha256": summary["plan_sha256"],
        "registry_backup": str(backup),
        "artifact_status": DELETED_BY_RETENTION,
        "protected_final_checkpoint_count": len(finals),
        "protected_final_checkpoints": [item.record() for item in finals],
        "bundle_verification": bundle_verification,
    }
    receipt_bytes = (_canonical_json(receipt) + "\n").encode("utf-8")
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    _write_immutable(output_dir / "application_receipt.json", receipt_bytes)
    _write_immutable(
        output_dir / "application_receipt.sha256",
        f"{receipt_sha256}  application_receipt.json\n".encode("utf-8"),
    )
    print(
        json.dumps(
            {"applied": True, **receipt, "receipt_sha256": receipt_sha256},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
