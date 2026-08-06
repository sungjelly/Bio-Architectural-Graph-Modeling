from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

from spatial_benchmark.configuration import compose_config
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.registry import Registry


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "train"
    / "enqueue_multicore_qkv_campaign.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "enqueue_multicore_qkv_campaign_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

LockedEnqueueError = _MODULE.LockedEnqueueError
enqueue_locked_campaign = _MODULE.enqueue_locked_campaign

_CAMPAIGN_ID = "cmp_20260728_adjacent_normal_10core_qkv_large_k"
_ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
_ARMS = ("k1000", "k5000", "matched_self")
_BASE_CONFIGS = {
    "k1000": _ROOT / "configs/experiment/full_core_qkv_k1000.yaml",
    "k5000": _ROOT / "configs/experiment/full_core_qkv_k5000.yaml",
    "matched_self": (
        _ROOT / "configs/experiment/full_core_qkv_matched_self.yaml"
    ),
}


def _checksum(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _campaign_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(_MODULE, "_PROJECT_ROOT", root)
    database = root / "state/tracking/test.sqlite3"
    registry = Registry(database)
    registry.create_campaign(
        _CAMPAIGN_ID,
        name="Synthetic locked enqueue campaign",
    )
    jobs: list[dict[str, Any]] = []
    for alias_index, alias in enumerate(_ALIASES, start=1):
        key = alias.lower().replace("-", "")
        prepared_reference = Path("prepared") / alias.lower()
        (root / prepared_reference).mkdir(parents=True)
        dataset_id = f"cosmx_{key}_adjacent_normal_test"
        dataset_version = "adjacent_normal_full_core_fit_v1"
        split_id = f"split_{key}"
        dataset_fingerprint = _checksum(f"dataset-{alias}")
        split_fingerprint = _checksum(f"split-{alias}")
        registry.register_dataset(
            dataset_id,
            dataset_version,
            display_name=f"Synthetic {alias}",
            protected_source_path=prepared_reference,
            processed_fingerprint=dataset_fingerprint,
            preprocessing_version="adjacent_normal_full_core_fit_v1",
            verification_status="verified",
        )
        registry.register_split(
            split_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            method="all-fit",
            unit="single_adjacent_normal_spatial_core",
            fingerprint=split_fingerprint,
            protected_path=prepared_reference,
            verification_status="verified",
        )
        for arm_index, arm in enumerate(_ARMS):
            config = compose_config(_BASE_CONFIGS[arm])
            gpu = (3 * (alias_index - 1) + arm_index) % 8
            config["campaign"] = {
                "campaign_id": _CAMPAIGN_ID,
                "display_name": "Synthetic locked enqueue campaign",
            }
            config["dataset"].update(
                {
                    "dataset_id": dataset_id,
                    "version": dataset_version,
                    "split_id": split_id,
                    "dataset_fingerprint": dataset_fingerprint,
                    "split_fingerprint": split_fingerprint,
                    "preprocessing_version": (
                        "adjacent_normal_full_core_fit_v1"
                    ),
                    "prepared_artifact_reference": (
                        prepared_reference.as_posix()
                    ),
                    "biological_unit_alias": alias,
                    "tissue_context": (
                        "pathology_confirmed_adjacent_normal"
                    ),
                    "validation_or_test_partition_present": False,
                }
            )
            config["experiment"] = {
                "variant_label": f"{key}_qkv_{arm}_full_core",
                "biological_unit_alias": alias,
                "tissue_context": "pathology_confirmed_adjacent_normal",
                "estimand": (
                    "held_in_full_core_whole_node_masked_reconstruction"
                ),
                "permitted_claim": (
                    "ten_core_adjacent_normal_transductive_"
                    "representation_capacity"
                ),
                "paired_within_core": True,
            }
            config["launcher"]["requested_gpu"] = str(gpu)
            config_path = Path("configs") / f"{key}_{arm}.yaml"
            absolute_config = root / config_path
            absolute_config.parent.mkdir(parents=True, exist_ok=True)
            absolute_config.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            jobs.append(
                {
                    "alias": alias,
                    "arm": arm,
                    "config": config_path.as_posix(),
                    "requested_gpu": gpu,
                    "config_sha256": canonical_sha256(config),
                }
            )
    materialization: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": _CAMPAIGN_ID,
        "job_count": 30,
        "jobs": jobs,
    }
    materialization["checksum"] = canonical_sha256(materialization)
    materialization_path = root / "campaign_materialization.json"
    materialization_path.write_text(
        json.dumps(materialization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return materialization_path, root / "enqueue_receipt.json", database


def _call(paths: tuple[Path, Path, Path]) -> dict[str, Any]:
    materialization, receipt, database = paths
    return enqueue_locked_campaign(
        materialization_path=materialization,
        receipt_path=receipt,
        database_path=database,
    )


def test_enqueue_is_idempotent_and_ignores_legitimate_retry_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _campaign_fixture(tmp_path, monkeypatch)

    first = _call(paths)
    second = _call(paths)

    assert first["complete"] is True
    assert {row["disposition"] for row in first["jobs"]} == {"enqueued"}
    assert {row["disposition"] for row in second["jobs"]} == {"reused"}
    registry = Registry(paths[2])
    root_jobs = registry.list_queue(limit=100)
    assert len(root_jobs) == 30
    with registry.connect() as connection:
        memberships = connection.execute(
            "SELECT count(*) FROM campaign_variants WHERE campaign_id = ?",
            (_CAMPAIGN_ID,),
        ).fetchone()[0]
    assert memberships == 30

    original = root_jobs[0]
    registry.enqueue(
        campaign_id=_CAMPAIGN_ID,
        configuration=original["canonical_config"],
        command=original["command"],
        experiment_config_reference=original[
            "experiment_config_reference"
        ],
        priority=int(original["priority"]),
        maximum_attempts=2,
        requested_gpu=str(original["requested_gpu"]),
        attempt_count=2,
        retry_of=str(original["job_id"]),
    )

    after_retry = _call(paths)

    assert after_retry["complete"] is True
    assert {row["disposition"] for row in after_retry["jobs"]} == {"reused"}
    assert len(registry.list_queue(limit=100)) == 31
    receipt = json.loads(paths[1].read_text(encoding="utf-8"))
    checksum = receipt.pop("checksum")
    assert checksum == canonical_sha256(receipt)


def test_two_concurrent_invocations_create_only_thirty_root_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _campaign_fixture(tmp_path, monkeypatch)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: _call(paths), range(2)))

    dispositions = [
        {row["disposition"] for row in result["jobs"]}
        for result in results
    ]
    assert {"enqueued"} in dispositions
    assert {"reused"} in dispositions
    registry = Registry(paths[2])
    assert len(registry.list_queue(limit=100)) == 30


def test_all_configs_are_preflighted_before_first_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _campaign_fixture(tmp_path, monkeypatch)
    materialization = json.loads(paths[0].read_text(encoding="utf-8"))
    last_config = tmp_path / materialization["jobs"][-1]["config"]
    config = yaml.safe_load(last_config.read_text(encoding="utf-8"))
    config["trainer"]["max_epochs"] = 299
    last_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(LockedEnqueueError, match="locked plan"):
        _call(paths)

    registry = Registry(paths[2])
    assert registry.list_queue(limit=100) == []


def test_existing_root_job_semantics_are_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _campaign_fixture(tmp_path, monkeypatch)
    _call(paths)
    registry = Registry(paths[2])
    job = registry.list_queue(limit=100)[0]
    with registry.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE queue_jobs SET command_json = ? WHERE job_id = ?",
            (json.dumps(["wrong-command"]), job["job_id"]),
        )

    with pytest.raises(
        LockedEnqueueError, match="wrong command"
    ):
        _call(paths)


def test_unplanned_existing_root_job_blocks_locked_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _campaign_fixture(tmp_path, monkeypatch)
    materialization = json.loads(paths[0].read_text(encoding="utf-8"))
    config_path = tmp_path / materialization["jobs"][0]["config"]
    extra_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    extra_config["experiment"]["variant_label"] = "unexpected_extra_root"
    registry = Registry(paths[2])
    registry.enqueue(
        campaign_id=_CAMPAIGN_ID,
        configuration=extra_config,
        command=["unexpected-command"],
        experiment_config_reference=Path("unexpected.yaml"),
        priority=0,
        maximum_attempts=1,
        requested_gpu="0",
    )

    with pytest.raises(LockedEnqueueError, match="unexpected root"):
        _call(paths)

    with registry.connect() as connection:
        root_count = connection.execute(
            """
            SELECT count(*) FROM queue_jobs
            WHERE campaign_id = ? AND retry_of IS NULL
            """,
            (_CAMPAIGN_ID,),
        ).fetchone()[0]
    assert root_count == 1
