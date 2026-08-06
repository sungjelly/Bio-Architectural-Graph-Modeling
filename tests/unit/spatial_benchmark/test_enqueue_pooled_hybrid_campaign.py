from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest

from spatial_benchmark.identifiers import canonical_sha256


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "train" / "enqueue_pooled_hybrid_campaign.py"
_SPEC = importlib.util.spec_from_file_location(
    "enqueue_pooled_hybrid_campaign_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

EnqueueError = _MODULE.PooledHybridEnqueueError


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("checksum", None)
    result["checksum"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pilot = [
        {
            "arm": arm,
            "seed": 0,
            "config": f"configs/pilot-{index}.yaml",
            "config_sha256": canonical_sha256({"pilot": arm}),
            "file_sha256": "f" * 64,
            "requested_gpu": _MODULE.PILOT_GPU_BY_ARM[arm],
        }
        for index, arm in enumerate(_MODULE.ARMS)
    ]
    gpu_by_seed = dict(zip(_MODULE.SEEDS, sorted(_MODULE.SAFE_GPU_IDS)))
    production = [
        {
            "arm": arm,
            "seed": seed,
            "config": f"configs/production-{arm}-{seed}.yaml",
            "config_sha256": canonical_sha256(
                {"production": arm, "seed": seed}
            ),
            "file_sha256": "f" * 64,
            "requested_gpu": gpu_by_seed[seed],
        }
        for arm in _MODULE.ARMS
        for seed in _MODULE.SEEDS
    ]
    return pilot, production


@pytest.fixture
def locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    project = tmp_path / "project"
    project.mkdir()
    contract = project / "contract.yaml"
    contract.write_text("frozen: true\n", encoding="utf-8")
    contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
    monkeypatch.setattr(_MODULE, "_PROJECT_ROOT", project)
    monkeypatch.setattr(_MODULE, "CONTRACT_SHA256", contract_sha)
    pilot, production = _jobs()
    payload = _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.MATERIALIZATION_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "counts": {
                "aliases": 10,
                "pilot_configs": 2,
                "production_configs": 14,
                "production_seeds": 7,
            },
            "registry_mutation_performed": False,
            "queue_mutation_performed": False,
            "training_performed": False,
            "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
            "allowed_gpu_ids": sorted(_MODULE.SAFE_GPU_IDS),
            "frozen_contract": {
                "reference": "contract.yaml",
                "sha256": contract_sha,
            },
            "cohort": {"aliases": list(_MODULE.ALIASES)},
            "graph_sources": {
                alias: {"sha256": canonical_sha256({"graph": alias})}
                for alias in _MODULE.ALIASES
            },
            "evaluation_mask_sources": {
                alias: {"sha256": canonical_sha256({"masks": alias})}
                for alias in _MODULE.ALIASES
            },
            "pilot_jobs": pilot,
            "production_jobs": production,
            "pilot_gate_receipt_reference": "pilot_gate_receipt.json",
        }
    )
    materialization = project / "materialization.json"
    _write_json(materialization, payload)
    return {
        "project": project,
        "materialization": payload,
        "materialization_path": materialization,
        "gate_path": project / "pilot_gate_receipt.json",
        "receipt_path": project / "enqueue_receipt.json",
        "database_path": project / "state" / "tracking.sqlite3",
    }


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"schema_version":1}\n',
        '{"value":NaN}\n',
        '{"value":Infinity}\n',
    ],
)
def test_materialization_requires_strict_json(
    locked: dict[str, Any], raw: str
) -> None:
    locked["materialization_path"].write_text(raw, encoding="utf-8")
    with pytest.raises(EnqueueError):
        _MODULE._load_materialization(locked["materialization_path"])


def test_materialization_rejects_checksum_tampering(
    locked: dict[str, Any],
) -> None:
    payload = deepcopy(locked["materialization"])
    payload["counts"]["production_configs"] = 140
    _write_json(locked["materialization_path"], payload)
    with pytest.raises(EnqueueError, match="checksum does not verify"):
        _MODULE._load_materialization(locked["materialization_path"])


@pytest.mark.parametrize(
    "mutation",
    ["missing_member", "duplicate_seed", "unsafe_gpu", "old_140_job_shape"],
)
def test_materialization_rejects_non_pooled_job_plans(
    locked: dict[str, Any], mutation: str
) -> None:
    payload = deepcopy(locked["materialization"])
    payload.pop("checksum")
    if mutation == "missing_member":
        payload["production_jobs"].pop()
    elif mutation == "duplicate_seed":
        payload["production_jobs"][-1]["seed"] = 0
    elif mutation == "unsafe_gpu":
        payload["production_jobs"][0]["requested_gpu"] = 4
    else:
        payload["production_jobs"][0]["alias"] = "ANC-01"
        payload["production_jobs"][0].pop("seed")
    _write_json(locked["materialization_path"], _signed(payload))
    with pytest.raises(EnqueueError):
        _MODULE._load_materialization(locked["materialization_path"])


def _gate(materialization: Mapping[str, Any]) -> dict[str, Any]:
    jobs = []
    for arm in _MODULE.ARMS:
        jobs.append(
            {
                "arm": arm,
                "seed": 0,
                "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
                "materialization_checksum": materialization["checksum"],
                "checkpoint_id": f"checkpoint-{arm}",
                "checkpoint_sha256": canonical_sha256(
                    {"checkpoint": arm}
                ),
                "checkpoint_epoch": 1,
                "checkpoint_role": "last",
                **{field: True for field in _MODULE._PILOT_JOB_TRUE_FIELDS},
            }
        )
    return _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.PILOT_GATE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "materialization_checksum": materialization["checksum"],
            "frozen_contract_sha256": _MODULE.CONTRACT_SHA256,
            "thresholds": dict(_MODULE.PILOT_GATE_THRESHOLDS),
            **{field: True for field in _MODULE._PILOT_GATE_TRUE_FIELDS},
            "jobs": jobs,
            "failure_reasons": [],
            "gate_passed": True,
            "production_authorized": True,
        }
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "binding",
        "failed",
        "threshold",
        "missing_arm",
        "parameter",
        "host_memory_check",
        "disk_check",
        "frozen_batch_check",
        "checkpoint_check",
    ],
)
def test_production_gate_fails_closed_on_resigned_tampering(
    locked: dict[str, Any], mutation: str
) -> None:
    gate = _gate(locked["materialization"])
    gate.pop("checksum")
    if mutation == "binding":
        gate["materialization_checksum"] = "0" * 64
    elif mutation == "failed":
        gate["gate_passed"] = False
    elif mutation == "threshold":
        gate["thresholds"]["peak_allocated_vram_gib_maximum"] = 20.6
    elif mutation == "missing_arm":
        gate["jobs"].pop()
    elif mutation == "parameter":
        gate["jobs"][0]["parameter_count"] += 1
    elif mutation == "host_memory_check":
        gate["jobs"][0]["peak_host_memory_passed"] = False
    elif mutation == "disk_check":
        gate["jobs"][0]["projected_disk_passed"] = False
    elif mutation == "frozen_batch_check":
        gate["same_frozen_precision_batches_all_cores"] = False
    else:
        gate["jobs"][0]["checkpoint_verified"] = False
    _write_json(locked["gate_path"], _signed(gate))
    with pytest.raises(EnqueueError):
        _MODULE._load_pilot_gate(
            locked["gate_path"],
            materialization=locked["materialization"],
        )


class _WriteSpyRegistry:
    def __init__(self) -> None:
        self.register_variant_calls = 0
        self.enqueue_calls = 0

    def get_campaign(self, _campaign_id: str) -> dict[str, Any]:
        return {"campaign_id": _MODULE.CAMPAIGN_ID}

    def register_variant(self, *_args: Any, **_kwargs: Any) -> None:
        self.register_variant_calls += 1

    def enqueue(self, *_args: Any, **_kwargs: Any) -> None:
        self.enqueue_calls += 1


def test_last_config_failure_occurs_before_any_registry_write(
    locked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _WriteSpyRegistry()
    count = 0
    monkeypatch.setattr(_MODULE, "Registry", lambda _path: spy)
    monkeypatch.setattr(_MODULE, "_existing_jobs_by_digest", lambda _registry: {})

    def validate(
        raw: Mapping[str, Any],
        *,
        stage: str,
        materialization: Mapping[str, Any],
        registry: Any,
    ) -> tuple[dict[str, Any], Path, str, int, str, int]:
        nonlocal count
        count += 1
        if count == 16:
            raise EnqueueError("synthetic final-config failure")
        config = {"stage": stage, "arm": raw["arm"], "seed": raw["seed"]}
        return (
            config,
            Path(str(raw["config"])),
            canonical_sha256(config),
            int(raw["requested_gpu"]),
            str(raw["arm"]),
            int(raw["seed"]),
        )

    monkeypatch.setattr(_MODULE, "_validate_job", validate)
    with pytest.raises(EnqueueError, match="final-config"):
        _MODULE.enqueue_stage(
            stage="pilot",
            materialization_path=locked["materialization_path"],
            gate_path=locked["gate_path"],
            receipt_path=locked["receipt_path"],
            database_path=locked["database_path"],
        )
    assert count == 16
    assert spy.register_variant_calls == 0
    assert spy.enqueue_calls == 0
    assert not locked["receipt_path"].exists()


def test_invalid_production_gate_precedes_registry_construction(
    locked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate(locked["materialization"])
    gate.pop("checksum")
    gate["production_authorized"] = False
    _write_json(locked["gate_path"], _signed(gate))
    constructed = False

    def forbidden(_path: Path) -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("invalid gate reached registry")

    monkeypatch.setattr(_MODULE, "Registry", forbidden)
    with pytest.raises(EnqueueError):
        _MODULE.enqueue_stage(
            stage="production",
            materialization_path=locked["materialization_path"],
            gate_path=locked["gate_path"],
            receipt_path=locked["receipt_path"],
            database_path=locked["database_path"],
        )
    assert constructed is False
    assert not locked["receipt_path"].exists()


def test_tampered_existing_receipt_precedes_registry_construction(
    locked: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = _signed(
        {
            "schema_version": 1,
            "receipt_kind": "pooled_hybrid_count_pilot_enqueue_v1",
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "stage": "pilot",
            "materialization_checksum": locked["materialization"]["checksum"],
            "pilot_gate_checksum": None,
            "complete": False,
            "jobs": [],
        }
    )
    existing["complete"] = True
    _write_json(locked["receipt_path"], existing)
    constructed = False

    def forbidden(_path: Path) -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("tampered receipt reached registry")

    monkeypatch.setattr(_MODULE, "Registry", forbidden)
    with pytest.raises(EnqueueError, match="checksum does not verify"):
        _MODULE.enqueue_stage(
            stage="pilot",
            materialization_path=locked["materialization_path"],
            gate_path=locked["gate_path"],
            receipt_path=locked["receipt_path"],
            database_path=locked["database_path"],
        )
    assert constructed is False
