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
_SCRIPT = (
    _ROOT
    / "scripts"
    / "train"
    / "enqueue_hybrid_count_campaign.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "enqueue_hybrid_count_campaign_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

EnqueueError = _MODULE.HybridCountEnqueueError
enqueue_stage = _MODULE.enqueue_stage


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("checksum", None)
    result["checksum"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pilot = [
        {
            "alias": "ANC-01",
            "arm": arm,
            "config": f"configs/pilot-{index}.yaml",
            "config_sha256": canonical_sha256(
                {"stage": "pilot", "arm": arm}
            ),
            "file_sha256": "f" * 64,
            "requested_gpu": index,
        }
        for index, arm in enumerate(_MODULE.ARMS)
    ]
    production = [
        {
            "alias": alias,
            "arm": arm,
            "config": f"configs/production-{alias.lower()}-{arm}.yaml",
            "config_sha256": canonical_sha256(
                {"stage": "production", "alias": alias, "arm": arm}
            ),
            "file_sha256": "f" * 64,
            "requested_gpu": sorted(_MODULE.SAFE_GPU_IDS)[
                index % len(_MODULE.SAFE_GPU_IDS)
            ],
        }
        for index, (alias, arm) in enumerate(
            (alias, arm)
            for alias in _MODULE.ALIASES
            for arm in _MODULE.ARMS
        )
    ]
    return pilot, production


def _materialization(project_root: Path) -> dict[str, Any]:
    contract = project_root / "contract.yaml"
    contract.write_text("frozen: true\n", encoding="utf-8")
    pilot, production = _jobs()
    return _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.MATERIALIZATION_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "counts": {
                "aliases": 10,
                "pilot_configs": 2,
                "production_configs": 20,
            },
            "registry_mutation_performed": False,
            "queue_mutation_performed": False,
            "training_performed": False,
            "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
            "allowed_gpu_ids": sorted(_MODULE.SAFE_GPU_IDS),
            "frozen_contract": {
                "reference": "contract.yaml",
                "sha256": _sha256_file(contract),
            },
            "cores": [
                {
                    "alias": alias,
                    "k1000_graph_sha256": canonical_sha256(
                        {"alias": alias}
                    ),
                    "k1000_directed_edges": 1_000,
                }
                for alias in _MODULE.ALIASES
            ],
            "pilot_jobs": pilot,
            "production_jobs": production,
            "pilot_gate_receipt_reference": "pilot_gate_receipt.json",
        }
    )


def _pilot_gate(
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    return _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.PILOT_GATE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "materialization_checksum": materialization["checksum"],
            "frozen_contract_sha256": materialization["frozen_contract"][
                "sha256"
            ],
            "thresholds": dict(_MODULE.PILOT_GATE_THRESHOLDS),
            "same_frozen_precision_batch": True,
            "same_evaluation_masks": True,
            "same_verified_graph": True,
            "failure_reasons": [],
            "gate_passed": True,
            "production_authorized": True,
            "jobs": [
                {
                    "alias": "ANC-01",
                    "arm": arm,
                    "verified_bundle": True,
                    "finite_losses_and_gradients": True,
                    "parameter_match": True,
                    "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
                    "precision_equivalence_passed": True,
                    "peak_vram_passed": True,
                    "projected_runtime_passed": True,
                    "runner_pilot_gate_passed": True,
                }
                for arm in _MODULE.ARMS
            ],
        }
    )


@pytest.fixture
def locked_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(_MODULE, "_PROJECT_ROOT", project_root)
    materialization = _materialization(project_root)
    materialization_path = project_root / "materialization.json"
    gate = _pilot_gate(materialization)
    gate_path = project_root / "pilot_gate_receipt.json"
    _write_json(materialization_path, materialization)
    _write_json(gate_path, gate)
    return {
        "project_root": project_root,
        "materialization": materialization,
        "materialization_path": materialization_path,
        "gate": gate,
        "gate_path": gate_path,
        "database_path": project_root / "state" / "tracking.sqlite3",
        "receipt_path": project_root / "enqueue_receipt.json",
    }


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"schema_version":1}\n',
        '{"value":NaN}\n',
        '{"value":-Infinity}\n',
    ],
)
def test_materialization_rejects_non_strict_json(
    locked_fixture: dict[str, Any],
    raw: str,
) -> None:
    path = locked_fixture["materialization_path"]
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(EnqueueError):
        _MODULE._load_materialization(path)


def test_materialization_checksum_tamper_fails_closed(
    locked_fixture: dict[str, Any],
) -> None:
    payload = deepcopy(locked_fixture["materialization"])
    payload["parameter_count"] += 1
    _write_json(locked_fixture["materialization_path"], payload)

    with pytest.raises(EnqueueError, match="checksum does not verify"):
        _MODULE._load_materialization(
            locked_fixture["materialization_path"]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("binding", "exact passing two-arm"),
        ("contract", "exact passing two-arm"),
        ("failed_gate", "exact passing two-arm"),
        ("duplicate_arm", "exact passing two-arm"),
        ("missing_job", "exactly two job mappings"),
        ("unverified_job", "unverified or invalid"),
    ],
)
def test_production_gate_requires_exact_bound_identity(
    locked_fixture: dict[str, Any],
    mutation: str,
    message: str,
) -> None:
    gate = deepcopy(locked_fixture["gate"])
    gate.pop("checksum")
    if mutation == "binding":
        gate["materialization_checksum"] = "0" * 64
    elif mutation == "contract":
        gate["frozen_contract_sha256"] = "1" * 64
    elif mutation == "failed_gate":
        gate["gate_passed"] = False
    elif mutation == "duplicate_arm":
        gate["jobs"][1]["arm"] = gate["jobs"][0]["arm"]
    elif mutation == "missing_job":
        gate["jobs"].pop()
    else:
        gate["jobs"][1]["verified_bundle"] = False
    gate = _signed(gate)
    _write_json(locked_fixture["gate_path"], gate)

    with pytest.raises(EnqueueError, match=message):
        _MODULE._load_pilot_gate(
            locked_fixture["gate_path"],
            materialization=locked_fixture["materialization"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "production_authorized",
        "threshold_value",
        "threshold_extra",
        "same_frozen_precision_batch",
        "same_evaluation_masks",
        "same_verified_graph",
        "failure_reasons",
        "parameter_count",
        "finite_losses_and_gradients",
        "parameter_match",
        "precision_equivalence_passed",
        "peak_vram_passed",
        "projected_runtime_passed",
        "runner_pilot_gate_passed",
    ],
)
def test_production_gate_rejects_resigned_evidence_tampering(
    locked_fixture: dict[str, Any],
    mutation: str,
) -> None:
    gate = deepcopy(locked_fixture["gate"])
    gate.pop("checksum")
    if mutation == "production_authorized":
        gate[mutation] = False
    elif mutation == "threshold_value":
        gate["thresholds"]["peak_allocated_vram_gib_maximum"] = 20.6
    elif mutation == "threshold_extra":
        gate["thresholds"]["unfrozen_threshold"] = 1
    elif mutation == "failure_reasons":
        gate[mutation] = ["hybrid-gat-k1000:synthetic_failure"]
    elif mutation in {
        "same_frozen_precision_batch",
        "same_evaluation_masks",
        "same_verified_graph",
    }:
        gate[mutation] = False
    elif mutation == "parameter_count":
        gate["jobs"][0][mutation] = _MODULE.EXPECTED_PARAMETER_COUNT + 1
    else:
        gate["jobs"][0][mutation] = False
    _write_json(locked_fixture["gate_path"], _signed(gate))

    with pytest.raises(EnqueueError, match="pilot gate"):
        _MODULE._load_pilot_gate(
            locked_fixture["gate_path"],
            materialization=locked_fixture["materialization"],
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


def test_invalid_last_config_preflight_performs_no_registry_writes(
    locked_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _WriteSpyRegistry()
    validated = 0
    total = 22

    monkeypatch.setattr(_MODULE, "Registry", lambda _path: spy)
    monkeypatch.setattr(
        _MODULE,
        "_existing_jobs_by_digest",
        lambda _registry: {},
    )

    def validate_job(
        raw: Mapping[str, Any],
        *,
        stage: str,
        materialization: Mapping[str, Any],
        registry: Any,
    ) -> tuple[dict[str, Any], Path, str, int, str]:
        nonlocal validated
        validated += 1
        if validated == total:
            raise EnqueueError("synthetic final-config preflight failure")
        alias = str(raw["alias"])
        arm = str(raw["arm"])
        config = {
            "experiment": {"biological_unit_alias": alias},
            "stage": stage,
            "arm": arm,
        }
        return (
            config,
            Path(str(raw["config"])),
            canonical_sha256(config),
            int(raw["requested_gpu"]),
            arm,
        )

    monkeypatch.setattr(_MODULE, "_validate_job", validate_job)

    with pytest.raises(EnqueueError, match="final-config"):
        enqueue_stage(
            stage="pilot",
            materialization_path=locked_fixture["materialization_path"],
            gate_path=locked_fixture["gate_path"],
            receipt_path=locked_fixture["receipt_path"],
            database_path=locked_fixture["database_path"],
        )

    assert validated == total
    assert spy.register_variant_calls == 0
    assert spy.enqueue_calls == 0
    assert not locked_fixture["receipt_path"].exists()


def test_invalid_production_gate_fails_before_registry_construction(
    locked_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = deepcopy(locked_fixture["gate"])
    gate.pop("checksum")
    gate["materialization_checksum"] = "0" * 64
    _write_json(locked_fixture["gate_path"], _signed(gate))
    registry_constructions = 0

    def forbidden_registry(_path: Path) -> None:
        nonlocal registry_constructions
        registry_constructions += 1
        raise AssertionError("invalid gate reached Registry")

    monkeypatch.setattr(_MODULE, "Registry", forbidden_registry)

    with pytest.raises(EnqueueError, match="exact passing two-arm"):
        enqueue_stage(
            stage="production",
            materialization_path=locked_fixture["materialization_path"],
            gate_path=locked_fixture["gate_path"],
            receipt_path=locked_fixture["receipt_path"],
            database_path=locked_fixture["database_path"],
        )

    assert registry_constructions == 0
    assert not locked_fixture["receipt_path"].exists()
