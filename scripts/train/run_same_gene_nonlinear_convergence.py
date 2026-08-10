#!/usr/bin/env python3
"""Launch the YAML-bound long-horizon same-gene convergence audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "cmp_20260810_same_gene_nonlinear_convergence"
CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "campaigns" / CAMPAIGN_ID
CONTRACT_PATH = CAMPAIGN_DIR / "frozen_task_contract.yaml"
CAMPAIGN_PATH = CAMPAIGN_DIR / "campaign.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a YAML mapping: {path}")
    return payload


def _verify_and_load_contract() -> tuple[dict[str, Any], dict[str, Any], str]:
    contract = _yaml(CONTRACT_PATH)
    campaign = _yaml(CAMPAIGN_PATH)
    contract_sha = _sha256(CONTRACT_PATH)
    if campaign.get("frozen_contract_sha256") != contract_sha:
        raise RuntimeError("campaign metadata does not bind the frozen contract")
    identity = contract.get("code_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("frozen contract has no code_identity mapping")
    paths = {
        "wrapper_sha256": Path(__file__).resolve(),
        "base_runner_sha256": PROJECT_ROOT
        / "scripts/train/run_same_gene_nonlinear.py",
        "model_sha256": PROJECT_ROOT
        / "src/spatial_benchmark/same_gene_nonlinear.py",
        "kernel_unit_test_sha256": PROJECT_ROOT
        / "tests/unit/spatial_benchmark/test_same_gene_nonlinear.py",
        "convergence_unit_test_sha256": PROJECT_ROOT
        / "tests/unit/spatial_benchmark/test_same_gene_nonlinear_convergence.py",
    }
    for key, path in paths.items():
        if identity.get(key) != _sha256(path):
            raise RuntimeError(f"frozen source identity mismatch: {key}")
    return contract, campaign, contract_sha


CONTRACT, CAMPAIGN, CONTRACT_SHA256 = _verify_and_load_contract()

import run_same_gene_nonlinear as runner  # noqa: E402


runner.CAMPAIGN_ID = str(CONTRACT["campaign_id"])
runner.CAMPAIGN_DISPLAY_NAME = str(CAMPAIGN["display_name"])
runner.CAMPAIGN_SCIENTIFIC_QUESTION = str(CONTRACT["question"])
runner.EXPERIMENT_FLAVOR = "convergence"
runner.FROZEN_CONTRACT_SHA256 = CONTRACT_SHA256
runner.EPOCH_CANDIDATES = tuple(int(value) for value in CONTRACT["model"]["epoch_candidates"])
runner.PILOT_EPOCH_CANDIDATES = runner.EPOCH_CANDIDATES
runner.PILOT_ARMS = tuple(str(value) for value in CONTRACT["resource_pilot"]["arms"])
runner.PILOT_REFIT_EPOCH_OVERRIDE = int(
    CONTRACT["resource_pilot"]["forced_final_refit_epochs"]
)
runner.ANCHOR_EPOCH = int(CONTRACT["model"]["anchor_epoch"])
runner.PROVENANCE_SOURCE_PATHS = (
    "src/spatial_benchmark/same_gene_jacobian.py",
    "src/spatial_benchmark/same_gene_nonlinear.py",
    "scripts/data/prepare_same_gene_jacobian.py",
    "scripts/train/run_same_gene_nonlinear.py",
    "scripts/train/run_same_gene_nonlinear_convergence.py",
    "tests/unit/spatial_benchmark/test_same_gene_nonlinear.py",
    "tests/unit/spatial_benchmark/test_same_gene_nonlinear_convergence.py",
    "experiments/campaigns/cmp_20260810_same_gene_nonlinear_convergence/README.md",
    "experiments/campaigns/cmp_20260810_same_gene_nonlinear_convergence/campaign.yaml",
    "experiments/campaigns/cmp_20260810_same_gene_nonlinear_convergence/frozen_task_contract.yaml",
)


if __name__ == "__main__":
    runner.main()
