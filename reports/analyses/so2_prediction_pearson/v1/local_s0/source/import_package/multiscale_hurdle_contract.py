"""Shared frozen identities for the multiscale hurdle-count campaign."""

from __future__ import annotations

from pathlib import Path


CAMPAIGN_ID = "cmp_20260729_multiscale_hurdle_count_pilot"
FROZEN_CONTRACT_SHA256 = (
    "54496bc73c8d6f2b3c2ec9885fba86b2021944462a106dbd10a23cfd88e25fb7"
)
FROZEN_CONTRACT_RELATIVE = (
    Path("experiments/campaigns")
    / CAMPAIGN_ID
    / "frozen_task_contract.yaml"
)
SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE = (
    Path("experiments/campaigns")
    / CAMPAIGN_ID
    / "contract_amendment_001.yaml"
)
SUPERSEDED_CONTRACT_AMENDMENT_SHA256 = (
    "3e263145fb33b86b3fcfd2b36daab7c14282b05c3ae15b0c8fbbfb0ed0f4638a"
)
ACTIVE_CONTRACT_AMENDMENT_RELATIVE = (
    Path("experiments/campaigns")
    / CAMPAIGN_ID
    / "contract_amendment_002.yaml"
)
ACTIVE_CONTRACT_AMENDMENT_SHA256 = (
    "6864fff7d35ce3af2312b8985a5fb7356b36bdd568bdcba4514ab0eaa3867cd2"
)
REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE = (
    Path("experiments/campaigns")
    / CAMPAIGN_ID
    / "contract_amendment_003.yaml"
)
REQUIRED_CONTRACT_SUPPLEMENT_SHA256 = (
    "6e89d83ccc909be16b7282d2dfb5cf826c0714ea6ce3923b33293cb96c4de09f"
)

ARMS = (
    "self",
    "self-regional",
    "self-regional-local",
    "self-regional-local-permuted",
)
ROUTING = {
    "self": ("surrogate", "surrogate"),
    "self-regional": ("true", "surrogate"),
    "self-regional-local": ("true", "true"),
    "self-regional-local-permuted": ("true", "permuted"),
}

LOCAL_SOURCE_PERMUTATION_SCHEMA = (
    "macroblock_spatial_antipode_sender_state_permutation_v1"
)
LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM = 0.99
LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM = 75.0
LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM = 0.90
LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM = 0.99


__all__ = [
    "ACTIVE_CONTRACT_AMENDMENT_RELATIVE",
    "ACTIVE_CONTRACT_AMENDMENT_SHA256",
    "ARMS",
    "CAMPAIGN_ID",
    "FROZEN_CONTRACT_RELATIVE",
    "FROZEN_CONTRACT_SHA256",
    "LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM",
    "LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM",
    "LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM",
    "LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM",
    "LOCAL_SOURCE_PERMUTATION_SCHEMA",
    "ROUTING",
    "REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE",
    "REQUIRED_CONTRACT_SUPPLEMENT_SHA256",
    "SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE",
    "SUPERSEDED_CONTRACT_AMENDMENT_SHA256",
]
