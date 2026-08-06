from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_PATH = _ROOT / "scripts/train/audit_adjacency_preparation_semantics.py"
_SPEC = importlib.util.spec_from_file_location("audit_adjacency_semantics", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_independent_graph_rebuild_has_exact_union_identity_and_position_null() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]]
    )
    raw_fov = np.asarray([20, 20, 20, 10, 10])

    rebuilt = _MODULE.rebuild_fixed_adjacencies(
        coordinates,
        raw_fov,
        alias="ANC-01",
        k=1,
        radius_um=1.1,
        null_seed=2026080202,
    )

    expected_off_diagonal = np.asarray(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=np.int64
    )
    np.testing.assert_array_equal(rebuilt["off_diagonal"], expected_off_diagonal)
    np.testing.assert_array_equal(
        rebuilt["identity"], np.stack([np.arange(5), np.arange(5)])
    )
    expected_null_off_diagonal = rebuilt["position_assignment"][
        expected_off_diagonal
    ]
    observed_null = rebuilt["position_permuted_null"]
    observed_null = observed_null[:, observed_null[0] != observed_null[1]]
    order = np.lexsort((expected_null_off_diagonal[1], expected_null_off_diagonal[0]))
    np.testing.assert_array_equal(observed_null, expected_null_off_diagonal[:, order])
    assert np.all(
        rebuilt["fov_group"][observed_null[0]]
        == rebuilt["fov_group"][observed_null[1]]
    )


def test_equal_core_standardizer_uses_equal_core_first_and_second_moments() -> None:
    moments = {
        "A": (np.asarray([0.0, 2.0]), np.asarray([0.0, 4.0])),
        "B": (np.asarray([2.0, 0.0]), np.asarray([4.0, 0.0])),
        "HELD-OUT": (np.asarray([999.0, 999.0]), np.asarray([999.0, 999.0])),
    }

    result = _MODULE.equal_core_standardizer(
        moments, ("A", "B"), scale_floor=1e-6
    )

    np.testing.assert_array_equal(result["mean"], np.asarray([1.0, 1.0]))
    np.testing.assert_array_equal(result["variance"], np.asarray([1.0, 1.0]))
    np.testing.assert_array_equal(result["scale"], np.asarray([1.0, 1.0]))
    assert len(result["checksum"]) == 64


def test_signed_receipt_detects_tampering_and_is_write_once(tmp_path: Path) -> None:
    receipt = _MODULE.signed_receipt(
        {"schema_version": 1, "receipt_kind": _MODULE.RECEIPT_KIND, "passed": True}
    )
    assert _MODULE._verify_signed(receipt, "test receipt") == receipt["checksum"]
    tampered = dict(receipt)
    tampered["passed"] = False
    with pytest.raises(_MODULE.SemanticAuditError, match="does not verify"):
        _MODULE._verify_signed(tampered, "test receipt")

    path = tmp_path / "receipt.json"
    _MODULE.write_once(path, receipt)
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(_MODULE.SemanticAuditError, match="refusing to overwrite"):
        _MODULE.write_once(path, receipt)
