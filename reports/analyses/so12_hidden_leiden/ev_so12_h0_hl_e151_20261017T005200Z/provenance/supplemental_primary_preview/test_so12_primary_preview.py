import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


_DIRECTORY = Path(__file__).resolve().parents[3] / "scripts/analysis"
sys.path.insert(0, str(_DIRECTORY))
_SPEC = importlib.util.spec_from_file_location("so12_primary_preview", _DIRECTORY / "render_so12_primary_preview.py")
preview = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preview)


def _primary(tmp_path, representation="h0"):
    folder = tmp_path / "clustering" / representation
    folder.mkdir(parents=True)
    (tmp_path / "cells.parquet").write_bytes(b"synthetic cell identity source")
    graph = {"input_sha256": {"cells.parquet": preview.sha256_file(tmp_path / "cells.parquet")}}
    graph_path = folder / "graph_receipt.json"
    graph_path.write_text(json.dumps(graph))
    labels_path = tmp_path / "clustering" / f"{representation}_labels.npy"
    np.save(labels_path, np.array([0, 1, 1, 0], dtype=np.int32))
    receipt = {"graph_receipt_sha256": preview.sha256_file(graph_path),
               "labels_sha256": preview.sha256_file(labels_path),
               "leiden": {"random_seed": 0, "resolution": 1.0, "n_iterations": -1, "cluster_count": 2}}
    receipt_path = folder / "leiden_seed0_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    return labels_path, receipt_path


def test_requires_genuine_receipt_and_checks_actual_primary_bytes(tmp_path):
    labels_path, receipt_path = _primary(tmp_path)
    labels, evidence = preview.load_verified_primary(tmp_path, "h0", expected_cells=4)
    np.testing.assert_array_equal(labels, [0, 1, 1, 0])
    assert evidence["leiden"]["cluster_count"] == 2
    np.save(labels_path, np.array([0, 0, 1, 1], dtype=np.int32))
    with pytest.raises(ValueError, match="label checksum"):
        preview.load_verified_primary(tmp_path, "h0", expected_cells=4)
    receipt_path.unlink()
    with pytest.raises(ValueError, match="not both available"):
        preview.load_verified_primary(tmp_path, "h0", expected_cells=4)


def test_rejects_seed_one_or_wrong_cell_identity(tmp_path):
    _, receipt_path = _primary(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["leiden"]["random_seed"] = 1
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="prescribed primary"):
        preview.load_verified_primary(tmp_path, "h0", expected_cells=4)
    receipt["leiden"]["random_seed"] = 0
    receipt_path.write_text(json.dumps(receipt))
    (tmp_path / "cells.parquet").write_bytes(b"different cell rows")
    with pytest.raises(ValueError, match="observation rows"):
        preview.load_verified_primary(tmp_path, "h0", expected_cells=4)
