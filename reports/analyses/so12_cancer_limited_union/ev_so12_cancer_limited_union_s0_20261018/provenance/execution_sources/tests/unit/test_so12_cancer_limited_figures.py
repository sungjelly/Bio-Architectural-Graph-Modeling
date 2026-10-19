import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/analysis'))
from so12_cancer_limited_figures import group_image, error_image, rgb, GROUP_COLORS, ERROR_COLORS


def test_full_cluster_spillover_and_missed_cells_stay_visible():
    owner = np.array([[0, 1, 2, 3, -1]], dtype=np.int32)
    labels = np.array([7, 7, 2, 2])
    valid = np.ones_like(owner, dtype=bool)
    roi = np.array([[True, False, True, False, True]])
    group = group_image(owner, valid, labels, [7])
    assert np.array_equal(group[0, 0], group[0, 1])
    assert np.array_equal(group[0, 2], group[0, 3])
    assert np.array_equal(group[0, 0], rgb(GROUP_COLORS['selected']))
    error, counts = error_image(owner, valid, roi, labels, [7], dict(tp=1, fp=1, fn=1, tn=1))
    assert counts == dict(tp=1, fp=1, fn=1, tn=1)
    for index, name in enumerate(['tp', 'fp', 'fn', 'tn']):
        assert np.array_equal(error[0, index], rgb(ERROR_COLORS[name]))
    assert (error[0, 4] == 255).all()


def test_error_map_rejects_forgiven_spillover():
    owner = np.array([[0, 1]], dtype=np.int32)
    with pytest.raises(RuntimeError, match='fp count differs'):
        error_image(owner, np.ones_like(owner, bool), np.array([[True, False]]),
                    np.array([0, 0]), [0], dict(tp=1, fp=0, fn=0, tn=0))
