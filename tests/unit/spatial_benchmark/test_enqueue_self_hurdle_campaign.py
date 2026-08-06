from __future__ import annotations

import pytest

from scripts.train.enqueue_self_hurdle_campaign import (
    SelfHurdleEnqueueError,
    _jobs,
)


def test_stage_matrix_requires_exactly_two_prespecified_cores() -> None:
    materialization = {
        "resource_jobs": [
            {"alias": "ANC-03", "stage": "resource"},
            {"alias": "ANC-05", "stage": "resource"},
        ]
    }
    assert len(_jobs(materialization, "resource")) == 2
    materialization["resource_jobs"][1]["alias"] = "ANC-06"
    with pytest.raises(SelfHurdleEnqueueError, match="exactly"):
        _jobs(materialization, "resource")

