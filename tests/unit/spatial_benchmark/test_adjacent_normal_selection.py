from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from spatial_benchmark.adjacent_normal_selection import (
    AdjacentNormalSelectionError,
    EXPECTED_LEGACY_LABEL_COUNTS,
    MINIMUM_CELLS,
    create_adjacent_normal_selection,
    load_adjacent_normal_route,
)
from spatial_benchmark.identifiers import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _clinical_tables(
    protected_dir: Path,
    *,
    duplicate_donor: bool = False,
    correction_index: int | None = None,
    unexpected_label: bool = False,
) -> tuple[Path, Path, Path, list[str]]:
    protected_dir.mkdir(parents=True, exist_ok=True)
    core_keys = [101 + index for index in range(7)] + [
        201 + index for index in range(7)
    ]
    restricted_donors = [
        f"SENSITIVE-DONOR-{index:02d}" for index in range(14)
    ]
    if duplicate_donor:
        restricted_donors[-1] = restricted_donors[0]
    labels = ["주변조직"] * 8 + ["주변조직 (N)"] * 6
    if unexpected_label:
        labels[0] = "주변조직-new-variant"

    legacy_path = protected_dir / "Gastric Study_Old.xlsx"
    pd.DataFrame(
        list(zip(core_keys, restricted_donors, labels))
    ).to_excel(legacy_path, header=False, index=False)

    corrections: list[object] = [np.nan] * 14
    if correction_index is not None:
        corrections[correction_index] = "SENSITIVE-CORRECTION-TEXT"
    review_path = protected_dir / "Gastric Study.xlsx"
    pd.DataFrame(
        {
            "슬라이드번호": core_keys,
            "진단명": ["Normal"] * 14,
            "결과": ["정상조직확인"] * 14,
            "비고 (수정진단)": corrections,
        }
    ).to_excel(review_path, index=False)

    map_path = protected_dir / "fov_core_map.csv"
    pd.DataFrame(
        {
            "slide": ["SO_1"] * 7 + ["SO_2"] * 7,
            "core_label": core_keys,
            "fov": list(range(1, 8)) * 2,
        }
    ).to_csv(map_path, index=False)
    return legacy_path, review_path, map_path, restricted_donors


def _metadata_tables(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for slide_index, slide in enumerate(("SO_1", "SO_2"), start=1):
        frames = []
        for fov, count in enumerate(
            range(MINIMUM_CELLS, MINIMUM_CELLS + 7),
            start=1,
        ):
            frames.append(
                pd.DataFrame(
                    {
                        "fov": np.full(count, fov, dtype=np.int32),
                        "cell_ID": np.arange(1, count + 1, dtype=np.int32),
                        "slide_ID": np.full(count, slide_index, dtype=np.int8),
                    }
                )
            )
        metadata = pd.concat(frames, ignore_index=True)
        metadata.to_csv(
            raw_dir / f"26040302{slide}_metadata_file.csv",
            index=False,
        )


def _valid_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, list[str]]:
    protected_dir = tmp_path / "protected" / "clinical"
    legacy, review, core_map, donors = _clinical_tables(protected_dir)
    raw_dir = tmp_path / "protected" / "raw"
    _metadata_tables(raw_dir)
    return legacy, review, core_map, raw_dir, donors


def test_balanced_selection_is_deterministic_provenanced_and_deidentified(
    tmp_path: Path,
) -> None:
    legacy, review, core_map, raw_dir, restricted_donors = _valid_inputs(
        tmp_path
    )
    first_path = tmp_path / "generated" / "selection-a.json"
    second_path = tmp_path / "generated" / "selection-b.json"

    first = create_adjacent_normal_selection(
        legacy_workbook=legacy,
        pathology_review_workbook=review,
        core_map_csv=core_map,
        raw_dir=raw_dir,
        output_path=first_path,
        chunksize=997,
    )
    second = create_adjacent_normal_selection(
        legacy_workbook=legacy,
        pathology_review_workbook=review,
        core_map_csv=core_map,
        raw_dir=raw_dir,
        output_path=second_path,
        chunksize=4096,
    )

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    public = first.to_public_dict()
    assert public == {
        "status": "created",
        "routes": [
            {"alias": "ANC-01", "slide": "SO_1", "fovs": [4]},
            {"alias": "ANC-02", "slide": "SO_1", "fovs": [3]},
            {"alias": "ANC-03", "slide": "SO_1", "fovs": [5]},
            {"alias": "ANC-04", "slide": "SO_1", "fovs": [2]},
            {"alias": "ANC-05", "slide": "SO_1", "fovs": [6]},
            {"alias": "ANC-06", "slide": "SO_2", "fovs": [4]},
            {"alias": "ANC-07", "slide": "SO_2", "fovs": [3]},
            {"alias": "ANC-08", "slide": "SO_2", "fovs": [5]},
            {"alias": "ANC-09", "slide": "SO_2", "fovs": [2]},
            {"alias": "ANC-10", "slide": "SO_2", "fovs": [6]},
        ],
    }
    public_text = json.dumps(public, sort_keys=True)
    assert "protected_core_key" not in public_text
    assert not any(value in public_text for value in restricted_donors)

    payload = json.loads(first_path.read_text(encoding="utf-8"))
    serialized = first_path.read_text(encoding="utf-8")
    assert payload["selection_policy"]["minimum_cells_inclusive"] == 5_001
    assert payload["aggregate_validation"]["legacy_source_label_counts"] == (
        EXPECTED_LEGACY_LABEL_COUNTS
    )
    assert payload["aggregate_validation"]["distinct_donor_count"] == 14
    assert payload["aggregate_validation"]["selected_count_by_slide"] == {
        "SO_1": 5,
        "SO_2": 5,
    }
    assert [record["protected_core_key"] for record in payload["cores"]] == [
        104,
        103,
        105,
        102,
        106,
        204,
        203,
        205,
        202,
        206,
    ]
    assert payload["confidentiality"]["contains_raw_donor_ids"] is False
    assert not any(value in serialized for value in restricted_donors)
    assert "donor_digest" not in serialized
    checksum = payload.pop("checksum")
    assert checksum["value"] == canonical_sha256(payload)
    assert os.stat(first_path).st_mode & 0o777 == 0o600
    loaded = load_adjacent_normal_route(first_path, "ANC-01")
    assert loaded == first.routes[0]
    assert asdict(loaded) == {
        "alias": "ANC-01",
        "slide": "SO_1",
        "fovs": (4,),
    }
    assert "protected_core" not in repr(loaded)


def test_cli_stdout_contains_only_public_alias_and_routing(
    tmp_path: Path,
) -> None:
    legacy, review, core_map, raw_dir, restricted_donors = _valid_inputs(
        tmp_path
    )
    output = tmp_path / "generated" / "cli-selection.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/data/select_adjacent_normal_cores.py",
            "--legacy-workbook",
            str(legacy),
            "--pathology-review-workbook",
            str(review),
            "--core-map",
            str(core_map),
            "--raw-dir",
            str(raw_dir),
            "--output",
            str(output),
            "--chunksize",
            "1301",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    public = json.loads(result.stdout)
    assert set(public) == {"status", "routes"}
    assert len(public["routes"]) == 10
    assert all(set(route) == {"alias", "slide", "fovs"} for route in public["routes"])
    assert "protected_core_key" not in result.stdout
    assert not any(value in result.stdout for value in restricted_donors)
    assert result.stderr == ""
    assert output.is_file()


@pytest.mark.parametrize(
    ("fixture_options", "message"),
    [
        ({"duplicate_donor": True}, "distinct donors"),
        ({"correction_index": 3}, "blank-correction"),
        ({"unexpected_label": True}, "unapproved"),
    ],
)
def test_selector_fails_closed_without_exposing_restricted_values(
    tmp_path: Path,
    fixture_options: dict[str, object],
    message: str,
) -> None:
    protected_dir = tmp_path / "protected" / "clinical"
    legacy, review, core_map, restricted_donors = _clinical_tables(
        protected_dir,
        **fixture_options,
    )
    with pytest.raises(AdjacentNormalSelectionError, match=message) as error:
        create_adjacent_normal_selection(
            legacy_workbook=legacy,
            pathology_review_workbook=review,
            core_map_csv=core_map,
            raw_dir=tmp_path / "protected" / "raw",
            output_path=tmp_path / "generated" / "selection.json",
        )
    text = str(error.value)
    assert not any(value in text for value in restricted_donors)
    assert "SENSITIVE-CORRECTION-TEXT" not in text


def test_output_cannot_modify_protected_inputs_or_overwrite_implicitly(
    tmp_path: Path,
) -> None:
    legacy, review, core_map, raw_dir, _ = _valid_inputs(tmp_path)
    protected_output = legacy.parent / "selection.json"
    with pytest.raises(AdjacentNormalSelectionError, match="immutable"):
        create_adjacent_normal_selection(
            legacy_workbook=legacy,
            pathology_review_workbook=review,
            core_map_csv=core_map,
            raw_dir=raw_dir,
            output_path=protected_output,
        )
    output = tmp_path / "generated" / "selection.json"
    output.parent.mkdir(parents=True)
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        create_adjacent_normal_selection(
            legacy_workbook=legacy,
            pathology_review_workbook=review,
            core_map_csv=core_map,
            raw_dir=raw_dir,
            output_path=output,
        )
    assert output.read_text(encoding="utf-8") == "sentinel"


def _write_rechecksummed(path: Path, payload: dict[str, object]) -> None:
    checksum = payload.get("checksum")
    assert isinstance(checksum, dict)
    digest_payload = dict(payload)
    digest_payload.pop("checksum")
    checksum["value"] = canonical_sha256(digest_payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def test_route_loader_fails_closed_on_tampering_and_inconsistent_records(
    tmp_path: Path,
) -> None:
    legacy, review, core_map, raw_dir, _ = _valid_inputs(tmp_path)
    source = tmp_path / "generated" / "selection.json"
    create_adjacent_normal_selection(
        legacy_workbook=legacy,
        pathology_review_workbook=review,
        core_map_csv=core_map,
        raw_dir=raw_dir,
        output_path=source,
    )
    original = json.loads(source.read_text(encoding="utf-8"))

    checksum_tamper = deepcopy(original)
    checksum_tamper["routes"][0]["fovs"] = [7]
    checksum_path = tmp_path / "generated" / "checksum-tamper.json"
    checksum_path.write_text(json.dumps(checksum_tamper), encoding="utf-8")
    with pytest.raises(AdjacentNormalSelectionError, match="checksum verification"):
        load_adjacent_normal_route(checksum_path, "ANC-01")

    schema_tamper = deepcopy(original)
    schema_tamper["schema_version"] = 99
    schema_path = tmp_path / "generated" / "schema-tamper.json"
    _write_rechecksummed(schema_path, schema_tamper)
    with pytest.raises(AdjacentNormalSelectionError, match="unsupported schema"):
        load_adjacent_normal_route(schema_path, "ANC-01")

    policy_tamper = deepcopy(original)
    policy_tamper["selection_policy"]["minimum_cells_inclusive"] = 1
    policy_path = tmp_path / "generated" / "policy-tamper.json"
    _write_rechecksummed(policy_path, policy_tamper)
    with pytest.raises(AdjacentNormalSelectionError, match="locked"):
        load_adjacent_normal_route(policy_path, "ANC-01")

    route_tamper = deepcopy(original)
    route_tamper["routes"][0]["fovs"] = [7]
    route_path = tmp_path / "generated" / "route-tamper.json"
    _write_rechecksummed(route_path, route_tamper)
    with pytest.raises(AdjacentNormalSelectionError, match="disagree"):
        load_adjacent_normal_route(route_path, "ANC-01")

    protected_key_tamper = deepcopy(original)
    protected_key_tamper["cores"][1]["protected_core_key"] = (
        protected_key_tamper["cores"][0]["protected_core_key"]
    )
    key_path = tmp_path / "generated" / "key-tamper.json"
    _write_rechecksummed(key_path, protected_key_tamper)
    with pytest.raises(AdjacentNormalSelectionError, match="not unique"):
        load_adjacent_normal_route(key_path, "ANC-01")

    alias_tamper = deepcopy(original)
    alias_tamper["routes"][1]["alias"] = "ANC-01"
    alias_tamper["cores"][1]["alias"] = "ANC-01"
    alias_path = tmp_path / "generated" / "alias-tamper.json"
    _write_rechecksummed(alias_path, alias_tamper)
    with pytest.raises(AdjacentNormalSelectionError, match="exact ten aliases"):
        load_adjacent_normal_route(alias_path, "ANC-01")

    with pytest.raises(AdjacentNormalSelectionError, match="absent"):
        load_adjacent_normal_route(source, "ANC-99")
