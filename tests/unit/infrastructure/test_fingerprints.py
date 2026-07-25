from __future__ import annotations

import os
from pathlib import Path

from spatial_benchmark.fingerprints import (
    build_path_fingerprint,
    fingerprint_dataset,
    fingerprint_path,
    fingerprint_split,
)


def test_path_fingerprint_is_sorted_and_does_not_dereference_symlinks(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "b.txt").write_text("beta", encoding="utf-8")
    (dataset / "a.txt").write_text("alpha", encoding="utf-8")
    protected = tmp_path / "protected.txt"
    protected.write_text("first secret value", encoding="utf-8")
    (dataset / "external-link").symlink_to(protected)

    before = fingerprint_path(dataset)
    manifest = build_path_fingerprint(dataset)
    protected.write_text("changed secret value", encoding="utf-8")
    after = fingerprint_path(dataset)

    assert before == after
    assert [entry["relative_path"] for entry in manifest.entries] == [
        "a.txt",
        "b.txt",
        "external-link",
    ]
    link = manifest.entries[-1]
    assert link["type"] == "symlink"
    assert "target" not in link


def test_dataset_fingerprint_does_not_mutate_inputs(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    file_path = source / "counts.bin"
    file_path.write_bytes(b"\x00\x01\x02")
    before = file_path.stat()

    first = fingerprint_dataset(
        {"counts": source},
        dataset_id="gastric_cosmx",
        source_version="v1",
    )
    second = fingerprint_dataset(
        {"counts": source},
        dataset_id="gastric_cosmx",
        source_version="v1",
    )
    after = file_path.stat()

    assert first == second
    assert (before.st_size, before.st_mtime_ns, before.st_mode) == (
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )


def test_split_fingerprint_is_row_order_independent() -> None:
    first = [
        {"sample_key": "sk_a", "split": "train", "fold": 0},
        {"sample_key": "sk_b", "split": "validation", "fold": 0},
    ]
    second = list(reversed(first))

    assert fingerprint_split(first, split_id="s1") == fingerprint_split(
        second, split_id="s1"
    )
    changed = [dict(first[0]), {**first[1], "split": "test"}]
    assert fingerprint_split(first, split_id="s1") != fingerprint_split(
        changed, split_id="s1"
    )
