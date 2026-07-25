"""Deterministic, read-only dataset and split fingerprint utilities.

Directory traversal never follows symbolic links.  Regular files are streamed
through SHA-256 by default; callers handling very large immutable datasets may
set ``max_file_bytes`` to cap content hashing.  Unhashed entries then include
size and nanosecond modification time so changes remain detectable, albeit with
weaker guarantees.  Absolute protected paths are never part of a digest or
returned manifest.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .identifiers import canonical_json, canonical_sha256


class FingerprintError(ValueError):
    """Raised when a stable read-only fingerprint cannot be produced."""


@dataclass(frozen=True, slots=True)
class PathFingerprint:
    """A digest plus a local manifest and non-sensitive aggregate statistics.

    Relative names in ``entries`` can themselves be sensitive.  Keep that
    detailed manifest protected; use ``summary()`` for tracked registries.
    """

    sha256: str
    file_count: int
    total_bytes: int
    hashed_bytes: int
    entries: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        """Return aggregate metadata without file names."""

        return {
            "algorithm": "sha256",
            "fingerprint": self.sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "hashed_bytes": self.hashed_bytes,
        }


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one regular file and fail if it changes while being read."""

    source = Path(path)
    before = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise FingerprintError(f"Not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    after = source.stat(follow_symlinks=False)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise FingerprintError(f"File changed while fingerprinting: {source}")
    return digest.hexdigest()


def _manifest_entry(
    path: Path,
    relative_path: str,
    *,
    max_file_bytes: int | None,
) -> dict[str, Any]:
    metadata = path.lstat()
    mode = metadata.st_mode
    entry: dict[str, Any] = {"relative_path": relative_path}
    if stat.S_ISLNK(mode):
        target = os.readlink(path)
        entry.update(
            {
                "type": "symlink",
                "target_sha256": hashlib.sha256(
                    target.encode("utf-8", errors="surrogateescape")
                ).hexdigest(),
                "target_is_absolute": Path(target).is_absolute(),
            }
        )
    elif stat.S_ISREG(mode):
        entry.update({"type": "file", "size": metadata.st_size})
        if max_file_bytes is None or metadata.st_size <= max_file_bytes:
            entry["sha256"] = sha256_file(path)
            entry["checksum_status"] = "hashed"
        else:
            entry["mtime_ns"] = metadata.st_mtime_ns
            entry["checksum_status"] = "metadata_only"
    elif stat.S_ISDIR(mode):
        entry["type"] = "directory"
    else:
        entry.update(
            {
                "type": "special",
                "mode": stat.S_IFMT(mode),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
            }
        )
    return entry


def build_path_fingerprint(
    path: str | Path,
    *,
    max_file_bytes: int | None = None,
) -> PathFingerprint:
    """Build a sorted manifest and digest without following links or mutating data."""

    if max_file_bytes is not None and max_file_bytes < 0:
        raise FingerprintError("max_file_bytes must be non-negative or null.")
    root = Path(path)
    if not root.exists() and not root.is_symlink():
        raise FileNotFoundError(f"Fingerprint input was not found: {root}")

    entries: list[dict[str, Any]] = []
    root_metadata = root.lstat()
    if stat.S_ISDIR(root_metadata.st_mode):
        stack: list[tuple[Path, str]] = [(root, "")]
        while stack:
            directory, prefix = stack.pop()
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda item: item.name)
            subdirectories: list[tuple[Path, str]] = []
            for child in children:
                child_path = Path(child.path)
                relative = f"{prefix}/{child.name}".lstrip("/")
                entry = _manifest_entry(
                    child_path,
                    relative,
                    max_file_bytes=max_file_bytes,
                )
                entries.append(entry)
                if child.is_dir(follow_symlinks=False):
                    subdirectories.append((child_path, relative))
            stack.extend(reversed(subdirectories))
    else:
        entries.append(
            _manifest_entry(root, ".", max_file_bytes=max_file_bytes)
        )

    entries.sort(key=lambda item: item["relative_path"])
    manifest = {"version": 1, "entries": entries}
    files = [entry for entry in entries if entry["type"] == "file"]
    return PathFingerprint(
        sha256=canonical_sha256(manifest),
        file_count=len(files),
        total_bytes=sum(int(entry["size"]) for entry in files),
        hashed_bytes=sum(
            int(entry["size"])
            for entry in files
            if entry["checksum_status"] == "hashed"
        ),
        entries=tuple(entries),
    )


def fingerprint_path(
    path: str | Path,
    *,
    max_file_bytes: int | None = None,
) -> str:
    """Return only the stable SHA-256 path fingerprint."""

    return build_path_fingerprint(path, max_file_bytes=max_file_bytes).sha256


def fingerprint_dataset(
    paths: str
    | Path
    | Mapping[str, str | Path]
    | Sequence[str | Path],
    *,
    dataset_id: str | None = None,
    source_version: str | None = None,
    max_file_bytes: int | None = None,
) -> str:
    """Fingerprint one or more dataset roots without embedding protected paths."""

    if isinstance(paths, (str, Path)):
        labeled = [("dataset", Path(paths))]
    elif isinstance(paths, Mapping):
        labeled = [(str(label), Path(value)) for label, value in paths.items()]
    else:
        labeled = [(f"input_{index:04d}", Path(value)) for index, value in enumerate(paths)]
    labels = [label for label, _ in labeled]
    if len(labels) != len(set(labels)):
        raise FingerprintError("Dataset fingerprint labels must be unique.")
    inputs = []
    for label, path in sorted(labeled, key=lambda pair: pair[0]):
        result = build_path_fingerprint(path, max_file_bytes=max_file_bytes)
        inputs.append(
            {
                "label": label,
                "content_fingerprint": result.sha256,
                "file_count": result.file_count,
                "total_bytes": result.total_bytes,
                "hashed_bytes": result.hashed_bytes,
            }
        )
    return canonical_sha256(
        {
            "version": 1,
            "dataset_id": dataset_id,
            "source_version": source_version,
            "inputs": inputs,
        }
    )


def _load_split_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise FingerprintError(
                        f"Split JSONL line {line_number} is not a mapping."
                    )
                records.append(dict(value))
        return records
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise FingerprintError(
            f"Unsupported split format {suffix!r}; use CSV, JSON, JSONL, or YAML."
        )
    if isinstance(value, Mapping) and "assignments" in value:
        value = value["assignments"]
    if isinstance(value, Mapping):
        return [
            {"sample_key": str(sample_key), "split": split_name}
            for sample_key, split_name in value.items()
        ]
    if not isinstance(value, list) or any(
        not isinstance(record, Mapping) for record in value
    ):
        raise FingerprintError("Split records must be a list of mappings.")
    return [dict(record) for record in value]


def fingerprint_split(
    records_or_path: str | Path | Iterable[Mapping[str, Any]],
    *,
    split_id: str | None = None,
    method: str | None = None,
) -> str:
    """Hash logical split assignments independent of input row order.

    The function returns only a digest.  Sample keys and any protected grouping
    fields participate in the local one-way hash but are not returned or logged.
    """

    if isinstance(records_or_path, (str, Path)):
        source = Path(records_or_path)
        if not source.is_file():
            raise FileNotFoundError(f"Split input was not found: {source}")
        records = _load_split_records(source)
    else:
        records = [dict(record) for record in records_or_path]
    if not records:
        raise FingerprintError("Cannot fingerprint an empty split.")
    for index, record in enumerate(records):
        if not record:
            raise FingerprintError(f"Split record {index} is empty.")
        if not all(isinstance(key, str) for key in record):
            raise FingerprintError("Split record keys must be strings.")
    canonical_records = sorted(
        (json.loads(canonical_json(record)) for record in records),
        key=canonical_json,
    )
    return canonical_sha256(
        {
            "version": 1,
            "split_id": split_id,
            "method": method,
            "assignments": canonical_records,
        }
    )


dataset_fingerprint = fingerprint_dataset
split_fingerprint = fingerprint_split


__all__ = [
    "FingerprintError",
    "PathFingerprint",
    "build_path_fingerprint",
    "dataset_fingerprint",
    "fingerprint_dataset",
    "fingerprint_path",
    "fingerprint_split",
    "sha256_file",
    "split_fingerprint",
]
