"""Validation-only standards locking and concrete execution matrices."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


FORMAT_VERSION = 1
ARTIFACT_KIND = "normal_spatial_benchmark_standards_lock"
LOCK_KIND = "validation_only_standards_lock"
REQUIRED_FINAL_CONDITIONS = (
    "b0",
    "b0_parameter_matched",
    "broad_field",
    "b1",
    "g1_true",
    "g1_rewired",
    "g2_true",
    "g2_zero",
    "g2_distance_only",
    "g2_permuted",
)
_TEST_STATE_KEYS = {
    "evaluate_test",
    "open_test",
    "sealed_test_opened",
    "test_metrics_used",
    "test_opened",
    "test_targets_evaluated",
}
_NONEMPTY_TEST_KEYS = {
    "test",
    "test_metrics",
    "test_results",
}
_REPRESENTATION_FIELDS = {
    "hidden_dim",
    "graph_layers",
    "edge_embedding_dim",
}


class StandardsLockError(ValueError):
    """Raised when validation evidence cannot support an immutable lock."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_job_hash(job: Mapping[str, Any]) -> str:
    """Return the full canonical hash for one expanded execution-matrix job."""

    if not isinstance(job, Mapping):
        raise StandardsLockError("A locked execution job must be a mapping")
    return _canonical_hash(dict(job))


def _manifest_content_hash(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("manifest_content_sha256", None)
    return _canonical_hash(core)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StandardsLockError(
            f"Could not read JSON artifact {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise StandardsLockError(
            f"JSON artifact must contain an object: {path}"
        )
    return dict(value)


def _has_content(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes, Sequence, Mapping)):
        return len(value) > 0
    return bool(value)


def _assert_recursively_sealed(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            item_context = f"{context}.{raw_key}"
            if key in _TEST_STATE_KEYS and item is not False:
                raise StandardsLockError(
                    f"Recommendation is not validation-only: {item_context}"
                )
            if key in _NONEMPTY_TEST_KEYS and _has_content(item):
                raise StandardsLockError(
                    f"Recommendation contains test results: {item_context}"
                )
            if key in {"split", "evaluation_split", "selection_split"} and (
                str(item).strip().lower() == "test"
            ):
                raise StandardsLockError(
                    f"Recommendation selects the test split: {item_context}"
                )
            _assert_recursively_sealed(item, context=item_context)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_recursively_sealed(
                item,
                context=f"{context}[{index}]",
            )


def _selection_stage(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    aliases = {
        "graph": "graph",
        "graph_structure": "graph",
        "mask": "mask",
        "masking": "mask",
        "curriculum": "mask",
        "representation": "representation",
        "hidden": "hidden",
        "hidden_dim": "hidden",
        "hidden_width": "hidden",
        "depth": "depth",
        "graph_depth": "depth",
        "graph_layers": "depth",
        "edge_embedding": "edge_embedding",
        "edge_embedding_dim": "edge_embedding",
        "g2_edge_embedding": "edge_embedding",
        "g3_eligibility": "g3_eligibility",
    }
    if text not in aliases:
        raise StandardsLockError(
            f"Unsupported recommendation selection stage {value!r}"
        )
    return aliases[text]


def _require_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise StandardsLockError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise StandardsLockError(f"{name} must be an integer") from exc
    if result != value:
        raise StandardsLockError(f"{name} must be an integer")
    return result


def _require_positive_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StandardsLockError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise StandardsLockError(f"{name} must be finite and positive")
    return result


def _merge_recommendations(
    recommendation_paths: Sequence[Path],
    *,
    minimum_confirmation_seeds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    if not recommendation_paths:
        raise StandardsLockError(
            "At least one validation recommendation is required"
        )
    graph_standard: dict[str, Any] | None = None
    mask_standard: dict[str, Any] | None = None
    representation: dict[str, Any] = {}
    g3_eligible = False
    sources: list[dict[str, Any]] = []
    for sequence_number, path in enumerate(recommendation_paths):
        recommendation = _read_json(path)
        _assert_recursively_sealed(
            recommendation,
            context=path.name,
        )
        if recommendation.get("locked") is not True:
            raise StandardsLockError(
                f"Recommendation is not locked: {path}"
            )
        if recommendation.get("test_metrics_used") is not False:
            raise StandardsLockError(
                f"Recommendation lacks an explicit unused-test seal: {path}"
            )
        required_seeds = _require_integer(
            recommendation.get("required_seeds"),
            name=f"{path.name}.required_seeds",
        )
        if required_seeds < minimum_confirmation_seeds:
            raise StandardsLockError(
                f"Recommendation {path.name} has only {required_seeds} "
                f"confirmation seeds; require at least "
                f"{minimum_confirmation_seeds}"
            )
        candidate_id = recommendation.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise StandardsLockError(
                f"Recommendation lacks a candidate_id: {path}"
            )
        standard = recommendation.get("standard")
        if not isinstance(standard, Mapping):
            raise StandardsLockError(
                f"Recommendation lacks a standard mapping: {path}"
            )
        stage = _selection_stage(recommendation.get("selection"))
        standard = dict(standard)
        if stage == "graph":
            candidate = {
                "k": _require_integer(standard.get("k"), name="graph.k"),
                "radius_um": _require_positive_float(
                    standard.get("radius_um"),
                    name="graph.radius_um",
                ),
                "symmetry": str(standard.get("symmetry")),
                "min_distance_um": float(
                    standard.get("min_distance_um", 0.0)
                ),
            }
            if candidate["k"] <= 0:
                raise StandardsLockError("graph.k must be positive")
            if candidate["symmetry"] not in {"union", "mutual"}:
                raise StandardsLockError(
                    "graph.symmetry must be union or mutual"
                )
            if (
                not math.isfinite(candidate["min_distance_um"])
                or candidate["min_distance_um"] < 0
            ):
                raise StandardsLockError(
                    "graph.min_distance_um must be finite and nonnegative"
                )
            if graph_standard is not None and graph_standard != candidate:
                raise StandardsLockError(
                    "Conflicting locked graph recommendations"
                )
            graph_standard = candidate
        elif stage == "mask":
            curriculum = str(standard.get("curriculum"))
            if curriculum not in {"P-only", "P+N", "P+N+B"}:
                raise StandardsLockError(
                    "Mask curriculum must be P-only, P+N, or P+N+B"
                )
            candidate = {"curriculum": curriculum}
            if mask_standard is not None and mask_standard != candidate:
                raise StandardsLockError(
                    "Conflicting locked mask recommendations"
                )
            mask_standard = candidate
        elif stage == "g3_eligibility":
            if standard.get("eligible") is not True:
                raise StandardsLockError(
                    "G3 eligibility recommendation is not positive"
                )
            g3_eligible = True
        else:
            if stage == "hidden":
                selected_fields = {"hidden_dim"}
            elif stage == "depth":
                selected_fields = {"graph_layers"}
            elif stage == "edge_embedding":
                selected_fields = {"edge_embedding_dim"}
            else:
                selected_fields = _REPRESENTATION_FIELDS
            supplied = selected_fields.intersection(standard)
            if supplied != selected_fields:
                missing = sorted(selected_fields.difference(supplied))
                raise StandardsLockError(
                    f"Representation recommendation {path.name} lacks "
                    f"{', '.join(missing)}"
                )
            for field in sorted(selected_fields):
                value = standard[field]
                if value is None:
                    raise StandardsLockError(
                        f"Representation field {field} cannot be null"
                    )
                selected_value = _require_integer(
                    value,
                    name=f"representation.{field}",
                )
                if (
                    field in representation
                    and representation[field] != selected_value
                ):
                    raise StandardsLockError(
                        f"Conflicting locked representation field {field}"
                    )
                representation[field] = selected_value
        sources.append(
            {
                "sequence": sequence_number,
                "file": path.name,
                "sha256": _sha256_file(path),
                "selection": stage,
                "candidate_id": candidate_id,
                "required_seeds": required_seeds,
                "test_metrics_used": False,
            }
        )
    missing_standards = []
    if graph_standard is None:
        missing_standards.append("graph")
    if mask_standard is None:
        missing_standards.append("mask")
    missing_standards.extend(
        sorted(_REPRESENTATION_FIELDS.difference(representation))
    )
    if missing_standards:
        raise StandardsLockError(
            "Recommendations do not lock: " + ", ".join(missing_standards)
        )
    assert graph_standard is not None
    assert mask_standard is not None
    if representation["hidden_dim"] not in {128, 256, 512}:
        raise StandardsLockError(
            "hidden_dim must be selected from 128, 256, or 512"
        )
    if representation["graph_layers"] not in {1, 2}:
        raise StandardsLockError(
            "graph_layers must be selected from 1 or 2"
        )
    if representation["edge_embedding_dim"] not in {16, 32, 64}:
        raise StandardsLockError(
            "edge_embedding_dim must be selected from 16, 32, or 64"
        )
    standards = {
        "graph": graph_standard,
        "masking": mask_standard,
        "representation": representation,
    }
    return standards, sources, g3_eligible


def _normalise_seeds(
    values: Iterable[int],
    *,
    name: str,
    exact_count: int | None = None,
    minimum_count: int | None = None,
) -> list[int]:
    seeds = [_require_integer(value, name=name) for value in values]
    if len(seeds) != len(set(seeds)):
        raise StandardsLockError(f"{name} must be unique")
    if any(seed < 0 for seed in seeds):
        raise StandardsLockError(f"{name} must be nonnegative")
    seeds = sorted(seeds)
    if exact_count is not None and len(seeds) != exact_count:
        raise StandardsLockError(
            f"{name} must contain exactly {exact_count} seeds"
        )
    if minimum_count is not None and len(seeds) < minimum_count:
        raise StandardsLockError(
            f"{name} must contain at least {minimum_count} seeds"
        )
    return seeds


def _common_training(
    standards: Mapping[str, Any],
    *,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    return {
        "curriculum": standards["masking"]["curriculum"],
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "amp": True,
    }


def _graph_values(standards: Mapping[str, Any]) -> dict[str, Any]:
    return dict(standards["graph"])


def _model_values(
    standards: Mapping[str, Any],
    model: str,
) -> dict[str, Any]:
    representation = standards["representation"]
    values = {"hidden_dim": representation["hidden_dim"]}
    if model in {"b0-matched", "g1", "g2"}:
        values["graph_layers"] = representation["graph_layers"]
    if model in {"g2", "g3"}:
        values["edge_embedding_dim"] = representation["edge_embedding_dim"]
    return values


def _condition_values(
    condition: str,
    seed: int,
    standards: Mapping[str, Any],
    *,
    max_epochs: int,
    patience: int,
    rewire_seed: int,
) -> dict[str, Any]:
    model_by_condition = {
        "b0": "b0",
        "b0_parameter_matched": "b0-matched",
        "broad_field": "broad-field",
        "b1": "b1",
        "g1_true": "g1",
        "g1_rewired": "g1",
        "g2_true": "g2",
        "g2_zero": "g2",
        "g2_distance_only": "g2",
        "g2_permuted": "g2",
    }
    model = model_by_condition[condition]
    values = {
        "model": model,
        "seed": int(seed),
        **_common_training(
            standards,
            max_epochs=max_epochs,
            patience=patience,
        ),
        **_model_values(standards, model),
    }
    if condition not in {
        "b0",
        "b0_parameter_matched",
        "broad_field",
    }:
        values.update(_graph_values(standards))
    if condition == "g1_rewired":
        values.update(
            {
                "rewired": True,
                "rewire_seed": int(rewire_seed),
                "swaps_per_edge": 1.0,
            }
        )
    edge_control_by_condition = {
        "g2_true": "none",
        "g2_zero": "zero",
        "g2_distance_only": "distance_only",
        "g2_permuted": "permuted",
    }
    if condition in edge_control_by_condition:
        values["edge_control"] = edge_control_by_condition[condition]
    if condition == "g2_permuted":
        values["rewire_seed"] = int(rewire_seed)
    return values


def _matrix_from_conditions(
    *,
    name: str,
    conditions: Sequence[str],
    seeds: Sequence[int],
    standards: Mapping[str, Any],
    max_epochs: int,
    patience: int,
    rewire_seed: int,
    open_test: bool,
    save_predictions: bool,
) -> dict[str, Any]:
    first_condition = conditions[0]
    first_seed_values = _condition_values(
        first_condition,
        seeds[0],
        standards,
        max_epochs=max_epochs,
        patience=patience,
        rewire_seed=rewire_seed,
    )
    fixed = {
        key: value
        for key, value in first_seed_values.items()
        if key not in {"model", "seed"}
    }
    include = [
        _condition_values(
            condition,
            seed,
            standards,
            max_epochs=max_epochs,
            patience=patience,
            rewire_seed=rewire_seed,
        )
        for condition in conditions[1:]
        for seed in seeds
    ]
    return {
        "name": name,
        "factors": {
            "model": [first_seed_values["model"]],
            "seed": list(seeds),
        },
        "fixed": fixed,
        "include": include,
        "open_test": bool(open_test),
        "save_predictions": bool(save_predictions),
    }


def build_confirmation_matrix(
    standards: Mapping[str, Any],
    *,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Build the paired validation-only top-candidate confirmation matrix."""

    confirmed_seeds = _normalise_seeds(
        seeds,
        name="confirmation_seeds",
        minimum_count=3,
    )
    fixed = {
        **_graph_values(standards),
        **_common_training(standards, max_epochs=100, patience=20),
        **_model_values(standards, "g1"),
    }
    return {
        "name": "validation_top_candidate_seed_confirmation",
        "factors": {
            "model": ["g1"],
            "seed": confirmed_seeds,
        },
        "fixed": fixed,
        "open_test": False,
        "save_predictions": False,
    }


def build_final_matrix(
    standards: Mapping[str, Any],
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    max_epochs: int = 200,
    patience: int = 25,
    rewire_seed: int = 271828,
) -> dict[str, Any]:
    """Build the complete, five-seed, sealed-test ladder matrix."""

    final_seeds = _normalise_seeds(
        seeds,
        name="final_seeds",
        exact_count=5,
    )
    return _matrix_from_conditions(
        name="final_locked_five_seed_ladder",
        conditions=REQUIRED_FINAL_CONDITIONS,
        seeds=final_seeds,
        standards=standards,
        max_epochs=max_epochs,
        patience=patience,
        rewire_seed=rewire_seed,
        open_test=True,
        save_predictions=True,
    )


def build_g3_matrix(
    standards: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    checkpoint_template: str,
    max_epochs: int = 200,
    patience: int = 25,
) -> dict[str, Any]:
    """Build G3 only after a positive eligibility lock and matched B0 paths."""

    final_seeds = _normalise_seeds(
        seeds,
        name="g3_seeds",
        exact_count=5,
    )
    if "{seed}" not in checkpoint_template:
        raise StandardsLockError(
            "G3 checkpoint template must contain the {seed} placeholder"
        )
    values = []
    for seed in final_seeds:
        checkpoint = checkpoint_template.format(seed=seed)
        if not checkpoint or "{" in checkpoint or "}" in checkpoint:
            raise StandardsLockError(
                "G3 checkpoint template did not produce concrete paths"
            )
        values.append(
            {
                "model": "g3",
                "seed": seed,
                **_graph_values(standards),
                **_model_values(standards, "g3"),
                **_common_training(
                    standards,
                    max_epochs=max_epochs,
                    patience=patience,
                ),
                "pretrained_b0_checkpoint": checkpoint,
                "g3_frozen_epochs": 10,
                "g3_joint_learning_rate": 1e-4,
            }
        )
    first = values[0]
    return {
        "name": "final_g3_conditional_five_seed_ladder",
        "factors": {
            "model": ["g3"],
            "seed": [first["seed"]],
        },
        "fixed": {
            key: value
            for key, value in first.items()
            if key not in {"model", "seed"}
        },
        "include": values[1:],
        "open_test": True,
        "save_predictions": True,
    }


def expand_matrix(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the launcher-compatible subset used by generated matrices."""

    factors = matrix.get("factors")
    if not isinstance(factors, Mapping) or not factors:
        raise StandardsLockError("Matrix requires non-empty factors")
    names = list(factors)
    combinations: list[dict[str, Any]] = [{}]
    for name in names:
        choices = factors[name]
        if not isinstance(choices, list) or not choices:
            raise StandardsLockError(
                f"Matrix factor {name!r} must be a non-empty list"
            )
        combinations = [
            {**combination, name: choice}
            for combination in combinations
            for choice in choices
        ]
    fixed = matrix.get("fixed", {})
    if not isinstance(fixed, Mapping):
        raise StandardsLockError("Matrix fixed values must be a mapping")
    jobs = [{**dict(fixed), **item} for item in combinations]
    includes = matrix.get("include", [])
    if not isinstance(includes, list) or not all(
        isinstance(item, Mapping) for item in includes
    ):
        raise StandardsLockError("Matrix include values must be mappings")
    jobs.extend(dict(item) for item in includes)
    unique: dict[str, dict[str, Any]] = {}
    for job in jobs:
        key = json.dumps(job, sort_keys=True, separators=(",", ":"))
        if key in unique:
            raise StandardsLockError("Generated matrix contains duplicate jobs")
        unique[key] = job
    return list(unique.values())


def condition_name(job: Mapping[str, Any]) -> str:
    """Return the prespecified final-control name for one expanded job."""

    model = str(job.get("model"))
    if model == "b0":
        return "b0"
    if model == "b0-matched":
        return "b0_parameter_matched"
    if model == "broad-field":
        return "broad_field"
    if model == "b1":
        return "b1"
    if model == "g1":
        return "g1_rewired" if bool(job.get("rewired")) else "g1_true"
    if model == "g2":
        control = str(job.get("edge_control", "none"))
        names = {
            "none": "g2_true",
            "zero": "g2_zero",
            "distance_only": "g2_distance_only",
            "permuted": "g2_permuted",
        }
        if control in names:
            return names[control]
    if model == "g3":
        return "g3_conditional"
    raise StandardsLockError(f"Unknown final control job: {dict(job)}")


def validate_final_matrix(
    matrix: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int],
) -> None:
    """Enforce exact required controls and five paired seeds."""

    seeds = _normalise_seeds(
        expected_seeds,
        name="expected_final_seeds",
        exact_count=5,
    )
    if matrix.get("open_test") is not True:
        raise StandardsLockError("Final matrix must explicitly open the test")
    if matrix.get("save_predictions") is not True:
        raise StandardsLockError(
            "Final matrix must preserve predictions for audit"
        )
    observed: dict[str, set[int]] = {
        name: set() for name in REQUIRED_FINAL_CONDITIONS
    }
    jobs = expand_matrix(matrix)
    for job in jobs:
        name = condition_name(job)
        if name not in observed:
            raise StandardsLockError(
                f"Unexpected condition in required final matrix: {name}"
            )
        seed = _require_integer(job.get("seed"), name=f"{name}.seed")
        if seed in observed[name]:
            raise StandardsLockError(
                f"Duplicate seed {seed} for final condition {name}"
            )
        observed[name].add(seed)
    expected = set(seeds)
    incomplete = {
        name: sorted(values)
        for name, values in observed.items()
        if values != expected
    }
    if incomplete:
        raise StandardsLockError(
            "Final matrix lacks exact five-seed controls: "
            + json.dumps(incomplete, sort_keys=True)
        )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(
            dict(value),
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )


def _parse_checksum_file(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            raise StandardsLockError("Invalid checksums.sha256 record")
        checksum, name = match.groups()
        if name in records:
            raise StandardsLockError("Duplicate checksums.sha256 record")
        records[name] = checksum
    return records


def load_standards_lock(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify and load one immutable standards-lock directory."""

    root = Path(path)
    manifest_path = root / "manifest.json"
    lock_path = root / "standards_lock.json"
    checksum_path = root / "checksums.sha256"
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Standards lock was not found: {root}")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("artifact_kind") != ARTIFACT_KIND
        or manifest.get("status") != "complete"
    ):
        raise StandardsLockError("Unsupported or incomplete lock artifact")
    if (
        manifest.get("manifest_content_sha256")
        != _manifest_content_hash(manifest)
    ):
        raise StandardsLockError("Lock manifest content checksum mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise StandardsLockError("Lock manifest lacks file checksums")
    actual_names = {
        item.name
        for item in root.iterdir()
        if item.is_file() and item.name != "manifest.json"
    }
    if actual_names != set(files):
        raise StandardsLockError(
            "Lock artifact file set differs from its manifest"
        )
    for name, expected in files.items():
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or _sha256_file(root / name) != expected
        ):
            raise StandardsLockError(
                f"Lock artifact checksum mismatch for {name}"
            )
    checksum_records = _parse_checksum_file(checksum_path)
    expected_checksum_records = {
        name: checksum
        for name, checksum in files.items()
        if name != "checksums.sha256"
    }
    if checksum_records != expected_checksum_records:
        raise StandardsLockError(
            "checksums.sha256 differs from the manifest"
        )
    lock = _read_json(lock_path)
    if (
        lock.get("artifact_kind") != LOCK_KIND
        or lock.get("status") != "locked"
        or lock.get("lock_id") != manifest.get("lock_id")
    ):
        raise StandardsLockError("Standards lock content is inconsistent")
    lock_core = dict(lock)
    lock_core.pop("lock_id", None)
    if lock.get("lock_id") != _canonical_hash(lock_core)[:20]:
        raise StandardsLockError("Standards lock ID is inconsistent")
    if manifest.get("artifact_id") != _canonical_hash(
        {
            "lock_id": lock["lock_id"],
            "files": dict(files),
        }
    )[:16]:
        raise StandardsLockError("Standards lock artifact ID is inconsistent")
    final_seeds = lock.get("final_execution", {}).get("seeds")
    if not isinstance(final_seeds, list):
        raise StandardsLockError("Standards lock lacks final seeds")
    matrices: dict[str, dict[str, Any]] = {}
    for name in sorted(files):
        if not name.endswith(".yaml"):
            continue
        value = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise StandardsLockError(f"Matrix {name} is not a mapping")
        matrices[name] = dict(value)
    final_name = lock["final_execution"]["matrix_file"]
    if final_name not in matrices:
        raise StandardsLockError("Final matrix is missing from lock artifact")
    validate_final_matrix(
        matrices[final_name],
        expected_seeds=final_seeds,
    )
    return manifest, lock, matrices


def _declared_test_matrices(
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    matrices: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the lock's checksum-bound test-opening matrices."""

    declarations: list[tuple[str, Any]] = [
        ("final_execution", lock.get("final_execution", {}).get("matrix_file"))
    ]
    g3 = lock.get("g3")
    if isinstance(g3, Mapping) and g3.get("enabled_by_lock") is True:
        declarations.append(("g3", g3.get("matrix_file")))

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise StandardsLockError("Lock manifest lacks file checksums")
    records: list[dict[str, Any]] = []
    for role, raw_name in declarations:
        if not isinstance(raw_name, str) or not raw_name:
            raise StandardsLockError(
                f"Standards lock lacks its {role} matrix filename"
            )
        matrix = matrices.get(raw_name)
        checksum = files.get(raw_name)
        if not isinstance(matrix, Mapping):
            raise StandardsLockError(
                f"Declared test matrix is missing: {raw_name}"
            )
        if (
            not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            raise StandardsLockError(
                f"Declared test matrix lacks a checksum: {raw_name}"
            )
        if matrix.get("open_test") is not True:
            raise StandardsLockError(
                f"Declared test matrix does not open test: {raw_name}"
            )
        if matrix.get("save_predictions") is not True:
            raise StandardsLockError(
                f"Declared test matrix does not preserve predictions: {raw_name}"
            )
        records.append(
            {
                "role": role,
                "filename": raw_name,
                "sha256": checksum,
                "matrix": dict(matrix),
            }
        )
    return records


def verify_locked_test_matrix(
    lock_path: str | os.PathLike[str],
    matrix_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify that a supplied test matrix is byte- and content-identical to the lock."""

    root = Path(lock_path).resolve()
    supplied = Path(matrix_path).resolve()
    if not supplied.is_file() or supplied.is_symlink():
        raise StandardsLockError(
            f"Supplied final matrix is not a regular file: {supplied}"
        )
    try:
        supplied_value = yaml.safe_load(supplied.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StandardsLockError(
            f"Could not read supplied final matrix: {supplied}"
        ) from exc
    if not isinstance(supplied_value, Mapping):
        raise StandardsLockError("Supplied final matrix must be a mapping")

    manifest, lock, matrices = load_standards_lock(root)
    supplied_checksum = _sha256_file(supplied)
    supplied_content_hash = _canonical_hash(dict(supplied_value))
    matches: list[dict[str, Any]] = []
    for record in _declared_test_matrices(manifest, lock, matrices):
        expected_content_hash = _canonical_hash(record["matrix"])
        if (
            supplied_checksum == record["sha256"]
            and supplied_content_hash == expected_content_hash
        ):
            matches.append(record)
    if len(matches) != 1:
        raise StandardsLockError(
            "Supplied open-test matrix is not exactly one checksum-bound "
            "final matrix declared by the standards lock"
        )
    match = matches[0]
    return {
        "authorization_version": 1,
        "standards_lock_path": str(root),
        "lock_id": str(lock["lock_id"]),
        "artifact_id": str(manifest["artifact_id"]),
        "lock_manifest_sha256": _sha256_file(root / "manifest.json"),
        "matrix_role": str(match["role"]),
        "final_matrix_file": str(match["filename"]),
        "final_matrix_sha256": str(match["sha256"]),
    }


def authorize_locked_test_job(
    lock_path: str | os.PathLike[str],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize one exact expanded job against a verified final matrix."""

    root = Path(lock_path).resolve()
    requested_job = dict(job)
    requested_hash = canonical_job_hash(requested_job)
    manifest, lock, matrices = load_standards_lock(root)
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in _declared_test_matrices(manifest, lock, matrices):
        for locked_job in expand_matrix(record["matrix"]):
            if (
                canonical_job_hash(locked_job) == requested_hash
                and locked_job == requested_job
            ):
                matches.append((record, locked_job))
    if len(matches) != 1:
        raise StandardsLockError(
            "Requested open-test job is not exactly authorized by one "
            "checksum-bound final matrix in the standards lock"
        )
    matrix_record, locked_job = matches[0]
    return {
        "authorization_version": 1,
        "standards_lock_path": str(root),
        "lock_id": str(lock["lock_id"]),
        "artifact_id": str(manifest["artifact_id"]),
        "lock_manifest_sha256": _sha256_file(root / "manifest.json"),
        "matrix_role": str(matrix_record["role"]),
        "final_matrix_file": str(matrix_record["filename"]),
        "final_matrix_sha256": str(matrix_record["sha256"]),
        "condition": condition_name(locked_job),
        "canonical_job_hash": requested_hash,
        "canonical_job": locked_job,
    }


def create_standards_lock(
    recommendation_paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    *,
    final_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    confirmation_seeds: Sequence[int] = (0, 1, 2),
    minimum_confirmation_seeds: int = 3,
    max_epochs: int = 200,
    patience: int = 25,
    rewire_seed: int = 271828,
    enable_g3: bool = False,
    g3_checkpoint_template: str | None = None,
) -> Path:
    """Atomically lock standards and emit non-executed concrete matrices."""

    destination = Path(output_dir).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite immutable standards lock: {destination}"
        )
    if minimum_confirmation_seeds < 3:
        raise StandardsLockError(
            "minimum_confirmation_seeds cannot be below three"
        )
    if max_epochs <= 0 or patience <= 0:
        raise StandardsLockError("Epoch and patience limits must be positive")
    final_seed_values = _normalise_seeds(
        final_seeds,
        name="final_seeds",
        exact_count=5,
    )
    confirmation_seed_values = _normalise_seeds(
        confirmation_seeds,
        name="confirmation_seeds",
        minimum_count=3,
    )
    paths = [Path(path).resolve() for path in recommendation_paths]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(
            "Recommendation files were not found: " + ", ".join(missing)
        )
    standards, sources, g3_eligible = _merge_recommendations(
        paths,
        minimum_confirmation_seeds=minimum_confirmation_seeds,
    )
    if enable_g3 and not g3_eligible:
        raise StandardsLockError(
            "G3 requires a positive validation-only g3_eligibility "
            "recommendation"
        )
    if enable_g3 and not g3_checkpoint_template:
        raise StandardsLockError(
            "Enabled G3 requires a matched B0 checkpoint template"
        )
    confirmation_matrix = build_confirmation_matrix(
        standards,
        seeds=confirmation_seed_values,
    )
    final_matrix = build_final_matrix(
        standards,
        seeds=final_seed_values,
        max_epochs=max_epochs,
        patience=patience,
        rewire_seed=rewire_seed,
    )
    validate_final_matrix(
        final_matrix,
        expected_seeds=final_seed_values,
    )
    g3_matrix = None
    if enable_g3:
        assert g3_checkpoint_template is not None
        g3_matrix = build_g3_matrix(
            standards,
            seeds=final_seed_values,
            checkpoint_template=g3_checkpoint_template,
            max_epochs=max_epochs,
            patience=patience,
        )
    confirmation_name = "matrix_top_candidate_confirmation.yaml"
    final_name = "matrix_final_locked_ladder.yaml"
    g3_name = "matrix_final_g3_conditional.yaml"
    lock_core = {
        "format_version": FORMAT_VERSION,
        "artifact_kind": LOCK_KIND,
        "status": "locked",
        "selection_scope": "validation_only",
        "test_metrics_used_for_selection": False,
        "source_recommendations": sources,
        "standards": standards,
        "confirmation": {
            "minimum_recommendation_seeds": minimum_confirmation_seeds,
            "seeds": confirmation_seed_values,
            "matrix_file": confirmation_name,
            "open_test": False,
        },
        "final_execution": {
            "required_seed_count": 5,
            "seeds": final_seed_values,
            "required_conditions": list(REQUIRED_FINAL_CONDITIONS),
            "matrix_file": final_name,
            "open_test": True,
            "save_predictions": True,
            "execution_started_by_utility": False,
        },
        "g3": {
            "conditional": True,
            "validation_eligible": bool(g3_eligible),
            "enabled_by_lock": bool(enable_g3),
            "matrix_file": g3_name if enable_g3 else None,
            "requirements": [
                "positive validation-only G3 eligibility recommendation",
                "matched B0 checkpoint path for every final seed",
            ],
        },
    }
    lock_id = _canonical_hash(lock_core)[:20]
    lock = {**lock_core, "lock_id": lock_id}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        _write_json(temporary / "standards_lock.json", lock)
        _write_yaml(temporary / confirmation_name, confirmation_matrix)
        _write_yaml(temporary / final_name, final_matrix)
        if g3_matrix is not None:
            _write_yaml(temporary / g3_name, g3_matrix)
        checksums = {
            path.name: _sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "checksums.sha256").write_text(
            "".join(
                f"{checksum}  {name}\n"
                for name, checksum in sorted(checksums.items())
            ),
            encoding="utf-8",
        )
        files = {
            path.name: _sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "artifact_id": _canonical_hash(
                {
                    "lock_id": lock_id,
                    "files": files,
                }
            )[:16],
            "lock_id": lock_id,
            "status": "complete",
            "files": files,
        }
        manifest["manifest_content_sha256"] = _manifest_content_hash(
            manifest
        )
        _write_json(temporary / "manifest.json", manifest)
        load_standards_lock(temporary)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "ARTIFACT_KIND",
    "LOCK_KIND",
    "REQUIRED_FINAL_CONDITIONS",
    "StandardsLockError",
    "authorize_locked_test_job",
    "build_confirmation_matrix",
    "build_final_matrix",
    "build_g3_matrix",
    "canonical_job_hash",
    "condition_name",
    "create_standards_lock",
    "expand_matrix",
    "load_standards_lock",
    "validate_final_matrix",
    "verify_locked_test_matrix",
]
