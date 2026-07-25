"""Strict, dependency-light YAML composition for BAGM experiments.

The supported composition surface is deliberately small.  A root file may
contain an ordered ``defaults`` list whose entries are one-key mappings such as
``{model: g1}``.  That entry resolves to ``<config_root>/model/g1.yaml``.
Group files may be namespaced (``model: {...}``) or contain only the group
body.  Referenced values are deep-merged in order and the root file is applied
last.  ``defaults`` is never materialized into the resolved configuration.

There is no interpolation, object construction, arbitrary import, or executable
YAML tag support.  Duplicate mapping keys, unknown groups, path traversal,
circular references, and shape-changing overrides are rejected.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .paths import CONFIG_ROOT


COMPOSABLE_GROUPS = frozenset(
    {
        "model",
        "masking",
        "dataset",
        "features",
        "graph",
        "trainer",
        "evaluation",
        "experiment",
        "sweep",
        "launcher",
    }
)

ALLOWED_ROOT_FIELDS = COMPOSABLE_GROUPS | frozenset(
    {
        "seed",
        "fold",
        "attempt",
        "campaign",
        "campaign_id",
        "name",
        "description",
        "version",
        "schema_version",
        "tags",
        "notes",
        "metadata",
        "preprocessing",
        "classification",
    }
)

REQUIRED_EXPERIMENT_GROUPS = (
    "model",
    "masking",
    "dataset",
    "features",
    "graph",
    "trainer",
    "evaluation",
)

SUPPORTED_MODEL_NAMES = frozenset(
    {"b0", "b0-matched", "broad-field", "b1", "g1", "g2", "g3"}
)
PRIMARY_METRIC_DIRECTIONS = {
    "val/masked_huber": "minimize",
    "val/auroc": "maximize",
    "val/auprc": "maximize",
    "val/concordance_index": "maximize",
    "val/integrated_brier_score": "minimize",
    "val/time_dependent_auroc": "maximize",
}
RUN_LIFECYCLE_STAGES = frozenset(
    {
        "diagnostic",
        "exploratory_screen",
        "validation_confirmation",
        "locked_final",
        "posthoc_evaluation",
        "unknown",
    }
)


class ConfigurationError(ValueError):
    """Raised when configuration composition or validation is unsafe."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConfigurationError("YAML mapping keys must be hashable.") from error
        if duplicate:
            raise ConfigurationError(
                f"Duplicate YAML mapping key at line {key_node.start_mark.line + 1}: "
                f"{key!r}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read one safe YAML mapping and reject duplicate or non-string keys."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {source}")
    try:
        loaded = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {source}: {error}") from error
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigurationError(f"YAML root must be a mapping: {source}")
    non_strings = [key for key in loaded if not isinstance(key, str)]
    if non_strings:
        raise ConfigurationError(
            f"YAML root keys must be strings in {source}: {non_strings!r}"
        )
    return dict(loaded)


def _compatible_scalar_types(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return True
    return type(left) is type(right)


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Deep-merge mappings while refusing accidental type/shape changes."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        if not isinstance(key, str):
            raise ConfigurationError("Configuration mapping keys must be strings.")
        if key not in result:
            result[key] = deepcopy(value)
            continue
        current = result[key]
        location = ".".join((*path, key))
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value, path=(*path, key))
        elif isinstance(current, Mapping) != isinstance(value, Mapping):
            raise ConfigurationError(
                f"Configuration override changes mapping shape at {location}."
            )
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ConfigurationError(
                    f"Configuration override changes sequence shape at {location}."
                )
            result[key] = deepcopy(value)
        elif not _compatible_scalar_types(current, value):
            raise ConfigurationError(
                f"Configuration override changes scalar type at {location}: "
                f"{type(current).__name__} -> {type(value).__name__}."
            )
        else:
            result[key] = deepcopy(value)
    return result


def _parse_default(entry: Any) -> tuple[str, str] | None:
    if entry == "_self_":
        return None
    if not isinstance(entry, Mapping) or len(entry) != 1:
        raise ConfigurationError(
            "Each defaults entry must be '_self_' or a one-key group mapping."
        )
    group, selection = next(iter(entry.items()))
    if group not in COMPOSABLE_GROUPS:
        raise ConfigurationError(f"Unknown configuration group: {group!r}")
    if not isinstance(selection, str) or not selection.strip():
        raise ConfigurationError(f"Default selection for {group} must be a string.")
    return str(group), selection


def _group_path(config_root: Path, group: str, selection: str) -> Path:
    relative = Path(selection)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigurationError(
            f"Unsafe configuration selection for {group}: {selection!r}"
        )
    if relative.suffix not in {"", ".yaml", ".yml"}:
        raise ConfigurationError(
            f"Configuration selection must be YAML: {group}/{selection}"
        )
    if not relative.suffix:
        relative = relative.with_suffix(".yaml")
    group_root = (config_root / group).resolve(strict=False)
    candidate = (group_root / relative).resolve(strict=False)
    if not candidate.is_relative_to(group_root):
        raise ConfigurationError(
            f"Configuration selection escapes group root: {group}/{selection}"
        )
    return candidate


def _compose_file(
    source: Path,
    *,
    config_root: Path,
    stack: tuple[Path, ...],
    expected_group: str | None,
) -> dict[str, Any]:
    resolved_source = source.resolve(strict=False)
    if resolved_source in stack:
        cycle = " -> ".join(path.as_posix() for path in (*stack, resolved_source))
        raise ConfigurationError(f"Circular configuration defaults: {cycle}")
    loaded = load_yaml_mapping(resolved_source)
    defaults = loaded.pop("defaults", [])
    if not isinstance(defaults, list):
        raise ConfigurationError(
            f"defaults must be a list in {resolved_source.as_posix()}"
        )
    if expected_group is not None and defaults:
        raise ConfigurationError(
            f"Group file {resolved_source} may not declare nested defaults."
        )
    composed: dict[str, Any] = {}
    selected_groups: set[str] = set()
    for entry in defaults:
        parsed = _parse_default(entry)
        if parsed is None:
            continue
        group, selection = parsed
        if group in selected_groups:
            raise ConfigurationError(
                f"Configuration group {group!r} is selected more than once in "
                f"{resolved_source.as_posix()}."
            )
        selected_groups.add(group)
        fragment = _compose_file(
            _group_path(config_root, group, selection),
            config_root=config_root,
            stack=(*stack, resolved_source),
            expected_group=group,
        )
        composed = deep_merge(composed, fragment)

    if expected_group is not None:
        other_groups = (
            set(loaded).intersection(COMPOSABLE_GROUPS) - {expected_group}
        )
        if other_groups:
            raise ConfigurationError(
                f"Group file {resolved_source} defines other groups: "
                f"{', '.join(sorted(other_groups))}"
            )
        if expected_group in loaded:
            body: dict[str, Any] = {expected_group: loaded[expected_group]}
            extra = set(loaded) - {expected_group}
            if extra:
                raise ConfigurationError(
                    f"Namespaced {expected_group} file has unexpected root keys: "
                    f"{', '.join(sorted(extra))}"
                )
        else:
            body = {expected_group: loaded}
    else:
        body = loaded
    return deep_merge(composed, body)


def compose_config(
    path: str | Path,
    *,
    config_root: str | Path | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Resolve a root experiment YAML and optionally validate required fields."""

    root = Path(config_root or CONFIG_ROOT).resolve(strict=False)
    source = Path(path)
    if not source.is_absolute():
        under_config_root = root / source
        under_project_root = root.parent / source
        source = (
            under_config_root
            if under_config_root.is_file() or not under_project_root.is_file()
            else under_project_root
        )
    resolved = _compose_file(
        source,
        config_root=root,
        stack=(),
        expected_group=None,
    )
    unknown = sorted(set(resolved) - ALLOWED_ROOT_FIELDS)
    if unknown:
        raise ConfigurationError(
            "Unknown resolved root fields: " + ", ".join(unknown)
        )
    if validate:
        validate_experiment_config(resolved)
    return resolved


def _mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a configuration mapping.")
    return value


def _positive_integer(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{field} must be an integer.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        adjective = "non-negative" if allow_zero else "positive"
        raise ConfigurationError(f"{field} must be {adjective}.")
    return value


def _positive_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ConfigurationError(f"{field} must be a positive number.")
    return float(value)


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] is None or mapping[key] == "":
        raise ConfigurationError(f"Missing required field {section}.{key}.")
    return mapping[key]


_OWNED_FIELDS = {
    "embedding_dim": "model",
    "use_edge_features": "features",
    "edge_features": "features",
    "edge_feature_names": "features",
    "neighbor_k": "graph",
    "masking_type": "masking",
    "masking_rate": "masking",
    "dataset_version": "dataset",
    "split_id": "dataset",
}


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Validate required groups, types, and scientific parameter ownership.

    A complete experiment requires model name and embedding dimension; masking
    type plus one rate (or a named rates mapping); dataset ID, version, and split
    ID; node/edge feature settings; graph neighbor count; learning rate and
    batch size; an explicit primary metric; seed, fold, and campaign.
    """

    unknown = sorted(set(config) - ALLOWED_ROOT_FIELDS)
    if unknown:
        raise ConfigurationError("Unknown root fields: " + ", ".join(unknown))
    for group in REQUIRED_EXPERIMENT_GROUPS:
        _mapping(config, group)

    seed = config.get("seed")
    fold = config.get("fold")
    _positive_integer(seed, "seed", allow_zero=True)
    _positive_integer(fold, "fold", allow_zero=True)
    if "attempt" in config:
        _positive_integer(config["attempt"], "attempt")
    campaign = config.get("campaign", config.get("campaign_id"))
    if not isinstance(campaign, (str, Mapping)) or not campaign:
        raise ConfigurationError("campaign or campaign_id must be non-empty.")
    if isinstance(campaign, Mapping):
        _required(campaign, "campaign_id", "campaign")

    classification = config.get("classification")
    if classification is not None:
        if not isinstance(classification, Mapping):
            raise ConfigurationError("classification must be a mapping.")
        allowed_classification = {
            "schema_version",
            "lifecycle_stage",
            "study_axis",
            "retention_class",
            "classification_confidence",
            "source_batch",
        }
        unknown_classification = sorted(
            set(classification) - allowed_classification
        )
        if unknown_classification:
            raise ConfigurationError(
                "Unknown classification fields: "
                + ", ".join(unknown_classification)
            )
        if classification.get("schema_version", 1) != 1:
            raise ConfigurationError(
                "classification.schema_version must be 1."
            )
        lifecycle_stage = _required(
            classification, "lifecycle_stage", "classification"
        )
        if lifecycle_stage not in RUN_LIFECYCLE_STAGES:
            raise ConfigurationError(
                "classification.lifecycle_stage must be one of: "
                + ", ".join(sorted(RUN_LIFECYCLE_STAGES))
            )
        for field in ("study_axis", "retention_class"):
            value = _required(classification, field, "classification")
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"classification.{field} must be a non-empty string."
                )
        confidence = _required(
            classification,
            "classification_confidence",
            "classification",
        )
        if confidence not in {"high", "medium", "low", "unknown"}:
            raise ConfigurationError(
                "classification.classification_confidence must be high, "
                "medium, low, or unknown."
            )
        source_batch = classification.get("source_batch")
        if source_batch is not None and (
            not isinstance(source_batch, str) or not source_batch.strip()
        ):
            raise ConfigurationError(
                "classification.source_batch must be a non-empty string."
            )

    model = _mapping(config, "model")
    model_name = str(_required(model, "name", "model")).strip().lower()
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ConfigurationError(
            f"model.name {model_name!r} is not supported by the existing trainer."
        )
    family = _required(model, "family", "model")
    if not isinstance(family, str) or not family.strip():
        raise ConfigurationError("model.family must be a non-empty string.")
    _positive_integer(_required(model, "embedding_dim", "model"), "model.embedding_dim")

    masking = _mapping(config, "masking")
    _required(masking, "type", "masking")
    if isinstance(masking.get("rate"), Mapping) and masking["rate"]:
        rates = dict(masking["rate"])
    elif "rate" in masking:
        rates = {"rate": masking["rate"]}
    elif isinstance(masking.get("rates"), Mapping) and masking["rates"]:
        rates = dict(masking["rates"])
    else:
        raise ConfigurationError("masking requires rate or a non-empty rates mapping.")
    for name, value in rates.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= float(value) <= 1
        ):
            raise ConfigurationError(
                f"masking rate {name!r} must be between zero and one."
            )

    dataset = _mapping(config, "dataset")
    for field in ("dataset_id", "version", "split_id"):
        _required(dataset, field, "dataset")

    features = _mapping(config, "features")
    use_edges = _required(features, "use_edge_features", "features")
    if not isinstance(use_edges, bool):
        raise ConfigurationError("features.use_edge_features must be boolean.")
    if use_edges and not (
        isinstance(features.get("edge_features"), (list, tuple, Mapping))
        or isinstance(features.get("edge_feature_names"), (list, tuple))
    ):
        raise ConfigurationError(
            "Enabled edge features require features.edge_features or "
            "features.edge_feature_names."
        )

    graph = _mapping(config, "graph")
    _positive_integer(
        _required(graph, "neighbor_k", "graph"),
        "graph.neighbor_k",
    )
    _positive_number(_required(graph, "radius_um", "graph"), "graph.radius_um")
    symmetry = _required(graph, "symmetry", "graph")
    if symmetry not in {"mutual", "union"}:
        raise ConfigurationError("graph.symmetry must be 'mutual' or 'union'.")

    trainer = _mapping(config, "trainer")
    _positive_number(
        _required(trainer, "learning_rate", "trainer"),
        "trainer.learning_rate",
    )
    _positive_integer(
        _required(trainer, "batch_size", "trainer"),
        "trainer.batch_size",
    )

    evaluation = _mapping(config, "evaluation")
    primary = _required(evaluation, "primary_metric", "evaluation")
    if not isinstance(primary, str) or "/" not in primary:
        raise ConfigurationError(
            "evaluation.primary_metric must be an explicit namespaced metric."
        )
    if primary not in PRIMARY_METRIC_DIRECTIONS:
        raise ConfigurationError(
            f"evaluation.primary_metric {primary!r} is absent from the metric registry."
        )
    direction = _required(evaluation, "primary_direction", "evaluation")
    expected_direction = PRIMARY_METRIC_DIRECTIONS[primary]
    if direction != expected_direction:
        raise ConfigurationError(
            f"evaluation.primary_direction for {primary} must be "
            f"{expected_direction!r}."
        )

    if model_name == "g1" and use_edges:
        raise ConfigurationError("G1 is topology-only and requires edge features off.")
    if model_name in {"g2", "g3"} and not use_edges:
        raise ConfigurationError(f"{model_name.upper()} requires edge features on.")
    if dataset.get("task") == "masked_expression_regression" and "expression" not in str(
        masking.get("type", "")
    ):
        raise ConfigurationError(
            "Masked-expression regression requires an expression-masking configuration."
        )

    for section_name in REQUIRED_EXPERIMENT_GROUPS:
        section = _mapping(config, section_name)
        for field, owner in _OWNED_FIELDS.items():
            if field in section and section_name != owner:
                raise ConfigurationError(
                    f"{section_name}.{field} is misplaced; it belongs in {owner}."
                )


__all__ = [
    "ALLOWED_ROOT_FIELDS",
    "COMPOSABLE_GROUPS",
    "ConfigurationError",
    "REQUIRED_EXPERIMENT_GROUPS",
    "PRIMARY_METRIC_DIRECTIONS",
    "RUN_LIFECYCLE_STAGES",
    "SUPPORTED_MODEL_NAMES",
    "compose_config",
    "deep_merge",
    "load_yaml_mapping",
    "validate_experiment_config",
]
