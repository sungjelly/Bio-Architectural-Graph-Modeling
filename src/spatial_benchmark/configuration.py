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

from .identifiers import canonical_sha256
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
    {
        "b0",
        "b0-matched",
        "b0-g2-matched",
        "broad-field",
        "b1",
        "g1",
        "g2",
        "g2-tokenized",
        "g3",
        "geometry-modulated-relative-qkv-gat",
        "hybrid-count-gat",
        "hybrid-count-matched-self",
        "mean-adjacency-sage",
        "multiscale-hurdle-count",
        "myjju-genemae",
        "qkv-gat",
        "qkv-gat-matched-self",
        "recurrent-relative-qkv-gat",
        "relative-qkv-gat",
        "self-hurdle-count",
    }
)
PRIMARY_METRIC_DIRECTIONS = {
    "analysis/attention_niche_qc_pass_fraction": "maximize",
    "fit/partial_gene/log1p_cp10k_masked_huber": "minimize",
    "fit/partial_gene/masked_huber": "minimize",
    "fit/uniform_per_cell/masked_huber": "minimize",
    "fit/whole_node/hybrid_loss": "minimize",
    "fit/whole_node/hurdle_loss": "minimize",
    "fit/whole_node/masked_huber": "minimize",
    "fit/whole_node/masked_token_accuracy_percent": "maximize",
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
            "scientific_variant",
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
        scientific_variant = classification.get("scientific_variant")
        if scientific_variant is not None and (
            not isinstance(scientific_variant, str)
            or not scientific_variant.strip()
        ):
            raise ConfigurationError(
                "classification.scientific_variant must be a non-empty string."
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
    if (
        model_name == "qkv-gat"
        and family.strip() != "edge_aware_qkv_graph_transformer"
    ):
        raise ConfigurationError(
            "qkv-gat requires "
            "model.family=edge_aware_qkv_graph_transformer."
        )
    if (
        model_name == "relative-qkv-gat"
        and family.strip() != "relative_geometry_qkv_graph_transformer"
    ):
        raise ConfigurationError(
            "relative-qkv-gat requires "
            "model.family=relative_geometry_qkv_graph_transformer."
        )
    if (
        model_name == "geometry-modulated-relative-qkv-gat"
        and family.strip()
        != "geometry_modulated_relative_qkv_graph_transformer"
    ):
        raise ConfigurationError(
            "geometry-modulated-relative-qkv-gat requires model.family="
            "geometry_modulated_relative_qkv_graph_transformer."
        )
    if (
        model_name == "recurrent-relative-qkv-gat"
        and family.strip()
        != "recurrent_relative_geometry_qkv_graph_transformer"
    ):
        raise ConfigurationError(
            "recurrent-relative-qkv-gat requires "
            "model.family=recurrent_relative_geometry_qkv_graph_transformer."
        )
    if model_name == "recurrent-relative-qkv-gat":
        locked_recurrent_architecture = {
            "graph_layers": 1,
            "unique_graph_blocks": 1,
            "recurrent_unroll_steps": 4,
            "effective_graph_depth": 4,
            "graph_block_weight_tying": "all_steps",
        }
        for field, expected in locked_recurrent_architecture.items():
            if model.get(field) != expected:
                raise ConfigurationError(
                    "recurrent-relative-qkv-gat requires "
                    f"model.{field}={expected!r}."
                )
    if (
        model_name == "qkv-gat-matched-self"
        and family.strip() != "qkv_parameter_matched_self_control"
    ):
        raise ConfigurationError(
            "qkv-gat-matched-self requires "
            "model.family=qkv_parameter_matched_self_control."
        )
    if (
        model_name == "g2-tokenized"
        and family.strip() != "tokenized_edge_conditioned_gatv2"
    ):
        raise ConfigurationError(
            "g2-tokenized requires "
            "model.family=tokenized_edge_conditioned_gatv2."
        )
    if (
        model_name == "hybrid-count-gat"
        and family.strip() != "hybrid_count_edge_conditioned_gatv2"
    ):
        raise ConfigurationError(
            "hybrid-count-gat requires "
            "model.family=hybrid_count_edge_conditioned_gatv2."
        )
    if (
        model_name == "hybrid-count-matched-self"
        and family.strip() != "hybrid_count_parameter_matched_self_control"
    ):
        raise ConfigurationError(
            "hybrid-count-matched-self requires "
            "model.family=hybrid_count_parameter_matched_self_control."
        )
    if (
        model_name == "mean-adjacency-sage"
        and family.strip() != "explicit_self_mean_adjacency_graphsage"
    ):
        raise ConfigurationError(
            "mean-adjacency-sage requires "
            "model.family=explicit_self_mean_adjacency_graphsage."
        )
    if (
        model_name == "multiscale-hurdle-count"
        and family.strip() != "additive_multiscale_hurdle_count"
    ):
        raise ConfigurationError(
            "multiscale-hurdle-count requires "
            "model.family=additive_multiscale_hurdle_count."
        )
    if (
        model_name == "myjju-genemae"
        and family.strip() != "myjju_dual_path_genemae"
    ):
        raise ConfigurationError(
            "myjju-genemae requires "
            "model.family=myjju_dual_path_genemae."
        )
    if (
        model_name == "self-hurdle-count"
        and family.strip() != "self_only_hurdle_count"
    ):
        raise ConfigurationError(
            "self-hurdle-count requires "
            "model.family=self_only_hurdle_count."
        )
    _positive_integer(_required(model, "embedding_dim", "model"), "model.embedding_dim")

    masking = _mapping(config, "masking")
    _required(masking, "type", "masking")
    masking_type = str(masking.get("type", "")).strip().lower()
    if masking_type == "uniform_per_cell_integer_count":
        minimum = _positive_integer(
            _required(masking, "count_min", "masking"),
            "masking.count_min",
            allow_zero=True,
        )
        maximum = _positive_integer(
            _required(masking, "count_max", "masking"),
            "masking.count_max",
        )
        if minimum != 0 or maximum != 1000:
            raise ConfigurationError(
                "uniform_per_cell_integer_count requires inclusive support "
                "from 0 through 1000."
            )
        if masking.get("positions_without_replacement") is not True:
            raise ConfigurationError(
                "uniform per-cell masking requires sampling without replacement."
            )
        if masking.get("model_seed_in_mask_derivation") is not False:
            raise ConfigurationError(
                "uniform per-cell mask derivation must exclude model seed."
            )
        if model_name in {
            "geometry-modulated-relative-qkv-gat",
            "relative-qkv-gat",
            "recurrent-relative-qkv-gat",
        }:
            if masking.get("independent_views_per_core_epoch") != 10:
                raise ConfigurationError(
                    "relative-QKV models require exactly ten independent mask "
                    "views per core epoch."
                )
            if masking.get("ratio_stratification_or_bins") is not False:
                raise ConfigurationError(
                    "relative-QKV mask views may not use ratio bins or strata."
                )
            expected_seed_fields = [
                "base_mask_seed",
                "core_alias",
                "global_epoch",
                "mask_view_index",
            ]
            if list(masking.get("mask_seed_derivation_fields", ())) != (
                expected_seed_fields
            ):
                raise ConfigurationError(
                    "relative-QKV mask seeds must derive only from base "
                    "seed, core alias, global epoch, and mask view index."
                )
    else:
        if isinstance(masking.get("rate"), Mapping) and masking["rate"]:
            rates = dict(masking["rate"])
        elif "rate" in masking:
            rates = {"rate": masking["rate"]}
        elif isinstance(masking.get("rates"), Mapping) and masking["rates"]:
            rates = dict(masking["rates"])
        else:
            raise ConfigurationError(
                "masking requires rate or a non-empty rates mapping."
            )
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
    if model_name in {
        "geometry-modulated-relative-qkv-gat",
        "relative-qkv-gat",
        "recurrent-relative-qkv-gat",
    }:
        if use_edges:
            raise ConfigurationError(
                "relative-QKV models prohibit ordinary edge features."
            )
        relative = features.get("relative_positional_encoding")
        if not isinstance(relative, Mapping):
            raise ConfigurationError(
                "relative-QKV models require relative_positional_encoding."
            )
        expected_relative_role = (
            "attention_logit_modulation_and_bias_only"
            if model_name == "geometry-modulated-relative-qkv-gat"
            else "attention_logit_bias_only"
        )
        if relative.get("role") != expected_relative_role:
            raise ConfigurationError(
                "relative geometry role must match the selected Relative-QKV "
                "score mechanism; relative geometry may affect attention "
                "logits only."
            )
        if model.get("uses_edge_inputs") is not False:
            raise ConfigurationError(
                "relative-QKV models must declare model.uses_edge_inputs=false."
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

    protocol = str(evaluation.get("protocol", "")).strip().lower()
    canonical_prediction_split = str(
        evaluation.get("canonical_prediction_split", "validation")
    ).strip().lower()
    artifact_contract = str(
        evaluation.get("artifact_contract", "predictive")
    ).strip().lower()
    held_in_fit_protocols = {
        "held_in_full_core_fixed_budget",
        "held_in_pooled_10core_fixed_budget",
        "held_in_pooled_14core_relative_qkv_fixed_continuation_epoch300",
        "held_in_pooled_14core_geometry_modulated_relative_qkv_seed_plateau",
        "held_in_pooled_14core_relative_qkv_seed_plateau",
        "held_in_pooled_14core_recurrent_relative_qkv_seed_plateau",
        "held_in_pooled_14core_untied8_relative_qkv_seed_plateau",
        "held_in_pooled_so1_14core_relative_qkv_plateau_min150",
        "held_in_pooled_6core_relative_qkv_fixed_budget",
        "held_in_pooled_6core_relative_qkv_joint_plateau",
        "held_in_pooled_6core_relative_qkv_seed_plateau",
    }
    if protocol == "posthoc_attention_routing_niche_v1":
        metadata = config.get("metadata", {})
        campaign = config.get("campaign", {})
        if (
            artifact_contract != "analysis_only"
            or canonical_prediction_split != "analysis"
            or list(evaluation.get("splits", [])) != ["analysis"]
            or primary != "analysis/attention_niche_qc_pass_fraction"
            or model_name != "relative-qkv-gat"
            or not isinstance(campaign, Mapping)
            or campaign.get("campaign_id")
            != "cmp_20260825_six_core_attention_routing_niches"
            or not isinstance(metadata, Mapping)
            or metadata.get("checkpoint_mutation_allowed") is not False
            or metadata.get("analysis_mask_views") != 10
            or metadata.get("analysis_mask_derivation_fields")
            != ["analysis_mask_seed", "core_alias", "mask_view_index"]
        ):
            raise ConfigurationError(
                "posthoc_attention_routing_niche_v1 requires the registered "
                "analysis-only relative-QKV contract with ten model-seed-"
                "independent mask views and immutable checkpoints."
            )
        required_outputs = metadata.get("required_analysis_outputs")
        if (
            not isinstance(required_outputs, list)
            or not required_outputs
            or any(
                not isinstance(value, str)
                or not value.strip()
                or Path(value).is_absolute()
                or ".." in Path(value).parts
                for value in required_outputs
            )
            or len(set(required_outputs)) != len(required_outputs)
        ):
            raise ConfigurationError(
                "posthoc analysis requires unique safe required_analysis_outputs."
            )
        locked_required_outputs = [
            "six_core_attention_niche_map.png",
            "six_core_attention_niche_map.pdf",
            "six_core_attention_niche_map.svg",
            "six_core_mutual_attention_network_overlay.png",
            "six_core_mutual_attention_network_overlay.pdf",
            *[
                f"core_{core:02d}_attention_niche_map.png"
                for core in (1, 9, 13, 15, 21, 23)
            ],
            "cell_attention_niche_assignments.parquet",
            "mutual_attention_edges.parquet",
            "directed_attention_edges.parquet",
            "attention_niche_summary.csv",
            "attention_niche_colors.json",
            "attention_niche_regions.geojson",
            "analysis_manifest.yaml",
            "analysis_qc_report.md",
            "README.md",
        ]
        locked_metadata = {
            "execution_role": "posthoc_readout_no_training",
            "upstream_campaign_id": (
                "cmp_20260824_cancer_6core_relative_qkv_multiseed"
            ),
            "discover_completed_catalog_verified_last_checkpoints": True,
            "core_aliases": [
                "CAN-01",
                "CAN-09",
                "CAN-13",
                "CAN-15",
                "CAN-21",
                "CAN-23",
            ],
            "core_numbers": [1, 9, 13, 15, 21, 23],
            "final_graph_layer": True,
            "analysis_mask_seed": 2026082501,
            "analysis_mask_views": 10,
            "analysis_mask_derivation_fields": [
                "analysis_mask_seed",
                "core_alias",
                "mask_view_index",
            ],
            "all_genes_visible_sensitivity": True,
            "uniform_routing_threshold": 1.0,
            "consensus_mutual_score_threshold": 1.0,
            "support_threshold": 0.60,
            "primary_top_neighbors": 8,
            "top_neighbor_sensitivity": [5, 8, 10],
            "primary_leiden_resolution": 1.0,
            "leiden_resolution_sensitivity": [0.5, 1.0, 1.5],
            "leiden_seed": 2026082502,
            "color_seed": 2026082503,
            "polygon_coordinate_alignment_rule": (
                "centroid_tolerance_or_polygon_covers_coordinate"
            ),
            "polygon_centroid_tolerance_um": 5.0,
            "spatial_max_gap_um": 75.0,
            "micro_niche_cell_threshold": 20,
            "minimum_free_disk_gib": 40.0,
            "checkpoint_mutation_allowed": False,
            "required_analysis_outputs": locked_required_outputs,
        }
        drifted_metadata = sorted(
            name
            for name, expected_value in locked_metadata.items()
            if metadata.get(name) != expected_value
        )
        if drifted_metadata:
            raise ConfigurationError(
                "posthoc attention-routing locked metadata drifted: "
                + ", ".join(drifted_metadata)
            )
    elif artifact_contract != "predictive":
        raise ConfigurationError(
            "evaluation.artifact_contract=analysis_only is restricted to the "
            "registered attention-routing post-hoc protocol."
        )
    elif protocol in held_in_fit_protocols:
        if canonical_prediction_split != "fit":
            raise ConfigurationError(
                f"{protocol} requires "
                "evaluation.canonical_prediction_split=fit."
            )
        if list(evaluation.get("splits", [])) != ["fit"]:
            raise ConfigurationError(
                f"{protocol} requires evaluation.splits=[fit]."
            )
        if not str(primary).startswith("fit/"):
            raise ConfigurationError(
                f"{protocol} requires a fit/* primary metric."
            )
        if trainer.get("restore_best") is not False:
            raise ConfigurationError(
                f"{protocol} requires trainer.restore_best=false."
            )
        if trainer.get("primary_checkpoint_role") != "last":
            raise ConfigurationError(
                f"{protocol} requires "
                "trainer.primary_checkpoint_role=last."
            )
        expected_checkpoint_policy = (
            "final_last_only_no_intermediate"
            if protocol
            == "held_in_pooled_14core_relative_qkv_fixed_continuation_epoch300"
            else (
                "atomic_latest_then_final_last_only"
                if protocol
                in {
                    "held_in_pooled_14core_relative_qkv_seed_plateau",
                    "held_in_pooled_14core_geometry_modulated_relative_qkv_seed_plateau",
                    "held_in_pooled_14core_recurrent_relative_qkv_seed_plateau",
                    "held_in_pooled_14core_untied8_relative_qkv_seed_plateau",
                    "held_in_pooled_so1_14core_relative_qkv_plateau_min150",
                }
                else (
                    "periodic_and_last"
                    if protocol
                    in {
                        "held_in_pooled_6core_relative_qkv_fixed_budget",
                        "held_in_pooled_6core_relative_qkv_joint_plateau",
                        "held_in_pooled_6core_relative_qkv_seed_plateau",
                    }
                    else "last_only"
                )
            )
        )
        if trainer.get("checkpoint_policy") != expected_checkpoint_policy:
            raise ConfigurationError(
                f"{protocol} requires "
                "trainer.checkpoint_policy="
                f"{expected_checkpoint_policy}."
            )
        if protocol == "held_in_pooled_6core_relative_qkv_joint_plateau":
            locked_values = {
                "max_epochs": 150,
                "minimum_global_epochs": 150,
                "fixed_epoch_budget": False,
                "continuation_policy": (
                    "joint_all_seed_plateau_25_epoch_blocks"
                ),
                "continuation_block_global_epochs": 25,
                "plateau_first_audit_epoch": 150,
                "plateau_window_global_epochs": 50,
                "plateau_consecutive_passing_audits": 2,
                "plateau_requires_all_five_seeds": True,
                "plateau_requires_common_final_epoch": True,
                "mask_views_per_core_step": 10,
                "optimizer_zero_grad_per_core_step": 1,
                "optimizer_steps_per_core_step": 1,
                "early_stopping": False,
            }
            for field, expected in locked_values.items():
                if trainer.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_6core_relative_qkv_joint_plateau "
                        f"requires trainer.{field}={expected!r}."
                    )
        if protocol == "held_in_pooled_6core_relative_qkv_seed_plateau":
            locked_values = {
                "max_epochs": 150,
                "minimum_global_epochs": 150,
                "fixed_epoch_budget": False,
                "continuation_block_global_epochs": 25,
                "plateau_first_audit_epoch": 150,
                "plateau_window_global_epochs": 50,
                "plateau_consecutive_passing_audits": 2,
                "plateau_requires_all_five_seeds": False,
                "plateau_requires_common_final_epoch": False,
                "mask_views_per_core_step": 10,
                "optimizer_zero_grad_per_core_step": 1,
                "optimizer_steps_per_core_step": 1,
                "early_stopping": False,
            }
            for field, expected in locked_values.items():
                if trainer.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_6core_relative_qkv_seed_plateau "
                        f"requires trainer.{field}={expected!r}."
                    )
            continuation_policy = trainer.get("continuation_policy")
            allowed_policies = {
                "seed0_training_loss_plateau_25_epoch_blocks",
                "independent_seed_training_loss_plateau_25_epoch_blocks",
            }
            if continuation_policy not in allowed_policies:
                raise ConfigurationError(
                    "held_in_pooled_6core_relative_qkv_seed_plateau requires "
                    "an independent per-seed plateau continuation policy."
                )
            seed = config.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4:
                raise ConfigurationError(
                    "held_in_pooled_6core_relative_qkv_seed_plateau requires "
                    "an integer model seed from 0 through 4."
                )
            if (
                continuation_policy
                == "seed0_training_loss_plateau_25_epoch_blocks"
                and seed != 0
            ):
                raise ConfigurationError(
                    "The legacy seed0 plateau policy is valid only for seed 0."
                )
            active_seeds = evaluation.get("active_model_seeds")
            if not isinstance(active_seeds, list) or seed not in active_seeds:
                raise ConfigurationError(
                    "The current model seed must appear in "
                    "evaluation.active_model_seeds."
                )
        if protocol == "held_in_pooled_14core_relative_qkv_seed_plateau":
            expected_aliases = [f"SO2-C{core}" for core in range(15, 29)]
            locked_values = {
                "batch_size": 2,
                "core_visits_per_global_epoch": 14,
                "cores_per_optimizer_update": 2,
                "optimizer_updates_per_global_epoch": 7,
                "max_epochs": None,
                "initial_global_epoch_budget": 150,
                "minimum_global_epochs": 150,
                "fixed_epoch_budget": False,
                "continuation_policy": (
                    "single_seed_training_loss_plateau_25_epoch_blocks"
                ),
                "continuation_block_global_epochs": 25,
                "plateau_first_audit_epoch": 150,
                "plateau_window_global_epochs": 50,
                "plateau_consecutive_passing_audits": 2,
                "maximum_scientific_epoch_cap": None,
                "mask_views_per_core_step": 10,
                "mask_views_per_rank_per_optimizer_update": 5,
                "optimizer_zero_grad_per_paired_core_update": 1,
                "optimizer_steps_per_paired_core_update": 1,
                "early_stopping": False,
                "distributed": True,
                "distributed_backend": "nccl",
                "distributed_world_size": 4,
                "rank_zero_only_artifact_writes": True,
                "checkpoint_every_global_epochs": 1,
                "epoch_metrics_csv": "results/epoch_metrics.csv",
                "epoch_metrics_fsync": True,
            }
            for field, expected in locked_values.items():
                if trainer.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_relative_qkv_seed_plateau "
                        f"requires trainer.{field}={expected!r}."
                    )
            launcher = _mapping(config, "launcher")
            locked_launcher = {
                "requested_gpu": "0,1,2,3",
                "requested_gpu_count": 4,
                "require_exact_visible_devices": "0,1,2,3",
                "distributed": True,
                "distributed_backend": "nccl",
                "process_count": 4,
                "elastic_max_restarts": 0,
                "hardware_preflight_receipt": (
                    "state/preflight/so2_14core_relative_qkv_ddp4.json"
                ),
            }
            for field, expected in locked_launcher.items():
                if launcher.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_relative_qkv_seed_plateau "
                        f"requires launcher.{field}={expected!r}."
                    )
            if model_name != "relative-qkv-gat":
                raise ConfigurationError(
                    "The SO2 14-core protocol requires model.name=relative-qkv-gat."
                )
            if config.get("seed") != 0 or evaluation.get("active_model_seeds") != [0]:
                raise ConfigurationError(
                    "The SO2 14-core production protocol is locked to model seed 0."
                )
            if dataset.get("core_aliases") != expected_aliases:
                raise ConfigurationError(
                    "The SO2 14-core protocol requires exact ordered cores 15--28."
                )
            if dataset.get("total_fit_cells") != 246063:
                raise ConfigurationError(
                    "The SO2 14-core protocol requires exactly 246,063 fit cells."
                )
        if (
            protocol
            == "held_in_pooled_14core_untied8_relative_qkv_seed_plateau"
        ):
            locked_architecture = {
                "name": "relative-qkv-gat",
                "family": "relative_geometry_qkv_graph_transformer",
                "embedding_dim": 256,
                "hidden_dim": 256,
                "graph_layers": 8,
                "unique_graph_blocks": 8,
                "effective_graph_depth": 8,
                "graph_block_weight_tying": "none",
                "attention_heads": 8,
                "attention_head_dim": 32,
                "ffn_dim": 1024,
                "decoder_dim": 1024,
                "relative_geometry_dim": 70,
                "positional_bias_hidden_dim": 128,
                "positional_bias_final_zero_init": True,
                "relative_geometry_role": "attention_logit_bias_only",
                "receiver_chunk_size": 512,
                "max_edges_per_chunk": 200000,
                "activation_checkpointing": True,
                "exact_receiver_partitioning": True,
                "fp32_attention_accumulation": True,
            }
            for field, expected in locked_architecture.items():
                if model.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_untied8_relative_qkv_seed_"
                        f"plateau requires model.{field}={expected!r}."
                    )
            if "recurrent_unroll_steps" in model:
                raise ConfigurationError(
                    "held_in_pooled_14core_untied8_relative_qkv_seed_plateau "
                    "prohibits model.recurrent_unroll_steps; all eight blocks "
                    "must be independently parameterized and applied once."
                )

            campaign_mapping = _mapping(config, "campaign")
            if campaign_mapping.get("campaign_id") != (
                "cmp_20260903_so2_14core_untied8_relative_qkv_seed0_batch2"
            ):
                raise ConfigurationError(
                    "The untied8 SO2 14-core protocol requires its registered "
                    "cmp_20260903 campaign."
                )
            if config.get("seed") != 0:
                raise ConfigurationError(
                    "The untied8 SO2 14-core exploratory protocol is locked "
                    "to model seed 0."
                )

            # These hashes freeze the complete resolved mappings, rather than
            # merely a subset of fields. In particular, dataset, features,
            # graph, masking, and trainer must remain byte-semantically equal
            # to the registered August 25 control after YAML composition.
            locked_section_hashes = {
                "model": (
                    "b67de73bced8495d102dbb7e0eba83e8a89895b9ac16b7fe5f5139e968392521"
                ),
                "dataset": (
                    "9f5faf106272b3457578f8514e7f1ffbca5a9eec9ea21f71d1795605e469bc9c"
                ),
                "features": (
                    "be087ab63c840f60687ca78811f76f270215456abc39eb1cedf5c921ebae014c"
                ),
                "graph": (
                    "c9dc31acee7ed2860da818a5f5ab188d2b8bd9755f22ad0076b3685f1618a11a"
                ),
                "masking": (
                    "07f55d3adaf92d0db845b87d30cdb322289ccc8d33bec071fab55b7b705d0dc2"
                ),
                "trainer": (
                    "78930fc1e325105428df27bffe27c4afc692ce221f028cbb6be1291011f9cee8"
                ),
                "evaluation": (
                    "1156016de36a249a834358c0bcf63c61f075bacddd2c32c4fac15aace630b05b"
                ),
                "launcher": (
                    "441bf17436fac7bab8db48a53c2359d138d0d59fa9ce0304517b023f9a9a30e5"
                ),
                "metadata": (
                    "90256784cb4c6a6d4acdb4acb4228355aa759bcd818734cfca81762aaa1bd4b9"
                ),
                "classification": (
                    "db0da446c71805d946bf784321525ff01dede97ca53e19ee398a7e05afc1a20a"
                ),
                "experiment": (
                    "5279349549d1fb8e8dc7ee705d8b4abc6708d94b09bd7aabf0698eadd00bff6b"
                ),
            }
            for section, expected_hash in locked_section_hashes.items():
                section_mapping = _mapping(config, section)
                if canonical_sha256(section_mapping) != expected_hash:
                    raise ConfigurationError(
                        "held_in_pooled_14core_untied8_relative_qkv_seed_"
                        f"plateau requires the frozen literal {section} "
                        "mapping; its canonical SHA-256 drifted."
                    )
        if (
            protocol
            == "held_in_pooled_14core_geometry_modulated_relative_qkv_seed_plateau"
        ):
            locked_architecture = {
                "name": "geometry-modulated-relative-qkv-gat",
                "family": "geometry_modulated_relative_qkv_graph_transformer",
                "embedding_dim": 256,
                "hidden_dim": 256,
                "graph_layers": 4,
                "unique_graph_blocks": 4,
                "effective_graph_depth": 4,
                "graph_block_weight_tying": "none",
                "attention_heads": 8,
                "attention_head_dim": 32,
                "ffn_dim": 1024,
                "decoder_dim": 1024,
                "relative_geometry_dim": 70,
                "geometry_hidden_dim": 128,
                "attention_score_mechanism": (
                    "geometry_modulated_cosine_qkv_v1"
                ),
                "qk_normalization": "per_head_l2",
                "qk_normalization_epsilon": 0.000001,
                "modulation_activation": "tanh",
                "modulation_amplitude": 0.5,
                "modulation_raw_range": [0.5, 1.5],
                "modulation_mean_normalization": True,
                "modulation_mean_clamp_min": 0.000001,
                "modulation_projection_bias": False,
                "modulation_final_zero_init": True,
                "geometry_bias_activation": "tanh",
                "geometry_bias_bound": 1.0,
                "geometry_bias_projection_bias": False,
                "geometry_bias_final_zero_init": True,
                "logit_scale_parameterization": "bounded_sigmoid",
                "logit_scale_minimum": 0.1,
                "logit_scale_initial": 1.8856180831641267,
                "logit_scale_maximum": 20.0,
                "relative_geometry_role": (
                    "attention_logit_modulation_and_bias_only"
                ),
                "relative_geometry_value_injection": False,
                "value_content_source": (
                    "expression_derived_node_embedding_only"
                ),
                "dropout": 0.10,
                "attention_dropout": 0.0,
                "receiver_chunk_size": 128,
                "max_edges_per_chunk": 50000,
                "activation_checkpointing": True,
                "exact_receiver_partitioning": True,
                "fp32_attention_scoring": True,
                "fp32_attention_accumulation": True,
                "implicit_self_loops": False,
                "trainable_node_identifiers": False,
                "trainable_edge_identifiers": False,
                "uses_graph_inputs": True,
                "uses_edge_inputs": False,
                "uses_relative_position": True,
                "edge_key_vectors": False,
                "edge_value_vectors": False,
                "edge_value_gates": False,
            }
            for field, expected in locked_architecture.items():
                if model.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_geometry_modulated_relative_"
                        f"qkv_seed_plateau requires model.{field}={expected!r}."
                    )
            if "recurrent_unroll_steps" in model:
                raise ConfigurationError(
                    "The geometry-modulated four-block protocol prohibits "
                    "recurrent_unroll_steps."
                )
            campaign_mapping = _mapping(config, "campaign")
            if campaign_mapping.get("campaign_id") != (
                "cmp_20260903_so2_14core_geometry_modulated_relative_qkv_"
                "seed0_batch2"
            ):
                raise ConfigurationError(
                    "The geometry-modulated SO2 protocol requires its "
                    "registered cmp_20260903 campaign."
                )
            if config.get("seed") != 0 or evaluation.get(
                "active_model_seeds"
            ) != [0]:
                raise ConfigurationError(
                    "The geometry-modulated SO2 exploratory protocol is "
                    "locked to model seed 0."
                )

            locked_section_hashes = {
                "model": (
                    "f7efd848610b2e500c218fa18618385c4799cdeb8a7c3f70f4d4d1a63fbd2609"
                ),
                "dataset": (
                    "9f5faf106272b3457578f8514e7f1ffbca5a9eec9ea21f71d1795605e469bc9c"
                ),
                "features": (
                    "3fc95a458dce1e6fe1254241a36a0e53869d615122503106ae5b4ecbd3d22f50"
                ),
                "graph": (
                    "c9dc31acee7ed2860da818a5f5ab188d2b8bd9755f22ad0076b3685f1618a11a"
                ),
                "masking": (
                    "07f55d3adaf92d0db845b87d30cdb322289ccc8d33bec071fab55b7b705d0dc2"
                ),
                "trainer": (
                    "78930fc1e325105428df27bffe27c4afc692ce221f028cbb6be1291011f9cee8"
                ),
                "evaluation": (
                    "84130a9543fde0cddd332fe93a7a89e05dc947894076163ce92b3cd6696bb5b3"
                ),
                "launcher": (
                    "cefd8c5c7e11b6c90bac43ceb3909c7354365aa052602e7a5908dd43e0572833"
                ),
                "metadata": (
                    "3b241c25db0d7ec27ee5fa5ebcd69476450d6e491d724d5f46a82e701b2b2462"
                ),
                "classification": (
                    "d9435f05c5459901df1635aa32015974d08c2f807d4ab39dbe50500589971d3f"
                ),
                "experiment": (
                    "2e4cbaae6110b5e17c7c060d8491c7380a2bdfa2cfc985030719e5932b1e88cb"
                ),
            }
            for section, expected_hash in locked_section_hashes.items():
                section_mapping = _mapping(config, section)
                if canonical_sha256(section_mapping) != expected_hash:
                    raise ConfigurationError(
                        "held_in_pooled_14core_geometry_modulated_relative_"
                        f"qkv_seed_plateau requires the frozen literal {section} "
                        "mapping; its canonical SHA-256 drifted."
                    )
        if (
            protocol
            == "held_in_pooled_14core_recurrent_relative_qkv_seed_plateau"
        ):
            expected_aliases = [f"SO2-C{core}" for core in range(15, 29)]
            locked_trainer = {
                "learning_rate": 0.0001,
                "batch_size": 2,
                "cores_per_optimizer_update": 2,
                "core_visits_per_global_epoch": 14,
                "optimizer_updates_per_global_epoch": 7,
                "weight_decay": 0.00001,
                "gradient_clip_norm": 1.0,
                "huber_delta": 1.0,
                "max_epochs": None,
                "initial_global_epoch_budget": 150,
                "minimum_global_epochs": 150,
                "fixed_epoch_budget": False,
                "continuation_policy": (
                    "single_seed_training_loss_plateau_25_epoch_blocks"
                ),
                "continuation_block_global_epochs": 25,
                "plateau_extension": True,
                "plateau_metric": "equal_core_mean_training_masked_huber",
                "plateau_first_audit_epoch": 150,
                "plateau_window_global_epochs": 50,
                "plateau_consecutive_passing_audits": 2,
                "plateau_relative_mean_improvement_max": 0.002,
                "plateau_normalized_absolute_slope_per_epoch_max": 0.0001,
                "maximum_scientific_epoch_cap": None,
                "mask_views_per_core_step": 10,
                "mask_views_per_rank_per_optimizer_update": 5,
                "optimizer_zero_grad_per_paired_core_update": 1,
                "optimizer_steps_per_paired_core_update": 1,
                "scheduler": "none",
                "early_stopping": False,
                "precision": "mixed",
                "amp": True,
                "deterministic": True,
                "restore_best": False,
                "primary_checkpoint_role": "last",
                "checkpoint_policy": "atomic_latest_then_final_last_only",
                "checkpoint_every_global_epochs": 1,
                "core_order_seed": 2026082402,
                "distributed": True,
                "distributed_backend": "nccl",
                "distributed_world_size": 4,
                "maximum_simultaneously_staged_cores_per_gpu": 1,
                "rank_zero_only_artifact_writes": True,
                "epoch_metrics_csv": "results/epoch_metrics.csv",
                "epoch_metrics_fsync": True,
            }
            for field, expected in locked_trainer.items():
                if trainer.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_recurrent_relative_qkv_seed_"
                        f"plateau requires trainer.{field}={expected!r}."
                    )

            launcher = _mapping(config, "launcher")
            locked_launcher = {
                "requested_gpu": "0,1,2,3",
                "requested_gpu_count": 4,
                "require_exact_visible_devices": "0,1,2,3",
                "distributed": True,
                "distributed_backend": "nccl",
                "process_count": 4,
                "elastic_max_restarts": 0,
                "hardware_preflight_receipt": (
                    "state/preflight/"
                    "so2_14core_recurrent_relative_qkv_ddp4.json"
                ),
            }
            for field, expected in locked_launcher.items():
                if launcher.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_recurrent_relative_qkv_seed_"
                        f"plateau requires launcher.{field}={expected!r}."
                    )

            locked_model = {
                "embedding_dim": 256,
                "hidden_dim": 256,
                "graph_layers": 1,
                "unique_graph_blocks": 1,
                "recurrent_unroll_steps": 4,
                "effective_graph_depth": 4,
                "graph_block_weight_tying": "all_steps",
                "attention_heads": 8,
                "attention_head_dim": 32,
                "ffn_dim": 1024,
                "decoder_dim": 1024,
                "relative_geometry_dim": 70,
                "positional_bias_hidden_dim": 128,
                "relative_geometry_role": "attention_logit_bias_only",
                "dropout": 0.10,
                "attention_dropout": 0.0,
                "activation_checkpointing": True,
                "exact_receiver_partitioning": True,
                "fp32_attention_accumulation": True,
                "implicit_self_loops": False,
                "trainable_node_identifiers": False,
                "trainable_edge_identifiers": False,
                "uses_graph_inputs": True,
                "uses_edge_inputs": False,
                "uses_relative_position": True,
                "edge_key_vectors": False,
                "edge_value_vectors": False,
                "edge_value_gates": False,
            }
            for field, expected in locked_model.items():
                if model.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_recurrent_relative_qkv_seed_"
                        f"plateau requires model.{field}={expected!r}."
                    )

            locked_graph = {
                "kind": "radial_stratified_knn",
                "neighbor_k": 200,
                "k": 200,
                "radius_um": 500.0,
                "maximum_range_um": 500.0,
                "symmetry": "union",
                "directed_selection_then_bidirectional_union": True,
                "self_loops": False,
                "cross_core_edges": False,
            }
            for field, expected in locked_graph.items():
                if graph.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_recurrent_relative_qkv_seed_"
                        f"plateau requires graph.{field}={expected!r}."
                    )

            locked_masking = {
                "type": "uniform_per_cell_integer_count",
                "count_min": 0,
                "count_max": 1000,
                "positions_without_replacement": True,
                "independent_views_per_core_epoch": 10,
                "ratio_stratification_or_bins": False,
                "mask_base_seed": 2026082401,
                "model_seed_in_mask_derivation": False,
            }
            for field, expected in locked_masking.items():
                if masking.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_recurrent_relative_qkv_seed_"
                        f"plateau requires masking.{field}={expected!r}."
                    )

            campaign_mapping = _mapping(config, "campaign")
            if campaign_mapping.get("campaign_id") != (
                "cmp_20260831_so2_14core_recurrent_relative_qkv_seed0_batch2"
            ):
                raise ConfigurationError(
                    "The recurrent SO2 14-core protocol requires its registered "
                    "cmp_20260831 campaign."
                )
            metadata = _mapping(config, "metadata")
            locked_gradient_diagnostics = {
                "enabled": True,
                "output_path": "results/gradient_direction_metrics.csv",
                "schema": "so2_full_gradient_direction_metrics_v1",
                "global_epoch_indexing": "one_based",
                "computation_rank": 0,
                "vector_dtype": "float32",
                "gradient_scope": (
                    "full_ddp_averaged_all_trainable_parameters"
                ),
                "read_timing": (
                    "after_amp_unscale_and_finite_check_before_gradient_"
                    "clipping_or_optimizer_step"
                ),
                "parameter_order": "trainable_parameter_registration_order",
                "missing_gradient_slices": "zero_filled",
                "persistence": "global_epoch_scalars_only",
                "ordered_columns": [
                    "schema",
                    "run_id",
                    "model_seed",
                    "global_epoch",
                    "trainable_parameter_count",
                    "optimizer_updates_observed",
                    "gradient_norm_mean_before_clip",
                    "gradient_norm_min_before_clip",
                    "gradient_norm_max_before_clip",
                    "consecutive_optimizer_step_cosine_mean",
                    "consecutive_optimizer_step_cosine_median",
                    "consecutive_optimizer_step_cosine_min",
                    "consecutive_optimizer_step_cosine_max",
                    "consecutive_optimizer_step_cosine_valid_pairs",
                    "epoch_aggregate_gradient_cosine_to_previous_epoch",
                    "resume_boundary_unavailable",
                ],
                "consecutive_optimizer_step_cosine": True,
                "epoch_aggregate_gradient_cosine_to_previous_epoch": True,
                "zero_norm_cosines": "invalid_and_omitted",
                "resume_boundary_policy": (
                    "boundary_pair_and_prior_epoch_aggregate_unavailable"
                ),
                "persist_gradient_tensors": False,
                "persist_per_optimizer_step_files": False,
                "persist_gradient_vectors_in_checkpoints": False,
                "affects_optimization_or_plateau_stopping": False,
            }
            gradient_diagnostics = metadata.get("gradient_diagnostics")
            if not isinstance(gradient_diagnostics, Mapping):
                raise ConfigurationError(
                    "The recurrent SO2 14-core protocol requires scalar-only "
                    "gradient diagnostics metadata."
                )
            for field, expected in locked_gradient_diagnostics.items():
                if gradient_diagnostics.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_recurrent_relative_qkv_seed_"
                        "plateau requires metadata.gradient_diagnostics."
                        f"{field}={expected!r}."
                    )
            if model_name != "recurrent-relative-qkv-gat":
                raise ConfigurationError(
                    "The recurrent SO2 14-core protocol requires "
                    "model.name=recurrent-relative-qkv-gat."
                )
            if config.get("seed") != 0 or evaluation.get("active_model_seeds") != [0]:
                raise ConfigurationError(
                    "The recurrent SO2 14-core exploratory protocol is locked "
                    "to model seed 0."
                )
            if dataset.get("core_aliases") != expected_aliases:
                raise ConfigurationError(
                    "The recurrent SO2 14-core protocol requires exact ordered "
                    "cores 15--28."
                )
            if dataset.get("total_fit_cells") != 246063:
                raise ConfigurationError(
                    "The recurrent SO2 14-core protocol requires exactly "
                    "246,063 fit cells."
                )
        if protocol == "held_in_pooled_so1_14core_relative_qkv_plateau_min150":
            expected_aliases = [f"SO1-C{core:02d}" for core in range(1, 15)]
            locked_values = {
                "batch_size": 2,
                "core_visits_per_global_epoch": 14,
                "cores_per_optimizer_update": 2,
                "optimizer_updates_per_global_epoch": 7,
                "execution_mode": "fresh_plateau_min150",
                "max_epochs": None,
                "initial_global_epoch_budget": 150,
                "minimum_global_epochs": 150,
                "fixed_epoch_budget": False,
                "continuation_policy": "strict_training_loss_plateau_25_epoch_blocks",
                "continuation_block_global_epochs": 25,
                "plateau_stopping_enabled": True,
                "plateau_diagnostic_only": False,
                "plateau_rule": (
                    "strict_absolute_relative_half_window_change_and_"
                    "normalized_absolute_slope"
                ),
                "plateau_audit_interval_global_epochs": 25,
                "plateau_window_global_epochs": 50,
                "plateau_consecutive_passing_audits": 2,
                "plateau_absolute_relative_half_window_change_max": 0.0005,
                "plateau_normalized_absolute_slope_per_epoch_max": 0.000025,
                "maximum_scientific_epoch_cap": None,
                "mask_views_per_core_step": 10,
                "mask_views_per_rank_per_optimizer_update": 5,
                "optimizer_zero_grad_per_paired_core_update": 1,
                "optimizer_steps_per_paired_core_update": 1,
                "early_stopping": False,
                "distributed": True,
                "distributed_backend": "nccl",
                "distributed_world_size": 4,
                "rank_zero_only_artifact_writes": True,
                "checkpoint_every_global_epochs": 1,
                "epoch_metrics_csv": "results/epoch_metrics.csv",
                "epoch_metrics_fsync": True,
            }
            for field, expected in locked_values.items():
                if trainer.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_so1_14core_relative_qkv_plateau_min150 "
                        f"requires trainer.{field}={expected!r}."
                    )
            launcher = _mapping(config, "launcher")
            locked_launcher = {
                "requested_gpu": "0,1,2,3",
                "requested_gpu_count": 4,
                "require_exact_visible_devices": "0,1,2,3",
                "distributed": True,
                "distributed_backend": "nccl",
                "process_count": 4,
                "elastic_max_restarts": 0,
                "hardware_preflight_receipt": (
                    "state/preflight/"
                    "so1_14core_relative_qkv_ddp4_plateau_min150.json"
                ),
            }
            for field, expected in locked_launcher.items():
                if launcher.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_so1_14core_relative_qkv_plateau_min150 "
                        f"requires launcher.{field}={expected!r}."
                    )
            if model_name != "relative-qkv-gat":
                raise ConfigurationError(
                    "The SO1 14-core protocol requires model.name=relative-qkv-gat."
                )
            if config.get("seed") != 0 or evaluation.get("active_model_seeds") != [0]:
                raise ConfigurationError(
                    "The SO1 14-core production protocol is locked to model seed 0."
                )
            if dataset.get("core_aliases") != expected_aliases:
                raise ConfigurationError(
                    "The SO1 14-core protocol requires exact ordered cores 1--14."
                )
            if dataset.get("total_fit_cells") != 161596:
                raise ConfigurationError(
                    "The SO1 14-core protocol requires exactly 161,596 fit cells."
                )
        if (
            protocol
            == "held_in_pooled_14core_relative_qkv_fixed_continuation_epoch300"
        ):
            expected_aliases = [f"SO2-C{core}" for core in range(15, 29)]
            locked_values = {
                "batch_size": 2,
                "core_visits_per_global_epoch": 14,
                "cores_per_optimizer_update": 2,
                "optimizer_updates_per_global_epoch": 7,
                "execution_mode": "resume_fixed_final_epoch",
                "required_resume_completed_global_epochs": 175,
                "required_source_run_id": (
                    "r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6"
                ),
                "required_source_checkpoint_sha256": (
                    "2e0f9d837fdbb78673f9788e355ce7a6f6843ffa1fc7a46a6b0c126d53fe7d8d"
                ),
                "fixed_final_global_epoch": 300,
                "max_epochs": 300,
                "initial_global_epoch_budget": 300,
                "minimum_global_epochs": 300,
                "fixed_epoch_budget": True,
                "continuation_policy": (
                    "fixed_epoch_300_from_confirmed_epoch_175"
                ),
                "plateau_extension": False,
                "plateau_stopping_enabled": False,
                "maximum_scientific_epoch_cap": 300,
                "mask_views_per_core_step": 10,
                "mask_views_per_rank_per_optimizer_update": 5,
                "optimizer_zero_grad_per_paired_core_update": 1,
                "optimizer_steps_per_paired_core_update": 1,
                "early_stopping": False,
                "distributed": True,
                "distributed_backend": "nccl",
                "distributed_world_size": 4,
                "rank_zero_only_artifact_writes": True,
                "checkpoint_every_global_epochs": None,
                "checkpoint_final_role": "final_epoch_300_last",
                "strict_plateau_diagnostic_only": True,
                "strict_plateau_audit_interval_global_epochs": 25,
                "strict_plateau_window_global_epochs": 50,
                "strict_plateau_absolute_relative_half_window_change_max": 0.0005,
                "strict_plateau_normalized_absolute_slope_per_epoch_max": 0.000025,
                "strict_plateau_consecutive_passing_audits": 2,
                "epoch_metrics_csv": "results/epoch_metrics.csv",
                "epoch_metrics_fsync": True,
            }
            for field, expected in locked_values.items():
                if trainer.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_relative_qkv_fixed_continuation_"
                        f"epoch300 requires trainer.{field}={expected!r}."
                    )
            launcher = _mapping(config, "launcher")
            locked_launcher = {
                "requested_gpu": "0,1,2,3",
                "requested_gpu_count": 4,
                "require_exact_visible_devices": "0,1,2,3",
                "distributed": True,
                "distributed_backend": "nccl",
                "process_count": 4,
                "elastic_max_restarts": 0,
                "hardware_preflight_receipt": (
                    "state/preflight/so2_14core_relative_qkv_ddp4_"
                    "resume175_fixed300.json"
                ),
            }
            for field, expected in locked_launcher.items():
                if launcher.get(field) != expected:
                    raise ConfigurationError(
                        "held_in_pooled_14core_relative_qkv_fixed_continuation_"
                        f"epoch300 requires launcher.{field}={expected!r}."
                    )
            resume_checkpoint = str(launcher.get("resume_checkpoint", "")).strip()
            if not resume_checkpoint.endswith(
                "/r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6/"
                "checkpoints/last.ckpt"
            ):
                raise ConfigurationError(
                    "The fixed continuation launcher must identify the locked "
                    "epoch-175 last checkpoint."
                )
            source_artifact_path = str(
                launcher.get("source_artifact_path", "")
            ).strip().rstrip("/")
            if not source_artifact_path.endswith(
                "/r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6"
            ):
                raise ConfigurationError(
                    "The fixed continuation launcher must identify the locked "
                    "source artifact bundle."
                )
            if model_name != "relative-qkv-gat":
                raise ConfigurationError(
                    "The SO2 fixed continuation requires model.name=relative-qkv-gat."
                )
            if config.get("seed") != 0 or evaluation.get("active_model_seeds") != [0]:
                raise ConfigurationError(
                    "The SO2 fixed continuation is locked to model seed 0."
                )
            if dataset.get("core_aliases") != expected_aliases:
                raise ConfigurationError(
                    "The SO2 fixed continuation requires exact ordered cores 15--28."
                )
            if dataset.get("total_fit_cells") != 246063:
                raise ConfigurationError(
                    "The SO2 fixed continuation requires exactly 246,063 fit cells."
                )
    elif canonical_prediction_split == "fit" or str(primary).startswith("fit/"):
        raise ConfigurationError(
            "fit prediction/metric semantics require "
            "an explicit held-in fit evaluation protocol."
        )

    token_task = "masked_expression_token_classification"
    token_family = "tokenized_edge_conditioned_gatv2"
    token_schema = "raw_count_tokens_0_1_2_3plus_v1"
    token_scale = "raw_count_token_0_1_2_3plus"
    token_metric = "fit/whole_node/masked_token_accuracy_percent"
    token_task_family = "masked_expression_token_classification"
    tokenization = dataset.get("tokenization")
    token_markers_present = any(
        (
            model_name == "g2-tokenized",
            family.strip() == token_family,
            dataset.get("task") == token_task,
            dataset.get("target_scale") == token_scale,
            "num_expression_tokens" in model,
            "tokenizer_schema" in model,
            evaluation.get("task_family") == token_task_family,
            primary == token_metric,
            tokenization is not None,
        )
    )
    if token_markers_present:
        expected = {
            "model.name": (model_name, "g2-tokenized"),
            "model.family": (family.strip(), token_family),
            "model.num_expression_tokens": (
                model.get("num_expression_tokens"),
                4,
            ),
            "model.tokenizer_schema": (
                model.get("tokenizer_schema"),
                token_schema,
            ),
            "dataset.task": (dataset.get("task"), token_task),
            "dataset.target_scale": (
                dataset.get("target_scale"),
                token_scale,
            ),
            "evaluation.task_family": (
                evaluation.get("task_family"),
                token_task_family,
            ),
            "evaluation.primary_metric": (primary, token_metric),
        }
        mismatches = [
            f"{field}={actual!r} (expected {required_value!r})"
            for field, (actual, required_value) in expected.items()
            if actual != required_value
        ]
        if not isinstance(tokenization, Mapping):
            mismatches.append(
                "dataset.tokenization must be a fixed-vocabulary mapping"
            )
        else:
            tokenization_expected = {
                "schema": token_schema,
                "num_output_tokens": 4,
                "mask_token_id": 4,
                "mask_token_is_output": False,
                "fixed_vocabulary": True,
                "fit_required": False,
                "source_scale": "raw_biological_probe_counts",
                "count_mapping": {
                    "0": 0,
                    "1": 1,
                    "2": 2,
                    "3+": 3,
                },
            }
            mismatches.extend(
                "dataset.tokenization."
                f"{field}={tokenization.get(field)!r} "
                f"(expected {required_value!r})"
                for field, required_value in tokenization_expected.items()
                if tokenization.get(field) != required_value
            )
        if mismatches:
            raise ConfigurationError(
                "Tokenized G2 requires one matching task/model/tokenizer/metric "
                "contract: " + "; ".join(mismatches)
            )

    if model_name == "g1" and use_edges:
        raise ConfigurationError("G1 is topology-only and requires edge features off.")
    if model_name in {
        "g2",
        "g2-tokenized",
        "g3",
        "hybrid-count-gat",
        "multiscale-hurdle-count",
        "myjju-genemae",
        "qkv-gat",
    } and not use_edges:
        raise ConfigurationError(f"{model_name.upper()} requires edge features on.")
    if model_name in {
        "hybrid-count-matched-self",
        "qkv-gat-matched-self",
        "self-hurdle-count",
    } and use_edges:
        raise ConfigurationError(
            f"{model_name.upper()} requires edge features off."
        )
    if (
        dataset.get("task") == "masked_expression_regression"
        and "expression" not in str(masking.get("type", ""))
        and masking_type != "uniform_per_cell_integer_count"
    ):
        raise ConfigurationError(
            "Masked-expression regression requires an expression-masking configuration."
        )
    if dataset.get("task") == token_task and "expression" not in str(
        masking.get("type", "")
    ):
        raise ConfigurationError(
            "Masked-expression token classification requires an "
            "expression-masking configuration."
        )
    if dataset.get("task") == "masked_expression_hybrid_count" and "expression" not in str(
        masking.get("type", "")
    ):
        raise ConfigurationError(
            "Hybrid-count masked expression requires an expression-masking "
            "configuration."
        )
    if dataset.get("task") == "masked_expression_hurdle_count" and "expression" not in str(
        masking.get("type", "")
    ):
        raise ConfigurationError(
            "Hurdle-count masked expression requires an expression-masking "
            "configuration."
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
