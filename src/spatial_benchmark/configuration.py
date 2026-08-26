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
        "hybrid-count-gat",
        "hybrid-count-matched-self",
        "mean-adjacency-sage",
        "multiscale-hurdle-count",
        "myjju-genemae",
        "qkv-gat",
        "qkv-gat-matched-self",
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
        if model_name == "relative-qkv-gat":
            if masking.get("independent_views_per_core_epoch") != 10:
                raise ConfigurationError(
                    "relative-qkv-gat requires exactly ten independent mask "
                    "views per core epoch."
                )
            if masking.get("ratio_stratification_or_bins") is not False:
                raise ConfigurationError(
                    "relative-qkv-gat mask views may not use ratio bins or strata."
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
                    "relative-qkv-gat mask seeds must derive only from base "
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
    if model_name == "relative-qkv-gat":
        if use_edges:
            raise ConfigurationError(
                "relative-qkv-gat prohibits ordinary edge features."
            )
        relative = features.get("relative_positional_encoding")
        if not isinstance(relative, Mapping):
            raise ConfigurationError(
                "relative-qkv-gat requires relative_positional_encoding."
            )
        if relative.get("role") != "attention_logit_bias_only":
            raise ConfigurationError(
                "relative geometry may affect attention logits only."
            )
        if model.get("uses_edge_inputs") is not False:
            raise ConfigurationError(
                "relative-qkv-gat must declare model.uses_edge_inputs=false."
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
        "held_in_pooled_14core_relative_qkv_seed_plateau",
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
                if protocol == "held_in_pooled_14core_relative_qkv_seed_plateau"
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
