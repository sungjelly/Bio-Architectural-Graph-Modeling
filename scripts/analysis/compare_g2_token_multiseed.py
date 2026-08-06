#!/usr/bin/env python3
"""Verify and compare the locked six-run tokenized-G2 width campaign.

The reported percentages are exact categorical accuracies for the fixed
raw-count vocabulary ``0``, ``1``, ``2``, and ``3+``.  They are held-in,
transductive reconstruction metrics, not validation or test accuracy.

This utility does not compute the conditional relaxed categorical Jacobian.
If every wider-model seed exceeds the strict 95% exact-accuracy gate, the
comparison is published with status ``conditional_analysis_required`` and the
CLI exits nonzero after writing the evidence.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import struct
import sys
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import yaml


_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from spatial_benchmark.run_archive import (  # noqa: E402
    RunValidationError,
    verify_run_bundle,
)


_CAMPAIGN_ID = "cmp_20260726_full_core_g2_count_tokens_multiseed"
_CAMPAIGN_NAME = "Full-core G2 raw-count token prediction across seeds"
_EVALUATION_PROTOCOL = "held_in_full_core_fixed_budget"
_TRAINING_PROTOCOL = "held_in_full_core_fixed_budget_token_classification"
_TASK = "masked_expression_token_classification"
_BASELINE = "g2_tokenized_width512_exact_k1000_full_core"
_WIDER = "g2_tokenized_width1024_exact_k1000_full_core"
_PILOT = "g2_tokenized_width1024_exact_k1000_resource_pilot"
_CONCLUSION_VARIANTS = (_BASELINE, _WIDER)
_EXPECTED_SEEDS = (0, 1, 2)
_EXPECTED_EPOCHS = 200
_PILOT_EPOCHS = 2
_EXPECTED_REPLICATES = 3
_PILOT_REPLICATES = 1
_EXPECTED_PARAMETERS = {
    _BASELINE: 7_062_880,
    _WIDER: 18_834_784,
    _PILOT: 18_834_784,
}
_SCIENTIFIC_TOKENS = {
    _BASELINE: "6dda5587",
    _WIDER: "56a25fc4",
    _PILOT: "d86b9849",
}
_EXPECTED_GRAPH_SHA256 = (
    "2469064e2fe14b48f642fca09a546d9e420d9fd851b9668996da62ee8246d060"
)
_EXPECTED_GRAPH_EDGES = 21_029_944
_EXPECTED_TOKEN_MATRIX_SHA256 = (
    "be7e62d6acefb03c4be69ac240ab7092e598c2e1a6f84afcd55cf365f49146b2"
)
_EXPECTED_TOKEN_COUNTS = (22_254_214, 1_099_754, 649_566, 241_466)
_EXPECTED_SHAPE = (24_245, 1_000)
_TOKEN_LABELS = ("zero", "one", "two", "three_or_more")
_DATA_FIELDS = (
    "dataset_id",
    "version",
    "dataset_fingerprint",
    "preprocessing_version",
    "split_id",
    "split_fingerprint",
)
_EXPECTED_EXPERIMENT = {
    _BASELINE: {
        "variant_label": _BASELINE,
        "estimand": "held_in_full_core_whole_node_exact_masked_token_reconstruction",
        "permitted_claim": "one_core_transductive_token_prediction_capacity",
        "paired_variant": "full_core_g2_tokenized_multiseed_width1024",
    },
    _WIDER: {
        "variant_label": _WIDER,
        "estimand": "held_in_full_core_whole_node_exact_masked_token_reconstruction",
        "permitted_claim": "one_core_transductive_token_prediction_capacity",
        "paired_variant": "full_core_g2_tokenized_multiseed_baseline",
    },
    _PILOT: {
        "variant_label": _PILOT,
        "estimand": "implementation_resource_feasibility_for_held_in_token_prediction",
        "permitted_claim": "diagnostic_runtime_and_memory_feasibility_only",
        "conclusion_eligible": False,
        "excluded_from_capacity_comparison": True,
    },
}
_EXPECTED_CLASSIFICATION = {
    _BASELINE: {
        "schema_version": 1,
        "lifecycle_stage": "exploratory_screen",
        "study_axis": "full_core_g2_token_width_capacity",
        "retention_class": "retain_exploratory_evidence",
        "classification_confidence": "high",
    },
    _WIDER: {
        "schema_version": 1,
        "lifecycle_stage": "exploratory_screen",
        "study_axis": "full_core_g2_token_width_capacity",
        "retention_class": "retain_exploratory_evidence",
        "classification_confidence": "high",
    },
    _PILOT: {
        "schema_version": 1,
        "lifecycle_stage": "diagnostic",
        "study_axis": "g2_token_width_resource_feasibility",
        "retention_class": "retain_diagnostic_evidence",
        "classification_confidence": "high",
    },
}
_EXPECTED_MODEL = {
    _BASELINE: {
        "name": "g2-tokenized",
        "family": "tokenized_edge_conditioned_gatv2",
        "tokenizer_schema": "raw_count_tokens_0_1_2_3plus_v1",
        "num_expression_tokens": 4,
        "embedding_dim": 512,
        "hidden_dim": 512,
        "graph_layers": 2,
        "attention_heads": 4,
        "ffn_dim": 512,
        "decoder_dim": 512,
        "edge_hidden_dim": 64,
        "edge_embedding_dim": 64,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "implicit_self_loops": False,
        "trainable_node_identifiers": False,
        "trainable_edge_identifiers": False,
        "receiver_chunk_size": 512,
        "activation_checkpointing": True,
        "exact_receiver_partitioning": True,
    },
    _WIDER: {
        "name": "g2-tokenized",
        "family": "tokenized_edge_conditioned_gatv2",
        "tokenizer_schema": "raw_count_tokens_0_1_2_3plus_v1",
        "num_expression_tokens": 4,
        "embedding_dim": 1024,
        "hidden_dim": 1024,
        "graph_layers": 2,
        "attention_heads": 8,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "edge_hidden_dim": 64,
        "edge_embedding_dim": 64,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "implicit_self_loops": False,
        "trainable_node_identifiers": False,
        "trainable_edge_identifiers": False,
        "receiver_chunk_size": 256,
        "activation_checkpointing": True,
        "exact_receiver_partitioning": True,
    },
}
_EXPECTED_MODEL[_PILOT] = _EXPECTED_MODEL[_WIDER]
_SHARED_DIGESTS = {
    "dataset": "c77ff5b7a1b24319007457d76b3b335d322fbafcc751023aa9005c3a8fe9a485",
    "graph": "e72135fd7295bd4598b85b3c2f8ef41ec3850fc3ef462a3db1c9877454bbf0f2",
    "features": "7dbd5cd9bda1646b96a26b85117c26e16d0dcd096cabf898cc5b61d3f8a6a179",
}
_CONCLUSION_DIGESTS = {
    **_SHARED_DIGESTS,
    "masking": "f9f0871ded533a118943f4a11e585336156f58fa3eff367801256e81c63fd454",
    "trainer": "42a9727a2331d990fa83bacf9bc5bc5cc03bc937769800d0bc736b92ad35a689",
    "evaluation": "b2b876ce4abcf36c0655ccb36e5efb6ee21545213eeddccdeb8faec8953b3dde",
}
_PILOT_DIGESTS = {
    **_SHARED_DIGESTS,
    "masking": "3116310866e53b051384891ec280af1b3b9e56e506d3545f813897bf1dbcc7dc",
    "trainer": "efc81355b18b7814c2c385119c804d0e8704a55b7e34e941a6c8f4e819f241f6",
    "evaluation": "ad895531e2ce3716f533fb2f88f2667b19fcd2c092d492513d00a424409bb6d9",
}
_TOKEN_SPEC = {
    "name": "raw_count_tokens_0_1_2_3plus_v1",
    "output_tokens": [
        {"id": 0, "label": "count0"},
        {"id": 1, "label": "count1"},
        {"id": 2, "label": "count2"},
        {"id": 3, "label": "count>=3"},
    ],
    "mask_input_token": {"id": 4, "label": "masked"},
    "num_output_tokens": 4,
    "vocabulary_size": 5,
}
_RUN_ID = re.compile(
    r"^r_\d{8}T\d{6}Z_(?P<scientific>[a-z0-9]+)_s(?P<seed>\d{3})_"
    r"f(?P<fold>\d{2})_a(?P<attempt>\d{2})_[a-z0-9_-]+$"
)
_COMPLETION_MARKERS = ("_SUCCESS", "_FAILED", "_PRUNED")
_TABLE_SUFFIXES = (".jsonl", ".parquet")
_OWNER_FILE = ".g2-token-comparison-owner.json"
_OUTPUT_FILES = frozenset(
    {
        _OWNER_FILE,
        "comparison.json",
        "per_run.csv",
        "paired.csv",
        "report.md",
    }
)
_OWNER = {
    "schema_version": 1,
    "owner": "compare_g2_token_multiseed.py",
    "artifact_kind": "g2_token_multiseed_comparison",
    "campaign_id": _CAMPAIGN_ID,
}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_REL_TOL = 1e-9
_ABS_TOL = 1e-9


class G2TokenComparisonError(RuntimeError):
    """Raised when supplied evidence violates the locked campaign contract."""


@dataclass(frozen=True)
class History:
    rows: int
    table_format: str
    schedule: tuple[tuple[Any, ...], ...]
    final_loss: float
    minimum_loss: float
    last_20_slope: float
    summed_duration_seconds: float
    peak_vram_gb: float


@dataclass(frozen=True)
class TokenMetrics:
    cross_entropy: float
    exact_percent: float
    balanced_percent: float
    nonzero_percent: float
    baselines: tuple[float, ...]
    recalls: tuple[float, ...]
    supports: tuple[int, ...]


@dataclass(frozen=True)
class Run:
    root: Path
    run_id: str
    variant: str
    seed: int
    fold: int
    attempt: int
    job_id: str
    parameter_count: int
    graph_sha256: str
    graph_edges: int
    mask_bundle_sha256: str
    mask_identity: tuple[tuple[Any, ...], ...]
    data_identity: tuple[Any, ...]
    token_audit_digest: str
    token_matrix_sha256: str
    modal_tokens_sha256: str
    metrics: TokenMetrics
    mask_metrics: tuple[TokenMetrics, ...]
    history: History
    total_duration_seconds: float
    training_duration_seconds: float


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise G2TokenComparisonError(f"{label} must be a mapping")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)
    except (OSError, ValueError) as error:
        raise G2TokenComparisonError(
            f"required JSON artifact is unreadable: {path}"
        ) from error


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)
    except (OSError, yaml.YAMLError) as error:
        raise G2TokenComparisonError(
            f"required YAML artifact is unreadable: {path}"
        ) from error


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise G2TokenComparisonError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise G2TokenComparisonError(f"{label} must be a finite number") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise G2TokenComparisonError(f"{label} is outside its finite range")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise G2TokenComparisonError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise G2TokenComparisonError(f"{label} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise G2TokenComparisonError(f"{label} must be an integer")
    if isinstance(value, str) and value.strip() != str(result):
        raise G2TokenComparisonError(f"{label} must be an integer")
    return result


def _sha256(value: Any, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise G2TokenComparisonError(f"{label} must be a lowercase SHA-256")
    return result


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise G2TokenComparisonError("artifact is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _int64_vector_sha256(values: Sequence[int]) -> str:
    header = json.dumps(
        {"dtype": "int64-le", "shape": [len(values)]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(struct.pack(f"<{len(values)}q", *values))
    return digest.hexdigest()


def _table_path(root: Path, stem: str) -> Path:
    matches = [
        root / f"{stem}{suffix}"
        for suffix in _TABLE_SUFFIXES
        if (root / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise G2TokenComparisonError(
            f"{root.name} requires exactly one {stem} JSONL or Parquet table"
        )
    return matches[0]


def _load_table(path: Path) -> tuple[Mapping[str, Any], ...]:
    if path.suffix == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    rows.append(_mapping(json.loads(line), f"{path.name}:{number}"))
        except (OSError, ValueError) as error:
            raise G2TokenComparisonError(f"unreadable table: {path}") from error
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet

            rows = [
                _mapping(row, f"row in {path.name}")
                for row in parquet.read_table(path).to_pylist()
            ]
        except Exception as error:
            raise G2TokenComparisonError(f"unreadable Parquet table: {path}") from error
    else:
        raise G2TokenComparisonError(f"unsupported table: {path}")
    if not rows:
        raise G2TokenComparisonError(f"required table is empty: {path}")
    return tuple(rows)


def _verify_success(root: Path) -> None:
    markers = [name for name in _COMPLETION_MARKERS if (root / name).is_file()]
    if markers != ["_SUCCESS"]:
        raise G2TokenComparisonError(f"{root.name} requires exactly one _SUCCESS marker")
    try:
        result = verify_run_bundle(root, require_success_contract=True)
    except (RunValidationError, OSError, ValueError) as error:
        raise G2TokenComparisonError(
            f"{root.name} failed finalized-bundle verification"
        ) from error
    if result.get("status") != "success":
        raise G2TokenComparisonError(f"{root.name} is not a verified success")


def _validate_config(
    config: Mapping[str, Any],
    *,
    variant: str,
    run_id: str,
) -> None:
    if config.get("version") != 1:
        raise G2TokenComparisonError(f"{run_id} config version is not v1")
    if _mapping(config.get("campaign"), f"{run_id} campaign") != {
        "campaign_id": _CAMPAIGN_ID,
        "display_name": _CAMPAIGN_NAME,
    }:
        raise G2TokenComparisonError(f"{run_id} has the wrong campaign")
    if _mapping(config.get("experiment"), f"{run_id} experiment") != _EXPECTED_EXPERIMENT[variant]:
        raise G2TokenComparisonError(f"{run_id} experiment contract drifted")
    if _mapping(config.get("classification"), f"{run_id} classification") != _EXPECTED_CLASSIFICATION[variant]:
        raise G2TokenComparisonError(f"{run_id} classification contract drifted")
    expected = _PILOT_DIGESTS if variant == _PILOT else _CONCLUSION_DIGESTS
    for section, expected_digest in expected.items():
        observed = _mapping(config.get(section), f"{run_id} config.{section}")
        if _digest(observed) != expected_digest:
            raise G2TokenComparisonError(f"{run_id} config.{section} drifted")
    if _mapping(config.get("model"), f"{run_id} config.model") != _EXPECTED_MODEL[variant]:
        raise G2TokenComparisonError(f"{run_id} exact model shape drifted")
    if _integer(config.get("fold"), f"{run_id} fold") != 0:
        raise G2TokenComparisonError(f"{run_id} fold must be 0")


def _lineage(
    root: Path,
    config: Mapping[str, Any],
    *,
    variant: str,
) -> tuple[int, int, int, str]:
    match = _RUN_ID.fullmatch(root.name)
    if match is None:
        raise G2TokenComparisonError(f"{root.name} is not a canonical run ID")
    if match.group("scientific") != _SCIENTIFIC_TOKENS[variant]:
        raise G2TokenComparisonError(f"{root.name} scientific identity is wrong")
    queue = _load_json(root / "provenance/queue.json")
    job_id = str(queue.get("job_id", ""))
    if re.fullmatch(r"q_[a-z0-9][a-z0-9_-]{2,63}", job_id) is None:
        raise G2TokenComparisonError(f"{root.name} queue job_id is invalid")
    seed = _integer(config.get("seed"), f"{root.name} config.seed")
    fold = _integer(config.get("fold"), f"{root.name} config.fold")
    attempt = _integer(config.get("attempt"), f"{root.name} config.attempt")
    queue_attempt = _integer(queue.get("attempt"), f"{root.name} queue.attempt")
    encoded = (
        int(match.group("seed")),
        int(match.group("fold")),
        int(match.group("attempt")),
    )
    if (seed, fold, attempt) != encoded or queue_attempt != attempt or attempt < 1:
        raise G2TokenComparisonError(f"{root.name} execution lineage disagrees")
    retry_of = queue.get("retry_of")
    if attempt == 1 and retry_of is not None:
        raise G2TokenComparisonError(f"{root.name} attempt 1 cannot be a retry")
    if attempt > 1 and not isinstance(retry_of, str):
        raise G2TokenComparisonError(f"{root.name} retry lacks retry_of")
    return seed, fold, attempt, job_id


def _validate_token_audit(root: Path) -> tuple[str, str, str]:
    run_id = root.name
    audit = _load_json(root / "diagnostics/expression_tokenization.json")
    expected_audit_keys = {
        "spec",
        "shape",
        "n_cells",
        "n_genes",
        "n_entries",
        "token_counts",
        "token_prevalence",
        "token_prevalence_percent",
        "all_output_tokens_present",
        "gene_class_coverage",
        "token_checksum_sha256",
        "per_gene_modal_tokens",
        "per_gene_modal_tokens_checksum_sha256",
        "thresholds_fitted",
        "fit_scope",
        "input_mask_token_is_output_class",
    }
    if set(audit) != expected_audit_keys:
        raise G2TokenComparisonError(f"{run_id} token audit schema drifted")
    if audit.get("spec") != _TOKEN_SPEC:
        raise G2TokenComparisonError(f"{run_id} token vocabulary drifted")
    if tuple(audit.get("shape", ())) != _EXPECTED_SHAPE:
        raise G2TokenComparisonError(f"{run_id} token matrix shape drifted")
    if (
        _integer(audit.get("n_cells"), f"{run_id} token n_cells") != _EXPECTED_SHAPE[0]
        or _integer(audit.get("n_genes"), f"{run_id} token n_genes") != _EXPECTED_SHAPE[1]
        or _integer(audit.get("n_entries"), f"{run_id} token n_entries")
        != _EXPECTED_SHAPE[0] * _EXPECTED_SHAPE[1]
    ):
        raise G2TokenComparisonError(f"{run_id} token dimensions disagree")
    counts = tuple(
        _integer(value, f"{run_id} token count") for value in audit.get("token_counts", ())
    )
    if counts != _EXPECTED_TOKEN_COUNTS:
        raise G2TokenComparisonError(f"{run_id} token counts drifted")
    if (
        audit.get("all_output_tokens_present") is not True
        or audit.get("thresholds_fitted") is not False
        or audit.get("fit_scope") != "all_nodes_transductive"
        or audit.get("input_mask_token_is_output_class") is not False
    ):
        raise G2TokenComparisonError(f"{run_id} token audit semantics drifted")
    expected_prevalence = [
        count / (_EXPECTED_SHAPE[0] * _EXPECTED_SHAPE[1]) for count in counts
    ]
    prevalence = list(audit.get("token_prevalence", ()))
    prevalence_percent = list(audit.get("token_prevalence_percent", ()))
    if len(prevalence) != 4 or len(prevalence_percent) != 4:
        raise G2TokenComparisonError(f"{run_id} token prevalence is incomplete")
    for token_id, expected in enumerate(expected_prevalence):
        if not _same(_finite(prevalence[token_id], "token prevalence"), expected):
            raise G2TokenComparisonError(f"{run_id} token prevalence drifted")
        if not _same(
            _finite(prevalence_percent[token_id], "token prevalence percent"),
            100.0 * expected,
        ):
            raise G2TokenComparisonError(f"{run_id} token prevalence percent drifted")
    coverage = _mapping(audit.get("gene_class_coverage"), f"{run_id} gene coverage")
    if set(coverage) != {
        "classes_present_per_gene",
        "genes_with_all_output_tokens",
        "genes_with_all_output_tokens_percent",
        "all_genes_have_all_output_tokens",
        "genes_containing_each_output_token",
        "gene_coverage_percent_by_output_token",
    }:
        raise G2TokenComparisonError(f"{run_id} gene-coverage schema drifted")
    if (
        coverage.get("all_genes_have_all_output_tokens") is not True
        or coverage.get("genes_with_all_output_tokens") != 1_000
        or coverage.get("genes_with_all_output_tokens_percent") != 100.0
        or coverage.get("genes_containing_each_output_token") != [1_000] * 4
        or coverage.get("gene_coverage_percent_by_output_token") != [100.0] * 4
    ):
        raise G2TokenComparisonError(f"{run_id} per-gene class coverage drifted")
    classes = list(coverage.get("classes_present_per_gene", ()))
    modes = [
        _integer(value, f"{run_id} per-gene modal token")
        for value in audit.get("per_gene_modal_tokens", ())
    ]
    if classes != [4] * 1_000 or len(modes) != 1_000 or any(
        value not in range(4) for value in modes
    ):
        raise G2TokenComparisonError(f"{run_id} per-gene token evidence drifted")
    modal_checksum = _sha256(
        audit.get("per_gene_modal_tokens_checksum_sha256"),
        f"{run_id} modal-token checksum",
    )
    if _int64_vector_sha256(modes) != modal_checksum:
        raise G2TokenComparisonError(f"{run_id} modal-token checksum disagrees")
    token_checksum = _sha256(
        audit.get("token_checksum_sha256"), f"{run_id} token checksum"
    )
    if token_checksum != _EXPECTED_TOKEN_MATRIX_SHA256:
        raise G2TokenComparisonError(f"{run_id} materialized token matrix drifted")
    return _digest(audit), token_checksum, modal_checksum


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise G2TokenComparisonError("at least two values are needed for slope")
    x_mean = (len(values) - 1) / 2
    y_mean = statistics.fmean(values)
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / sum((index - x_mean) ** 2 for index in range(len(values)))


def _validate_history(
    root: Path,
    *,
    epochs: int,
    graph_edges: int,
    summary: Mapping[str, Any],
    convergence: Mapping[str, Any],
) -> History:
    run_id = root.name
    path = _table_path(root, "metrics/history")
    rows = _load_table(path)
    if len(rows) != epochs:
        raise G2TokenComparisonError(
            f"{run_id} requires exactly {epochs} complete history rows"
        )
    normalized: dict[int, tuple[float, float, int, tuple[Any, ...]]] = {}
    for number, row in enumerate(rows, 1):
        if (
            row.get("run_id") != run_id
            or row.get("split") != "fit"
            or row.get("training_protocol") != _TRAINING_PROTOCOL
        ):
            raise G2TokenComparisonError(f"{run_id} history row {number} is misaligned")
        epoch = _integer(row.get("epoch"), f"{run_id} history epoch")
        if epoch in normalized:
            raise G2TokenComparisonError(f"{run_id} repeats epoch {epoch}")
        mode = str(row.get("mask_mode", ""))
        if mode not in {"partial", "node", "block"}:
            raise G2TokenComparisonError(f"{run_id} has an unknown mask mode")
        mask_seed = _integer(row.get("mask_seed"), "mask_seed")
        mask_checksum = _sha256(row.get("mask_checksum"), "mask_checksum")
        edge_seed = _integer(row.get("edge_dropout_seed"), "edge_dropout_seed")
        edge_checksum = _sha256(row.get("edge_checksum"), "edge_checksum")
        n_masked = _integer(row.get("n_masked_entries"), "n_masked_entries")
        n_nodes = _integer(row.get("n_target_nodes"), "n_target_nodes")
        n_edges = _integer(row.get("n_edges_used"), "n_edges_used")
        if n_masked <= 0 or n_nodes <= 0 or n_edges != graph_edges:
            raise G2TokenComparisonError(f"{run_id} history evidence is incomplete")
        loss = _finite(row.get("train_loss"), "train_loss", minimum=0.0)
        gradient = _finite(row.get("gradient_norm"), "gradient_norm", minimum=0.0)
        duration = _finite(row.get("duration_seconds"), "duration", minimum=0.0)
        peak = _integer(row.get("peak_cuda_memory_bytes"), "peak_cuda_memory_bytes")
        if peak < 0 or gradient < 0:
            raise G2TokenComparisonError(f"{run_id} has invalid resource diagnostics")
        schedule = (
            epoch,
            mode,
            mask_seed,
            mask_checksum,
            edge_seed,
            edge_checksum,
            n_masked,
            n_nodes,
            n_edges,
        )
        normalized[epoch] = (loss, duration, peak, schedule)
    if tuple(sorted(normalized)) != tuple(range(epochs)):
        raise G2TokenComparisonError(f"{run_id} history epochs are not contiguous")
    ordered = [normalized[index] for index in range(epochs)]
    losses = [item[0] for item in ordered]
    final_loss = losses[-1]
    minimum_loss = min(losses)
    slope = _slope(losses[-min(20, epochs) :])
    if (
        _integer(summary.get("fixed_epoch_budget"), "fixed_epoch_budget") != epochs
        or _integer(summary.get("final_epoch"), "final_epoch") != epochs - 1
        or _integer(convergence.get("final_epoch"), "convergence final_epoch")
        != epochs - 1
        or convergence.get("all_epochs_completed") is not True
        or convergence.get("all_losses_and_gradients_finite") is not True
        or convergence.get("objective") != "masked_token_cross_entropy"
    ):
        raise G2TokenComparisonError(f"{run_id} did not complete finite fixed training")
    checks = (
        (final_loss, convergence.get("final_train_loss"), "final loss"),
        (minimum_loss, convergence.get("minimum_observed_train_loss"), "minimum loss"),
        (slope, convergence.get("last_20_epoch_loss_slope"), "loss slope"),
    )
    for observed, recorded, label in checks:
        if not _same(observed, _finite(recorded, f"{run_id} {label}")):
            raise G2TokenComparisonError(f"{run_id} {label} disagrees")
    return History(
        rows=epochs,
        table_format=path.suffix[1:],
        schedule=tuple(item[3] for item in ordered),
        final_loss=final_loss,
        minimum_loss=minimum_loss,
        last_20_slope=slope,
        summed_duration_seconds=sum(item[1] for item in ordered),
        peak_vram_gb=max(item[2] for item in ordered) / (1024**3),
    )


_BASELINE_FIELDS = (
    "baseline_uniform_accuracy_percent",
    "baseline_empirical_frequency_accuracy_percent",
    "baseline_always_zero_accuracy_percent",
    "baseline_always_zero_balanced_accuracy_percent",
    "baseline_per_gene_modal_accuracy_percent",
    "baseline_per_gene_modal_balanced_accuracy_percent",
    "baseline_per_gene_modal_nonzero_accuracy_percent",
)


def _validate_token_row(
    row: Mapping[str, Any],
    *,
    run_id: str,
    number: int,
) -> tuple[TokenMetrics, tuple[Any, ...]]:
    prefix = f"{run_id} whole-node row {number}"
    replicate = _integer(row.get("mask_replicate"), f"{prefix} replicate")
    seed = _integer(row.get("mask_seed"), f"{prefix} mask seed")
    checksum = _sha256(row.get("mask_checksum"), f"{prefix} mask checksum")
    n_masked = _integer(row.get("n_masked"), f"{prefix} n_masked")
    supports = tuple(
        _integer(row.get(f"token_{token}_support"), f"{prefix} token {token} support")
        for token in range(4)
    )
    confusion = tuple(
        tuple(
            _integer(
                row.get(f"confusion_{target}_{predicted}"),
                f"{prefix} confusion {target},{predicted}",
            )
            for predicted in range(4)
        )
        for target in range(4)
    )
    if any(value < 0 for row_values in confusion for value in row_values):
        raise G2TokenComparisonError(f"{prefix} confusion counts cannot be negative")
    if n_masked <= 0 or sum(supports) != n_masked:
        raise G2TokenComparisonError(f"{prefix} support does not sum to n_masked")
    if any(sum(confusion[token]) != supports[token] for token in range(4)):
        raise G2TokenComparisonError(f"{prefix} confusion and supports disagree")
    if any(value <= 0 for value in supports):
        raise G2TokenComparisonError(f"{prefix} does not contain all output tokens")
    recalls = tuple(
        _finite(row.get(f"token_{token}_recall_percent"), f"{prefix} recall")
        for token in range(4)
    )
    expected_recalls = tuple(
        100.0 * confusion[token][token] / supports[token] for token in range(4)
    )
    if any(not _same(left, right) for left, right in zip(recalls, expected_recalls)):
        raise G2TokenComparisonError(f"{prefix} token recall disagrees")
    exact = _finite(row.get("masked_token_accuracy_percent"), f"{prefix} exact")
    balanced = _finite(
        row.get("masked_token_balanced_accuracy_percent"), f"{prefix} balanced"
    )
    nonzero = _finite(
        row.get("masked_nonzero_token_accuracy_percent"), f"{prefix} nonzero"
    )
    cross_entropy = _finite(
        row.get("masked_token_cross_entropy"), f"{prefix} cross entropy", minimum=0.0
    )
    expected_exact = 100.0 * sum(confusion[token][token] for token in range(4)) / n_masked
    expected_balanced = statistics.fmean(expected_recalls)
    expected_nonzero = (
        100.0
        * sum(confusion[token][token] for token in range(1, 4))
        / sum(supports[1:])
    )
    if not (
        _same(exact, expected_exact)
        and _same(balanced, expected_balanced)
        and _same(nonzero, expected_nonzero)
    ):
        raise G2TokenComparisonError(f"{prefix} percentage metrics do not reconcile")
    baselines = tuple(
        _finite(row.get(field), f"{prefix} {field}") for field in _BASELINE_FIELDS
    )
    empirical = 100.0 * sum((support / n_masked) ** 2 for support in supports)
    if not (
        _same(baselines[0], 25.0)
        and _same(baselines[1], empirical)
        and _same(baselines[2], 100.0 * supports[0] / n_masked)
        and _same(baselines[3], 25.0)
    ):
        raise G2TokenComparisonError(f"{prefix} chance or zero baseline disagrees")
    if any(value < 0.0 or value > 100.0 for value in (*recalls, *baselines, exact, balanced, nonzero)):
        raise G2TokenComparisonError(f"{prefix} percentage lies outside 0-100")
    identity = (
        replicate,
        str(row.get("mask_entry_id", "")),
        seed,
        checksum,
        n_masked,
        supports,
        baselines,
    )
    return (
        TokenMetrics(
            cross_entropy=cross_entropy,
            exact_percent=exact,
            balanced_percent=balanced,
            nonzero_percent=nonzero,
            baselines=baselines,
            recalls=recalls,
            supports=supports,
        ),
        identity,
    )


def _validate_mask_manifest(
    root: Path,
    *,
    replicates: int,
    expected_bundle_checksum: str,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    run_id = root.name
    provenance = _load_json(root / "provenance/fixed_evaluation_masks.json")
    if (
        provenance.get("used_for_checkpoint_selection") is not False
        or provenance.get("used_for_gradient_updates") is not False
    ):
        raise G2TokenComparisonError(
            f"{run_id} fixed evaluation masks were used for fitting or selection"
        )
    manifest = _mapping(
        provenance.get("bundle_manifest"),
        f"{run_id} fixed-mask bundle manifest",
    )
    checksum = _sha256(
        manifest.get("bundle_checksum"),
        f"{run_id} fixed-mask bundle checksum",
    )
    core = dict(manifest)
    core.pop("bundle_checksum", None)
    core.pop("bundle_id", None)
    if (
        checksum != expected_bundle_checksum
        or _digest(core) != checksum
        or manifest.get("bundle_id") != checksum[:16]
        or manifest.get("format_version") != 1
        or manifest.get("n_genes") != _EXPECTED_SHAPE[1]
        or manifest.get("replicates") != replicates
    ):
        raise G2TokenComparisonError(
            f"{run_id} fixed-mask manifest identity or checksum disagrees"
        )
    splits = _mapping(manifest.get("splits"), f"{run_id} fixed-mask splits")
    fit = _mapping(splits.get("fit"), f"{run_id} fixed-mask fit split")
    if set(splits) != {"fit"} or (
        fit.get("n_nodes") != _EXPECTED_SHAPE[0]
        or fit.get("n_eligible_nodes") != _EXPECTED_SHAPE[0]
    ):
        raise G2TokenComparisonError(
            f"{run_id} fixed-mask split dimensions drifted"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 3 * replicates:
        raise G2TokenComparisonError(
            f"{run_id} fixed-mask manifest has the wrong entry count"
        )
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    expected_modes = {
        "partial_gene": "partial",
        "whole_node": "node",
        "spatial_block": "block",
    }
    for number, raw_entry in enumerate(entries, 1):
        entry = _mapping(raw_entry, f"{run_id} fixed-mask entry {number}")
        spec = _mapping(entry.get("spec"), f"{run_id} fixed-mask spec {number}")
        label = str(spec.get("label", ""))
        replicate = _integer(
            entry.get("replicate"),
            f"{run_id} fixed-mask entry {number} replicate",
        )
        key = (label, replicate)
        summary = _mapping(
            entry.get("summary"),
            f"{run_id} fixed-mask entry {number} summary",
        )
        if (
            label not in expected_modes
            or spec.get("mode") != expected_modes.get(label)
            or key in by_key
            or entry.get("split") != "fit"
            or tuple(entry.get("shape", ())) != _EXPECTED_SHAPE
            or tuple(summary.get("shape", ())) != _EXPECTED_SHAPE
            or summary.get("mode") != expected_modes.get(label)
            or _integer(
                summary.get("n_masked_entries"),
                f"{run_id} fixed-mask entry {number} n_masked",
            )
            <= 0
            or _integer(
                entry.get("seed"),
                f"{run_id} fixed-mask entry {number} seed",
            )
            != _integer(
                summary.get("seed"),
                f"{run_id} fixed-mask entry {number} summary seed",
            )
        ):
            raise G2TokenComparisonError(
                f"{run_id} fixed-mask manifest entry {number} drifted"
            )
        _sha256(
            entry.get("mask_checksum"),
            f"{run_id} fixed-mask entry {number} checksum",
        )
        if not str(entry.get("entry_id", "")):
            raise G2TokenComparisonError(
                f"{run_id} fixed-mask entry {number} lacks an ID"
            )
        by_key[key] = entry
    expected_keys = {
        (mode, replicate)
        for mode in expected_modes
        for replicate in range(replicates)
    }
    if set(by_key) != expected_keys:
        raise G2TokenComparisonError(
            f"{run_id} fixed-mask manifest is missing configured entries"
        )
    return by_key


def _validate_evaluation(
    root: Path,
    *,
    replicates: int,
    summary: Mapping[str, Any],
    final: Mapping[str, Any],
) -> tuple[
    TokenMetrics,
    tuple[TokenMetrics, ...],
    tuple[tuple[Any, ...], ...],
    str,
]:
    run_id = root.name
    if dict(_mapping(summary.get("metrics"), f"{run_id} summary metrics")) != dict(final):
        raise G2TokenComparisonError(f"{run_id} summary and final metrics disagree")
    expected_bundle_checksum = _sha256(
        summary.get("evaluation_mask_bundle_sha256"),
        f"{run_id} evaluation mask bundle",
    )
    manifest_entries = _validate_mask_manifest(
        root,
        replicates=replicates,
        expected_bundle_checksum=expected_bundle_checksum,
    )
    path = _table_path(root, "metrics/evaluation_replicates")
    selected: dict[int, tuple[TokenMetrics, tuple[Any, ...]]] = {}
    rows = _load_table(path)
    expected_keys = {
        (mode, replicate)
        for mode in ("partial_gene", "whole_node", "spatial_block")
        for replicate in range(replicates)
    }
    observed_keys: set[tuple[str, int]] = set()
    for number, row in enumerate(rows, 1):
        mode = str(row.get("mask_mode", ""))
        replicate = _integer(
            row.get("mask_replicate"),
            f"{run_id} evaluation row {number} replicate",
        )
        key = (mode, replicate)
        if key in observed_keys:
            raise G2TokenComparisonError(
                f"{run_id} repeats evaluation row {mode}/{replicate}"
            )
        observed_keys.add(key)
        if row.get("split") != "fit":
            raise G2TokenComparisonError(f"{run_id} evaluation row is not fit")
        row_checksum = _sha256(
            row.get("mask_checksum"),
            f"{run_id} evaluation row {number} mask checksum",
        )
        n_masked = _integer(
            row.get("n_masked"),
            f"{run_id} evaluation row {number} n_masked",
        )
        if n_masked <= 0:
            raise G2TokenComparisonError(
                f"{run_id} evaluation row has no masked entries"
            )
        manifest_entry = manifest_entries.get(key)
        if manifest_entry is None:
            raise G2TokenComparisonError(
                f"{run_id} evaluation row is absent from the fixed-mask manifest"
            )
        manifest_summary = _mapping(
            manifest_entry.get("summary"),
            f"{run_id} fixed-mask summary for {mode}/{replicate}",
        )
        if (
            row.get("mask_entry_id") != manifest_entry.get("entry_id")
            or _integer(row.get("mask_seed"), "evaluation mask seed")
            != _integer(manifest_entry.get("seed"), "manifest mask seed")
            or row_checksum != manifest_entry.get("mask_checksum")
            or n_masked != manifest_summary.get("n_masked_entries")
        ):
            raise G2TokenComparisonError(
                f"{run_id} evaluation row and fixed-mask manifest disagree"
            )
        if mode != "whole_node":
            continue
        metric, identity = _validate_token_row(row, run_id=run_id, number=number)
        selected[replicate] = (metric, identity)
    if observed_keys != expected_keys:
        raise G2TokenComparisonError(
            f"{run_id} evaluation table does not contain exactly all configured "
            "mode/replicate pairs"
        )
    if tuple(sorted(selected)) != tuple(range(replicates)):
        raise G2TokenComparisonError(
            f"{run_id} requires whole-node replicates 0 through {replicates - 1}"
        )
    ordered = [selected[index] for index in range(replicates)]

    def mean(attribute: str) -> float:
        return statistics.fmean(getattr(item[0], attribute) for item in ordered)

    aggregate = TokenMetrics(
        cross_entropy=mean("cross_entropy"),
        exact_percent=mean("exact_percent"),
        balanced_percent=mean("balanced_percent"),
        nonzero_percent=mean("nonzero_percent"),
        baselines=tuple(
            statistics.fmean(item[0].baselines[index] for item in ordered)
            for index in range(len(_BASELINE_FIELDS))
        ),
        recalls=tuple(
            statistics.fmean(item[0].recalls[index] for item in ordered)
            for index in range(4)
        ),
        supports=tuple(sum(item[0].supports[index] for item in ordered) for index in range(4)),
    )
    final_fields: dict[str, float] = {
        "fit/whole_node/masked_token_cross_entropy": aggregate.cross_entropy,
        "fit/whole_node/masked_token_accuracy_percent": aggregate.exact_percent,
        "fit/whole_node/masked_token_balanced_accuracy_percent": aggregate.balanced_percent,
        "fit/whole_node/masked_nonzero_token_accuracy_percent": aggregate.nonzero_percent,
    }
    final_fields.update(
        {
            f"fit/whole_node/{field}": aggregate.baselines[index]
            for index, field in enumerate(_BASELINE_FIELDS)
        }
    )
    final_fields.update(
        {
            f"fit/whole_node/token_{token}_recall_percent": aggregate.recalls[token]
            for token in range(4)
        }
    )
    for field, expected in final_fields.items():
        if not _same(_finite(final.get(field), f"{run_id} final {field}"), expected):
            raise G2TokenComparisonError(f"{run_id} final metric {field} disagrees")
    if (
        summary.get("primary_metric_name")
        != "fit/whole_node/masked_token_accuracy_percent"
        or not _same(
            _finite(summary.get("primary_metric_value"), f"{run_id} primary value"),
            aggregate.exact_percent,
        )
    ):
        raise G2TokenComparisonError(f"{run_id} primary metric disagrees")
    return (
        aggregate,
        tuple(item[0] for item in ordered),
        tuple(item[1] for item in ordered),
        path.suffix[1:],
    )


def _load_run(path: str | Path, *, pilot: bool) -> Run:
    try:
        root = Path(path).resolve(strict=True)
    except OSError as error:
        raise G2TokenComparisonError(f"run directory does not exist: {path}") from error
    if not root.is_dir():
        raise G2TokenComparisonError(f"run input is not a directory: {path}")
    _verify_success(root)
    config = _load_yaml(root / "config.resolved.yaml")
    summary = _load_json(root / "summary.json")
    final = _load_json(root / "metrics/final.json")
    convergence = _load_json(root / "diagnostics/training_convergence.json")
    experiment = _mapping(config.get("experiment"), f"{root.name} experiment")
    variant = str(experiment.get("variant_label", ""))
    allowed = {_PILOT} if pilot else set(_CONCLUSION_VARIANTS)
    if variant not in allowed:
        raise G2TokenComparisonError(f"{root.name} has an unsupported variant")
    _validate_config(config, variant=variant, run_id=root.name)
    seed, fold, attempt, job_id = _lineage(root, config, variant=variant)
    epochs = _PILOT_EPOCHS if pilot else _EXPECTED_EPOCHS
    replicates = _PILOT_REPLICATES if pilot else _EXPECTED_REPLICATES
    if pilot and seed != 0:
        raise G2TokenComparisonError(f"{root.name} pilot seed must be 0")
    if not pilot and seed not in _EXPECTED_SEEDS:
        raise G2TokenComparisonError(f"{root.name} conclusion seed is invalid")
    if (
        summary.get("run_id") != root.name
        or summary.get("status") != "success"
        or summary.get("training_exit_status") != "success"
        or summary.get("evaluation_protocol") != _EVALUATION_PROTOCOL
        or summary.get("task_family") != _TASK
        or summary.get("model_name") != "g2-tokenized"
        or summary.get("generalization_estimate") is not False
        or _integer(summary.get("model_seed"), "model seed") != seed
        or summary.get("diagnostic_resource_pilot") is not pilot
        or summary.get("conclusion_eligible") is pilot
        or _integer(summary.get("evaluation_mask_replicates_per_mode"), "replicates")
        != replicates
        or summary.get("evaluation_metrics_include_all_configured_replicates_per_mode")
        is not True
    ):
        raise G2TokenComparisonError(f"{root.name} summary is outside the protocol")
    graph_sha256 = _sha256(summary.get("graph_sha256"), "graph checksum")
    graph_edges = _integer(summary.get("graph_directed_edges"), "graph edges")
    if graph_sha256 != _EXPECTED_GRAPH_SHA256 or graph_edges != _EXPECTED_GRAPH_EDGES:
        raise G2TokenComparisonError(f"{root.name} graph identity drifted")
    audit_digest, token_checksum, modal_checksum = _validate_token_audit(root)
    metrics, mask_metrics, mask_identity, _ = _validate_evaluation(
        root,
        replicates=replicates,
        summary=summary,
        final=final,
    )
    history = _validate_history(
        root,
        epochs=epochs,
        graph_edges=graph_edges,
        summary=summary,
        convergence=convergence,
    )
    parameter_sources = (
        _integer(summary.get("parameter_count"), "summary parameter count"),
        _integer(final.get("resource/parameter_count"), "final parameter count"),
    )
    if len(set(parameter_sources)) != 1 or parameter_sources[0] != _EXPECTED_PARAMETERS[variant]:
        raise G2TokenComparisonError(f"{root.name} parameter count disagrees")
    total = _finite(final.get("resource/total_duration_seconds"), "total duration", minimum=0)
    training = _finite(
        final.get("resource/training_duration_seconds"), "training duration", minimum=0
    )
    peak = _finite(final.get("resource/peak_vram_gb"), "peak VRAM", minimum=0)
    if (
        not _same(total, _finite(summary.get("duration_seconds"), "summary duration"))
        or not _same(peak, _finite(summary.get("peak_vram_gb"), "summary peak VRAM"))
        or not _same(peak, history.peak_vram_gb)
        or training > total + _ABS_TOL
        or history.summed_duration_seconds > training + _ABS_TOL
    ):
        raise G2TokenComparisonError(f"{root.name} resource records disagree")
    dataset = _mapping(config.get("dataset"), f"{root.name} dataset")
    data_identity = tuple(dataset.get(field) for field in _DATA_FIELDS)
    if any(value is None or value == "" for value in data_identity):
        raise G2TokenComparisonError(f"{root.name} dataset identity is incomplete")
    _sha256(data_identity[2], "dataset fingerprint")
    _sha256(data_identity[5], "split fingerprint")
    return Run(
        root=root,
        run_id=root.name,
        variant=variant,
        seed=seed,
        fold=fold,
        attempt=attempt,
        job_id=job_id,
        parameter_count=parameter_sources[0],
        graph_sha256=graph_sha256,
        graph_edges=graph_edges,
        mask_bundle_sha256=_sha256(
            summary.get("evaluation_mask_bundle_sha256"), "evaluation mask bundle"
        ),
        mask_identity=mask_identity,
        data_identity=data_identity,
        token_audit_digest=audit_digest,
        token_matrix_sha256=token_checksum,
        modal_tokens_sha256=modal_checksum,
        metrics=metrics,
        mask_metrics=mask_metrics,
        history=history,
        total_duration_seconds=total,
        training_duration_seconds=training,
    )


def _statistics(values: Sequence[float]) -> dict[str, float | int]:
    if len(values) != 3:
        raise G2TokenComparisonError("seed aggregate requires exactly three values")
    return {
        "n": 3,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _metric_dict(metrics: TokenMetrics) -> dict[str, Any]:
    result: dict[str, Any] = {
        "whole_node_cross_entropy": metrics.cross_entropy,
        "whole_node_exact_accuracy_percent": metrics.exact_percent,
        "whole_node_balanced_accuracy_percent": metrics.balanced_percent,
        "whole_node_nonzero_accuracy_percent": metrics.nonzero_percent,
    }
    result.update(
        {field: metrics.baselines[index] for index, field in enumerate(_BASELINE_FIELDS)}
    )
    for token in range(4):
        result[f"token_{token}_recall_percent"] = metrics.recalls[token]
        result[f"token_{token}_support"] = metrics.supports[token]
    return result


def _reconcile_registry(
    registry_path: str | Path,
    *,
    runs: Sequence[Run],
) -> dict[str, Any]:
    supplied = Path(registry_path)
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise G2TokenComparisonError(
            f"registry database does not exist: {supplied}"
        ) from error
    if not resolved.is_file():
        raise G2TokenComparisonError(
            f"registry input is not a regular file: {supplied}"
        )
    try:
        connection = sqlite3.connect(
            resolved.as_uri() + "?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            """
            SELECT job_id, run_id, status, failure_category, last_error
            FROM queue_jobs
            WHERE campaign_id = ?
            ORDER BY created_at, job_id
            """,
            (_CAMPAIGN_ID,),
        ).fetchall()
    except sqlite3.Error as error:
        raise G2TokenComparisonError(
            "registry database is unreadable or lacks the queue_jobs contract"
        ) from error
    finally:
        if "connection" in locals():
            connection.close()

    if len(rows) != 7:
        raise G2TokenComparisonError(
            f"registry campaign must contain exactly 7 jobs; found {len(rows)}"
        )
    expected_by_run = {run.run_id: run.job_id for run in runs}
    if len(expected_by_run) != 7:
        raise G2TokenComparisonError(
            "registry reconciliation requires seven distinct supplied run IDs"
        )
    observed_by_run: dict[str, sqlite3.Row] = {}
    for row in rows:
        run_id = str(row["run_id"] or "")
        job_id = str(row["job_id"] or "")
        if not run_id or run_id in observed_by_run:
            raise G2TokenComparisonError(
                "registry campaign has a missing or duplicate run_id"
            )
        observed_by_run[run_id] = row
        if (
            row["status"] != "completed"
            or row["failure_category"] is not None
            or row["last_error"] is not None
        ):
            raise G2TokenComparisonError(
                f"registry job {job_id} is not a clean completed job"
            )
    if set(observed_by_run) != set(expected_by_run):
        missing = sorted(set(expected_by_run) - set(observed_by_run))
        extra = sorted(set(observed_by_run) - set(expected_by_run))
        raise G2TokenComparisonError(
            f"registry run membership disagrees; missing={missing}, extra={extra}"
        )
    for run_id, expected_job_id in expected_by_run.items():
        if observed_by_run[run_id]["job_id"] != expected_job_id:
            raise G2TokenComparisonError(
                f"registry and bundle queue job IDs disagree for {run_id}"
            )
    return {
        "status": "passed",
        "read_only": True,
        "registry_path_exported": False,
        "campaign_id": _CAMPAIGN_ID,
        "expected_job_count": 7,
        "observed_job_count": 7,
        "expected_run_membership_matches": True,
        "all_jobs_completed": True,
        "jobs_with_failure_category": 0,
        "jobs_with_last_error": 0,
        "jobs": [
            {
                "job_id": str(row["job_id"]),
                "run_id": str(row["run_id"]),
                "status": str(row["status"]),
            }
            for row in rows
        ],
    }


def compare_g2_token_multiseed(
    run_directories: Sequence[str | Path],
    *,
    resource_pilot: str | Path,
    registry: str | Path,
) -> dict[str, Any]:
    """Return a strict machine-readable comparison for the locked campaign."""

    if len(run_directories) != 6:
        raise G2TokenComparisonError("exactly six finalized run directories are required")
    runs = tuple(_load_run(path, pilot=False) for path in run_directories)
    if len({run.root for run in runs}) != 6:
        raise G2TokenComparisonError("run directories must be distinct")
    grouped: dict[str, dict[int, Run]] = {_BASELINE: {}, _WIDER: {}}
    for run in runs:
        if run.seed in grouped[run.variant]:
            raise G2TokenComparisonError(f"{run.variant} duplicates seed {run.seed}")
        grouped[run.variant][run.seed] = run
    for variant in _CONCLUSION_VARIANTS:
        if tuple(sorted(grouped[variant])) != _EXPECTED_SEEDS:
            raise G2TokenComparisonError(
                f"{variant} requires seeds 0, 1, and 2 exactly once"
            )
    identity_sets = {
        "graph": {(run.graph_sha256, run.graph_edges) for run in runs},
        "evaluation masks": {
            (run.mask_bundle_sha256, run.mask_identity) for run in runs
        },
        "dataset and split": {run.data_identity for run in runs},
        "token vocabulary and matrix": {
            (
                run.token_audit_digest,
                run.token_matrix_sha256,
                run.modal_tokens_sha256,
            )
            for run in runs
        },
    }
    for label, identities in identity_sets.items():
        if len(identities) != 1:
            raise G2TokenComparisonError(f"the six runs do not share one {label}")
    for seed in _EXPECTED_SEEDS:
        if grouped[_BASELINE][seed].history.schedule != grouped[_WIDER][seed].history.schedule:
            raise G2TokenComparisonError(
                f"seed {seed} training mask or edge-dropout schedules are not paired"
            )
    pilot = _load_run(resource_pilot, pilot=True)
    if (
        pilot.graph_sha256 != runs[0].graph_sha256
        or pilot.data_identity != runs[0].data_identity
        or pilot.token_audit_digest != runs[0].token_audit_digest
        or pilot.mask_identity != (runs[0].mask_identity[0],)
    ):
        raise G2TokenComparisonError(
            "resource pilot graph, data, token vocabulary, or first mask does "
            "not align"
        )
    projected = pilot.training_duration_seconds / _PILOT_EPOCHS * _EXPECTED_EPOCHS
    pilot_criteria = {
        "all_losses_and_gradients_finite": True,
        "peak_vram_at_most_20_5_gib": pilot.history.peak_vram_gb <= 20.5,
        "projected_200_epoch_training_at_most_6_hours": projected <= 21_600.0,
    }
    pilot_passes = all(pilot_criteria.values())
    registry_reconciliation = _reconcile_registry(
        registry,
        runs=(*runs, pilot),
    )

    per_run: list[dict[str, Any]] = []
    per_mask: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    for variant in _CONCLUSION_VARIANTS:
        variant_runs = [grouped[variant][seed] for seed in _EXPECTED_SEEDS]
        for run in variant_runs:
            per_run.append(
                {
                    "run_id": run.run_id,
                    "variant_label": variant,
                    "seed": run.seed,
                    "fold": run.fold,
                    "attempt": run.attempt,
                    "job_id": run.job_id,
                    "parameter_count": run.parameter_count,
                    **_metric_dict(run.metrics),
                    "total_duration_seconds": run.total_duration_seconds,
                    "training_duration_seconds": run.training_duration_seconds,
                    "summed_epoch_duration_seconds": run.history.summed_duration_seconds,
                    "peak_vram_gb": run.history.peak_vram_gb,
                    "final_train_loss": run.history.final_loss,
                    "minimum_train_loss": run.history.minimum_loss,
                    "last_20_epoch_loss_slope": run.history.last_20_slope,
                }
            )
            for replicate, metrics in enumerate(run.mask_metrics):
                per_mask.append(
                    {
                        "run_id": run.run_id,
                        "variant_label": variant,
                        "seed": run.seed,
                        "mask_replicate": replicate,
                        **_metric_dict(metrics),
                    }
                )
        metric_names = (
            "cross_entropy",
            "exact_percent",
            "balanced_percent",
            "nonzero_percent",
        )
        aggregates[variant] = {
            "model_seeds": list(_EXPECTED_SEEDS),
            "parameter_count": variant_runs[0].parameter_count,
            "metrics": {
                name: _statistics([getattr(run.metrics, name) for run in variant_runs])
                for name in metric_names
            },
            "baselines": {
                field: _statistics(
                    [run.metrics.baselines[index] for run in variant_runs]
                )
                for index, field in enumerate(_BASELINE_FIELDS)
            },
            "token_recalls_percent": {
                str(token): _statistics(
                    [run.metrics.recalls[token] for run in variant_runs]
                )
                for token in range(4)
            },
            "resources": {
                "total_duration_seconds": _statistics(
                    [run.total_duration_seconds for run in variant_runs]
                ),
                "training_duration_seconds": _statistics(
                    [run.training_duration_seconds for run in variant_runs]
                ),
                "peak_vram_gb": _statistics(
                    [run.history.peak_vram_gb for run in variant_runs]
                ),
            },
            "convergence": {
                "final_train_loss": _statistics(
                    [run.history.final_loss for run in variant_runs]
                ),
                "minimum_train_loss": _statistics(
                    [run.history.minimum_loss for run in variant_runs]
                ),
                "last_20_epoch_loss_slope": _statistics(
                    [run.history.last_20_slope for run in variant_runs]
                ),
            },
        }

    paired: list[dict[str, Any]] = []
    for seed in _EXPECTED_SEEDS:
        baseline = grouped[_BASELINE][seed]
        wider = grouped[_WIDER][seed]
        paired.append(
            {
                "seed": seed,
                "baseline_exact_accuracy_percent": baseline.metrics.exact_percent,
                "wider_exact_accuracy_percent": wider.metrics.exact_percent,
                "wider_minus_baseline_exact_percentage_points": (
                    wider.metrics.exact_percent - baseline.metrics.exact_percent
                ),
                "baseline_balanced_accuracy_percent": baseline.metrics.balanced_percent,
                "wider_balanced_accuracy_percent": wider.metrics.balanced_percent,
                "wider_minus_baseline_balanced_percentage_points": (
                    wider.metrics.balanced_percent - baseline.metrics.balanced_percent
                ),
                "baseline_nonzero_accuracy_percent": baseline.metrics.nonzero_percent,
                "wider_nonzero_accuracy_percent": wider.metrics.nonzero_percent,
                "wider_minus_baseline_nonzero_percentage_points": (
                    wider.metrics.nonzero_percent - baseline.metrics.nonzero_percent
                ),
                "baseline_cross_entropy": baseline.metrics.cross_entropy,
                "wider_cross_entropy": wider.metrics.cross_entropy,
                "baseline_minus_wider_cross_entropy": (
                    baseline.metrics.cross_entropy - wider.metrics.cross_entropy
                ),
                "wider_not_worse_on_exact": (
                    wider.metrics.exact_percent >= baseline.metrics.exact_percent
                ),
                "wider_not_worse_on_balanced": (
                    wider.metrics.balanced_percent >= baseline.metrics.balanced_percent
                ),
            }
        )
    baseline_exact = aggregates[_BASELINE]["metrics"]["exact_percent"]["mean"]
    wider_exact = aggregates[_WIDER]["metrics"]["exact_percent"]["mean"]
    baseline_balanced = aggregates[_BASELINE]["metrics"]["balanced_percent"]["mean"]
    wider_balanced = aggregates[_WIDER]["metrics"]["balanced_percent"]["mean"]
    exact_delta = float(wider_exact) - float(baseline_exact)
    balanced_delta = float(wider_balanced) - float(baseline_balanced)
    h1_criteria = {
        "mean_exact_gain_at_least_0_5_percentage_points": exact_delta >= 0.5,
        "mean_balanced_gain_at_least_1_0_percentage_point": balanced_delta >= 1.0,
        "no_seed_decreases_on_exact_accuracy": all(
            bool(row["wider_not_worse_on_exact"]) for row in paired
        ),
        "no_seed_decreases_on_balanced_accuracy": all(
            bool(row["wider_not_worse_on_balanced"]) for row in paired
        ),
    }
    h1_passes = all(h1_criteria.values())
    wider_seed_exact = {
        str(seed): grouped[_WIDER][seed].metrics.exact_percent
        for seed in _EXPECTED_SEEDS
    }
    jacobian_eligible = all(value > 95.0 for value in wider_seed_exact.values())
    if not pilot_passes:
        status = "protocol_failed"
    elif jacobian_eligible:
        status = "conditional_analysis_required"
    else:
        status = "complete"
    first = grouped[_BASELINE][0]
    data_identity = dict(zip(_DATA_FIELDS, first.data_identity, strict=True))
    return {
        "schema_version": 1,
        "artifact_kind": "g2_token_multiseed_comparison",
        "campaign_id": _CAMPAIGN_ID,
        "status": status,
        "scope": {
            "estimand": "held-in whole-node exact masked raw-count-token reconstruction",
            "tokens": {"0": "count 0", "1": "count 1", "2": "count 2", "3": "count >=3"},
            "percentages_are_classification_accuracy": True,
            "generalization_estimate": False,
            "entrywise_holdout": False,
            "evaluation_entries_may_overlap_training_masks": True,
            "evaluation_mask_seed_derivation": (
                "separate_from_epoch_mask_seed_derivation"
            ),
            "recorded_evaluation_mask_seed_namespace": (
                "held_in_evaluation_disjoint_from_epoch_masks"
            ),
            "namespace_label_establishes_entrywise_disjointness": False,
            "per_gene_modal_baseline_scope": "all_fit_transductive_reference",
            "model_seeds_are_independent_biological_replicates": False,
            "technical_masks_are_independent_biological_replicates": False,
        },
        "verified_identity": {
            "all_six_runs_match": True,
            "locked_config_sections_verified": [
                "dataset",
                "graph",
                "features",
                "masking",
                "trainer",
                "evaluation",
                "model",
                "experiment",
                "classification",
            ],
            "variants": list(_CONCLUSION_VARIANTS),
            "seeds_per_variant": list(_EXPECTED_SEEDS),
            "fixed_epochs_per_run": _EXPECTED_EPOCHS,
            "paired_training_schedules": True,
            "graph_sha256": first.graph_sha256,
            "graph_directed_edges": first.graph_edges,
            "evaluation_mask_bundle_sha256": first.mask_bundle_sha256,
            "token_matrix_sha256": first.token_matrix_sha256,
            "per_gene_modal_tokens_sha256": first.modal_tokens_sha256,
            "token_counts": list(_EXPECTED_TOKEN_COUNTS),
            **data_identity,
        },
        "aggregation": {
            "model_seed_count_per_variant": 3,
            "whole_node_mask_count_per_run": 3,
            "percentage_cross_entropy_recall_and_baseline_per_run": (
                "unweighted arithmetic mean across fixed masks"
            ),
            "token_support_per_run": "sum across fixed masks",
            "seed_standard_deviation_ddof": 1,
        },
        "reproducibility_notice": {
            "comparison_contract": "launched_resolved_configuration_snapshots",
            "source_configs_match_launched_resolved_snapshots": True,
            "exact_reproduction_instruction": (
                "Use the checked source experiment definitions and verify the "
                "resulting config against each bundle's config.resolved.yaml."
            ),
            "legacy_mask_namespace_label_implies_entrywise_holdout": False,
        },
        "baseline_evidence": {
            "uniform_empirical_and_all_zero_recomputed_from_supports": True,
            "per_gene_modal_vector_checksum_verified": True,
            "per_gene_modal_metrics_reconciled_across_aligned_runs": True,
            "per_gene_modal_metrics_independently_recomputed_by_comparator": False,
            "per_gene_modal_interpretation": "all_fit_transductive_reference",
        },
        "registry_reconciliation": registry_reconciliation,
        "resource_pilot": {
            "status": "passed" if pilot_passes else "failed",
            "run_id": pilot.run_id,
            "seed": pilot.seed,
            "parameter_count": pilot.parameter_count,
            "peak_vram_gb": pilot.history.peak_vram_gb,
            "training_duration_seconds": pilot.training_duration_seconds,
            "projected_200_epoch_training_seconds": projected,
            "criteria": pilot_criteria,
            "thresholds": {
                "maximum_peak_vram_gib": 20.5,
                "maximum_projected_200_epoch_training_seconds": 21_600.0,
            },
        },
        "runs": per_run,
        "whole_node_masks": per_mask,
        "variant_aggregates": aggregates,
        "paired_seed_deltas": {
            "direction_convention": (
                "positive accuracy deltas and positive baseline-minus-wider "
                "cross-entropy favor the wider model"
            ),
            "per_seed": paired,
            "aggregates": {
                "wider_minus_baseline_exact_percentage_points": _statistics(
                    [
                        float(row["wider_minus_baseline_exact_percentage_points"])
                        for row in paired
                    ]
                ),
                "wider_minus_baseline_balanced_percentage_points": _statistics(
                    [
                        float(row["wider_minus_baseline_balanced_percentage_points"])
                        for row in paired
                    ]
                ),
                "wider_minus_baseline_nonzero_percentage_points": _statistics(
                    [
                        float(row["wider_minus_baseline_nonzero_percentage_points"])
                        for row in paired
                    ]
                ),
                "baseline_minus_wider_cross_entropy": _statistics(
                    [float(row["baseline_minus_wider_cross_entropy"]) for row in paired]
                ),
            },
        },
        "h1_width_gate": {
            "thresholds": {
                "minimum_mean_exact_gain_percentage_points": 0.5,
                "minimum_mean_balanced_gain_percentage_points": 1.0,
                "maximum_decreasing_seeds_on_each_primary_metric": 0,
            },
            "observed": {
                "baseline_mean_exact_accuracy_percent": baseline_exact,
                "wider_mean_exact_accuracy_percent": wider_exact,
                "mean_exact_gain_percentage_points": exact_delta,
                "baseline_mean_balanced_accuracy_percent": baseline_balanced,
                "wider_mean_balanced_accuracy_percent": wider_balanced,
                "mean_balanced_gain_percentage_points": balanced_delta,
            },
            "criteria": h1_criteria,
            "passes": h1_passes,
        },
        "relaxed_jacobian_gate": {
            "metric": "per-seed mean whole-node exact masked-token accuracy percent",
            "operator": "strictly_greater_than",
            "threshold_percent": 95.0,
            "wider_seed_values_percent": wider_seed_exact,
            "eligible": jacobian_eligible,
            "status": "required" if jacobian_eligible else "skipped",
            "jacobians_computed": False,
            "reason": (
                "Conditional analysis required: every wider-model seed is strictly "
                "above 95% exact accuracy; run the predefined relaxed categorical "
                "sensitivity comparison before claiming campaign completion."
                if jacobian_eligible
                else "At least one wider-model seed is not strictly above 95% exact "
                "accuracy, so the relaxed categorical Jacobian is skipped."
            ),
        },
    }


def _markdown(result: Mapping[str, Any]) -> str:
    aggregates = _mapping(result["variant_aggregates"], "variant aggregates")
    h1 = _mapping(result["h1_width_gate"], "H1")
    jacobian = _mapping(result["relaxed_jacobian_gate"], "Jacobian")
    pilot = _mapping(result["resource_pilot"], "pilot")
    lines = [
        "# Full-core tokenized G2 width comparison",
        "",
        (
            "Six checksum-verified fixed-200-epoch runs were compared: two widths "
            "by seeds 0, 1, and 2. These percentages are held-in categorical "
            "reconstruction metrics, not validation/test accuracy."
        ),
        "",
        f"Comparison status: `{result['status']}`.",
        "",
        "## Whole-node results",
        "",
        "| Variant | Exact % | Balanced % | Nonzero % | Cross-entropy | Parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in _CONCLUSION_VARIANTS:
        record = aggregates[variant]
        metrics = record["metrics"]
        lines.append(
            f"| `{variant}` | {metrics['exact_percent']['mean']:.4f} ± "
            f"{metrics['exact_percent']['std']:.4f} | "
            f"{metrics['balanced_percent']['mean']:.4f} ± "
            f"{metrics['balanced_percent']['std']:.4f} | "
            f"{metrics['nonzero_percent']['mean']:.4f} ± "
            f"{metrics['nonzero_percent']['std']:.4f} | "
            f"{metrics['cross_entropy']['mean']:.6f} | "
            f"{int(record['parameter_count']):,} |"
        )
    lines.extend(
        [
            "",
            "### Per-seed whole-node means",
            "",
            (
                "| Variant | Seed | Exact % | Balanced % | Nonzero % | "
                "Cross-entropy | Train s | Peak GiB |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["runs"]:
        lines.append(
            f"| `{row['variant_label']}` | {row['seed']} | "
            f"{row['whole_node_exact_accuracy_percent']:.4f} | "
            f"{row['whole_node_balanced_accuracy_percent']:.4f} | "
            f"{row['whole_node_nonzero_accuracy_percent']:.4f} | "
            f"{row['whole_node_cross_entropy']:.6f} | "
            f"{row['training_duration_seconds']:.2f} | "
            f"{row['peak_vram_gb']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Per-mask whole-node results",
            "",
            (
                "| Variant | Seed | Mask | Exact % | Balanced % | Nonzero % | "
                "Cross-entropy |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["whole_node_masks"]:
        lines.append(
            f"| `{row['variant_label']}` | {row['seed']} | "
            f"{row['mask_replicate']} | "
            f"{row['whole_node_exact_accuracy_percent']:.4f} | "
            f"{row['whole_node_balanced_accuracy_percent']:.4f} | "
            f"{row['whole_node_nonzero_accuracy_percent']:.4f} | "
            f"{row['whole_node_cross_entropy']:.6f} |"
        )
    baseline = aggregates[_BASELINE]["baselines"]
    baseline_labels = {
        "baseline_uniform_accuracy_percent": "Uniform-class exact",
        "baseline_empirical_frequency_accuracy_percent": (
            "Empirical-frequency random exact"
        ),
        "baseline_always_zero_accuracy_percent": "All-zero exact",
        "baseline_always_zero_balanced_accuracy_percent": "All-zero balanced",
        "baseline_per_gene_modal_accuracy_percent": "Per-gene modal exact",
        "baseline_per_gene_modal_balanced_accuracy_percent": (
            "Per-gene modal balanced"
        ),
        "baseline_per_gene_modal_nonzero_accuracy_percent": (
            "Per-gene modal nonzero"
        ),
    }
    lines.extend(
        [
            "",
            "## Baseline context",
            "",
            (
                f"The all-zero exact baseline is "
                f"{baseline['baseline_always_zero_accuracy_percent']['mean']:.4f}%; "
                f"the per-gene modal exact baseline is "
                f"{baseline['baseline_per_gene_modal_accuracy_percent']['mean']:.4f}%; "
                f"its balanced accuracy is "
                f"{baseline['baseline_per_gene_modal_balanced_accuracy_percent']['mean']:.4f}%. "
                "The modal baseline is an all-fit transductive reference. Exact "
                "accuracy must therefore be read with balanced and nonzero accuracy."
            ),
            "",
            "| Reference metric | Mean % | Seed SD |",
            "|---|---:|---:|",
        ]
    )
    for field in _BASELINE_FIELDS:
        lines.append(
            f"| {baseline_labels[field]} | {baseline[field]['mean']:.4f} | "
            f"{baseline[field]['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Token recall and support",
            "",
            (
                "Recall is averaged across the three fixed masks and then across "
                "seeds. Support is the mean per-run sum across those masks."
            ),
            "",
            "| Variant | Token | Mean recall % | Seed SD | Mean support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in _CONCLUSION_VARIANTS:
        variant_rows = [
            row for row in result["runs"] if row["variant_label"] == variant
        ]
        for token in range(4):
            recall = aggregates[variant]["token_recalls_percent"][str(token)]
            support = statistics.fmean(
                float(row[f"token_{token}_support"]) for row in variant_rows
            )
            lines.append(
                f"| `{variant}` | {token} (`{_TOKEN_LABELS[token]}`) | "
                f"{recall['mean']:.4f} | {recall['std']:.4f} | "
                f"{support:.0f} |"
            )
    lines.extend(
        [
            "",
            "## Resources and convergence",
            "",
            (
                "| Variant | Total runtime s | Training s | Peak GiB | "
                "Final loss | Minimum loss | Last-20 slope |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in _CONCLUSION_VARIANTS:
        record = aggregates[variant]
        resources = record["resources"]
        convergence = record["convergence"]
        lines.append(
            f"| `{variant}` | "
            f"{resources['total_duration_seconds']['mean']:.2f} | "
            f"{resources['training_duration_seconds']['mean']:.2f} | "
            f"{resources['peak_vram_gb']['mean']:.3f} | "
            f"{convergence['final_train_loss']['mean']:.6f} | "
            f"{convergence['minimum_train_loss']['mean']:.6f} | "
            f"{convergence['last_20_epoch_loss_slope']['mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Locked gates",
            "",
            (
                f"- H1-width: **{'PASS' if h1['passes'] else 'FAIL'}**; mean exact "
                f"gain {h1['observed']['mean_exact_gain_percentage_points']:.4f} pp, "
                f"mean balanced gain "
                f"{h1['observed']['mean_balanced_gain_percentage_points']:.4f} pp."
            ),
            (
                f"- Resource pilot: **{str(pilot['status']).upper()}**; "
                f"{pilot['peak_vram_gb']:.3f} GiB peak and "
                f"{pilot['projected_200_epoch_training_seconds'] / 3600:.3f} h projected."
            ),
            (
                f"- Relaxed Jacobian: **{str(jacobian['status']).upper()}**. "
                f"{jacobian['reason']}"
            ),
            "",
            "## Limits",
            "",
            (
                "All cells, graph construction, preprocessing summaries, and baselines "
                "come from one held-in core. Evaluation masks use a separate seed "
                "derivation but are not an entry-wise holdout and may overlap epoch "
                "training masks. Seeds and masks are technical repeats, not independent "
                "biological replication or evidence of patient generalization."
            ),
            (
                "The exact comparison contract is the resolved configuration saved "
                "inside each launched bundle; the checked source experiment definitions "
                "compose to those snapshots. The legacy mask-namespace label denotes "
                "separate seed derivation, not entry-wise holdout."
            ),
            (
                "Registry reconciliation: **PASSED**. The campaign contains exactly "
                "the supplied pilot plus six science runs; all seven jobs are completed "
                "with no failure category or last error."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise G2TokenComparisonError("CSV output requires at least one row")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise G2TokenComparisonError("CSV rows do not share one ordered schema")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_file(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _renameat2(source: Path, destination: Path, flag: int) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameat2", None)
    if function is None:
        raise G2TokenComparisonError("renameat2 is required for atomic publication")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    if function(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        flag,
    ) == 0:
        return
    number = ctypes.get_errno()
    if flag == _RENAME_NOREPLACE and number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise G2TokenComparisonError(f"output already exists: {destination}")
    raise OSError(number, os.strerror(number), destination.as_posix())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_owned(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise G2TokenComparisonError("existing output is not an owned directory")
    entries = {entry.name: entry for entry in path.iterdir()}
    if set(entries) != set(_OUTPUT_FILES) or any(
        entry.is_symlink() or not entry.is_file() for entry in entries.values()
    ):
        raise G2TokenComparisonError("owned output must contain exactly its five files")
    if dict(_load_json(path / _OWNER_FILE)) != _OWNER:
        raise G2TokenComparisonError("output ownership marker disagrees")


def _remove_owned(path: Path, *, complete: bool) -> None:
    entries = {entry.name: entry for entry in path.iterdir()}
    valid = set(entries) == set(_OUTPUT_FILES) if complete else set(entries).issubset(_OUTPUT_FILES)
    if not valid or any(entry.is_symlink() or not entry.is_file() for entry in entries.values()):
        raise G2TokenComparisonError(f"refusing unsafe cleanup: {path}")
    for entry in entries.values():
        entry.unlink()
    path.rmdir()


def write_comparison(
    result: Mapping[str, Any],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish JSON, CSV, and Markdown comparison artifacts."""

    requested = Path(output_directory)
    if requested.is_symlink():
        raise G2TokenComparisonError("output directory cannot be a symlink")
    destination = requested.resolve(strict=False)
    project = _ROOT.resolve()
    home = Path.home().resolve()
    filesystem_root = Path(destination.anchor)
    if destination == filesystem_root or any(
        destination == protected or protected.is_relative_to(destination)
        for protected in (project, home)
    ):
        raise G2TokenComparisonError(f"refusing protected output: {destination}")
    if destination.is_relative_to(project) and not destination.is_relative_to(
        project / "reports"
    ):
        raise G2TokenComparisonError(
            "repository-local comparison output must be under reports/"
        )
    exists = destination.exists()
    if exists and not overwrite:
        raise G2TokenComparisonError(f"output already exists: {destination}")
    if exists:
        _validate_owned(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.writing-", dir=destination.parent)
    )
    committed = False
    try:
        _write_file(
            staging / "comparison.json",
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
        )
        _write_file(staging / "per_run.csv", _csv_text(result["runs"]))
        paired = _mapping(result["paired_seed_deltas"], "paired deltas")["per_seed"]
        _write_file(staging / "paired.csv", _csv_text(paired))
        _write_file(staging / "report.md", _markdown(result))
        _write_file(
            staging / _OWNER_FILE,
            json.dumps(_OWNER, indent=2, sort_keys=True) + "\n",
        )
        _sync_directory(staging)
        if exists:
            _validate_owned(destination)
            _renameat2(staging, destination, _RENAME_EXCHANGE)
        else:
            _renameat2(staging, destination, _RENAME_NOREPLACE)
        committed = True
        try:
            _sync_directory(destination.parent)
        except OSError as error:
            warnings.warn(
                f"comparison was committed but parent fsync failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
    except BaseException as error:
        if not committed and staging.exists():
            try:
                _remove_owned(staging, complete=False)
            except BaseException as cleanup_error:
                warnings.warn(f"could not clean staging directory: {cleanup_error}")
        if isinstance(error, G2TokenComparisonError):
            raise
        raise G2TokenComparisonError("comparison publication failed") from error
    if exists:
        try:
            _remove_owned(staging, complete=True)
        except BaseException as cleanup_error:
            warnings.warn(f"old owned output remains at {staging}: {cleanup_error}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exactly six finalized token-G2 runs and their required "
            "two-epoch resource pilot."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        required=True,
        type=Path,
        help="Conclusion-eligible run; repeat exactly six times.",
    )
    parser.add_argument("--resource-pilot", required=True, type=Path)
    parser.add_argument(
        "--registry",
        required=True,
        type=Path,
        help="Authoritative SQLite registry; opened read-only.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if len(arguments.runs) != 6:
        parser.error("--run must be supplied exactly six times")
    result = compare_g2_token_multiseed(
        arguments.runs,
        resource_pilot=arguments.resource_pilot,
        registry=arguments.registry,
    )
    output = write_comparison(result, arguments.output, overwrite=arguments.overwrite)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "status": result["status"],
                "h1_width_passes": result["h1_width_gate"]["passes"],
                "relaxed_jacobian_status": result["relaxed_jacobian_gate"]["status"],
                "resource_pilot": result["resource_pilot"]["status"],
                "registry_reconciliation": result["registry_reconciliation"][
                    "status"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
