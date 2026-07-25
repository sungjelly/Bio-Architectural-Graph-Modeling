"""Authoritative local SQLite registry for BAGM experiments.

The registry is intentionally dependency-free and stores only small metadata.
Large artifacts remain in the canonical artifact tree and are referenced by
path and checksum.  Every state transition uses an explicit transaction and
all values are bound parameters.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean, median, pstdev
from typing import Any, Iterator, Mapping, Sequence

from .identifiers import canonical_json, scientific_id as calculate_scientific_id
from .identifiers import scientific_payload
from .paths import current_paths


SCHEMA_VERSION = 3
DEFAULT_BUSY_TIMEOUT_MS = 30_000
_ANY_GPU = object()
RUN_STATUSES = frozenset(
    {
        "pending",
        "running",
        "finalizing",
        "completed",
        "failed",
        "pruned",
        "cancelled",
    }
)
QUEUE_STATUSES = frozenset(
    {
        "queued",
        "claimed",
        "running",
        "completed",
        "failed",
        "pruned",
        "cancelled",
        "stale",
    }
)

_RUN_TRANSITIONS = {
    "pending": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"finalizing", "failed", "pruned", "cancelled"}),
    "finalizing": frozenset({"completed", "failed"}),
}
_QUEUE_TRANSITIONS = {
    "queued": frozenset({"claimed", "cancelled"}),
    "claimed": frozenset({"running", "failed", "cancelled", "queued", "stale"}),
    "running": frozenset(
        {"completed", "failed", "pruned", "cancelled", "stale"}
    ),
    "stale": frozenset({"failed", "cancelled"}),
}


class RegistryError(RuntimeError):
    """Base class for registry errors."""


class RegistryConflictError(RegistryError):
    """Raised when an identifier already has different canonical content."""


class InvalidTransitionError(RegistryError):
    """Raised when a state transition is not permitted."""


def utc_now() -> str:
    """Return a lexically sortable UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _json(value: Any) -> str:
    return canonical_json(value)


def _decode_json(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _bool_int(value: bool | None) -> int | None:
    return None if value is None else int(bool(value))


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for name in (
        "config_json",
        "canonical_config_json",
        "command_json",
        "metadata_json",
        "metrics_json",
        "stratification_json",
        "details_json",
        "rules_json",
    ):
        if name in result:
            result[name.removesuffix("_json")] = _decode_json(result.pop(name))
    for name in (
        "attempt_known",
        "dirty_status",
        "fold_known",
        "preferred",
        "seed_known",
        "use_edge_features",
    ):
        if result.get(name) is not None:
            result[name] = bool(result[name])
    return result


_MIGRATION_1 = (
    """
    CREATE TABLE campaigns (
        campaign_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        scientific_question TEXT,
        status TEXT NOT NULL DEFAULT 'planned',
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE variants (
        scientific_id TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        canonical_config_json TEXT NOT NULL,
        model_family TEXT,
        masking_type TEXT,
        dataset_id TEXT,
        dataset_version TEXT,
        split_id TEXT,
        use_edge_features INTEGER,
        embedding_dim INTEGER,
        neighbor_k INTEGER,
        learning_rate REAL,
        batch_size INTEGER,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        scientific_id TEXT NOT NULL REFERENCES variants(scientific_id),
        repro_id TEXT NOT NULL,
        status TEXT NOT NULL,
        model_family TEXT,
        masking_type TEXT,
        dataset_id TEXT,
        dataset_version TEXT,
        split_id TEXT,
        use_edge_features INTEGER,
        embedding_dim INTEGER,
        neighbor_k INTEGER,
        learning_rate REAL,
        batch_size INTEGER,
        seed INTEGER NOT NULL,
        fold INTEGER NOT NULL,
        attempt INTEGER NOT NULL,
        git_commit TEXT,
        dirty_status INTEGER,
        preprocessing_version TEXT,
        dataset_fingerprint TEXT,
        split_fingerprint TEXT,
        start_time TEXT,
        end_time TEXT,
        duration_seconds REAL,
        host TEXT,
        gpu_model TEXT,
        peak_vram_gb REAL,
        parameter_count INTEGER,
        primary_metric_name TEXT,
        primary_metric_value REAL,
        artifact_path TEXT,
        failure_category TEXT,
        retry_of TEXT REFERENCES runs(run_id),
        promoted_at TEXT,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE evaluations (
        evaluation_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id),
        checkpoint_name TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        split_id TEXT NOT NULL,
        status TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        artifact_path TEXT,
        created_at TEXT NOT NULL,
        finished_at TEXT
    )
    """,
    """
    CREATE TABLE metrics (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(run_id),
        evaluation_id TEXT REFERENCES evaluations(evaluation_id),
        name TEXT NOT NULL,
        value REAL NOT NULL,
        step INTEGER,
        split TEXT,
        recorded_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE artifacts (
        artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(run_id),
        evaluation_id TEXT REFERENCES evaluations(evaluation_id),
        kind TEXT NOT NULL,
        path TEXT NOT NULL,
        sha256 TEXT,
        size_bytes INTEGER,
        status TEXT NOT NULL DEFAULT 'present',
        created_at TEXT NOT NULL,
        UNIQUE(run_id, path)
    )
    """,
    """
    CREATE TABLE datasets (
        dataset_id TEXT NOT NULL,
        dataset_version TEXT NOT NULL,
        display_name TEXT NOT NULL,
        protected_source_path TEXT,
        raw_fingerprint TEXT,
        preprocessing_version TEXT,
        processed_fingerprint TEXT,
        aggregate_sample_count INTEGER,
        graph_count INTEGER,
        node_feature_schema TEXT,
        edge_feature_schema TEXT,
        creation_date TEXT,
        status TEXT NOT NULL,
        verification_status TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(dataset_id, dataset_version)
    )
    """,
    """
    CREATE TABLE splits (
        split_id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        dataset_version TEXT NOT NULL,
        method TEXT NOT NULL,
        seed INTEGER,
        fold_count INTEGER,
        unit TEXT NOT NULL,
        stratification_json TEXT NOT NULL,
        fingerprint TEXT,
        protected_path TEXT,
        verification_status TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(dataset_id, dataset_version)
            REFERENCES datasets(dataset_id, dataset_version)
    )
    """,
    """
    CREATE TABLE queue_jobs (
        job_id TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        experiment_config_reference TEXT,
        canonical_config_json TEXT NOT NULL,
        command_json TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        attempt_count INTEGER NOT NULL,
        maximum_attempts INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        claimed_at TEXT,
        started_at TEXT,
        heartbeat_at TEXT,
        finished_at TEXT,
        run_id TEXT REFERENCES runs(run_id),
        worker_id TEXT,
        requested_gpu TEXT,
        failure_category TEXT,
        retry_of TEXT REFERENCES queue_jobs(job_id),
        last_error TEXT
    )
    """,
    """
    CREATE TABLE failures (
        failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT REFERENCES runs(run_id),
        job_id TEXT REFERENCES queue_jobs(job_id),
        category TEXT NOT NULL,
        message TEXT NOT NULL,
        details_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX runs_campaign_idx ON runs(campaign_id)",
    "CREATE INDEX runs_variant_idx ON runs(scientific_id, repro_id)",
    "CREATE INDEX metrics_run_name_idx ON metrics(run_id, name)",
    "CREATE INDEX artifacts_run_idx ON artifacts(run_id)",
    """
    CREATE INDEX queue_claim_idx
    ON queue_jobs(status, priority DESC, created_at, job_id)
    """,
)

_MIGRATION_2 = (
    """
    CREATE TABLE campaign_variants (
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        scientific_id TEXT NOT NULL REFERENCES variants(scientific_id),
        created_at TEXT NOT NULL,
        PRIMARY KEY(campaign_id, scientific_id)
    )
    """,
    """
    INSERT INTO campaign_variants(campaign_id, scientific_id, created_at)
    SELECT campaign_id, scientific_id, created_at FROM variants
    """,
    """
    CREATE INDEX campaign_variants_variant_idx
    ON campaign_variants(scientific_id, campaign_id)
    """,
)

_MIGRATION_3 = (
    """
    CREATE TABLE run_aliases (
        alias_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        alias_type TEXT NOT NULL
            CHECK(alias_type IN ('canonical', 'semantic')),
        preferred INTEGER NOT NULL DEFAULT 0
            CHECK(preferred IN (0, 1)),
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(run_id, alias_type)
    )
    """,
    """
    CREATE UNIQUE INDEX run_aliases_preferred_idx
    ON run_aliases(run_id) WHERE preferred = 1
    """,
    """
    CREATE INDEX run_aliases_run_idx
    ON run_aliases(run_id, alias_type)
    """,
    """
    CREATE TABLE run_categories (
        run_id TEXT PRIMARY KEY REFERENCES runs(run_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        lifecycle_stage TEXT NOT NULL CHECK(length(trim(lifecycle_stage)) > 0),
        study_axis TEXT NOT NULL CHECK(length(trim(study_axis)) > 0),
        source_batch TEXT NOT NULL CHECK(length(trim(source_batch)) > 0),
        variant_label TEXT,
        model_key TEXT,
        dataset_key TEXT,
        masking_key TEXT,
        graph_key TEXT,
        feature_key TEXT,
        embedding_key TEXT,
        seed_known INTEGER NOT NULL CHECK(seed_known IN (0, 1)),
        fold_known INTEGER NOT NULL CHECK(fold_known IN (0, 1)),
        attempt_known INTEGER NOT NULL CHECK(attempt_known IN (0, 1)),
        retention_class TEXT NOT NULL
            CHECK(length(trim(retention_class)) > 0),
        category_key TEXT NOT NULL CHECK(length(trim(category_key)) > 0),
        rules_json TEXT NOT NULL,
        classification_confidence TEXT NOT NULL
            CHECK(classification_confidence IN ('high', 'medium', 'low', 'unknown')),
        timestamp_basis TEXT NOT NULL CHECK(length(trim(timestamp_basis)) > 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX run_categories_browse_idx
    ON run_categories(
        lifecycle_stage, study_axis, model_key, dataset_key, category_key
    )
    """,
    """
    CREATE INDEX run_categories_retention_idx
    ON run_categories(retention_class, source_batch)
    """,
    """
    CREATE UNIQUE INDEX artifacts_artifact_run_unique_idx
    ON artifacts(artifact_id, run_id)
    """,
    """
    CREATE TABLE checkpoint_catalog (
        artifact_id INTEGER PRIMARY KEY REFERENCES artifacts(artifact_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        run_id TEXT NOT NULL REFERENCES runs(run_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        role TEXT NOT NULL CHECK(length(trim(role)) > 0),
        best_epoch INTEGER CHECK(best_epoch IS NULL OR best_epoch >= 0),
        monitored_metric TEXT,
        monitored_mode TEXT
            CHECK(
                monitored_mode IS NULL
                OR monitored_mode IN ('min', 'max', 'unknown')
            ),
        monitored_value REAL,
        retention_class TEXT NOT NULL
            CHECK(length(trim(retention_class)) > 0),
        duplicate_group TEXT,
        duplicate_count INTEGER NOT NULL DEFAULT 1 CHECK(duplicate_count >= 1),
        verification_status TEXT NOT NULL
            CHECK(length(trim(verification_status)) > 0),
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(artifact_id, run_id)
            REFERENCES artifacts(artifact_id, run_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX checkpoint_catalog_run_idx
    ON checkpoint_catalog(run_id, role)
    """,
    """
    CREATE INDEX checkpoint_catalog_browse_idx
    ON checkpoint_catalog(retention_class, verification_status, duplicate_group)
    """,
)


class Registry:
    """Versioned SQLite registry with WAL concurrency and transactional states."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        initialize: bool = True,
    ) -> None:
        paths = current_paths()
        if path is not None and str(path) == ":memory:":
            raise ValueError(
                "Registry(':memory:') is unsupported because the registry uses "
                "multiple transactional connections; use an explicit temporary "
                "filesystem path instead."
            )
        self.path = Path(
            path or (paths.state_root / "tracking" / "bagm.sqlite3")
        ).resolve(strict=False)
        self.busy_timeout_ms = int(busy_timeout_ms)
        if self.busy_timeout_ms < 1:
            raise ValueError("busy_timeout_ms must be positive.")
        if initialize:
            self.initialize()

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms:d}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def transaction(
        self, *, immediate: bool = False
    ) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
            ).fetchone()
            current = int(row["version"])
            if current > SCHEMA_VERSION:
                raise RegistryError(
                    f"Registry schema {current} is newer than supported "
                    f"version {SCHEMA_VERSION}."
                )
            if current < 1:
                for statement in _MIGRATION_1:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, utc_now()),
                )
            if current < 2:
                for statement in _MIGRATION_2:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (2, utc_now()),
                )
            if current < 3:
                for statement in _MIGRATION_3:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (3, utc_now()),
                )

    def schema_version(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
            ).fetchone()
        return int(row["version"])

    def integrity_check(self) -> list[str]:
        with self.connect() as connection:
            return [
                str(row[0])
                for row in connection.execute("PRAGMA integrity_check").fetchall()
            ]

    def table_names(self) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        return {str(row["name"]) for row in rows}

    def create_campaign(
        self,
        campaign_id: str,
        *,
        name: str,
        scientific_question: str | None = None,
        config: Mapping[str, Any] | None = None,
        status: str = "planned",
    ) -> dict[str, Any]:
        now = utc_now()
        content = _json(config or {})
        try:
            with self.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO campaigns(
                        campaign_id, name, scientific_question, status,
                        config_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        campaign_id,
                        name,
                        scientific_question,
                        status,
                        content,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as error:
            existing = self.get_campaign(campaign_id)
            if existing and (
                existing["name"] != name
                or existing["scientific_question"] != scientific_question
                or existing["config"] != (config or {})
            ):
                raise RegistryConflictError(
                    f"Campaign {campaign_id!r} already has different content."
                ) from error
            if existing is None:
                raise
        result = self.get_campaign(campaign_id)
        assert result is not None
        return result

    def get_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
        return _row_dict(row)

    def register_variant(
        self,
        scientific_id: str,
        *,
        campaign_id: str,
        configuration: Mapping[str, Any],
        **fields: Any,
    ) -> dict[str, Any]:
        # A variant excludes execution dimensions such as seed, fold, attempt,
        # campaign, timestamps, host, and output paths by definition.
        config_text = _json(scientific_payload(configuration))
        values = _experiment_fields(configuration)
        values.update({key: value for key, value in fields.items() if value is not None})
        with self.transaction(immediate=True) as connection:
            existing = connection.execute(
                """
                SELECT canonical_config_json FROM variants
                WHERE scientific_id = ?
                """,
                (scientific_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO variants(
                        scientific_id, campaign_id, canonical_config_json,
                        model_family, masking_type, dataset_id, dataset_version,
                        split_id, use_edge_features, embedding_dim, neighbor_k,
                        learning_rate, batch_size, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scientific_id,
                        campaign_id,
                        config_text,
                        values.get("model_family"),
                        values.get("masking_type"),
                        values.get("dataset_id"),
                        values.get("dataset_version"),
                        values.get("split_id"),
                        _bool_int(values.get("use_edge_features")),
                        values.get("embedding_dim"),
                        values.get("neighbor_k"),
                        values.get("learning_rate"),
                        values.get("batch_size"),
                        utc_now(),
                    ),
                )
            elif json.loads(str(existing["canonical_config_json"])) != json.loads(
                config_text
            ):
                raise RegistryConflictError(
                    f"Variant {scientific_id!r} already has different content."
                )
            # Scientific variants are global identities. Campaign membership is
            # a many-to-many association; the legacy variants.campaign_id column
            # records only the first campaign that introduced the variant.
            connection.execute(
                """
                INSERT OR IGNORE INTO campaign_variants(
                    campaign_id, scientific_id, created_at
                ) VALUES (?, ?, ?)
                """,
                (campaign_id, scientific_id, utc_now()),
            )
        result = self.get_variant(scientific_id)
        assert result is not None
        return result

    def get_variant(self, scientific_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM variants WHERE scientific_id = ?", (scientific_id,)
            ).fetchone()
        return _row_dict(row)

    def create_run(
        self,
        run_id: str,
        *,
        campaign_id: str,
        scientific_id: str,
        repro_id: str,
        seed: int,
        fold: int,
        attempt: int,
        configuration: Mapping[str, Any],
        status: str = "pending",
        artifact_path: str | Path | None = None,
        retry_of: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        if status not in RUN_STATUSES:
            raise ValueError(f"Unknown run status: {status}")
        values = _experiment_fields(configuration)
        values.update({key: value for key, value in fields.items() if value is not None})
        now = utc_now()
        with self.transaction(immediate=True) as connection:
            alias_collision = connection.execute(
                "SELECT run_id FROM run_aliases WHERE alias_id = ?",
                (run_id,),
            ).fetchone()
            if alias_collision is not None:
                raise RegistryConflictError(
                    f"Run ID {run_id!r} is already an alias for "
                    f"{alias_collision['run_id']!r}."
                )
            membership = connection.execute(
                """
                SELECT 1 FROM campaign_variants
                WHERE campaign_id = ? AND scientific_id = ?
                """,
                (campaign_id, scientific_id),
            ).fetchone()
            if membership is None:
                raise RegistryError(
                    f"Variant {scientific_id!r} is not registered with "
                    f"campaign {campaign_id!r}."
                )
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, campaign_id, scientific_id, repro_id, status,
                    model_family, masking_type, dataset_id, dataset_version,
                    split_id, use_edge_features, embedding_dim, neighbor_k,
                    learning_rate, batch_size, seed, fold, attempt, git_commit,
                    dirty_status, preprocessing_version, dataset_fingerprint,
                    split_fingerprint, start_time, end_time, duration_seconds,
                    host, gpu_model, peak_vram_gb, parameter_count,
                    primary_metric_name, primary_metric_value, artifact_path,
                    failure_category, retry_of, config_json, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    campaign_id,
                    scientific_id,
                    repro_id,
                    status,
                    values.get("model_family"),
                    values.get("masking_type"),
                    values.get("dataset_id"),
                    values.get("dataset_version"),
                    values.get("split_id"),
                    _bool_int(values.get("use_edge_features")),
                    values.get("embedding_dim"),
                    values.get("neighbor_k"),
                    values.get("learning_rate"),
                    values.get("batch_size"),
                    int(seed),
                    int(fold),
                    int(attempt),
                    values.get("git_commit"),
                    _bool_int(values.get("dirty_status")),
                    values.get("preprocessing_version"),
                    values.get("dataset_fingerprint"),
                    values.get("split_fingerprint"),
                    values.get("start_time"),
                    values.get("end_time"),
                    values.get("duration_seconds"),
                    values.get("host"),
                    values.get("gpu_model"),
                    values.get("peak_vram_gb"),
                    values.get("parameter_count"),
                    values.get("primary_metric_name"),
                    values.get("primary_metric_value"),
                    str(artifact_path) if artifact_path is not None else None,
                    values.get("failure_category"),
                    retry_of,
                    _json(configuration),
                    now,
                    now,
                ),
            )
        result = self.get_run(run_id)
        assert result is not None
        return result

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _row_dict(row)

    def resolve_run_id(self, run_reference: str) -> str:
        """Resolve an immutable primary ID or a globally unique alias."""

        reference = str(run_reference).strip()
        if not reference:
            raise ValueError("run_reference must be non-empty.")
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id FROM runs WHERE run_id = ?
                UNION ALL
                SELECT run_id FROM run_aliases WHERE alias_id = ?
                """,
                (reference, reference),
            ).fetchall()
        identifiers = {str(row["run_id"]) for row in rows}
        if not identifiers:
            raise RegistryError(f"Unknown run or alias: {reference}")
        if len(identifiers) != 1:
            raise RegistryConflictError(
                f"Run reference {reference!r} resolves ambiguously: "
                f"{sorted(identifiers)}"
            )
        return next(iter(identifiers))

    def get_run_aliases(self, run_reference: str) -> list[dict[str, Any]]:
        """Return aliases for a primary ID or alias, preferred first."""

        run_id = self.resolve_run_id(run_reference)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM run_aliases
                WHERE run_id = ?
                ORDER BY preferred DESC,
                         CASE alias_type WHEN 'semantic' THEN 0 ELSE 1 END,
                         alias_id
                """,
                (run_id,),
            ).fetchall()
        return [_row_dict(row) for row in rows if row is not None]

    def get_run_category(self, run_reference: str) -> dict[str, Any] | None:
        """Return explicit browsing semantics without interpreting placeholders."""

        run_id = self.resolve_run_id(run_reference)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_categories WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return _row_dict(row)

    def register_run_semantics(
        self,
        run_reference: str,
        *,
        lifecycle_stage: str,
        study_axis: str,
        source_batch: str,
        seed_known: bool,
        fold_known: bool,
        attempt_known: bool,
        retention_class: str,
        category_key: str,
        classification_confidence: str,
        timestamp_basis: str,
        variant_label: str | None = None,
        model_key: str | None = None,
        dataset_key: str | None = None,
        masking_key: str | None = None,
        graph_key: str | None = None,
        feature_key: str | None = None,
        embedding_key: str | None = None,
        rules: Mapping[str, Any] | None = None,
        canonical_alias: str | None = None,
        semantic_alias: str | None = None,
        preferred_alias_type: str | None = None,
        alias_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        update: bool = False,
    ) -> dict[str, Any]:
        """Register query semantics and optional aliases transactionally.

        Existing content is idempotent. Different category or alias content is
        rejected unless ``update=True`` is explicit. The three ``*_known``
        flags distinguish trustworthy execution dimensions from legacy
        placeholder values stored in ``runs``.
        """

        run_id = self.resolve_run_id(run_reference)
        for name, value in (
            ("seed_known", seed_known),
            ("fold_known", fold_known),
            ("attempt_known", attempt_known),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be an explicit boolean.")
        confidence = _required_registry_text(
            "classification_confidence", classification_confidence
        )
        if confidence not in {"high", "medium", "low", "unknown"}:
            raise ValueError(
                "classification_confidence must be high, medium, low, or unknown."
            )
        category_values: dict[str, Any] = {
            "lifecycle_stage": _required_registry_text(
                "lifecycle_stage", lifecycle_stage
            ),
            "study_axis": _required_registry_text("study_axis", study_axis),
            "source_batch": _required_registry_text("source_batch", source_batch),
            "variant_label": _optional_registry_text(variant_label),
            "model_key": _optional_registry_text(model_key),
            "dataset_key": _optional_registry_text(dataset_key),
            "masking_key": _optional_registry_text(masking_key),
            "graph_key": _optional_registry_text(graph_key),
            "feature_key": _optional_registry_text(feature_key),
            "embedding_key": _optional_registry_text(embedding_key),
            "seed_known": int(seed_known),
            "fold_known": int(fold_known),
            "attempt_known": int(attempt_known),
            "retention_class": _required_registry_text(
                "retention_class", retention_class
            ),
            "category_key": _required_registry_text("category_key", category_key),
            "rules_json": _json(rules or {}),
            "classification_confidence": confidence,
            "timestamp_basis": _required_registry_text(
                "timestamp_basis", timestamp_basis
            ),
        }
        aliases = {
            alias_type: alias
            for alias_type, alias in (
                ("canonical", canonical_alias),
                ("semantic", semantic_alias),
            )
            if alias is not None
        }
        normalized_aliases = {
            alias_type: _required_registry_text(f"{alias_type}_alias", alias)
            for alias_type, alias in aliases.items()
        }
        metadata_by_type = alias_metadata or {}
        unknown_metadata_types = set(metadata_by_type) - {"canonical", "semantic"}
        if unknown_metadata_types:
            raise ValueError(
                f"Unknown alias metadata types: {sorted(unknown_metadata_types)}"
            )
        for alias_type, metadata in metadata_by_type.items():
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    f"Alias metadata for {alias_type!r} must be a mapping."
                )
        if preferred_alias_type is not None and preferred_alias_type not in {
            "canonical",
            "semantic",
        }:
            raise ValueError(
                "preferred_alias_type must be canonical, semantic, or None."
            )
        desired_preferred = preferred_alias_type

        category_columns = tuple(category_values)
        now = utc_now()
        try:
            with self.transaction(immediate=True) as connection:
                existing_category = connection.execute(
                    "SELECT * FROM run_categories WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if existing_category is None:
                    connection.execute(
                        f"""
                        INSERT INTO run_categories(
                            run_id, {", ".join(category_columns)},
                            created_at, updated_at
                        ) VALUES (
                            ?, {", ".join("?" for _ in category_columns)}, ?, ?
                        )
                        """,
                        (
                            run_id,
                            *(category_values[name] for name in category_columns),
                            now,
                            now,
                        ),
                    )
                else:
                    category_changed = any(
                        existing_category[name] != category_values[name]
                        for name in category_columns
                    )
                    if category_changed and not update:
                        raise RegistryConflictError(
                            f"Run {run_id!r} already has different categories."
                        )
                    if category_changed:
                        connection.execute(
                            f"""
                            UPDATE run_categories
                            SET {", ".join(f"{name} = ?" for name in category_columns)},
                                updated_at = ?
                            WHERE run_id = ?
                            """,
                            (
                                *(category_values[name] for name in category_columns),
                                now,
                                run_id,
                            ),
                        )

                existing_alias_rows = connection.execute(
                    "SELECT * FROM run_aliases WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
                existing_by_type = {
                    str(row["alias_type"]): row for row in existing_alias_rows
                }
                current_preferred = next(
                    (
                        str(row["alias_type"])
                        for row in existing_alias_rows
                        if bool(row["preferred"])
                    ),
                    None,
                )
                effective_preferred = desired_preferred
                if effective_preferred is None and normalized_aliases:
                    effective_preferred = current_preferred or (
                        "semantic"
                        if "semantic" in normalized_aliases
                        else "canonical"
                    )
                if effective_preferred is not None:
                    available_types = set(existing_by_type) | set(normalized_aliases)
                    if effective_preferred not in available_types:
                        raise ValueError(
                            f"Preferred alias type {effective_preferred!r} is not "
                            "registered or supplied."
                        )
                    if (
                        current_preferred is not None
                        and current_preferred != effective_preferred
                        and not update
                    ):
                        raise RegistryConflictError(
                            f"Run {run_id!r} already prefers "
                            f"{current_preferred!r}; pass update=True to change it."
                        )
                    if current_preferred != effective_preferred:
                        connection.execute(
                            "UPDATE run_aliases SET preferred = 0 WHERE run_id = ?",
                            (run_id,),
                        )

                for alias_type, alias_id in normalized_aliases.items():
                    primary_collision = connection.execute(
                        "SELECT run_id FROM runs WHERE run_id = ?",
                        (alias_id,),
                    ).fetchone()
                    if (
                        primary_collision is not None
                        and str(primary_collision["run_id"]) != run_id
                    ):
                        raise RegistryConflictError(
                            f"Alias {alias_id!r} is another run's primary ID."
                        )
                    alias_collision = connection.execute(
                        "SELECT run_id, alias_type FROM run_aliases WHERE alias_id = ?",
                        (alias_id,),
                    ).fetchone()
                    if (
                        alias_collision is not None
                        and str(alias_collision["run_id"]) != run_id
                    ):
                        raise RegistryConflictError(
                            f"Alias {alias_id!r} already belongs to "
                            f"{alias_collision['run_id']!r}."
                        )
                    metadata_text = _json(metadata_by_type.get(alias_type, {}))
                    preferred = int(alias_type == effective_preferred)
                    existing_alias = existing_by_type.get(alias_type)
                    if existing_alias is None:
                        connection.execute(
                            """
                            INSERT INTO run_aliases(
                                alias_id, run_id, alias_type, preferred,
                                metadata_json, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                alias_id,
                                run_id,
                                alias_type,
                                preferred,
                                metadata_text,
                                now,
                            ),
                        )
                        continue
                    alias_changed = (
                        str(existing_alias["alias_id"]) != alias_id
                        or bool(existing_alias["preferred"]) != bool(preferred)
                        or str(existing_alias["metadata_json"]) != metadata_text
                    )
                    if alias_changed and not update:
                        raise RegistryConflictError(
                            f"Run {run_id!r} already has a different "
                            f"{alias_type!r} alias."
                        )
                    if alias_changed:
                        connection.execute(
                            """
                            UPDATE run_aliases
                            SET alias_id = ?, preferred = ?, metadata_json = ?
                            WHERE run_id = ? AND alias_type = ?
                            """,
                            (
                                alias_id,
                                preferred,
                                metadata_text,
                                run_id,
                                alias_type,
                            ),
                        )

                if (
                    effective_preferred is not None
                    and effective_preferred not in normalized_aliases
                    and current_preferred != effective_preferred
                ):
                    connection.execute(
                        """
                        UPDATE run_aliases SET preferred = 1
                        WHERE run_id = ? AND alias_type = ?
                        """,
                        (run_id, effective_preferred),
                    )
        except sqlite3.IntegrityError as error:
            raise RegistryConflictError(
                f"Run semantics for {run_id!r} violate registry uniqueness."
            ) from error
        return {
            "run_id": run_id,
            "aliases": self.get_run_aliases(run_id),
            "category": self.get_run_category(run_id),
        }

    def show_run(self, run_id: str) -> dict[str, Any] | None:
        try:
            resolved_run_id = self.resolve_run_id(run_id)
        except RegistryConflictError:
            raise
        except RegistryError:
            return None
        run = self.get_run(resolved_run_id)
        if run is None:
            return None
        with self.connect() as connection:
            metrics = connection.execute(
                """
                SELECT name, value, step, split, recorded_at
                FROM metrics WHERE run_id = ?
                ORDER BY metric_id
                """,
                (resolved_run_id,),
            ).fetchall()
            artifacts = connection.execute(
                """
                SELECT kind, path, sha256, size_bytes, status, created_at
                FROM artifacts WHERE run_id = ?
                ORDER BY artifact_id
                """,
                (resolved_run_id,),
            ).fetchall()
            failures = connection.execute(
                """
                SELECT category, message, details_json, created_at
                FROM failures WHERE run_id = ?
                ORDER BY failure_id
                """,
                (resolved_run_id,),
            ).fetchall()
            checkpoints = connection.execute(
                """
                SELECT c.*, a.kind AS artifact_kind, a.path,
                       a.sha256, a.size_bytes,
                       a.status AS artifact_status
                FROM checkpoint_catalog c
                JOIN artifacts a ON a.artifact_id = c.artifact_id
                WHERE c.run_id = ?
                ORDER BY c.role, c.artifact_id
                """,
                (resolved_run_id,),
            ).fetchall()
        run["metrics"] = [dict(row) for row in metrics]
        run["artifacts"] = [dict(row) for row in artifacts]
        run["failures"] = [
            {
                **{key: value for key, value in dict(row).items() if key != "details_json"},
                "details": _decode_json(row["details_json"]),
            }
            for row in failures
        ]
        run["aliases"] = self.get_run_aliases(resolved_run_id)
        run["category"] = self.get_run_category(resolved_run_id)
        run["checkpoints"] = [
            _row_dict(row) for row in checkpoints if row is not None
        ]
        return run

    def transition_run(
        self,
        run_id: str,
        status: str,
        **updates: Any,
    ) -> dict[str, Any]:
        if status not in RUN_STATUSES:
            raise ValueError(f"Unknown run status: {status}")
        allowed_columns = {
            "start_time",
            "end_time",
            "duration_seconds",
            "host",
            "gpu_model",
            "peak_vram_gb",
            "parameter_count",
            "primary_metric_name",
            "primary_metric_value",
            "artifact_path",
            "failure_category",
        }
        unknown = set(updates) - allowed_columns
        if unknown:
            raise ValueError(f"Unknown run update fields: {sorted(unknown)}")
        with self.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RegistryError(f"Unknown run: {run_id}")
            current = str(row["status"])
            if current != status and status not in _RUN_TRANSITIONS.get(
                current, frozenset()
            ):
                raise InvalidTransitionError(
                    f"Run {run_id} cannot transition {current!r} -> {status!r}."
                )
            assignments = ["status = ?", "updated_at = ?"]
            values: list[Any] = [status, utc_now()]
            for column, value in updates.items():
                assignments.append(f"{column} = ?")
                values.append(
                    str(value) if column == "artifact_path" and value is not None else value
                )
            values.append(run_id)
            connection.execute(
                f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?",
                values,
            )
        result = self.get_run(run_id)
        assert result is not None
        return result

    def record_metric(
        self,
        run_id: str,
        name: str,
        value: float,
        *,
        step: int | None = None,
        split: str | None = None,
        evaluation_id: str | None = None,
    ) -> int:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Registry metrics must be finite.")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO metrics(
                    run_id, evaluation_id, name, value, step, split, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    evaluation_id,
                    name,
                    numeric,
                    step,
                    split,
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def record_artifact(
        self,
        run_id: str,
        *,
        kind: str,
        path: str | Path,
        sha256: str | None = None,
        size_bytes: int | None = None,
        status: str = "present",
        evaluation_id: str | None = None,
    ) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO artifacts(
                    run_id, evaluation_id, kind, path, sha256, size_bytes,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    evaluation_id,
                    kind,
                    str(path),
                    sha256,
                    size_bytes,
                    status,
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def record_artifacts(
        self,
        run_id: str,
        artifacts: Sequence[Mapping[str, Any]],
    ) -> None:
        """Record a bundle inventory atomically using parameterized SQL."""

        rows = list(artifacts)
        if not rows:
            return
        with self.transaction(immediate=True) as connection:
            connection.executemany(
                """
                INSERT INTO artifacts(
                    run_id, evaluation_id, kind, path, sha256, size_bytes,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        row.get("evaluation_id"),
                        str(row["kind"]),
                        str(row["path"]),
                        row.get("sha256"),
                        row.get("size_bytes"),
                        str(row.get("status", "present")),
                        utc_now(),
                    )
                    for row in rows
                ],
            )

    def register_checkpoint_metadata(
        self,
        artifact_id: int,
        *,
        run_id: str,
        role: str,
        retention_class: str,
        verification_status: str,
        best_epoch: int | None = None,
        monitored_metric: str | None = None,
        monitored_mode: str | None = None,
        monitored_value: float | None = None,
        duplicate_group: str | None = None,
        duplicate_count: int = 1,
        metadata: Mapping[str, Any] | None = None,
        update: bool = False,
    ) -> dict[str, Any]:
        """Attach searchable checkpoint semantics to one artifact record."""

        resolved_run_id = self.resolve_run_id(run_id)
        if isinstance(artifact_id, bool) or int(artifact_id) < 1:
            raise ValueError("artifact_id must be a positive integer.")
        identifier = int(artifact_id)
        if best_epoch is not None and (
            isinstance(best_epoch, bool) or int(best_epoch) < 0
        ):
            raise ValueError("best_epoch must be a non-negative integer or None.")
        if monitored_mode not in {None, "min", "max", "unknown"}:
            raise ValueError("monitored_mode must be min, max, unknown, or None.")
        numeric_value: float | None = None
        if monitored_value is not None:
            numeric_value = float(monitored_value)
            if not math.isfinite(numeric_value):
                raise ValueError("monitored_value must be finite.")
        if isinstance(duplicate_count, bool) or int(duplicate_count) < 1:
            raise ValueError("duplicate_count must be a positive integer.")
        values: dict[str, Any] = {
            "run_id": resolved_run_id,
            "role": _required_registry_text("role", role),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "monitored_metric": _optional_registry_text(monitored_metric),
            "monitored_mode": monitored_mode,
            "monitored_value": numeric_value,
            "retention_class": _required_registry_text(
                "retention_class", retention_class
            ),
            "duplicate_group": _optional_registry_text(duplicate_group),
            "duplicate_count": int(duplicate_count),
            "verification_status": _required_registry_text(
                "verification_status", verification_status
            ),
            "metadata_json": _json(metadata or {}),
        }
        columns = tuple(values)
        now = utc_now()
        try:
            with self.transaction(immediate=True) as connection:
                artifact = connection.execute(
                    """
                    SELECT run_id, kind FROM artifacts WHERE artifact_id = ?
                    """,
                    (identifier,),
                ).fetchone()
                if artifact is None:
                    raise RegistryError(f"Unknown artifact: {identifier}")
                if str(artifact["run_id"]) != resolved_run_id:
                    raise RegistryConflictError(
                        f"Artifact {identifier} belongs to {artifact['run_id']!r}, "
                        f"not {resolved_run_id!r}."
                    )
                if "checkpoint" not in str(artifact["kind"]).lower():
                    raise RegistryError(
                        f"Artifact {identifier} is not classified as a checkpoint."
                    )
                existing = connection.execute(
                    "SELECT * FROM checkpoint_catalog WHERE artifact_id = ?",
                    (identifier,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        f"""
                        INSERT INTO checkpoint_catalog(
                            artifact_id, {", ".join(columns)}, created_at, updated_at
                        ) VALUES (
                            ?, {", ".join("?" for _ in columns)}, ?, ?
                        )
                        """,
                        (
                            identifier,
                            *(values[name] for name in columns),
                            now,
                            now,
                        ),
                    )
                else:
                    changed = any(existing[name] != values[name] for name in columns)
                    if changed and not update:
                        raise RegistryConflictError(
                            f"Checkpoint artifact {identifier} already has "
                            "different metadata."
                        )
                    if changed:
                        connection.execute(
                            f"""
                            UPDATE checkpoint_catalog
                            SET {", ".join(f"{name} = ?" for name in columns)},
                                updated_at = ?
                            WHERE artifact_id = ?
                            """,
                            (
                                *(values[name] for name in columns),
                                now,
                                identifier,
                            ),
                        )
        except sqlite3.IntegrityError as error:
            raise RegistryConflictError(
                f"Checkpoint metadata for artifact {identifier} conflicts."
            ) from error
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT c.*, a.kind AS artifact_kind, a.path,
                       a.sha256, a.size_bytes,
                       a.status AS artifact_status
                FROM checkpoint_catalog c
                JOIN artifacts a ON a.artifact_id = c.artifact_id
                WHERE c.artifact_id = ?
                """,
                (identifier,),
            ).fetchone()
        result = _row_dict(row)
        assert result is not None
        return result

    def list_checkpoint_catalog(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        role: str | None = None,
        retention_class: str | None = None,
        verification_status: str | None = None,
        duplicate_group: str | None = None,
        lifecycle_stage: str | None = None,
        study_axis: str | None = None,
        limit: int | None = 1000,
    ) -> list[dict[str, Any]]:
        """List checkpoints using a fixed, parameterized filter vocabulary."""

        filter_columns = {
            "run_id": "c.run_id",
            "role": "c.role",
            "retention_class": "c.retention_class",
            "verification_status": "c.verification_status",
            "duplicate_group": "c.duplicate_group",
            "lifecycle_stage": "rc.lifecycle_stage",
            "study_axis": "rc.study_axis",
            "source_batch": "rc.source_batch",
            "variant_label": "rc.variant_label",
            "model_key": "rc.model_key",
            "dataset_key": "rc.dataset_key",
            "masking_key": "rc.masking_key",
            "graph_key": "rc.graph_key",
            "feature_key": "rc.feature_key",
            "embedding_key": "rc.embedding_key",
            "seed_known": "rc.seed_known",
            "fold_known": "rc.fold_known",
            "attempt_known": "rc.attempt_known",
            "category_key": "rc.category_key",
            "run_retention_class": "rc.retention_class",
        }
        selected: dict[str, Any] = dict(filters or {})
        explicit = {
            "run_id": run_id,
            "role": role,
            "retention_class": retention_class,
            "verification_status": verification_status,
            "duplicate_group": duplicate_group,
            "lifecycle_stage": lifecycle_stage,
            "study_axis": study_axis,
        }
        for name, value in explicit.items():
            if value is None:
                continue
            if name in selected and selected[name] != value:
                raise ValueError(f"Conflicting checkpoint filter: {name}")
            selected[name] = value
        unknown = set(selected) - set(filter_columns)
        if unknown:
            raise ValueError(f"Unknown checkpoint filters: {sorted(unknown)}")
        if "run_id" in selected:
            selected["run_id"] = self.resolve_run_id(str(selected["run_id"]))
        if limit is not None and (
            isinstance(limit, bool) or int(limit) < 1
        ):
            raise ValueError("limit must be a positive integer or None.")

        clauses: list[str] = []
        parameters: list[Any] = []
        for name, value in selected.items():
            column = filter_columns[name]
            if value is None:
                clauses.append(f"{column} IS NULL")
            else:
                clauses.append(f"{column} = ?")
                parameters.append(
                    int(value)
                    if name in {"seed_known", "fold_known", "attempt_known"}
                    else value
                )
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT ?"
            parameters.append(int(limit))
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, a.kind AS artifact_kind, a.path,
                       a.sha256, a.size_bytes,
                       a.status AS artifact_status,
                       r.scientific_id, r.repro_id, r.model_family,
                       r.masking_type, r.dataset_id, r.dataset_version,
                       r.use_edge_features, r.embedding_dim, r.neighbor_k,
                       r.seed, r.fold, r.attempt,
                       rc.lifecycle_stage, rc.study_axis, rc.source_batch,
                       rc.variant_label, rc.model_key, rc.dataset_key,
                       rc.masking_key, rc.graph_key, rc.feature_key,
                       rc.embedding_key, rc.seed_known, rc.fold_known,
                       rc.attempt_known,
                       rc.retention_class AS run_retention_class,
                       rc.category_key, rc.classification_confidence,
                       rc.timestamp_basis,
                       (
                           SELECT ra.alias_id FROM run_aliases ra
                           WHERE ra.run_id = c.run_id
                           ORDER BY ra.preferred DESC,
                                    CASE ra.alias_type
                                        WHEN 'semantic' THEN 0 ELSE 1 END,
                                    ra.alias_id
                           LIMIT 1
                       ) AS preferred_alias
                FROM checkpoint_catalog c
                JOIN artifacts a ON a.artifact_id = c.artifact_id
                JOIN runs r ON r.run_id = c.run_id
                LEFT JOIN run_categories rc ON rc.run_id = c.run_id
                {where}
                ORDER BY COALESCE(rc.category_key, ''), c.run_id,
                         c.role, c.artifact_id
                {limit_sql}
                """,
                parameters,
            ).fetchall()
        return [_row_dict(row) for row in rows if row is not None]

    def record_failure(
        self,
        *,
        category: str,
        message: str,
        run_id: str | None = None,
        job_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        if run_id is None and job_id is None:
            raise ValueError("A failure must reference a run or queue job.")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO failures(
                    run_id, job_id, category, message, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    job_id,
                    category,
                    message,
                    _json(details or {}),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def create_evaluation(
        self,
        evaluation_id: str,
        *,
        run_id: str,
        checkpoint_name: str,
        dataset_id: str,
        split_id: str,
        status: str = "pending",
        metrics: Mapping[str, Any] | None = None,
        artifact_path: str | Path | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO evaluations(
                    evaluation_id, run_id, checkpoint_name, dataset_id,
                    split_id, status, metrics_json, artifact_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    run_id,
                    checkpoint_name,
                    dataset_id,
                    split_id,
                    status,
                    _json(metrics or {}),
                    str(artifact_path) if artifact_path is not None else None,
                    utc_now(),
                ),
            )
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
        result = _row_dict(row)
        assert result is not None
        return result

    def register_dataset(
        self,
        dataset_id: str,
        dataset_version: str,
        *,
        display_name: str,
        protected_source_path: str | Path | None = None,
        raw_fingerprint: str | None = None,
        preprocessing_version: str | None = None,
        processed_fingerprint: str | None = None,
        aggregate_sample_count: int | None = None,
        graph_count: int | None = None,
        node_feature_schema: str | None = None,
        edge_feature_schema: str | None = None,
        creation_date: str | None = None,
        status: str = "available",
        verification_status: str = "unverified",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        values = (
            dataset_id,
            dataset_version,
            display_name,
            str(protected_source_path) if protected_source_path else None,
            raw_fingerprint,
            preprocessing_version,
            processed_fingerprint,
            aggregate_sample_count,
            graph_count,
            node_feature_schema,
            edge_feature_schema,
            creation_date,
            status,
            verification_status,
            _json(metadata or {}),
            utc_now(),
        )
        try:
            with self.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO datasets(
                        dataset_id, dataset_version, display_name,
                        protected_source_path, raw_fingerprint,
                        preprocessing_version, processed_fingerprint,
                        aggregate_sample_count, graph_count, node_feature_schema,
                        edge_feature_schema, creation_date, status,
                        verification_status, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        except sqlite3.IntegrityError as error:
            existing = self.get_dataset(dataset_id, dataset_version)
            comparable = existing and {
                key: existing[key]
                for key in (
                    "display_name",
                    "protected_source_path",
                    "raw_fingerprint",
                    "preprocessing_version",
                    "processed_fingerprint",
                    "aggregate_sample_count",
                    "graph_count",
                    "node_feature_schema",
                    "edge_feature_schema",
                    "creation_date",
                    "status",
                    "verification_status",
                    "metadata",
                )
            }
            desired = {
                "display_name": display_name,
                "protected_source_path": (
                    str(protected_source_path) if protected_source_path else None
                ),
                "raw_fingerprint": raw_fingerprint,
                "preprocessing_version": preprocessing_version,
                "processed_fingerprint": processed_fingerprint,
                "aggregate_sample_count": aggregate_sample_count,
                "graph_count": graph_count,
                "node_feature_schema": node_feature_schema,
                "edge_feature_schema": edge_feature_schema,
                "creation_date": creation_date,
                "status": status,
                "verification_status": verification_status,
                "metadata": metadata or {},
            }
            if comparable != desired:
                raise RegistryConflictError(
                    f"Dataset {dataset_id!r}/{dataset_version!r} conflicts."
                ) from error
            if existing is None:
                raise
        result = self.get_dataset(dataset_id, dataset_version)
        assert result is not None
        return result

    def get_dataset(
        self, dataset_id: str, dataset_version: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM datasets
                WHERE dataset_id = ? AND dataset_version = ?
                """,
                (dataset_id, dataset_version),
            ).fetchone()
        return _row_dict(row)

    def register_split(
        self,
        split_id: str,
        *,
        dataset_id: str,
        dataset_version: str,
        method: str,
        unit: str,
        seed: int | None = None,
        fold_count: int | None = None,
        stratification: Sequence[str] | Mapping[str, Any] | None = None,
        fingerprint: str | None = None,
        protected_path: str | Path | None = None,
        verification_status: str = "unverified",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            with self.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO splits(
                        split_id, dataset_id, dataset_version, method, seed,
                        fold_count, unit, stratification_json, fingerprint,
                        protected_path, verification_status, metadata_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        split_id,
                        dataset_id,
                        dataset_version,
                        method,
                        seed,
                        fold_count,
                        unit,
                        _json(stratification or []),
                        fingerprint,
                        str(protected_path) if protected_path else None,
                        verification_status,
                        _json(metadata or {}),
                        utc_now(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            existing = self.get_split(split_id)
            if existing is None:
                raise
            desired = {
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "method": method,
                "seed": seed,
                "fold_count": fold_count,
                "unit": unit,
                "stratification": stratification or [],
                "fingerprint": fingerprint,
                "protected_path": str(protected_path) if protected_path else None,
                "verification_status": verification_status,
                "metadata": metadata or {},
            }
            if any(existing[key] != value for key, value in desired.items()):
                raise RegistryConflictError(
                    f"Split {split_id!r} already has different content."
                ) from error
        result = self.get_split(split_id)
        assert result is not None
        return result

    def get_split(self, split_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM splits WHERE split_id = ?", (split_id,)
            ).fetchone()
        return _row_dict(row)

    def enqueue(
        self,
        *,
        campaign_id: str,
        configuration: Mapping[str, Any],
        command: Sequence[str],
        experiment_config_reference: str | Path | None = None,
        priority: int = 0,
        maximum_attempts: int = 1,
        requested_gpu: str | None = "0",
        job_id: str | None = None,
        attempt_count: int = 1,
        retry_of: str | None = None,
    ) -> dict[str, Any]:
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive.")
        if not 1 <= attempt_count <= maximum_attempts:
            raise ValueError("attempt_count must be within maximum_attempts.")
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("command must be a non-empty sequence of strings.")
        identifier = job_id or _job_id(configuration)
        now = utc_now()
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO queue_jobs(
                    job_id, campaign_id, experiment_config_reference,
                    canonical_config_json, command_json, priority, status,
                    attempt_count, maximum_attempts, created_at, requested_gpu,
                    retry_of
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    campaign_id,
                    (
                        str(experiment_config_reference)
                        if experiment_config_reference is not None
                        else None
                    ),
                    _json(configuration),
                    _json(list(command)),
                    int(priority),
                    int(attempt_count),
                    int(maximum_attempts),
                    now,
                    requested_gpu,
                    retry_of,
                ),
            )
        result = self.get_job(identifier)
        assert result is not None
        return result

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM queue_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _row_dict(row)

    def list_queue(
        self, *, statuses: Sequence[str] | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        parameters: list[Any] = []
        where = ""
        if statuses:
            unknown = set(statuses) - QUEUE_STATUSES
            if unknown:
                raise ValueError(f"Unknown queue statuses: {sorted(unknown)}")
            where = "WHERE status IN (" + ",".join("?" for _ in statuses) + ")"
            parameters.extend(statuses)
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM queue_jobs
                {where}
                ORDER BY
                    CASE status WHEN 'running' THEN 0 WHEN 'claimed' THEN 1
                                WHEN 'queued' THEN 2 ELSE 3 END,
                    priority DESC, created_at, job_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [_row_dict(row) for row in rows if row is not None]

    def claim_next(
        self,
        worker_id: str,
        *,
        requested_gpu: str | None | object = _ANY_GPU,
    ) -> dict[str, Any] | None:
        """Atomically claim at most one queued job for *worker_id*."""

        now = utc_now()
        with self.transaction(immediate=True) as connection:
            gpu_clause = ""
            parameters: list[Any] = []
            if requested_gpu is None:
                gpu_clause = "AND requested_gpu IS NULL"
            elif requested_gpu is not _ANY_GPU:
                gpu_clause = "AND (requested_gpu IS NULL OR requested_gpu = ?)"
                parameters.append(str(requested_gpu))
            row = connection.execute(
                f"""
                SELECT job_id FROM queue_jobs
                WHERE status = 'queued'
                  {gpu_clause}
                ORDER BY priority DESC, created_at, job_id
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["job_id"])
            cursor = connection.execute(
                """
                UPDATE queue_jobs
                SET status = 'claimed', claimed_at = ?, heartbeat_at = ?,
                    worker_id = ?
                WHERE job_id = ? AND status = 'queued'
                """,
                (now, now, worker_id, job_id),
            )
            if cursor.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM queue_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _row_dict(claimed)

    def transition_job(
        self,
        job_id: str,
        status: str,
        *,
        worker_id: str | None = None,
        run_id: str | None = None,
        failure_category: str | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        if status not in QUEUE_STATUSES:
            raise ValueError(f"Unknown queue status: {status}")
        now = utc_now()
        with self.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT status, worker_id FROM queue_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise RegistryError(f"Unknown queue job: {job_id}")
            current = str(row["status"])
            if current != status and status not in _QUEUE_TRANSITIONS.get(
                current, frozenset()
            ):
                raise InvalidTransitionError(
                    f"Job {job_id} cannot transition {current!r} -> {status!r}."
                )
            if worker_id is not None and row["worker_id"] not in (None, worker_id):
                raise RegistryError(
                    f"Job {job_id} is owned by worker {row['worker_id']!r}."
                )
            assignments = ["status = ?"]
            values: list[Any] = [status]
            if status == "running":
                assignments.extend(["started_at = ?", "heartbeat_at = ?"])
                values.extend([now, now])
            if status in {"completed", "failed", "pruned", "cancelled"}:
                assignments.append("finished_at = ?")
                values.append(now)
            for column, value in (
                ("worker_id", worker_id),
                ("run_id", run_id),
                ("failure_category", failure_category),
                ("last_error", last_error),
            ):
                if value is not None:
                    assignments.append(f"{column} = ?")
                    values.append(value)
            values.append(job_id)
            connection.execute(
                f"UPDATE queue_jobs SET {', '.join(assignments)} WHERE job_id = ?",
                values,
            )
        result = self.get_job(job_id)
        assert result is not None
        return result

    def heartbeat(self, job_id: str, worker_id: str) -> None:
        with self.transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE queue_jobs SET heartbeat_at = ?
                WHERE job_id = ? AND worker_id = ?
                  AND status IN ('claimed', 'running')
                """,
                (utc_now(), job_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise RegistryError(
                    f"Cannot heartbeat unowned or inactive job {job_id}."
                )

    def mark_stale(self, *, heartbeat_before: str) -> list[str]:
        """Atomically mark abandoned claims and their live runs as stale/failed.

        A stale queue record is terminal evidence about worker ownership.  Its
        associated run cannot remain ``running`` indefinitely, so pending or
        running run rows are failed in the same transaction.  Scratch content
        is deliberately left untouched for operator review because a detached
        process cannot be ruled out solely from a database heartbeat.
        """

        finished = utc_now()
        with self.transaction(immediate=True) as connection:
            rows = connection.execute(
                """
                SELECT job_id, run_id FROM queue_jobs
                WHERE status IN ('claimed', 'running')
                  AND heartbeat_at IS NOT NULL
                  AND heartbeat_at < ?
                """,
                (heartbeat_before,),
            ).fetchall()
            identifiers = [str(row["job_id"]) for row in rows]
            if identifiers:
                placeholders = ",".join("?" for _ in identifiers)
                connection.execute(
                    f"""
                    UPDATE queue_jobs
                    SET status = 'stale', finished_at = ?,
                        failure_category = 'missing_heartbeat',
                        last_error = 'Worker heartbeat exceeded stale threshold.'
                    WHERE job_id IN ({placeholders})
                      AND status IN ('claimed', 'running')
                    """,
                    [finished, *identifiers],
                )
                run_ids = [str(row["run_id"]) for row in rows if row["run_id"]]
                if run_ids:
                    run_placeholders = ",".join("?" for _ in run_ids)
                    connection.execute(
                        f"""
                        UPDATE runs
                        SET status = 'failed', end_time = ?,
                            failure_category = 'missing_heartbeat', updated_at = ?
                        WHERE run_id IN ({run_placeholders})
                          AND status IN ('pending', 'running', 'finalizing')
                        """,
                        [finished, finished, *run_ids],
                    )
                connection.executemany(
                    """
                    INSERT INTO failures(
                        run_id, job_id, category, message, details_json, created_at
                    ) VALUES (?, ?, 'missing_heartbeat', ?, ?, ?)
                    """,
                    [
                        (
                            row["run_id"],
                            row["job_id"],
                            "Worker heartbeat exceeded stale threshold.",
                            _json({"heartbeat_before": heartbeat_before}),
                            finished,
                        )
                        for row in rows
                    ],
                )
        return identifiers

    def record_deferred_stale_artifact(
        self, *, run_id: str, scratch_path: str | Path
    ) -> None:
        """Model preserved stale scratch without moving a possibly active path."""

        path = str(Path(scratch_path))
        now = utc_now()
        with self.transaction(immediate=True) as connection:
            run = connection.execute(
                "SELECT status, failure_category FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if (
                run is None
                or run["status"] != "failed"
                or run["failure_category"] != "missing_heartbeat"
            ):
                raise RegistryError(
                    f"Run {run_id} is not a failed stale run awaiting review."
                )
            connection.execute(
                "UPDATE runs SET artifact_path = ?, updated_at = ? WHERE run_id = ?",
                (path, now, run_id),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO artifacts(
                    run_id, evaluation_id, kind, path, sha256, size_bytes,
                    status, created_at
                ) VALUES (?, NULL, 'deferred_stale_scratch', ?, NULL, NULL,
                          'deferred', ?)
                """,
                (run_id, path, now),
            )

    def create_retry(self, job_id: str, *, priority: int | None = None) -> dict[str, Any]:
        original = self.get_job(job_id)
        if original is None:
            raise RegistryError(f"Unknown queue job: {job_id}")
        if original["status"] not in {"failed", "stale", "pruned"}:
            raise InvalidTransitionError(
                f"Only failed, stale, or pruned jobs may be retried: {job_id}"
            )
        next_attempt = int(original["attempt_count"]) + 1
        maximum = int(original["maximum_attempts"])
        if next_attempt > maximum:
            raise RegistryError(
                f"Job {job_id} exhausted maximum_attempts={maximum}."
            )
        return self.enqueue(
            campaign_id=str(original["campaign_id"]),
            configuration=original["canonical_config"],
            command=original["command"],
            experiment_config_reference=original["experiment_config_reference"],
            priority=int(original["priority"] if priority is None else priority),
            maximum_attempts=maximum,
            requested_gpu=original["requested_gpu"],
            attempt_count=next_attempt,
            retry_of=job_id,
        )

    def register_run_for_job(
        self,
        job_id: str,
        *,
        run_id: str,
        scientific_id: str,
        repro_id: str,
        artifact_path: str | Path,
        resolved_configuration: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job is None:
            raise RegistryError(f"Unknown queue job: {job_id}")
        retry_run = None
        if job["retry_of"]:
            previous = self.get_job(str(job["retry_of"]))
            retry_run = previous["run_id"] if previous else None
        config = dict(resolved_configuration or job["canonical_config"])
        seed = _execution_integer(config.get("seed"), 0)
        fold = _execution_integer(config.get("fold"), 0)
        run = self.create_run(
            run_id,
            campaign_id=str(job["campaign_id"]),
            scientific_id=scientific_id,
            repro_id=repro_id,
            seed=seed,
            fold=fold,
            attempt=int(job["attempt_count"]),
            configuration=config,
            artifact_path=artifact_path,
            retry_of=retry_run,
            **fields,
        )
        self.transition_job(
            job_id,
            "running",
            worker_id=str(job["worker_id"]),
            run_id=run_id,
        )
        return run

    def begin_run_and_job_finalization(
        self,
        *,
        job_id: str,
        run_id: str,
        worker_id: str,
        artifact_path: str | Path,
        end_time: str,
        duration_seconds: float,
        primary_metric_name: str | None = None,
        primary_metric_value: float | None = None,
        peak_vram_gb: float | None = None,
        parameter_count: int | None = None,
        artifacts: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Commit verified content metadata and enter recoverable finalization.

        The caller publishes and verifies an unmarked bundle first. This
        transaction records its artifacts and moves the run to ``finalizing``;
        the queue job deliberately remains running. The marker and terminal
        completion transaction follow, allowing startup reconciliation after a
        hard crash at either boundary.
        """

        finished = utc_now()
        with self.transaction(immediate=True) as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            job = connection.execute(
                "SELECT status, worker_id, run_id FROM queue_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if run is None or job is None:
                raise RegistryError(
                    f"Missing run/job during completion: {run_id}/{job_id}."
                )
            if str(run["status"]) != "running":
                raise InvalidTransitionError(
                    f"Run {run_id} is not running during completion."
                )
            if str(job["status"]) != "running":
                raise InvalidTransitionError(
                    f"Job {job_id} is not running during completion."
                )
            if job["worker_id"] != worker_id or job["run_id"] != run_id:
                raise RegistryError(
                    f"Job {job_id} ownership/run changed during completion."
                )
            connection.execute(
                """
                UPDATE runs
                SET status = 'finalizing', end_time = ?, duration_seconds = ?,
                    primary_metric_name = ?, primary_metric_value = ?,
                    peak_vram_gb = ?, parameter_count = ?, artifact_path = ?,
                    updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (
                    end_time,
                    float(duration_seconds),
                    primary_metric_name,
                    primary_metric_value,
                    peak_vram_gb,
                    parameter_count,
                    str(artifact_path),
                    finished,
                    run_id,
                ),
            )
            connection.execute(
                """
                UPDATE queue_jobs
                SET heartbeat_at = ?, run_id = ?
                WHERE job_id = ? AND status = 'running' AND worker_id = ?
                """,
                (finished, run_id, job_id, worker_id),
            )
            if primary_metric_name is not None and primary_metric_value is not None:
                connection.execute(
                    """
                    INSERT INTO metrics(
                        run_id, evaluation_id, name, value, step, split, recorded_at
                    ) VALUES (?, NULL, ?, ?, NULL, ?, ?)
                    """,
                    (
                        run_id,
                        primary_metric_name,
                        float(primary_metric_value),
                        (
                            primary_metric_name.split("/", 1)[0]
                            if "/" in primary_metric_name
                            else None
                        ),
                        finished,
                    ),
                )
            if artifacts:
                connection.executemany(
                    """
                    INSERT INTO artifacts(
                        run_id, evaluation_id, kind, path, sha256, size_bytes,
                        status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            row.get("evaluation_id"),
                            str(row["kind"]),
                            str(row["path"]),
                            row.get("sha256"),
                            row.get("size_bytes"),
                            str(row.get("status", "present")),
                            finished,
                        )
                        for row in artifacts
                    ],
                )

    def complete_run_and_job(
        self,
        *,
        job_id: str,
        run_id: str,
        worker_id: str | None = None,
    ) -> None:
        """Atomically complete a marker-bound finalizing run and queue job."""

        run_before = self.get_run(run_id)
        if run_before is None or not run_before.get("artifact_path"):
            raise RegistryError(f"Finalizing run has no artifact path: {run_id}")
        from .run_archive import verify_run_bundle

        verify_run_bundle(Path(str(run_before["artifact_path"])))
        finished = utc_now()
        with self.transaction(immediate=True) as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            job = connection.execute(
                "SELECT status, worker_id, run_id FROM queue_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if run is None or job is None:
                raise RegistryError(
                    f"Missing run/job during terminal completion: {run_id}/{job_id}."
                )
            if str(run["status"]) != "finalizing" or str(job["status"]) != "running":
                raise InvalidTransitionError(
                    f"Run/job are not finalizing/running: {run_id}/{job_id}."
                )
            if job["run_id"] != run_id:
                raise RegistryError(f"Job {job_id} no longer references {run_id}.")
            if worker_id is not None and job["worker_id"] != worker_id:
                raise RegistryError(
                    f"Job {job_id} is owned by {job['worker_id']!r}, not "
                    f"{worker_id!r}."
                )
            connection.execute(
                """
                UPDATE runs SET status = 'completed', updated_at = ?
                WHERE run_id = ? AND status = 'finalizing'
                """,
                (finished, run_id),
            )
            connection.execute(
                """
                UPDATE queue_jobs
                SET status = 'completed', finished_at = ?, heartbeat_at = ?
                WHERE job_id = ? AND status = 'running' AND run_id = ?
                """,
                (finished, finished, job_id, run_id),
            )

    def list_finalizing_runs(self) -> list[dict[str, Any]]:
        """Return recoverable finalization records in deterministic order."""

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT r.run_id, r.artifact_path, q.job_id, q.worker_id
                FROM runs r JOIN queue_jobs q ON q.run_id = r.run_id
                WHERE r.status = 'finalizing' AND q.status = 'running'
                ORDER BY r.updated_at, r.run_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def fail_run_and_job_finalization(
        self,
        *,
        job_id: str,
        run_id: str,
        worker_id: str,
        artifact_path: str | Path,
        failure_category: str,
        last_error: str,
        duration_seconds: float,
    ) -> None:
        """Compensate an unmarked publish that failed during finalization."""

        finished = utc_now()
        with self.transaction(immediate=True) as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            job = connection.execute(
                "SELECT status, worker_id, run_id FROM queue_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if run is None or job is None:
                raise RegistryError(
                    f"Missing run/job during finalization compensation: "
                    f"{run_id}/{job_id}."
                )
            if str(run["status"]) not in {"running", "finalizing", "completed"}:
                raise InvalidTransitionError(
                    f"Run {run_id} cannot be compensated from {run['status']!r}."
                )
            if str(job["status"]) not in {"running", "completed"}:
                raise InvalidTransitionError(
                    f"Job {job_id} cannot be compensated from {job['status']!r}."
                )
            if job["worker_id"] != worker_id or job["run_id"] != run_id:
                raise RegistryError(
                    f"Job {job_id} ownership/run changed during compensation."
                )
            connection.execute(
                """
                UPDATE runs
                SET status = 'failed', end_time = ?, duration_seconds = ?,
                    artifact_path = ?, failure_category = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    finished,
                    float(duration_seconds),
                    str(artifact_path),
                    failure_category,
                    finished,
                    run_id,
                ),
            )
            connection.execute(
                """
                UPDATE queue_jobs
                SET status = 'failed', finished_at = ?, heartbeat_at = ?,
                    failure_category = ?, last_error = ?
                WHERE job_id = ?
                """,
                (
                    finished,
                    finished,
                    failure_category,
                    last_error,
                    job_id,
                ),
            )

    def promote_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None or run["status"] != "completed":
            raise RegistryError(f"Only a completed run can be promoted: {run_id}")
        artifact_value = run.get("artifact_path")
        if not artifact_value:
            raise RegistryError(f"Run {run_id} has no artifact path to promote.")
        with self.connect() as connection:
            artifact_row = connection.execute(
                """
                SELECT COUNT(*) AS artifact_count,
                       MAX(CASE WHEN kind = 'legacy_manifest' THEN 1 ELSE 0 END)
                           AS is_legacy_native
                FROM artifacts WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if artifact_row is None or int(artifact_row["artifact_count"]) < 1:
            raise RegistryError(
                f"Run {run_id} has no registered artifacts to verify."
            )
        issues = self.verify_artifacts(run_id=run_id)
        if issues:
            raise RegistryError(
                f"Run {run_id} failed artifact verification: "
                + _json(issues[:10])
            )
        if not bool(artifact_row["is_legacy_native"]):
            try:
                from .run_archive import verify_run_bundle

                verify_run_bundle(Path(str(artifact_value)))
            except Exception as error:
                raise RegistryError(
                    f"Run {run_id} failed canonical bundle verification: {error}"
                ) from error
        with self.transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET promoted_at = COALESCE(promoted_at, ?), updated_at = ?
                WHERE run_id = ? AND status = 'completed'
                """,
                (utc_now(), utc_now(), run_id),
            )
            if cursor.rowcount != 1:
                raise RegistryError(
                    f"Only a completed run can be promoted: {run_id}"
                )
        result = self.get_run(run_id)
        assert result is not None
        return result

    def summarize_variants(
        self, *, campaign_id: str | None = None
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        where = ""
        if campaign_id is not None:
            where = "WHERE campaign_id = ?"
            parameters.append(campaign_id)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs {where} ORDER BY created_at, run_id",
                parameters,
            ).fetchall()
            metric_rows = connection.execute(
                """
                SELECT run_id, name, value FROM metrics
                WHERE name IN ('val/auprc', 'calibration/brier')
                ORDER BY metric_id
                """
            ).fetchall()
            job_where = "WHERE retry_of IS NULL"
            job_parameters: list[Any] = []
            if campaign_id is not None:
                job_where += " AND campaign_id = ?"
                job_parameters.append(campaign_id)
            job_rows = connection.execute(
                f"""
                SELECT campaign_id, canonical_config_json, run_id
                FROM queue_jobs {job_where}
                ORDER BY created_at, job_id
                """,
                job_parameters,
            ).fetchall()
        latest_metric: dict[tuple[str, str], float] = {}
        for row in metric_rows:
            latest_metric[(str(row["run_id"]), str(row["name"]))] = float(
                row["value"]
            )
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        run_to_group: dict[str, tuple[str, str, str]] = {}
        for row in rows:
            item = dict(row)
            key = (
                str(item["campaign_id"]),
                str(item["scientific_id"]),
                str(item["repro_id"]),
            )
            grouped.setdefault(key, []).append(item)
            run_to_group[str(item["run_id"])] = key

        expected_by_group: dict[tuple[str, str, str], int] = {}
        unassigned_expected: dict[tuple[str, str], int] = {}
        pending_config: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in job_rows:
            config = json.loads(str(row["canonical_config_json"]))
            scientific_identifier = calculate_scientific_id(config)
            campaign = str(row["campaign_id"])
            pending_config[(campaign, scientific_identifier)] = config
            attached = run_to_group.get(str(row["run_id"])) if row["run_id"] else None
            if attached is None:
                key = (campaign, scientific_identifier)
                unassigned_expected[key] = unassigned_expected.get(key, 0) + 1
            else:
                expected_by_group[attached] = expected_by_group.get(attached, 0) + 1

        groups_per_variant: dict[tuple[str, str], int] = {}
        for campaign, scientific_identifier, _ in grouped:
            key = (campaign, scientific_identifier)
            groups_per_variant[key] = groups_per_variant.get(key, 0) + 1
        summaries: list[dict[str, Any]] = []
        for (
            group_campaign_id,
            scientific_id,
            reproduction_id,
        ), group in sorted(grouped.items()):
            completed = [item for item in group if item["status"] == "completed"]
            failed = [item for item in group if item["status"] == "failed"]
            primary = [
                (str(item["run_id"]), float(item["primary_metric_value"]))
                for item in completed
                if item["primary_metric_value"] is not None
            ]
            durations = [
                float(item["duration_seconds"])
                for item in completed
                if item["duration_seconds"] is not None
            ]
            peaks = [
                float(item["peak_vram_gb"])
                for item in completed
                if item["peak_vram_gb"] is not None
            ]
            auprc = [
                latest_metric[(str(item["run_id"]), "val/auprc")]
                for item in completed
                if (str(item["run_id"]), "val/auprc") in latest_metric
            ]
            brier = [
                latest_metric[(str(item["run_id"]), "calibration/brier")]
                for item in completed
                if (str(item["run_id"]), "calibration/brier") in latest_metric
            ]
            direction = _primary_direction(
                completed[0]["config_json"] if completed else group[0]["config_json"]
            )
            ordered_primary = sorted(
                primary, key=lambda item: item[1], reverse=direction != "minimize"
            )
            values = [value for _, value in primary]
            planned = expected_by_group.get(
                (group_campaign_id, scientific_id, reproduction_id), 0
            )
            variant_key = (group_campaign_id, scientific_id)
            if groups_per_variant.get(variant_key) == 1:
                planned += unassigned_expected.get(variant_key, 0)
            intended_units = {
                (int(item["seed"]), int(item["fold"])) for item in group
            }
            completed_units = {
                (int(item["seed"]), int(item["fold"])) for item in completed
            }
            failed_units = {
                (int(item["seed"]), int(item["fold"])) for item in failed
            }
            metric_names = {
                str(item["primary_metric_name"])
                for item in completed
                if item["primary_metric_name"]
            }
            if not metric_names:
                metric_names = {
                    str(item["primary_metric_name"])
                    for item in group
                    if item["primary_metric_name"]
                }
            if len(metric_names) > 1:
                raise RegistryError(
                    f"Variant/repro group mixes primary metrics: {sorted(metric_names)}"
                )
            primary_name = next(iter(metric_names), None)
            if primary_name is None:
                decoded_config = _decode_json(str(group[0]["config_json"]))
                evaluation = (
                    decoded_config.get("evaluation", {})
                    if isinstance(decoded_config, Mapping)
                    else {}
                )
                primary_name = (
                    evaluation.get("primary_metric")
                    if isinstance(evaluation, Mapping)
                    else None
                )
            summaries.append(
                {
                    "scientific_id": scientific_id,
                    "repro_id": reproduction_id,
                    "campaign_id": group_campaign_id,
                    "n_expected_runs": planned or len(intended_units),
                    "n_completed_runs": len(completed),
                    "n_failed_runs": len(failed_units - completed_units),
                    "n_failed_attempts": len(failed),
                    "primary_metric_name": primary_name,
                    "primary_direction": direction,
                    "primary_mean": mean(values) if values else None,
                    "primary_std": pstdev(values) if values else None,
                    "primary_min": min(values) if values else None,
                    "primary_max": max(values) if values else None,
                    "auprc_mean": mean(auprc) if auprc else None,
                    "brier_mean": mean(brier) if brier else None,
                    "median_duration": median(durations) if durations else None,
                    "maximum_peak_vram": max(peaks) if peaks else None,
                    "best_run_id": ordered_primary[0][0] if ordered_primary else None,
                    "worst_run_id": ordered_primary[-1][0] if ordered_primary else None,
                }
            )
        # A queued variant has no repro_id until code, data, and environment
        # provenance are captured by a worker. Keep those planned executions
        # visible rather than inventing a reproduction identity.
        for (pending_campaign, scientific_identifier), count in sorted(
            unassigned_expected.items()
        ):
            if groups_per_variant.get((pending_campaign, scientific_identifier)) == 1:
                continue
            config = pending_config[(pending_campaign, scientific_identifier)]
            evaluation = config.get("evaluation", {})
            primary_name = (
                evaluation.get("primary_metric")
                if isinstance(evaluation, Mapping)
                else None
            )
            summaries.append(
                {
                    "scientific_id": scientific_identifier,
                    "repro_id": None,
                    "campaign_id": pending_campaign,
                    "n_expected_runs": count,
                    "n_completed_runs": 0,
                    "n_failed_runs": 0,
                    "n_failed_attempts": 0,
                    "primary_metric_name": primary_name,
                    "primary_direction": _primary_direction(config),
                    "primary_mean": None,
                    "primary_std": None,
                    "primary_min": None,
                    "primary_max": None,
                    "auprc_mean": None,
                    "brier_mean": None,
                    "median_duration": None,
                    "maximum_peak_vram": None,
                    "best_run_id": None,
                    "worst_run_id": None,
                }
            )
        summaries.sort(
            key=lambda item: (
                str(item["campaign_id"]),
                str(item["scientific_id"]),
                str(item["repro_id"] or ""),
            )
        )
        return summaries

    def verify_artifacts(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        where = ""
        if run_id is not None:
            where = "WHERE a.run_id = ?"
            parameters.append(run_id)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT a.*, r.status AS run_status
                FROM artifacts a JOIN runs r ON r.run_id = a.run_id
                {where}
                ORDER BY a.run_id, a.artifact_id
                """,
                parameters,
            ).fetchall()
            runs = connection.execute(
                (
                    """
                    SELECT r.run_id, r.status, r.artifact_path,
                           EXISTS(
                               SELECT 1 FROM artifacts a
                               WHERE a.run_id = r.run_id
                                 AND a.kind = 'legacy_manifest'
                           ) AS is_legacy_native
                           , EXISTS(
                               SELECT 1 FROM artifacts a
                               WHERE a.run_id = r.run_id
                                 AND a.kind = 'deferred_stale_scratch'
                                 AND a.status = 'deferred'
                           ) AS is_deferred_stale
                    FROM runs r
                    """
                    + ("WHERE r.run_id = ?" if run_id is not None else "")
                ),
                parameters,
            ).fetchall()
        issues: list[dict[str, Any]] = []
        for row in rows:
            path = Path(str(row["path"]))
            if not path.exists():
                issues.append(
                    {"run_id": row["run_id"], "path": str(path), "issue": "missing"}
                )
                continue
            if row["size_bytes"] is not None and path.is_file():
                if path.stat().st_size != int(row["size_bytes"]):
                    issues.append(
                        {
                            "run_id": row["run_id"],
                            "path": str(path),
                            "issue": "size_mismatch",
                        }
                    )
            if row["sha256"] and path.is_file():
                if _sha256_file(path) != str(row["sha256"]):
                    issues.append(
                        {
                            "run_id": row["run_id"],
                            "path": str(path),
                            "issue": "checksum_mismatch",
                        }
                    )
        marker_for_status = {
            "completed": "_SUCCESS",
            "failed": "_FAILED",
            "pruned": "_PRUNED",
        }
        for row in runs:
            if bool(row["is_legacy_native"]) or bool(row["is_deferred_stale"]):
                # Legacy bundles retain their native manifest/checksum contract;
                # they are never rewritten merely to add canonical markers.
                continue
            marker = marker_for_status.get(str(row["status"]))
            artifact_path = row["artifact_path"]
            if marker and artifact_path and not (Path(str(artifact_path)) / marker).is_file():
                issues.append(
                    {
                        "run_id": row["run_id"],
                        "path": str(artifact_path),
                        "issue": f"missing_{marker}",
                    }
                )
        return issues


def _experiment_fields(configuration: Mapping[str, Any]) -> dict[str, Any]:
    model = configuration.get("model", {})
    masking = configuration.get("masking", {})
    dataset = configuration.get("dataset", {})
    features = configuration.get("features", {})
    graph = configuration.get("graph", {})
    trainer = configuration.get("trainer", {})
    evaluation = configuration.get("evaluation", {})
    return {
        "model_family": (
            model.get("family", model.get("name")) if isinstance(model, Mapping) else None
        ),
        "masking_type": masking.get("type") if isinstance(masking, Mapping) else None,
        "dataset_id": (
            dataset.get("dataset_id") if isinstance(dataset, Mapping) else None
        ),
        "dataset_version": (
            dataset.get("version") if isinstance(dataset, Mapping) else None
        ),
        "split_id": dataset.get("split_id") if isinstance(dataset, Mapping) else None,
        "use_edge_features": (
            features.get("use_edge_features")
            if isinstance(features, Mapping)
            else None
        ),
        "embedding_dim": (
            model.get("embedding_dim") if isinstance(model, Mapping) else None
        ),
        "neighbor_k": graph.get("neighbor_k") if isinstance(graph, Mapping) else None,
        "learning_rate": (
            trainer.get("learning_rate") if isinstance(trainer, Mapping) else None
        ),
        "batch_size": (
            trainer.get("batch_size") if isinstance(trainer, Mapping) else None
        ),
        "preprocessing_version": (
            dataset.get("preprocessing_version")
            if isinstance(dataset, Mapping)
            else None
        ),
        "dataset_fingerprint": (
            dataset.get("dataset_fingerprint")
            if isinstance(dataset, Mapping)
            else None
        ),
        "split_fingerprint": (
            dataset.get("split_fingerprint")
            if isinstance(dataset, Mapping)
            else None
        ),
        "primary_metric_name": (
            evaluation.get("primary_metric")
            if isinstance(evaluation, Mapping)
            else None
        ),
    }


def _primary_direction(config_text: str) -> str:
    try:
        config = json.loads(config_text)
        evaluation = config.get("evaluation", {})
        value = evaluation.get("primary_direction", "maximize")
        return str(value)
    except (TypeError, ValueError, AttributeError):
        return "maximize"


def _job_id(configuration: Mapping[str, Any]) -> str:
    payload = f"{_json(configuration)}\0{utc_now()}".encode("utf-8")
    return "q_" + hashlib.sha256(payload).hexdigest()[:20]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _execution_integer(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result >= 0 else default


def _required_registry_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _optional_registry_text(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "QUEUE_STATUSES",
    "RUN_STATUSES",
    "SCHEMA_VERSION",
    "InvalidTransitionError",
    "Registry",
    "RegistryConflictError",
    "RegistryError",
    "utc_now",
]
