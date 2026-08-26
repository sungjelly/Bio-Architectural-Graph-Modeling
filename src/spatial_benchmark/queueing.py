"""Persistent one-GPU queue worker for BAGM.

The worker owns one project lock, claims one SQLite job atomically, starts one
fresh subprocess, and never changes scientific parameters during a retry.
Every claimed execution receives a unique run ID and canonical scratch bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoint_catalog import (
    index_checkpoint_catalog,
    run_semantics_from_configuration,
)
from .configuration import (
    ConfigurationError,
    load_yaml_mapping,
    validate_experiment_config,
)
from .identifiers import (
    canonical_json,
    canonical_sha256,
    create_run_id,
    repro_id,
    scientific_id,
)
from .paths import ProjectPaths, current_paths
from .registry import Registry, RegistryError, utc_now
from .run_archive import (
    COMPLETION_MARKERS,
    RunArchive,
    RunArchiveError,
    deidentify_prediction_rows,
    verify_run_bundle,
    verify_unmarked_run_bundle,
)


FAILURE_CATEGORIES = frozenset(
    {
        "cuda_out_of_memory",
        "nonfinite_loss",
        "nonzero_exit",
        "missing_heartbeat",
        "insufficient_disk_space",
        "missing_dataset",
        "invalid_configuration",
        "artifact_finalization_failure",
        "interrupted",
        "unknown",
    }
)


class QueueWorkerError(RuntimeError):
    """Base class for queue worker errors."""


class WorkerLockError(QueueWorkerError):
    """Raised when another one-GPU worker owns the project lock."""


class InsufficientDiskSpaceError(QueueWorkerError):
    """Raised before claiming when the configured free-space floor is violated."""


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    """Configuration for one persistent local worker."""

    worker_id: str
    gpu: str | None = "0"
    poll_seconds: float = 5.0
    heartbeat_seconds: float = 30.0
    stale_after_seconds: float = 900.0
    min_free_gb: float = 25.0
    once: bool = False
    auto_retry: bool = False
    allow_test_jobs: bool = False
    parallel_gpu_workers: bool = False

    def validate(self) -> None:
        if not self.worker_id.strip():
            raise ValueError("worker_id must be non-empty.")
        for name in ("poll_seconds", "heartbeat_seconds", "stale_after_seconds"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.min_free_gb < 0:
            raise ValueError("min_free_gb must be non-negative.")
        if self.parallel_gpu_workers:
            if self.gpu is None or not str(self.gpu).strip():
                raise ValueError(
                    "parallel_gpu_workers requires one explicit GPU."
                )
            if "," in str(self.gpu):
                raise ValueError(
                    "parallel_gpu_workers requires exactly one GPU index."
                )


class ProjectWorkerLock:
    """Non-blocking advisory lock for exactly one project worker."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: Any = None

    def __enter__(self) -> "ProjectWorkerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            raise WorkerLockError(
                f"Another BAGM worker owns {self.path}."
            ) from error
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "acquired_at": utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def classify_failure(
    *,
    return_code: int | None = None,
    output: str = "",
    exception: BaseException | None = None,
    finalization: bool = False,
    interrupted: bool = False,
) -> str:
    """Classify an execution failure without altering retry parameters."""

    if interrupted:
        return "interrupted"
    if finalization or isinstance(exception, RunArchiveError):
        return "artifact_finalization_failure"
    if isinstance(exception, InsufficientDiskSpaceError):
        return "insufficient_disk_space"
    if isinstance(exception, ConfigurationError):
        return "invalid_configuration"
    if isinstance(exception, (FileNotFoundError, MissingDatasetError)):
        return "missing_dataset"
    lowered = output.lower()
    if (
        "cuda out of memory" in lowered
        or "cudnn_status_alloc_failed" in lowered
        or "cuda error: out of memory" in lowered
    ):
        return "cuda_out_of_memory"
    if any(
        token in lowered
        for token in (
            "loss is nan",
            "loss became nan",
            "non-finite loss",
            "nonfinite loss",
            "loss is inf",
            "infinite loss",
        )
    ):
        return "nonfinite_loss"
    if return_code not in (None, 0):
        return "nonzero_exit"
    return "unknown"


class MissingDatasetError(QueueWorkerError):
    """Raised when a registered protected dataset reference is unavailable."""


def _analysis_only_artifact_contract(configuration: Mapping[str, Any]) -> bool:
    evaluation = configuration.get("evaluation", {})
    return bool(
        isinstance(evaluation, Mapping)
        and evaluation.get("artifact_contract") == "analysis_only"
    )


def _resolve_prepared_artifact_reference(
    reference: object,
    *,
    paths: ProjectPaths,
) -> Path:
    """Resolve a registered prepared input against its owning BAGM root.

    Absolute references remain absolute. Relative references beginning with
    ``data/`` are rooted at ``BAGM_DATA_ROOT`` (with that leading component
    removed); all other legacy references retain project-root semantics. Both
    relative forms are prevented from escaping their selected root.
    """

    raw = Path(str(reference)).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    if not raw.parts:
        raise ConfigurationError(
            "dataset.prepared_artifact_reference cannot be empty."
        )
    if raw.parts[0] == "data":
        root = paths.data_root.resolve(strict=False)
        relative = Path(*raw.parts[1:])
    else:
        root = paths.project_root.resolve(strict=False)
        relative = raw
    candidate = (root / relative).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ConfigurationError(
            "dataset.prepared_artifact_reference escapes its configured root."
        )
    return candidate


def command_for_config(
    configuration: Mapping[str, Any],
    *,
    paths: ProjectPaths | None = None,
) -> list[str]:
    """Build the current spatial benchmark command from a resolved config.

    A launcher ``command`` list is permitted only for explicitly test-only
    fixtures. Scientific jobs are derived from the validated configuration so
    arbitrary argv cannot change behavior without changing the identifiers.
    The output placeholder is rendered by the worker.
    """

    selected_paths = paths or current_paths()
    launcher = configuration.get("launcher", {})
    if isinstance(launcher, Mapping) and launcher.get("command") is not None:
        metadata = configuration.get("metadata", {})
        if not (
            isinstance(metadata, Mapping)
            and bool(metadata.get("test_only_dummy"))
        ):
            raise ConfigurationError(
                "launcher.command is restricted to test_only_dummy fixtures; "
                "scientific commands are derived from resolved configuration."
            )
        command = launcher["command"]
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
            raise ConfigurationError("launcher.command must be a list of arguments.")
        result = [str(part) for part in command]
        if not result or any(not part for part in result):
            raise ConfigurationError("launcher.command cannot contain empty values.")
        return result

    evaluation = _section(configuration, "evaluation")
    protocol = str(evaluation.get("protocol", "")).strip().lower()
    if protocol == "posthoc_attention_routing_niche_v1":
        model = _section(configuration, "model")
        campaign = _section(configuration, "campaign")
        if (
            str(model.get("name", "")).strip().lower() != "relative-qkv-gat"
            or campaign.get("campaign_id")
            != "cmp_20260825_six_core_attention_routing_niches"
            or evaluation.get("artifact_contract") != "analysis_only"
        ):
            raise ConfigurationError(
                "posthoc_attention_routing_niche_v1 requires the registered "
                "relative-QKV analysis-only campaign."
            )
        script = (
            selected_paths.project_root
            / "scripts/analysis/run_attention_routing_niches.py"
        )
        return [
            sys.executable,
            str(script),
            "--config",
            "{run_scratch}/config.resolved.yaml",
            "--run-id",
            "{run_id}",
            "--run-scratch",
            "{run_scratch}",
        ]
    if protocol == "grouped_core_adjacency_ablation_v1":
        model = _section(configuration, "model")
        model_name = str(model.get("name", "")).strip().lower()
        if model_name != "mean-adjacency-sage":
            raise ConfigurationError(
                "grouped_core_adjacency_ablation_v1 requires "
                "model.name=mean-adjacency-sage."
            )
        script = (
            selected_paths.project_root
            / "scripts/train/run_adjacency_ablation.py"
        )
        return [
            sys.executable,
            str(script),
            "--config",
            "{run_scratch}/config.resolved.yaml",
            "--run-scratch",
            "{run_scratch}",
        ]
    if protocol == "held_in_pooled_10core_fixed_budget":
        model = _section(configuration, "model")
        model_name = str(model.get("name", "")).strip().lower()
        if model_name not in {
            "hybrid-count-gat",
            "hybrid-count-matched-self",
            "myjju-genemae",
        }:
            raise ConfigurationError(
                "held_in_pooled_10core_fixed_budget requires a supported "
                "pooled ten-core model."
            )
        runner = (
            "run_myjju_genemae_pooled.py"
            if model_name == "myjju-genemae"
            else "run_pooled_hybrid_count_capacity.py"
        )
        script = selected_paths.project_root / "scripts/train" / runner
        return [
            sys.executable,
            str(script),
            "--config",
            "{run_scratch}/config.resolved.yaml",
            "--run-scratch",
            "{run_scratch}",
        ]
    if protocol in {
        "held_in_pooled_6core_relative_qkv_fixed_budget",
        "held_in_pooled_6core_relative_qkv_joint_plateau",
        "held_in_pooled_6core_relative_qkv_seed_plateau",
    }:
        model = _section(configuration, "model")
        model_name = str(model.get("name", "")).strip().lower()
        if model_name != "relative-qkv-gat":
            raise ConfigurationError(
                f"{protocol} requires "
                "model.name=relative-qkv-gat."
            )
        script = (
            selected_paths.project_root
            / "scripts/train/run_pooled_relative_qkv.py"
        )
        return [
            sys.executable,
            str(script),
            "--config",
            "{run_scratch}/config.resolved.yaml",
            "--run-scratch",
            "{run_scratch}",
        ]
    if protocol in {
        "held_in_pooled_14core_relative_qkv_seed_plateau",
        "held_in_pooled_14core_relative_qkv_fixed_continuation_epoch300",
    }:
        model = _section(configuration, "model")
        campaign = _section(configuration, "campaign")
        launcher = _section(configuration, "launcher")
        if (
            str(model.get("name", "")).strip().lower() != "relative-qkv-gat"
            or campaign.get("campaign_id")
            != "cmp_20260825_so2_14core_relative_qkv_seed0_batch2"
            or launcher.get("requested_gpu") != "0,1,2,3"
            or launcher.get("process_count") != 4
            or launcher.get("elastic_max_restarts") != 0
        ):
            raise ConfigurationError(
                "The SO2 14-core protocol requires its registered four-rank "
                "Relative-QKV campaign with GPUs 0,1,2,3 and no elastic restarts."
            )
        script = (
            selected_paths.project_root
            / "scripts/train/run_so2_14core_relative_qkv.py"
        )
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            "--max-restarts=0",
            str(script),
            "--config",
            "{run_scratch}/config.resolved.yaml",
            "--run-scratch",
            "{run_scratch}",
        ]
    if protocol == "held_in_pooled_so1_14core_relative_qkv_plateau_min150":
        model = _section(configuration, "model")
        campaign = _section(configuration, "campaign")
        launcher = _section(configuration, "launcher")
        if (
            str(model.get("name", "")).strip().lower() != "relative-qkv-gat"
            or campaign.get("campaign_id")
            != "cmp_20260826_so1_14core_relative_qkv_seed0_batch2_plateau_min150"
            or launcher.get("requested_gpu") != "0,1,2,3"
            or launcher.get("process_count") != 4
            or launcher.get("elastic_max_restarts") != 0
        ):
            raise ConfigurationError(
                "The SO1 14-core protocol requires its registered four-rank "
                "Relative-QKV campaign with GPUs 0,1,2,3 and no elastic restarts."
            )
        script = (
            selected_paths.project_root
            / "scripts/train/run_so1_14core_relative_qkv.py"
        )
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            "--max-restarts=0",
            str(script),
            "--config",
            "{run_scratch}/config.resolved.yaml",
            "--run-scratch",
            "{run_scratch}",
        ]
    if protocol == "held_in_full_core_fixed_budget":
        model_name = ""
        if "model" in configuration:
            model = _section(configuration, "model")
            model_name = str(model.get("name", "")).strip().lower()
        campaign = configuration.get("campaign", {})
        metadata = configuration.get("metadata", {})
        synthetic_recovery = (
            isinstance(campaign, Mapping)
            and campaign.get("campaign_id")
            == "cmp_20260729_multiscale_hurdle_count_pilot"
            and model_name == "multiscale-hurdle-count"
            and isinstance(metadata, Mapping)
            and metadata.get("execution_role")
            == "stage0_synthetic_recovery"
        )
        if synthetic_recovery:
            script = (
                selected_paths.project_root
                / "scripts"
                / "diagnostics"
                / "run_multiscale_synthetic_recovery.py"
            )
        else:
            runner_name = (
                "run_hybrid_count_capacity.py"
                if model_name
                in {"hybrid-count-gat", "hybrid-count-matched-self"}
                else (
                    "run_multiscale_hurdle_capacity.py"
                    if model_name == "multiscale-hurdle-count"
                    else (
                        "run_self_hurdle_capacity.py"
                        if model_name == "self-hurdle-count"
                        else "run_full_core_capacity.py"
                    )
                )
            )
            script = (
                selected_paths.project_root
                / "scripts/train"
                / runner_name
            )
        return [
            sys.executable,
            str(script),
            "--config",
            "{run_scratch}/config.resolved.yaml",
            "--run-scratch",
            "{run_scratch}",
        ]

    dataset = _section(configuration, "dataset")
    model = _section(configuration, "model")
    masking = _section(configuration, "masking")
    graph = _section(configuration, "graph")
    trainer = _section(configuration, "trainer")
    prepared = dataset.get("prepared_artifact_reference")
    if not prepared:
        raise ConfigurationError(
            "dataset.prepared_artifact_reference is required for the "
            "spatial benchmark launcher."
        )
    prepared_path = _resolve_prepared_artifact_reference(
        prepared,
        paths=selected_paths,
    )
    script = selected_paths.project_root / "scripts/train/run_spatial_benchmark.py"
    command = [
        sys.executable,
        str(script),
        "--prepared",
        str(prepared_path),
        "--output",
        "{run_scratch}/training_output",
        "--model",
        str(model["name"]),
        "--seed",
        str(int(configuration.get("seed", 0))),
        "--device",
        "cuda",
    ]
    optional: tuple[tuple[Mapping[str, Any], str, str], ...] = (
        (graph, "k", "--k"),
        (graph, "radius_um", "--radius-um"),
        (graph, "symmetry", "--symmetry"),
        (graph, "min_distance_um", "--min-distance-um"),
        (model, "hidden_dim", "--hidden-dim"),
        (model, "graph_layers", "--graph-layers"),
        (model, "edge_embedding_dim", "--edge-embedding-dim"),
        (masking, "curriculum", "--curriculum"),
        (trainer, "max_epochs", "--max-epochs"),
        (trainer, "early_stopping_patience", "--patience"),
        (trainer, "learning_rate", "--learning-rate"),
        (graph, "edge_dropout", "--edge-dropout"),
    )
    for section, key, option in optional:
        if section.get(key) is not None:
            command.extend([option, str(section[key])])
    if trainer.get("amp") is True:
        command.append("--amp")
    elif trainer.get("amp") is False:
        command.append("--no-amp")
    return command


class QueueWorker:
    """One-lock, one-subprocess queue runner.

    The backward-compatible default takes the project-global worker lock.
    Explicit GPU-parallel workers instead take one deterministic lock per
    physical GPU, so two workers cannot intentionally claim the same device
    while independent registry-backed jobs can use distinct devices.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        settings: WorkerSettings,
        paths: ProjectPaths | None = None,
    ) -> None:
        settings.validate()
        self.registry = registry
        self.settings = settings
        self.paths = paths or current_paths()
        lock_name = "bagm-worker.lock"
        if settings.parallel_gpu_workers:
            gpu_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(settings.gpu))
            lock_name = f"bagm-worker-gpu-{gpu_key}.lock"
        self.lock_path = self.paths.state_root / "locks" / lock_name
        self._stop = threading.Event()
        self._child: subprocess.Popen[bytes] | None = None
        self._child_record: Path | None = None

    def request_stop(self) -> None:
        self._stop.set()
        child = self._child
        if child is not None and child.poll() is None:
            child.terminate()

    def _signal_handler(self, _signum: int, _frame: object) -> None:
        self.request_stop()

    def run(self) -> int:
        """Run until signalled, or process one job when ``once`` is set."""

        old_handlers: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal_handler)
        try:
            with ProjectWorkerLock(self.lock_path):
                self._assert_no_surviving_children()
                self._reconcile_finalizing_runs()
                self._mark_stale_jobs()
                processed = 0
                while not self._stop.is_set():
                    self._assert_disk_space()
                    job = self.registry.claim_next(
                        self.settings.worker_id,
                        requested_gpu=self.settings.gpu,
                    )
                    if job is None:
                        if self.settings.once:
                            break
                        self._stop.wait(self.settings.poll_seconds)
                        continue
                    self.execute_claimed(job)
                    self._reconcile_finalizing_runs()
                    processed += 1
                    if self.settings.once:
                        break
                return processed
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)

    def execute_claimed(self, job: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one already-claimed job and return its final queue record."""

        job_id = str(job["job_id"])
        if job.get("worker_id") != self.settings.worker_id:
            raise QueueWorkerError(
                f"Claimed job {job_id} does not belong to this worker."
            )
        configuration = dict(job["canonical_config"])
        # attempt is an execution dimension: retries preserve the scientific
        # configuration but the resolved run configuration must state the
        # actual attempt that produced its artifacts.
        configuration["attempt"] = int(job["attempt_count"])
        archive: RunArchive | None = None
        run_id: str | None = None
        published_path: Path | None = None
        success_marked = False
        started_monotonic = time.monotonic()
        try:
            scientific_identifier = scientific_id(configuration)
            self.registry.register_variant(
                scientific_identifier,
                campaign_id=str(job["campaign_id"]),
                configuration=configuration,
            )
            provenance = _provenance(configuration, self.paths.project_root)
            reproduction_identifier = repro_id(
                configuration,
                git_commit=provenance["git_commit"],
                dirty_fingerprint=provenance["dirty_fingerprint"],
                dataset_fingerprint=provenance["dataset_fingerprint"],
                split_fingerprint=provenance["split_fingerprint"],
                preprocessing_version=provenance["preprocessing_version"],
                environment_fingerprint=provenance["environment_fingerprint"],
            )
            run_id = create_run_id(
                seed=_execution_int(configuration.get("seed"), 0),
                fold=_execution_int(configuration.get("fold"), 0),
                attempt=int(job["attempt_count"]),
                scientific_id_value=scientific_identifier,
            )
            archive = RunArchive.create(
                run_id,
                paths=self.paths,
                resolved_config=configuration,
            )
            stdout_path, stderr_path = archive.prepare_log_files()
            command = _render_command(
                job["command"],
                run_id=run_id,
                run_scratch=archive.scratch_path,
                artifact_dir=archive.artifact_path,
                project_root=self.paths.project_root,
            )
            self._write_provenance(
                archive,
                provenance=provenance,
                command=command,
                configuration=configuration,
                job=job,
            )
            self.registry.register_run_for_job(
                job_id,
                run_id=run_id,
                scientific_id=scientific_identifier,
                repro_id=reproduction_identifier,
                artifact_path=archive.artifact_path,
                resolved_configuration=configuration,
                start_time=utc_now(),
                host=socket.gethostname(),
                gpu_model=provenance.get("gpu_model"),
                git_commit=provenance["git_commit"],
                dirty_status=provenance["dirty_fingerprint"] is not None,
            )
            run_semantics = run_semantics_from_configuration(
                primary_run_id=run_id,
                configuration=configuration,
                scientific_id_value=scientific_identifier,
            )
            self.registry.register_run_semantics(run_id, **run_semantics)
            self.registry.transition_run(run_id, "running", start_time=utc_now())
            # Validation occurs after the run record/scratch bundle exist so an
            # invalid configuration or missing dataset remains a preserved,
            # uniquely identified failed attempt.
            self._validate_job(
                configuration,
                queued_command=job["command"],
                requested_gpu=job.get("requested_gpu"),
            )
            environment = self._subprocess_environment(
                job=job,
                run_id=run_id,
                archive=archive,
                configuration=configuration,
            )
            with stdout_path.open("ab", buffering=0) as stdout_handle, stderr_path.open(
                "ab", buffering=0
            ) as stderr_handle:
                self._child = subprocess.Popen(
                    command,
                    cwd=self.paths.project_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                    start_new_session=False,
                )
                try:
                    self._record_child_process(
                        job_id=job_id,
                        run_id=run_id,
                        command=command,
                    )
                except BaseException:
                    self._child.terminate()
                    self._child.wait()
                    raise
                try:
                    return_code = self._wait_with_heartbeat(job_id)
                except BaseException:
                    if self._child.poll() is None:
                        self._child.terminate()
                        self._child.wait()
                    observed_return_code = self._child.returncode
                    self._archive_child_record(
                        return_code=(
                            int(observed_return_code)
                            if observed_return_code is not None
                            else -int(signal.SIGTERM)
                        )
                    )
                    raise
                self._archive_child_record(return_code=return_code)
            self._child = None
            if self._stop.is_set():
                raise InterruptedError("Worker received a shutdown signal.")
            if return_code != 0:
                tail = _output_tail(stdout_path, stderr_path)
                category = classify_failure(
                    return_code=return_code,
                    output=tail,
                )
                raise ProcessExecutionError(
                    f"Experiment subprocess exited with status {return_code}.",
                    category=category,
                    return_code=return_code,
                )

            self._adapt_spatial_benchmark_output(
                archive, command, configuration=configuration
            )
            self._write_test_dummy_outputs(archive, configuration)
            if not (archive.scratch_path / "summary.json").exists():
                archive.write_summary(
                    {
                        "run_id": run_id,
                        "training_exit_status": "success",
                        "return_code": 0,
                    }
                )
            scratch_summary = _read_json_mapping(
                archive.scratch_path / "summary.json"
            )
            metric_name, metric_value = _primary_metric(
                scratch_summary, configuration
            )
            analysis_only = _analysis_only_artifact_contract(configuration)
            archive.write_manifest(
                {
                    "run_id": run_id,
                    "campaign_id": job["campaign_id"],
                    "job_id": job_id,
                    "scientific_id": scientific_identifier,
                    "repro_id": reproduction_identifier,
                    "lifecycle_status_source": "registry_and_completion_marker",
                    "schema_version": 1,
                    "primary_metric_name": metric_name,
                    "primary_metric_value": metric_value,
                    "artifact_roles": (
                        {
                            "analysis_outputs": "conclusion_bearing_posthoc_readout",
                            "primary_checkpoint": None,
                            "canonical_predictions": None,
                        }
                        if analysis_only
                        else {
                            "primary_checkpoint": str(
                                _section(configuration, "trainer").get(
                                    "primary_checkpoint_role", "best"
                                )
                            ),
                            "canonical_predictions": str(
                                _section(configuration, "evaluation").get(
                                    "canonical_prediction_split", "validation"
                                )
                            ),
                        }
                    ),
                    "checkpoint_catalog": {
                        "schema_version": 1,
                        "semantic_alias": run_semantics["semantic_alias"],
                        "lifecycle_stage": run_semantics["lifecycle_stage"],
                        "study_axis": run_semantics["study_axis"],
                        "retention_class": run_semantics["retention_class"],
                        "category_key": run_semantics["category_key"],
                        "classification_confidence": run_semantics[
                            "classification_confidence"
                        ],
                    },
                }
            )
            try:
                artifact_path = archive.publish_success_pending()
                published_path = artifact_path
            except BaseException as error:
                raise ArtifactFinalizationError(str(error)) from error

            duration = time.monotonic() - started_monotonic
            artifact_records = self._bundle_artifact_records(artifact_path)
            self.registry.begin_run_and_job_finalization(
                job_id=job_id,
                run_id=run_id,
                worker_id=self.settings.worker_id,
                artifact_path=artifact_path,
                end_time=utc_now(),
                duration_seconds=duration,
                primary_metric_name=metric_name,
                primary_metric_value=metric_value,
                peak_vram_gb=_peak_vram_from_summary(scratch_summary),
                parameter_count=_optional_nonnegative_integer(
                    scratch_summary.get("parameter_count")
                ),
                artifacts=artifact_records,
            )
            if not analysis_only:
                index_checkpoint_catalog(
                    self.registry,
                    self.paths,
                    run_reference=run_id,
                    verify=True,
                )
            archive.mark_success()
            success_marked = True
            verify_run_bundle(artifact_path)
            self.registry.complete_run_and_job(
                job_id=job_id,
                run_id=run_id,
                worker_id=self.settings.worker_id,
            )
        except BaseException as error:
            self._child = None
            category = (
                error.category
                if isinstance(error, ProcessExecutionError)
                else classify_failure(
                    exception=error,
                    finalization=isinstance(error, ArtifactFinalizationError),
                    interrupted=isinstance(error, InterruptedError),
                )
            )
            final_path: Path | None = None
            trace = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            if (
                archive is not None
                and run_id is not None
                and published_path is not None
                and not success_marked
            ):
                try:
                    self.registry.fail_run_and_job_finalization(
                        job_id=job_id,
                        run_id=run_id,
                        worker_id=self.settings.worker_id,
                        artifact_path=published_path,
                        failure_category="artifact_finalization_failure",
                        last_error=str(error),
                        duration_seconds=time.monotonic() - started_monotonic,
                    )
                    archive.mark_published_failure()
                except BaseException as compensation_error:
                    trace += (
                        "\nFinalization compensation also failed:\n"
                        + "".join(
                            traceback.format_exception(
                                type(compensation_error),
                                compensation_error,
                                compensation_error.__traceback__,
                            )
                        )
                    )
            if archive is not None and archive.scratch_path.is_dir():
                try:
                    final_path = archive.finalize_failure(
                        error,
                        failure_category=category,
                        traceback_text=trace,
                    )
                    if run_id is not None:
                        self._record_bundle_artifacts(run_id, final_path)
                except BaseException as final_error:
                    trace += (
                        "\nFailure bundle finalization also failed:\n"
                        + "".join(
                            traceback.format_exception(
                                type(final_error),
                                final_error,
                                final_error.__traceback__,
                            )
                        )
                    )
            run_snapshot = (
                self.registry.get_run(run_id) if run_id is not None else None
            )
            recoverable_finalizing = bool(
                success_marked
                and run_snapshot is not None
                and run_snapshot["status"] == "finalizing"
            )
            if run_id is not None and run_snapshot is not None:
                run = self.registry.get_run(run_id)
                if run and run["status"] in {"pending", "running"}:
                    self.registry.transition_run(
                        run_id,
                        "failed",
                        end_time=utc_now(),
                        duration_seconds=time.monotonic() - started_monotonic,
                        artifact_path=(
                            final_path
                            or (archive.scratch_path if archive is not None else None)
                        ),
                        failure_category=category,
                    )
            if not recoverable_finalizing:
                self.registry.record_failure(
                    category=category,
                    message=str(error),
                    run_id=run_id,
                    job_id=job_id,
                    details={"exception_type": type(error).__name__},
                )
            latest = self.registry.get_job(job_id)
            if (
                not recoverable_finalizing
                and latest
                and latest["status"] in {"claimed", "running"}
            ):
                self.registry.transition_job(
                    job_id,
                    "failed",
                    worker_id=self.settings.worker_id,
                    run_id=run_id,
                    failure_category=category,
                    last_error=str(error),
                )
            if (
                not recoverable_finalizing
                and
                self.settings.auto_retry
                and int(job["attempt_count"]) < int(job["maximum_attempts"])
            ):
                self.registry.create_retry(job_id)
        result = self.registry.get_job(job_id)
        assert result is not None
        return result

    def _validate_job(
        self,
        configuration: Mapping[str, Any],
        *,
        queued_command: Sequence[str],
        requested_gpu: Any,
    ) -> None:
        metadata = configuration.get("metadata", {})
        is_test = isinstance(metadata, Mapping) and bool(
            metadata.get("test_only_dummy")
        )
        if is_test and not self.settings.allow_test_jobs:
            raise ConfigurationError(
                "Test-only dummy jobs require --allow-test-jobs."
            )
        if requested_gpu is not None and str(requested_gpu) != str(
            self.settings.gpu
        ):
            raise ConfigurationError(
                f"Job requests GPU {requested_gpu!r}, but worker is bound to "
                f"{self.settings.gpu!r}."
            )
        validate_experiment_config(configuration)
        if not is_test:
            expected_command = command_for_config(configuration, paths=self.paths)
            if list(queued_command) != expected_command:
                raise ConfigurationError(
                    "Queued argv does not match the command derived from the "
                    "resolved scientific configuration."
                )
        dataset = _section(configuration, "dataset")
        record = self.registry.get_dataset(
            str(dataset["dataset_id"]), str(dataset["version"])
        )
        if record is None:
            raise MissingDatasetError(
                f"Dataset is not registered: {dataset['dataset_id']}/"
                f"{dataset['version']}"
            )
        configured_dataset_fingerprint = str(
            dataset.get("dataset_fingerprint", "")
        )
        registered_dataset_fingerprints = {
            str(value)
            for value in (
                record.get("processed_fingerprint"),
                record.get("raw_fingerprint"),
            )
            if value
        }
        if (
            not configured_dataset_fingerprint
            or configured_dataset_fingerprint not in registered_dataset_fingerprints
        ):
            raise ConfigurationError(
                "Configuration dataset_fingerprint does not match the "
                "authoritative dataset registry."
            )
        if str(dataset.get("preprocessing_version", "")) != str(
            record.get("preprocessing_version", "")
        ):
            raise ConfigurationError(
                "Configuration preprocessing_version does not match the "
                "authoritative dataset registry."
            )
        split = self.registry.get_split(str(dataset["split_id"]))
        if split is None:
            raise MissingDatasetError(
                f"Split is not registered: {dataset['split_id']}"
            )
        if (
            split["dataset_id"] != dataset["dataset_id"]
            or split["dataset_version"] != dataset["version"]
        ):
            raise ConfigurationError(
                "Registered split does not belong to the configured dataset/version."
            )
        if str(dataset.get("split_fingerprint", "")) != str(
            split.get("fingerprint", "")
        ):
            raise ConfigurationError(
                "Configuration split_fingerprint does not match the "
                "authoritative split registry."
            )
        reference = dataset.get("prepared_artifact_reference")
        if reference and not is_test:
            path = _resolve_prepared_artifact_reference(
                reference,
                paths=self.paths,
            )
            if not path.exists():
                raise MissingDatasetError(
                    f"Prepared dataset artifact does not exist: {path}"
                )

    def _assert_disk_space(self) -> None:
        target = self.paths.scratch_root
        target.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target).free
        required = int(self.settings.min_free_gb * (1024**3))
        if free < required:
            raise InsufficientDiskSpaceError(
                f"Free disk {free / (1024**3):.2f} GiB is below "
                f"{self.settings.min_free_gb:.2f} GiB; no job was claimed."
            )

    def _record_child_process(
        self, *, job_id: str, run_id: str, command: Sequence[str]
    ) -> None:
        assert self._child is not None
        directory = self.paths.state_root / "pids"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{job_id}.json"
        payload = {
            "job_id": job_id,
            "run_id": run_id,
            "pid": self._child.pid,
            "proc_start_ticks": _process_start_ticks(self._child.pid),
            # Record the physical device selected by this worker.  Queue jobs
            # may be GPU-agnostic, so the worker binding is the authoritative
            # assignment for excluding unrelated live children when explicit
            # per-GPU workers are enabled.
            "requested_gpu": self.settings.gpu,
            "command_sha256": canonical_sha256(list(command)),
            "recorded_at": utc_now(),
        }
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise QueueWorkerError(
                f"Child process record already exists: {path}"
            ) from error
        self._child_record = path

    def _archive_child_record(self, *, return_code: int) -> None:
        path = self._child_record
        if path is None or not path.is_file():
            self._child_record = None
            return
        directory = self.paths.state_root / "logs" / "process_records"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{path.stem}-{int(time.time())}.json"
        if destination.exists():
            raise QueueWorkerError(
                f"Process-record archive collision: {destination}"
            )
        path.rename(destination)
        exit_path = destination.with_suffix(".exit.json")
        with exit_path.open("x", encoding="utf-8") as handle:
            json.dump(
                {"return_code": int(return_code), "observed_at": utc_now()},
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._child_record = None

    def _child_record_gpu(
        self, record: Mapping[str, Any], *, path: Path
    ) -> str:
        """Resolve the physical GPU assigned to a child-process record.

        Current records carry the worker's GPU directly.  Records written by
        older workers did not, so resolve those through their immutable queue
        job.  An absent or ambiguous assignment cannot safely be treated as a
        different GPU.
        """

        if "requested_gpu" in record:
            requested_gpu = record.get("requested_gpu")
        else:
            job_id = record.get("job_id")
            job = self.registry.get_job(str(job_id)) if job_id else None
            requested_gpu = job.get("requested_gpu") if job else None
        if requested_gpu is None or not str(requested_gpu).strip():
            raise QueueWorkerError(
                "Child-process record has no resolvable GPU assignment; "
                f"refusing a parallel worker: {path}"
            )
        gpu = str(requested_gpu)
        if "," in gpu:
            raise QueueWorkerError(
                "Child-process record has an ambiguous GPU assignment; "
                f"refusing a parallel worker: {path}"
            )
        return gpu

    def _assert_no_surviving_children(self) -> None:
        directory = self.paths.state_root / "pids"
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                pid = int(record["pid"])
                start_ticks = int(record["proc_start_ticks"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise QueueWorkerError(
                    f"Unreadable child-process record requires review: {path}"
                ) from error
            if self.settings.parallel_gpu_workers:
                record_gpu = self._child_record_gpu(record, path=path)
                if record_gpu != str(self.settings.gpu):
                    # Another per-GPU worker owns this record.  Do not block on
                    # its live process or move its exited-process evidence.
                    continue
            if _process_start_ticks(pid) == start_ticks:
                raise QueueWorkerError(
                    f"Recorded experiment child PID {pid} for job "
                    f"{record.get('job_id')} is still alive; refusing a new worker."
                )
            directory_out = self.paths.state_root / "logs" / "process_records"
            directory_out.mkdir(parents=True, exist_ok=True)
            destination = directory_out / f"orphan-exited-{path.name}"
            if destination.exists():
                raise QueueWorkerError(
                    f"Process-record archive collision: {destination}"
                )
            path.rename(destination)

    def _mark_stale_jobs(self) -> list[str]:
        identifiers = self.registry.mark_stale(
            heartbeat_before=self._stale_heartbeat_before()
        )
        for job_id in identifiers:
            job = self.registry.get_job(job_id)
            if not job or not job.get("run_id"):
                continue
            run_id = str(job["run_id"])
            scratch_path = self.paths.scratch_root / "active_runs" / run_id
            if scratch_path.is_dir():
                self.registry.record_deferred_stale_artifact(
                    run_id=run_id,
                    scratch_path=scratch_path,
                )
        return identifiers

    def _reconcile_finalizing_runs(self) -> list[str]:
        """Finish marker/registry transitions left by a hard worker crash."""

        reconciled: list[str] = []
        for record in self.registry.list_finalizing_runs(
            recoverer_worker_id=self.settings.worker_id,
            heartbeat_before=self._stale_heartbeat_before(),
        ):
            run_id = str(record["run_id"])
            artifact_path = Path(str(record["artifact_path"]))
            published_configuration = load_yaml_mapping(
                artifact_path / "config.resolved.yaml"
            )
            analysis_only = _analysis_only_artifact_contract(
                published_configuration
            )
            archive = RunArchive.from_published(run_id, paths=self.paths)
            markers = [
                marker
                for marker in COMPLETION_MARKERS
                if (artifact_path / marker).exists()
            ]
            if not markers:
                verify_unmarked_run_bundle(artifact_path)
                if not analysis_only:
                    index_checkpoint_catalog(
                        self.registry,
                        self.paths,
                        run_reference=run_id,
                        verify=True,
                    )
                archive.mark_success()
            elif markers != ["_SUCCESS"]:
                raise ArtifactFinalizationError(
                    f"Finalizing run {run_id} has incompatible markers: {markers}"
                )
            else:
                if not analysis_only:
                    index_checkpoint_catalog(
                        self.registry,
                        self.paths,
                        run_reference=run_id,
                        verify=True,
                    )
            verify_run_bundle(artifact_path)
            self.registry.complete_run_and_job(
                job_id=str(record["job_id"]),
                run_id=run_id,
            )
            reconciled.append(run_id)
        return reconciled

    def _stale_heartbeat_before(self) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.settings.stale_after_seconds
        )
        return cutoff.isoformat(timespec="microseconds").replace("+00:00", "Z")

    def _wait_with_heartbeat(self, job_id: str) -> int:
        assert self._child is not None
        next_heartbeat = time.monotonic() + self.settings.heartbeat_seconds
        termination_requested = False
        while True:
            code = self._child.poll()
            if code is not None:
                return int(code)
            if self._stop.is_set():
                if not termination_requested and self._child.poll() is None:
                    self._child.terminate()
                    termination_requested = True
                # Never finalize or move scratch while the child may still be
                # writing. Keep the worker lock and wait for an actual exit;
                # service policy intentionally does not escalate to SIGKILL.
                time.sleep(min(0.2, self.settings.heartbeat_seconds))
                continue
            self._stop.wait(min(0.2, self.settings.heartbeat_seconds))
            if time.monotonic() >= next_heartbeat:
                self.registry.heartbeat(job_id, self.settings.worker_id)
                next_heartbeat = time.monotonic() + self.settings.heartbeat_seconds

    def _subprocess_environment(
        self,
        *,
        job: Mapping[str, Any],
        run_id: str,
        archive: RunArchive,
        configuration: Mapping[str, Any],
    ) -> dict[str, str]:
        environment = dict(os.environ)
        selected_gpu = self.settings.gpu
        environment["CUDA_VISIBLE_DEVICES"] = (
            "" if selected_gpu is None else str(selected_gpu)
        )
        environment.update(
            {
                "BAGM_ROOT": str(self.paths.project_root),
                "BAGM_RUN_ID": run_id,
                "BAGM_RUN_SCRATCH": str(archive.scratch_path),
                "BAGM_ARTIFACT_DIR": str(archive.artifact_path),
                "BAGM_CONFIG_PATH": str(
                    archive.scratch_path / "config.resolved.yaml"
                ),
                "BAGM_JOB_ID": str(job["job_id"]),
                "PYTHONUNBUFFERED": "1",
            }
        )
        if len(canonical_json(configuration).encode("utf-8")) < 64 * 1024:
            environment["BAGM_CONFIG_JSON"] = canonical_json(configuration)
        return environment

    def _write_provenance(
        self,
        archive: RunArchive,
        *,
        provenance: Mapping[str, Any],
        command: Sequence[str],
        configuration: Mapping[str, Any],
        job: Mapping[str, Any],
    ) -> None:
        archive.write_json(
            "provenance/git.json",
            {
                "commit": provenance["git_commit"],
                "dirty": provenance["dirty_fingerprint"] is not None,
                "dirty_fingerprint": provenance["dirty_fingerprint"],
                "untracked_file_count": len(provenance["untracked_files"]),
            },
        )
        archive.write_text(
            "provenance/uncommitted_changes.patch",
            str(provenance["uncommitted_changes_patch"]),
        )
        archive.write_json(
            "provenance/untracked_files.json",
            {
                "files": provenance["untracked_files"],
                "note": (
                    "Hashes cover every non-ignored untracked file. Eligible "
                    "small source/configuration text is also embedded in the "
                    "uncommitted patch; protected/runtime paths are never copied."
                ),
            },
        )
        archive.write_text(
            "provenance/environment.txt",
            str(provenance["environment_text"]),
        )
        archive.write_json(
            "provenance/hardware.json",
            {
                "host": socket.gethostname(),
                "requested_gpu": job.get("requested_gpu"),
                "cuda_visible_devices": self.settings.gpu,
                **dict(provenance["hardware"]),
            },
        )
        archive.write_json(
            "provenance/data_fingerprints.json",
            {
                "dataset_fingerprint": provenance["dataset_fingerprint"],
                "preprocessing_version": provenance["preprocessing_version"],
            },
        )
        archive.write_json(
            "provenance/split_fingerprint.json",
            {"split_fingerprint": provenance["split_fingerprint"]},
        )
        archive.write_text(
            "provenance/command.txt",
            canonical_json(
                {"argv": list(command), "cwd": str(self.paths.project_root)}
            )
            + "\n",
        )
        archive.write_json(
            "provenance/queue.json",
            {
                "job_id": job["job_id"],
                "attempt": job["attempt_count"],
                "retry_of": job.get("retry_of"),
                "config_sha256": canonical_sha256(configuration),
            },
        )

    def _adapt_spatial_benchmark_output(
        self,
        archive: RunArchive,
        command: Sequence[str],
        *,
        configuration: Mapping[str, Any],
    ) -> None:
        """Convert the existing benchmark bundle into the canonical run bundle."""

        output = _option_value(command, "--output")
        if output is None:
            return
        root = Path(output)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            return
        from .experiment import load_run_manifest

        manifest = load_run_manifest(root)
        moves = {
            "model_state.pt": "checkpoints/best.ckpt",
            "metrics.json": "provenance/native_metrics.json",
            # The native NPZ contains cell/FOV/block routing identifiers and has
            # not passed the canonical salted sample-key validator. Preserve it
            # as restricted provenance, never as an exportable prediction table.
            "predictions.npz": "provenance/native_predictions.restricted.npz",
            "manifest.json": "provenance/legacy_training_manifest.json",
        }
        for source_name, relative in moves.items():
            source = root / source_name
            if not source.exists():
                continue
            destination = archive.scratch_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise RunArchiveError(
                    f"Refusing to overwrite adapted artifact: {destination}"
                )
            source.rename(destination)
        if root.is_dir() and not any(root.iterdir()):
            root.rmdir()
        restricted_predictions = (
            archive.scratch_path / "provenance/native_predictions.restricted.npz"
        )
        if restricted_predictions.is_file():
            self._write_canonical_native_predictions(
                archive,
                manifest=manifest,
                configuration=configuration,
                restricted_path=restricted_predictions,
            )
        training = manifest.get("training", {})
        resources = manifest.get("resources", {})
        if not isinstance(training, Mapping):
            training = {}
        if not isinstance(resources, Mapping):
            resources = {}
        history = training.get("history", []) if isinstance(training, Mapping) else []
        if isinstance(history, list) and history:
            clean_history = [dict(row) for row in history if isinstance(row, Mapping)]
        else:
            clean_history = []
        if not clean_history:
            clean_history = [
                {
                    "epoch": training.get("best_epoch", 0),
                    "validation_loss": training.get("best_validation_loss"),
                    "history_reconstructed_from_summary": True,
                }
            ]
        archive.write_table("metrics/history", clean_history)
        for row in clean_history:
            step = _execution_int(row.get("epoch"), 0)
            for source_name, metric_name in (
                ("train_loss", "train/loss"),
                ("validation_loss", "val/masked_huber"),
            ):
                value = _optional_nonnegative_float(row.get(source_name))
                if value is not None:
                    archive.append_metric_event(
                        {"name": metric_name, "value": value, "step": step}
                    )
        graph_record = manifest.get("graph", {})
        archive.write_json(
            "diagnostics/graph_statistics.json",
            {
                "graph": graph_record if isinstance(graph_record, Mapping) else {},
                "split_node_counts": manifest.get("split_node_counts", {}),
                "source": "native_training_manifest",
            },
        )
        native_config = manifest.get("config", {})
        native_training = (
            native_config.get("training", {})
            if isinstance(native_config, Mapping)
            else {}
        )
        mode_counts: dict[str, int] = {}
        masked_entries = 0
        masked_nodes = 0
        for row in clean_history:
            mode = str(row.get("mask_mode", "unknown"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            masked_entries += _execution_int(row.get("n_masked_entries"), 0)
            masked_nodes += _execution_int(row.get("n_target_nodes"), 0)
        archive.write_json(
            "diagnostics/mask_statistics.json",
            {
                "requested_rates": {
                    "partial_gene": native_training.get("partial_gene_rate"),
                    "whole_node": native_training.get("node_rate"),
                    "spatial_block": native_training.get("block_node_rate"),
                }
                if isinstance(native_training, Mapping)
                else {},
                "curriculum": (
                    native_training.get("curriculum")
                    if isinstance(native_training, Mapping)
                    else None
                ),
                "epoch_mode_counts": mode_counts,
                "masked_entry_count_across_epochs": masked_entries,
                "masked_node_count_across_epochs": masked_nodes,
                "source": "native_training_history",
            },
        )
        archive.write_summary(
            {
                "run_id": archive.run_id,
                "training_exit_status": "success",
                "legacy_run_id": manifest.get("run_id"),
                "best_epoch": training.get("best_epoch"),
                "primary_metric_name": "val/masked_huber",
                "primary_metric_value": training.get("best_validation_loss"),
                "duration_seconds": manifest.get("timing", {}).get(
                    "runtime_seconds"
                ),
                "peak_vram_gb": (
                    float(resources.get("peak_cuda_memory_bytes", 0)) / (1024**3)
                ),
                "canonical_predictions_available": True,
                "prediction_note": (
                    "One declared validation mask evaluation is stored with "
                    "salted stable sample keys; the full native NPZ remains "
                    "restricted provenance."
                ),
            }
        )
        primary_value = training.get("best_validation_loss")
        if primary_value is not None:
            archive.write_json(
                "metrics/final.json",
                {"val/masked_huber": float(primary_value)},
            )

    def _write_canonical_native_predictions(
        self,
        archive: RunArchive,
        *,
        manifest: Mapping[str, Any],
        configuration: Mapping[str, Any],
        restricted_path: Path,
    ) -> None:
        """Convert one declared validation evaluation to privacy-safe JSONL."""

        salt = os.environ.get("BAGM_SAMPLE_KEY_SALT", "")
        if len(salt.encode("utf-8")) < 16:
            raise ArtifactFinalizationError(
                "BAGM_SAMPLE_KEY_SALT (at least 16 bytes) is required to "
                "finalize canonical validation predictions."
            )
        evaluations = manifest.get("evaluations", [])
        candidates = [
            dict(item)
            for item in evaluations
            if isinstance(item, Mapping) and item.get("split") == "validation"
        ] if isinstance(evaluations, list) else []
        if not candidates:
            raise ArtifactFinalizationError(
                "Native run declares no validation prediction evaluation."
            )
        selected = sorted(
            candidates,
            key=lambda item: (
                {"partial": 0, "node": 1, "block": 2}.get(
                    str(item.get("mask_mode", "")), 99
                ),
                int(item.get("mask_replicate", 0)),
                str(item.get("prefix", "")),
            ),
        )[0]
        required_keys = (
            "prediction_key",
            "y_true_key",
            "mask_key",
            "cell_ids_key",
        )
        if any(not selected.get(key) for key in required_keys):
            raise ArtifactFinalizationError(
                "Native validation prediction declaration is incomplete."
            )
        dataset = _section(configuration, "dataset")
        namespace = (
            f"bagm:{dataset['dataset_id']}:{dataset['version']}:validation"
        )
        graph_record = manifest.get("graph", {})
        graph_id = (
            graph_record.get("graph_id")
            if isinstance(graph_record, Mapping)
            else None
        )
        with np.load(restricted_path, allow_pickle=False) as arrays:
            missing = [
                str(selected[key])
                for key in required_keys
                if str(selected[key]) not in arrays.files
            ]
            if missing:
                raise ArtifactFinalizationError(
                    "Native prediction archive is missing declared arrays: "
                    + ", ".join(missing)
                )
            target = np.asarray(arrays[str(selected["y_true_key"])])
            prediction = np.asarray(arrays[str(selected["prediction_key"])])
            mask = np.asarray(arrays[str(selected["mask_key"])], dtype=bool)
            cell_ids = np.asarray(arrays[str(selected["cell_ids_key"])])
            if (
                target.shape != prediction.shape
                or target.shape != mask.shape
                or target.ndim != 2
                or cell_ids.shape != (target.shape[0],)
            ):
                raise ArtifactFinalizationError(
                    "Native validation arrays have incompatible shapes."
                )

            def rows() -> Any:
                for index in range(target.shape[0]):
                    selected_targets = np.flatnonzero(mask[index])
                    if selected_targets.size == 0:
                        continue
                    truth = target[index, selected_targets].astype(
                        np.float64, copy=False
                    )
                    estimate = prediction[index, selected_targets].astype(
                        np.float64, copy=False
                    )
                    difference = np.abs(estimate - truth)
                    huber = np.where(
                        difference < 1.0,
                        0.5 * difference * difference,
                        difference - 0.5,
                    )
                    protected_row = {
                        "_protected_local_index": str(cell_ids[index]),
                        "run_id": archive.run_id,
                        "graph_id": graph_id,
                        "dataset_id": str(dataset["dataset_id"]),
                        "split": "validation",
                        "fold": int(configuration.get("fold", 0)),
                        "y_true": truth.tolist(),
                        "y_pred": estimate.tolist(),
                        "target_indices": selected_targets.astype(int).tolist(),
                        "sample_loss": float(np.mean(huber)),
                        "node_count": int(target.shape[0]),
                        "effective_mask_rate": float(
                            selected_targets.size / target.shape[1]
                        ),
                        "masking_type": str(selected.get("mask_mode")),
                        "mask_replicate": int(selected.get("mask_replicate", 0)),
                    }
                    yield deidentify_prediction_rows(
                        [protected_row],
                        identifier_fields=["_protected_local_index"],
                        salt=salt,
                        namespace=namespace,
                    )[0]

            archive.write_prediction_jsonl_stream("validation", rows())
        archive.write_json(
            "diagnostics/canonical_prediction_selection.json",
            {
                "source": "provenance/native_predictions.restricted.npz",
                "split": "validation",
                "mask_mode": selected.get("mask_mode"),
                "mask_replicate": selected.get("mask_replicate"),
                "prediction_key": selected.get("prediction_key"),
                "sample_key_method": "HMAC-SHA256 with untracked local salt",
            },
        )

    def _record_bundle_artifacts(self, run_id: str, root: Path) -> None:
        self.registry.record_artifacts(
            run_id,
            self._bundle_artifact_records(root),
        )

    def _bundle_artifact_records(self, root: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name in COMPLETION_MARKERS:
                continue
            records.append(
                {
                    "kind": _artifact_kind(path.relative_to(root)),
                    "path": path,
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        return records

    def _write_test_dummy_outputs(
        self,
        archive: RunArchive,
        configuration: Mapping[str, Any],
    ) -> None:
        metadata = configuration.get("metadata", {})
        if not isinstance(metadata, Mapping) or not metadata.get("test_only_dummy"):
            return
        if not (archive.scratch_path / "metrics/final.json").exists():
            archive.write_json(
                "metrics/final.json",
                {"val/masked_huber": 0.0},
            )
        if not (archive.scratch_path / "metrics/events.jsonl").exists():
            archive.append_metric_event(
                {"name": "val/masked_huber", "value": 0.0, "step": 0}
            )
        if not any((archive.scratch_path / "metrics").glob("history.*")):
            archive.write_table(
                "metrics/history",
                [{"epoch": 0, "val/masked_huber": 0.0, "test_only_dummy": True}],
            )
        dataset = _section(configuration, "dataset")
        if not any((archive.scratch_path / "predictions").glob("validation.*")):
            archive.write_predictions(
                "validation",
                [
                    {
                        "run_id": archive.run_id,
                        "sample_key": "sk_test_only_nonclinical_fixture",
                        "dataset_id": str(dataset["dataset_id"]),
                        "split": "validation",
                        "fold": int(configuration.get("fold", 0)),
                        "y_true": 0.0,
                        "y_pred": 0.0,
                        "sample_loss": 0.0,
                    }
                ],
            )


class ProcessExecutionError(QueueWorkerError):
    def __init__(self, message: str, *, category: str, return_code: int) -> None:
        super().__init__(message)
        self.category = category
        self.return_code = return_code


class ArtifactFinalizationError(QueueWorkerError):
    """Raised when zero exit did not satisfy the artifact completion contract."""


def _section(configuration: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = configuration.get(name)
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"{name} must be a mapping.")
    return section


def _render_command(
    command: Sequence[str],
    *,
    run_id: str,
    run_scratch: Path,
    artifact_dir: Path,
    project_root: Path,
) -> list[str]:
    substitutions = {
        "{run_id}": run_id,
        "{run_scratch}": str(run_scratch),
        "{artifact_dir}": str(artifact_dir),
        "{project_root}": str(project_root),
    }
    rendered: list[str] = []
    for part in command:
        value = str(part)
        for token, replacement in substitutions.items():
            value = value.replace(token, replacement)
        rendered.append(value)
    return rendered


def _option_value(command: Sequence[str], option: str) -> str | None:
    try:
        index = list(command).index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def _output_tail(stdout_path: Path, stderr_path: Path, limit: int = 64 * 1024) -> str:
    chunks: list[str] = []
    for path in (stdout_path, stderr_path):
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - limit))
                chunks.append(handle.read().decode("utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _git_output(project_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _provenance(
    configuration: Mapping[str, Any], project_root: Path
) -> dict[str, Any]:
    dataset_value = configuration.get("dataset", {})
    dataset = dataset_value if isinstance(dataset_value, Mapping) else {}
    commit = _git_output(project_root, "rev-parse", "HEAD") or "unknown"
    status = _git_output(project_root, "status", "--short", "--untracked-files=all")
    tracked_diff = _git_output(project_root, "diff", "--binary", "HEAD")
    untracked_files = _git_untracked_manifest(project_root)
    untracked_patch = _git_untracked_patch(project_root, untracked_files)
    dirty_fingerprint = (
        hashlib.sha256(
            canonical_json(
                {
                    "status": status or "",
                    "tracked_diff": tracked_diff or "",
                    "untracked_files": untracked_files,
                }
            ).encode("utf-8")
        ).hexdigest()
        if status or tracked_diff or untracked_files
        else None
    )
    hardware = _hardware_snapshot()
    environment_payload, environment_text = _environment_snapshot()
    environment_payload["hardware_runtime"] = hardware
    environment_fingerprint = canonical_sha256(environment_payload)
    environment_text += (
        "\n[hardware_runtime]\n"
        + canonical_json(hardware)
        + "\n[bagm_repro]\n"
        + f"environment_fingerprint={environment_fingerprint}"
        + "\n"
    )
    return {
        "git_commit": commit,
        "dirty_fingerprint": dirty_fingerprint,
        "dataset_fingerprint": str(
            dataset.get("dataset_fingerprint") or "unverified"
        ),
        "split_fingerprint": str(dataset.get("split_fingerprint") or "unverified"),
        "preprocessing_version": str(
            dataset.get("preprocessing_version") or "unverified"
        ),
        "environment_fingerprint": environment_fingerprint,
        "environment_text": environment_text,
        "uncommitted_changes_patch": (
            (tracked_diff or "")
            + ("\n" if tracked_diff and untracked_patch else "")
            + untracked_patch
        ),
        "untracked_files": untracked_files,
        "gpu_model": hardware.get("gpu_model"),
        "hardware": hardware,
    }


def _git_untracked_manifest(project_root: Path) -> list[dict[str, Any]]:
    """Hash non-ignored untracked files without copying their contents."""

    try:
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    records: list[dict[str, Any]] = []
    for encoded in sorted(part for part in result.stdout.split(b"\0") if part):
        relative_text = os.fsdecode(encoded)
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        path = project_root / relative
        try:
            info = path.lstat()
            if path.is_symlink():
                target = os.readlink(path)
                record = {
                    "path": relative.as_posix(),
                    "type": "symlink",
                    "target_sha256": hashlib.sha256(
                        target.encode("utf-8", errors="surrogateescape")
                    ).hexdigest(),
                }
            elif path.is_file():
                record = {
                    "path": relative.as_posix(),
                    "type": "file",
                    "size": info.st_size,
                    "sha256": _sha256_file(path),
                }
            else:
                record = {
                    "path": relative.as_posix(),
                    "type": "special",
                    "size": info.st_size,
                }
        except OSError as error:
            record = {
                "path": relative.as_posix(),
                "type": "unreadable",
                "error": type(error).__name__,
            }
        records.append(record)
    return records


def _git_untracked_patch(
    project_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_file_bytes: int = 2 * 1024 * 1024,
    maximum_total_bytes: int = 20 * 1024 * 1024,
) -> str:
    """Capture reconstructable untracked source/config text without secrets."""

    allowed_roots = {
        "src",
        "scripts",
        "configs",
        "tests",
        "docs",
        "experiments",
        "ops",
    }
    allowed_root_files = {
        "AGENTS.md",
        "README.md",
        "Makefile",
        "pyproject.toml",
        ".env.example",
    }
    chunks: list[str] = []
    total = 0
    for record in records:
        if record.get("type") != "file":
            continue
        relative = Path(str(record["path"]))
        if not (
            (relative.parts and relative.parts[0] in allowed_roots)
            or relative.as_posix() in allowed_root_files
        ):
            continue
        if any(
            part.lower() in {".env", "credentials", "secrets", "private"}
            for part in relative.parts
        ):
            continue
        size = int(record.get("size", 0))
        if size > maximum_file_bytes or total + size > maximum_total_bytes:
            continue
        path = project_root / relative
        try:
            raw = path.read_bytes()
            raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", str(relative)],
            cwd=project_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        if result.returncode not in {0, 1}:
            continue
        chunks.append(result.stdout)
        total += size
    return "".join(chunks)


def _environment_snapshot() -> tuple[dict[str, Any], str]:
    """Return a dependency-version fingerprint and human-readable snapshot."""

    try:
        from importlib.metadata import distributions

        packages = sorted(
            {
                (
                    distribution.metadata.get("Name") or "unknown"
                ).strip().lower(): distribution.version
                for distribution in distributions()
            }.items()
        )
    except Exception:
        packages = []
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve(strict=False)),
        "packages": packages,
    }
    fingerprint = canonical_sha256(payload)
    lines = [
        f"python={payload['python']}",
        f"implementation={payload['implementation']}",
        f"platform={payload['platform']}",
        f"executable={payload['executable']}",
        f"environment_fingerprint={fingerprint}",
        "",
        "[packages]",
        *(f"{name}=={version}" for name, version in packages),
    ]
    return payload, "\n".join(lines) + "\n"


def _hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu_model": None,
        "gpus": [],
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None:
        for line in result.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) == 6:
                snapshot["gpus"].append(
                    dict(
                        zip(
                            (
                                "index",
                                "uuid",
                                "name",
                                "driver_version",
                                "memory_total_mib",
                                "compute_capability",
                            ),
                            values,
                            strict=True,
                        )
                    )
                )
    names = [str(item["name"]) for item in snapshot["gpus"]]
    snapshot["gpu_model"] = "; ".join(names) or None
    try:
        import torch

        snapshot["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
    except Exception as error:
        snapshot["torch"] = {"inspection_error": type(error).__name__}
    return snapshot


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _primary_metric(
    summary: Mapping[str, Any], configuration: Mapping[str, Any]
) -> tuple[str | None, float | None]:
    evaluation = configuration.get("evaluation", {})
    name = (
        str(evaluation.get("primary_metric"))
        if isinstance(evaluation, Mapping) and evaluation.get("primary_metric")
        else None
    )
    declared_name = summary.get("primary_metric_name")
    if declared_name is not None:
        if not isinstance(declared_name, str) or "/" not in declared_name:
            raise ArtifactFinalizationError(
                "Summary primary_metric_name must be explicitly namespaced."
            )
        if name is not None and declared_name != name:
            raise ArtifactFinalizationError(
                "Summary primary_metric_name does not match the resolved "
                "evaluation.primary_metric."
            )
    value = summary.get("primary_metric_value")
    if value is None and name:
        metrics = summary.get("metrics", {})
        if isinstance(metrics, Mapping):
            value = metrics.get(name)
    if value is None:
        return name, None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ArtifactFinalizationError("Primary metric is NaN or infinite.")
    return name, numeric


def _artifact_kind(relative: Path) -> str:
    if len(relative.parts) > 1:
        return relative.parts[0]
    if relative.name.startswith("_"):
        return "completion_marker"
    return "metadata"


def _optional_nonnegative_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _peak_vram_from_summary(summary: Mapping[str, Any]) -> float | None:
    """Read the registry's historical GB field from either supported spelling.

    Resource artifacts report binary GiB.  The schema-v3 registry predates that
    explicit unit spelling and calls its column ``peak_vram_gb``.  Accepting
    ``peak_vram_gib`` keeps newer runners unit-explicit while preserving the
    existing registry schema.  When both are present they must agree.
    """

    legacy = _optional_nonnegative_float(summary.get("peak_vram_gb"))
    explicit = _optional_nonnegative_float(summary.get("peak_vram_gib"))
    if (
        legacy is not None
        and explicit is not None
        and not math.isclose(legacy, explicit, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ArtifactFinalizationError(
            "Summary peak_vram_gb and peak_vram_gib disagree."
        )
    return legacy if legacy is not None else explicit


def _optional_nonnegative_integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _execution_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return default
    return converted if converted >= 0 else default


def _process_start_ticks(pid: int) -> int | None:
    """Return Linux proc start ticks, which disambiguate PID reuse."""

    try:
        content = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        tail = content[content.rindex(")") + 2 :].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "FAILURE_CATEGORIES",
    "ArtifactFinalizationError",
    "InsufficientDiskSpaceError",
    "MissingDatasetError",
    "ProjectWorkerLock",
    "QueueWorker",
    "QueueWorkerError",
    "WorkerLockError",
    "WorkerSettings",
    "classify_failure",
    "command_for_config",
]
