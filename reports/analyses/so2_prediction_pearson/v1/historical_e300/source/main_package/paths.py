"""Project-root-aware paths for Bio-Architectural Graph Modeling.

Reusable code must resolve repository resources through this module rather than
through the process working directory.  ``BAGM_ROOT`` overrides the repository
root.  The data, artifact, state, scratch, cache, export, report, result, and config
roots each have a corresponding ``BAGM_*_ROOT`` override; relative overrides are
resolved beneath the selected project root.

Import-time constants represent the environment at import time.  Long-running
processes and tests that may change their environment should call
``current_paths()`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


class PathConfigurationError(ValueError):
    """Raised when a BAGM path override is invalid."""


def _absolute(path: Path, *, base: Path | None = None) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = (base or Path.cwd()) / expanded
    return expanded.resolve(strict=False)


def discover_project_root(anchor: str | Path | None = None) -> Path:
    """Find the nearest repository root without depending on the caller's CWD.

    A directory containing ``.git`` is preferred.  A directory containing both
    ``AGENTS.md`` and ``README.md`` is accepted for source archives without Git
    metadata.  The explicit *anchor* (or this source module) is searched first.
    The current working directory is only a final fallback for installed copies.
    """

    starts = [Path(anchor) if anchor is not None else Path(__file__), Path.cwd()]
    checked: set[Path] = set()
    fallback: Path | None = None
    for start in starts:
        start = _absolute(start)
        if start.is_file():
            start = start.parent
        for candidate in (start, *start.parents):
            if candidate in checked:
                continue
            checked.add(candidate)
            if (candidate / ".git").exists():
                return candidate
            if (
                fallback is None
                and (candidate / "AGENTS.md").is_file()
                and (candidate / "README.md").is_file()
            ):
                fallback = candidate
    if fallback is not None:
        return fallback
    raise PathConfigurationError(
        "Could not discover the BAGM project root; set BAGM_ROOT explicitly."
    )


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved locations owned by the single BAGM project."""

    project_root: Path
    config_root: Path
    data_root: Path
    artifact_root: Path
    state_root: Path
    scratch_root: Path
    cache_root: Path
    export_root: Path
    report_root: Path
    result_root: Path | None = None

    def __post_init__(self) -> None:
        """Preserve direct-constructor compatibility for older callers."""

        if self.result_root is None:
            object.__setattr__(
                self,
                "result_root",
                _absolute(Path("results"), base=self.project_root),
            )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        anchor: str | Path | None = None,
    ) -> "ProjectPaths":
        """Resolve paths from an environment mapping without creating anything."""

        values = os.environ if environ is None else environ
        configured_root = values.get("BAGM_ROOT")
        project_root = (
            _absolute(Path(configured_root))
            if configured_root
            else discover_project_root(anchor)
        )

        def location(variable: str, default: str) -> Path:
            raw = values.get(variable)
            if raw is None or not raw.strip():
                return _absolute(Path(default), base=project_root)
            return _absolute(Path(raw), base=project_root)

        return cls(
            project_root=project_root,
            config_root=location("BAGM_CONFIG_ROOT", "configs"),
            data_root=location("BAGM_DATA_ROOT", "data"),
            artifact_root=location("BAGM_ARTIFACT_ROOT", "artifacts"),
            state_root=location("BAGM_STATE_ROOT", "state"),
            scratch_root=location("BAGM_SCRATCH_ROOT", "scratch"),
            cache_root=location("BAGM_CACHE_ROOT", "cache"),
            export_root=location("BAGM_EXPORT_ROOT", "exports"),
            report_root=location("BAGM_REPORT_ROOT", "reports"),
            result_root=location("BAGM_RESULT_ROOT", "results"),
        )

    def validate(self, *, require_project_root: bool = True) -> None:
        """Validate path types without creating or modifying directories."""

        if require_project_root and not self.project_root.is_dir():
            raise PathConfigurationError(
                f"BAGM project root is not a directory: {self.project_root}"
            )
        for field_name in (
            "config_root",
            "data_root",
            "artifact_root",
            "state_root",
            "scratch_root",
            "cache_root",
            "export_root",
            "report_root",
            "result_root",
        ):
            value = getattr(self, field_name)
            if value.exists() and not value.is_dir():
                raise PathConfigurationError(
                    f"{field_name} exists but is not a directory: {value}"
                )

    def ensure_runtime_directories(self) -> None:
        """Create only generated-runtime roots, never data or configuration roots."""

        self.validate()
        for path in (
            self.artifact_root,
            self.state_root,
            self.scratch_root,
            self.cache_root,
            self.export_root,
            self.report_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def current_paths(
    environ: Mapping[str, str] | None = None,
    *,
    anchor: str | Path | None = None,
) -> ProjectPaths:
    """Return paths resolved from the current environment."""

    return ProjectPaths.from_environment(environ, anchor=anchor)


_DEFAULT_PATHS = current_paths()

PROJECT_ROOT = _DEFAULT_PATHS.project_root
CONFIG_ROOT = _DEFAULT_PATHS.config_root
DATA_ROOT = _DEFAULT_PATHS.data_root
ARTIFACT_ROOT = _DEFAULT_PATHS.artifact_root
STATE_ROOT = _DEFAULT_PATHS.state_root
SCRATCH_ROOT = _DEFAULT_PATHS.scratch_root
CACHE_ROOT = _DEFAULT_PATHS.cache_root
EXPORT_ROOT = _DEFAULT_PATHS.export_root
REPORT_ROOT = _DEFAULT_PATHS.report_root
RESULT_ROOT = _DEFAULT_PATHS.result_root


__all__ = [
    "ARTIFACT_ROOT",
    "CACHE_ROOT",
    "CONFIG_ROOT",
    "DATA_ROOT",
    "EXPORT_ROOT",
    "PROJECT_ROOT",
    "REPORT_ROOT",
    "RESULT_ROOT",
    "SCRATCH_ROOT",
    "STATE_ROOT",
    "PathConfigurationError",
    "ProjectPaths",
    "current_paths",
    "discover_project_root",
]
