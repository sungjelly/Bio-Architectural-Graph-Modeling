from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "preflight_so2_14core_relative_qkv_ddp",
    PROJECT_ROOT / "scripts/diagnostics/preflight_so2_14core_relative_qkv_ddp.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_PREFLIGHT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PREFLIGHT
_SPEC.loader.exec_module(_PREFLIGHT)


def test_preflight_import_and_locked_pair() -> None:
    assert _PREFLIGHT.PREFLIGHT_ALIASES == ("SO2-C22", "SO2-C23")
    assert _PREFLIGHT.PREFLIGHT_SCHEMA == (
        "so2_14core_relative_qkv_ddp4_preflight_v1"
    )
    assert _PREFLIGHT.RECURRENT_PREFLIGHT_SCHEMA == (
        "so2_14core_recurrent_relative_qkv_ddp4_preflight_v1"
    )
    assert _PREFLIGHT.UNTIED8_PREFLIGHT_SCHEMA == (
        "so2_14core_untied8_relative_qkv_ddp4_preflight_v1"
    )
    assert _PREFLIGHT.UNTIED8_PREFLIGHT_MAX_VRAM_GIB == 22.0
    assert _PREFLIGHT.UNTIED8_MINIMUM_VRAM_HEADROOM_GIB == 2.0
    parsed = _PREFLIGHT.build_parser().parse_args(["--config", "experiment.yaml"])
    assert parsed.config == Path("experiment.yaml")
    assert parsed.output is None
    assert parsed.prior_c23_receipt is None
