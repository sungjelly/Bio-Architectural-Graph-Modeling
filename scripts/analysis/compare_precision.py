#!/usr/bin/env python3
"""Audit mixed-precision equivalence against a paired FP32 run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.artifacts import sha256_file  # noqa: E402
from spatial_benchmark.experiment import load_run_manifest  # noqa: E402


def _canonical_hash(value: dict) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _paired_config(manifest: dict) -> dict:
    config = copy.deepcopy(manifest["config"])
    config["training"].pop("amp", None)
    config["training"].pop("amp_dtype", None)
    return {
        "model_name": manifest["model_name"],
        "model_seed": manifest["model_seed"],
        "prepared_artifact": manifest["prepared_artifact"],
        "graph": manifest["graph"],
        "config": config,
    }


def _metrics_by_prefix(run: Path, manifest: dict) -> dict[str, dict]:
    metrics = json.loads(
        (run / manifest["metrics_file"]).read_text(encoding="utf-8")
    )
    result = {}
    for record in metrics["validation"]:
        prefix = (
            f"validation__{record['mask_mode']}"
            f"__r{record['mask_replicate']}"
        )
        result[prefix] = record["metrics"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-run", required=True, type=Path)
    parser.add_argument("--amp-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-relative-huber-difference",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--max-masked-prediction-mae",
        type=float,
        default=0.01,
    )
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite precision audit: {destination}"
        )
    fp32_root = args.fp32_run.resolve()
    amp_root = args.amp_run.resolve()
    fp32_manifest = load_run_manifest(fp32_root)
    amp_manifest = load_run_manifest(amp_root)
    if fp32_manifest["sealed_test_opened"] or amp_manifest["sealed_test_opened"]:
        raise ValueError("Precision equivalence must use validation-only runs.")
    if fp32_manifest["config"]["training"]["amp"] is not False:
        raise ValueError("Reference run is not FP32.")
    if amp_manifest["config"]["training"]["amp"] is not True:
        raise ValueError("Candidate run did not enable AMP.")
    if _paired_config(fp32_manifest) != _paired_config(amp_manifest):
        raise ValueError("Precision runs differ outside AMP settings.")
    fp32_metrics = _metrics_by_prefix(fp32_root, fp32_manifest)
    amp_metrics = _metrics_by_prefix(amp_root, amp_manifest)
    if set(fp32_metrics) != set(amp_metrics):
        raise ValueError("Precision runs do not share evaluation records.")
    fp32_predictions = np.load(
        fp32_root / fp32_manifest["artifacts"]["predictions"],
        allow_pickle=False,
    )
    amp_predictions = np.load(
        amp_root / amp_manifest["artifacts"]["predictions"],
        allow_pickle=False,
    )
    rows = []
    try:
        if set(fp32_predictions.files) != set(amp_predictions.files):
            raise ValueError("Precision prediction arrays differ.")
        for prefix in sorted(fp32_metrics):
            mask_key = f"{prefix}__mask"
            prediction_key = f"{prefix}__prediction"
            mask_a = np.asarray(fp32_predictions[mask_key], dtype=bool)
            mask_b = np.asarray(amp_predictions[mask_key], dtype=bool)
            if not np.array_equal(mask_a, mask_b):
                raise ValueError(f"Fixed masks differ for {prefix}.")
            fp32_prediction = np.asarray(
                fp32_predictions[prediction_key]
            )
            amp_prediction = np.asarray(
                amp_predictions[prediction_key]
            )
            masked_difference = np.abs(
                fp32_prediction[mask_a] - amp_prediction[mask_a]
            )
            fp32_huber = float(fp32_metrics[prefix]["huber"])
            amp_huber = float(amp_metrics[prefix]["huber"])
            relative_huber = abs(amp_huber - fp32_huber) / fp32_huber
            masked_mae = float(masked_difference.mean())
            rows.append(
                {
                    "evaluation": prefix,
                    "fp32_huber": fp32_huber,
                    "amp_huber": amp_huber,
                    "relative_huber_difference": relative_huber,
                    "masked_prediction_mae": masked_mae,
                    "masked_prediction_p99_absolute_difference": float(
                        np.quantile(masked_difference, 0.99)
                    ),
                    "passes": (
                        relative_huber
                        <= args.max_relative_huber_difference
                        and masked_mae <= args.max_masked_prediction_mae
                    ),
                }
            )
    finally:
        fp32_predictions.close()
        amp_predictions.close()
    decision = {
        "format_version": 1,
        "artifact_kind": "mixed_precision_equivalence_audit",
        "status": "complete",
        "decision": "pass" if all(row["passes"] for row in rows) else "fail",
        "thresholds": {
            "max_relative_huber_difference": (
                args.max_relative_huber_difference
            ),
            "max_masked_prediction_mae": args.max_masked_prediction_mae,
        },
        "fp32_run_id": fp32_manifest["run_id"],
        "amp_run_id": amp_manifest["run_id"],
        "prepared_artifact_id": fp32_manifest["prepared_artifact"][
            "artifact_id"
        ],
        "test_metrics_used": False,
        "evaluations": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        decision_path = temporary / "decision.json"
        decision_path.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format_version": 1,
            "artifact_kind": "mixed_precision_equivalence_audit",
            "status": "complete",
            "artifact_id": _canonical_hash(decision)[:16],
            "files": {"decision.json": sha256_file(decision_path)},
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(decision, sort_keys=True))
    return 0 if decision["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
