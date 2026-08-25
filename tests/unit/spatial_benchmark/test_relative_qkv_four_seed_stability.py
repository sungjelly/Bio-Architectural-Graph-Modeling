from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import spatial_benchmark.relative_qkv_four_seed_stability as stability_module
from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES
from spatial_benchmark.pooled_relative_qkv_training import PooledRelativeQKVCoreBatch
from spatial_benchmark.relative_qkv_four_seed_stability import (
    FourSeedAnalysisResult,
    FourSeedStabilityError,
    SeedCompactExtraction,
    ValidatedMember,
    canonical_sha256,
    compute_four_seed_analysis,
    evenly_spaced_receivers,
    parse_seed_path_specs,
    receiver_centered_logits,
    scalar_summary,
    top_mutual_pair_positions,
    vectorized_mutual_pair_scores,
    verify_four_seed_analysis_bundle,
    write_deterministic_npz,
    write_four_seed_analysis_bundle,
)


def test_seed_path_specs_require_exactly_four_distinct_active_seeds(
    tmp_path: Path,
) -> None:
    values = [f"{seed}={tmp_path / f'seed-{seed}'}" for seed in range(4)]
    parsed = parse_seed_path_specs(values, label="run")
    assert tuple(parsed) == (0, 1, 2, 3)
    assert parsed[2] == (tmp_path / "seed-2").resolve()

    with pytest.raises(FourSeedStabilityError, match="exactly seeds"):
        parse_seed_path_specs(values[:3], label="run")
    with pytest.raises(FourSeedStabilityError, match="Duplicate"):
        parse_seed_path_specs(values + [values[0]], label="run")
    with pytest.raises(FourSeedStabilityError, match="SEED=PATH"):
        parse_seed_path_specs(["invalid"], label="run")


def test_evenly_spaced_receiver_probes_are_exact_and_endpoint_inclusive() -> None:
    first = evenly_spaced_receivers(1_001, count=64)
    second = evenly_spaced_receivers(1_001, count=64)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 64
    assert first[0] == 0
    assert first[-1] == 1_000
    assert len(np.unique(first)) == 64

    with pytest.raises(FourSeedStabilityError, match=r"\[1, n_nodes\]"):
        evenly_spaced_receivers(10, count=11)


def test_receiver_probes_follow_locked_floor_formula_for_real_can21_size() -> None:
    n_nodes = 4_897
    receivers = evenly_spaced_receivers(n_nodes, count=64)
    expected = (np.arange(64, dtype=np.int64) * (n_nodes - 1)) // 63

    np.testing.assert_array_equal(receivers, expected)
    assert receivers[[21, 42, 49]].tolist() == [1_632, 3_264, 3_808]


def test_fixed_attention_normalization_uses_fp64_host_accumulation() -> None:
    # Simulate FP32 softmax probabilities whose device reduction order differs
    # from NumPy's host reduction order.  Their represented probability mass is
    # within the locked 1e-6 gate, while a second FP32 reduction is not.
    edge_count = 181
    exponents = (np.arange(edge_count, dtype=np.int64) * 2) % 5
    numerators = np.power(
        np.float32(10.0),
        -exponents.astype(np.float32),
        dtype=np.float32,
    )
    device_order_denominator = np.add.accumulate(
        numerators,
        dtype=np.float32,
    )[-1]
    attention = (numerators / device_order_denominator).reshape(-1, 1)
    assert abs(float(attention.sum(dtype=np.float32)) - 1.0) > 1e-6
    assert abs(float(attention.sum(dtype=np.float64)) - 1.0) < 1e-6

    edge_index = np.vstack(
        (
            np.arange(edge_count, dtype=np.int64),
            np.zeros(edge_count, dtype=np.int64),
        )
    )
    edge_ids = np.arange(edge_count, dtype=np.int64)
    zeros = np.zeros_like(attention)
    stability_module._validate_fixed_attention(
        edge_index=edge_index,
        edge_ids=edge_ids,
        attention=attention,
        content=zeros,
        bias=zeros,
        combined=zeros,
        receivers=np.asarray([0], dtype=np.int64),
    )

    invalid = attention.copy()
    invalid[0, 0] += np.float32(1e-3)
    with pytest.raises(FourSeedStabilityError, match="maximum_abs_error"):
        stability_module._validate_fixed_attention(
            edge_index=edge_index,
            edge_ids=edge_ids,
            attention=invalid,
            content=zeros,
            bias=zeros,
            combined=zeros,
            receivers=np.asarray([0], dtype=np.int64),
        )


def test_receiver_centered_logits_ignore_softmax_null_offsets() -> None:
    groups = ("a", "a", "b", "b", "b")
    logits = np.asarray(
        [[1.0, -2.0], [2.0, 3.0], [4.0, 1.0], [7.0, -1.0], [8.0, 5.0]],
        dtype=np.float64,
    )
    offsets = {"a": np.asarray([100.0, -7.0]), "b": np.asarray([-20.0, 13.0])}
    shifted = logits + np.stack([offsets[group] for group in groups])

    first = receiver_centered_logits(logits, groups)
    second = receiver_centered_logits(shifted, groups)
    np.testing.assert_allclose(first, second, atol=1e-14, rtol=0.0)
    for group in set(groups):
        selected = np.asarray([value == group for value in groups])
        np.testing.assert_allclose(first[selected].mean(axis=0), 0.0, atol=1e-14)


def _exact_replay_record(
    *,
    shape: tuple[int, ...],
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> dict[str, object]:
    return {
        "byte_identical": True,
        "within_tolerance": True,
        "dtype": "float32",
        "shape": list(shape),
        "element_count": int(np.prod(shape)),
        "failing_element_count": 0,
        "first_sha256": "a" * 64,
        "second_sha256": "a" * 64,
        "maximum_absolute_difference": 0.0,
        "mean_absolute_difference": 0.0,
        "atol": atol,
        "rtol": rtol,
    }


def test_core_receipt_gate_requires_every_exact_prediction_and_attention_channel() -> None:
    row: dict[str, object] = {
        "n_nodes": 4,
        "prediction_replay": _exact_replay_record(shape=(4, 1000)),
        "selected_attention_replay": {
            "edge_index_sha256": "b" * 64,
            "edge_index_reload_sha256": "b" * 64,
            "attention_heads": 8,
            "selected_directed_edges": 12,
            "selected_receivers": [0, 1, 2, 3],
            "maximum_logit_composition_error": 0.0,
            "maximum_receiver_head_normalization_error": 5e-7,
            "channels": {
                    name: _exact_replay_record(shape=(12, 8))
                for name in ("attention", "content", "positional_bias", "combined")
            },
        },
    }
    stability_module._validate_core_replay_receipt(row, seed=2, alias="CAN-01")

    channels = row["selected_attention_replay"]["channels"]
    channels["content"]["failing_element_count"] = 1
    with pytest.raises(FourSeedStabilityError, match="exact strict replay"):
        stability_module._validate_core_replay_receipt(row, seed=2, alias="CAN-01")


def test_vectorized_mutual_pair_math_and_deterministic_top_ties() -> None:
    # Receiver-major/source-major ordering for reciprocal pairs 0-1 and 1-2.
    edge_index = np.asarray(
        [[1, 0, 2, 1], [0, 1, 1, 2]],
        dtype=np.int64,
    )
    directional = np.asarray([0.5, 0.9, 0.4, 0.7], dtype=np.float32)
    result = vectorized_mutual_pair_scores(
        edge_index,
        directional,
        n_nodes=3,
    )
    np.testing.assert_array_equal(result.canonical_edge_ids, [1, 3])
    np.testing.assert_array_equal(result.pair_keys, [1, 5])
    np.testing.assert_allclose(result.scores, [0.5, 0.4])
    np.testing.assert_array_equal(
        top_mutual_pair_positions(result.scores, result.pair_keys, top_k=1),
        [0],
    )
    np.testing.assert_array_equal(
        top_mutual_pair_positions(
            np.asarray([1.0, 1.0]),
            np.asarray([9, 4]),
            top_k=2,
        ),
        [1, 0],
    )

    with pytest.raises(FourSeedStabilityError, match="reciprocal"):
        vectorized_mutual_pair_scores(
            edge_index[:, :-1],
            directional[:-1],
            n_nodes=3,
        )


def test_four_seed_scalar_summary_uses_sample_sd_and_empirical_quantiles() -> None:
    report = scalar_summary([1.0, 2.0, 3.0, 4.0])
    assert report["mean"] == pytest.approx(2.5)
    assert report["sample_standard_deviation"] == pytest.approx(
        np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    )
    assert report["range"] == pytest.approx(3.0)
    assert report["ensemble_member_count"] == 4
    assert report["spread_label"] == "four-seed ensemble spread"
    assert report["calibrated_confidence_interval"] is False
    assert report["empirical_quantiles"]["0.05"] == pytest.approx(1.15)

    with pytest.raises(FourSeedStabilityError, match="exactly four"):
        scalar_summary([1.0, 2.0, 3.0])


def test_locked_analysis_protocol_freezes_statistics_execution_and_claim_scope(
    tmp_path: Path,
) -> None:
    campaign = (
        Path(__file__).resolve().parents[3]
        / "experiments/campaigns/cmp_20260824_cancer_6core_relative_qkv_multiseed"
    )
    protocol = stability_module.verify_locked_gradient_protocol(
        campaign / "analysis_protocol_selected_gradient_stability_v1.yaml",
        campaign / "analysis_protocol_selected_gradient_stability_v1.sha256",
    )
    cohort_manifest = tmp_path / "cohort.json"
    graph_manifest = tmp_path / "graph.json"
    cohort_manifest.write_text("{}\n", encoding="utf-8")
    graph_manifest.write_text("{}\n", encoding="utf-8")
    protocol = json.loads(json.dumps(protocol))
    protocol["prepared_inputs"]["cohort_manifest_sha256"] = (
        stability_module.sha256_file(cohort_manifest)
    )
    protocol["prepared_inputs"]["graph_manifest_sha256"] = (
        stability_module.sha256_file(graph_manifest)
    )
    stability_module._validate_analysis_protocol_inputs(
        protocol,
        cohort_manifest_path=cohort_manifest,
        graph_manifest_path=graph_manifest,
    )

    protocol["statistical_summary_semantics"]["standard_deviation"]["ddof"] = 0
    with pytest.raises(FourSeedStabilityError, match="statistical/logit"):
        stability_module._validate_analysis_protocol_inputs(
            protocol,
            cohort_manifest_path=cohort_manifest,
            graph_manifest_path=graph_manifest,
        )


def test_gpu_binding_requires_single_idle_physical_device_and_uuid_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = {
        "records": [
            {
                "index": 2,
                "uuid": "GPU-test-uuid",
                "pci_bus_id": "00000000:02:00.0",
                "name": "test",
                "driver_version": "test",
                "memory_total_mib": 1,
            }
        ]
    }
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(stability_module, "_nvidia_smi_inventory", lambda: inventory)
    monkeypatch.setattr(stability_module, "_nvidia_compute_applications", lambda: ())
    device, preflight = stability_module._preflight_gpu_binding("cuda:0")
    assert device == torch.device("cuda:0")
    assert preflight["selected_physical_gpu"]["uuid"] == "GPU-test-uuid"

    monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(
        stability_module,
        "_nvidia_compute_applications",
        lambda: (
            {
                "pid": stability_module.os.getpid(),
                "gpu_uuid": "GPU-test-uuid",
                "process_name": "python",
            },
        ),
    )
    finalized = stability_module._finalize_gpu_binding(
        preflight,
        properties=type("Properties", (), {"uuid": "GPU-test-uuid"})(),
        device=device,
    )
    assert finalized["analysis_pid"] == stability_module.os.getpid()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    with pytest.raises(FourSeedStabilityError, match="exactly one"):
        stability_module._preflight_gpu_binding("cuda:0")


def test_deterministic_npz_and_atomic_bundle_refuse_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    arrays = {
        "z": np.arange(4, dtype=np.float32),
        "a": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
    }
    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite NPZ"):
        write_deterministic_npz(first, arrays)

    report = {
        "schema": "test_schema",
        "status": "complete",
        "campaign_id": "cmp_20260824_cancer_6core_relative_qkv_multiseed",
        "active_model_seeds": [0, 1, 2, 3],
        "deferred_model_seeds": [4],
        "five_seed_campaign_completion_claim_allowed": False,
        "calibrated_biological_confidence_interval": False,
        "generalization_claim_supported": False,
        "causal_claim_supported": False,
        "members": [
            {
                "seed": seed,
                "run_id": f"run-{seed}",
                "completed_global_epochs": 175 + 25 * (seed % 2),
                "checkpoint_sha256": str(seed) * 64,
            }
            for seed in range(4)
        ],
        "fixed_input_identity": {"node_count": 8},
        "training_loss": {
            "final_summary": scalar_summary([1.0, 2.0, 3.0, 4.0]),
        },
        "fixed_held_in_metrics": {
            "summary": {
                "huber": scalar_summary([1.0, 2.0, 3.0, 4.0]),
            }
        },
        "embedding_stability": {"linear_cka": np.eye(4).tolist()},
        "attention_head_stability": {
            "matching_method": "Hungarian",
            "head_top_edge_fraction": 0.05,
            "head_top_edge_count": 1,
        },
        "mutual_attention_routing_stability": {
            "selection": "test top pairs",
        },
        "selected_gradient_stability": {"request_count": 24},
        "limitations": ["test limitation"],
    }
    result = FourSeedAnalysisResult(
        report=report,
        tables={"tiny": pa.table({"value": [1, 2]})},
        arrays={"tiny": arrays},
    )
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("schema: test\n", encoding="utf-8")
    output = tmp_path / "bundle"
    with pytest.raises(FourSeedStabilityError, match="manifest identity"):
        write_four_seed_analysis_bundle(
            result,
            output,
            provenance_files={"protocol.yaml": protocol},
        )
    assert not output.exists()

    def structural_verifier(path: Path) -> dict[str, object]:
        manifest_path = Path(path) / "manifest.json"
        staged = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {"manifest_content_sha256": staged["manifest_content_sha256"]}

    monkeypatch.setattr(
        stability_module,
        "verify_four_seed_analysis_bundle",
        structural_verifier,
    )
    manifest = write_four_seed_analysis_bundle(
        result,
        output,
        provenance_files={"protocol.yaml": protocol},
    )
    assert manifest["manifest_content_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    )
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()
    assert (output / "tables/tiny.parquet").is_file()
    assert (output / "arrays/tiny.npz").is_file()
    loaded = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert loaded == manifest
    with pytest.raises(FourSeedStabilityError, match="manifest identity"):
        verify_four_seed_analysis_bundle(output)
    with pytest.raises(FileExistsError, match="overwrite analysis bundle"):
        write_four_seed_analysis_bundle(
            result,
            output,
            provenance_files={"protocol.yaml": protocol},
        )
    raced_output = tmp_path / "raced-bundle"
    with pytest.raises(FourSeedStabilityError, match="changed before publication"):
        write_four_seed_analysis_bundle(
            result,
            raced_output,
            provenance_files={"protocol.yaml": protocol},
            provenance_expected_sha256={"protocol.yaml": "0" * 64},
        )
    assert not raced_output.exists()

    concurrent_output = tmp_path / "concurrent-bundle"
    original_publish = stability_module._atomic_publish_directory_no_replace

    def create_concurrent_destination(source: Path, destination: Path) -> None:
        destination.mkdir()
        original_publish(source, destination)

    monkeypatch.setattr(
        stability_module,
        "_atomic_publish_directory_no_replace",
        create_concurrent_destination,
    )
    with pytest.raises(FileExistsError, match="concurrently created"):
        write_four_seed_analysis_bundle(
            result,
            concurrent_output,
            provenance_files={"protocol.yaml": protocol},
        )
    assert concurrent_output.is_dir()
    assert not any(concurrent_output.iterdir())


def test_pipeline_gates_before_cuda_runs_seeds_serially_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = tmp_path / "cohort"
    graph = tmp_path / "graph"
    cohort.mkdir()
    graph.mkdir()
    (cohort / "manifest.json").write_text("{}\n", encoding="utf-8")
    (graph / "manifest.json").write_text("{}\n", encoding="utf-8")
    protocol = tmp_path / "protocol.yaml"
    protocol_sha = tmp_path / "protocol.sha256"
    requests = tmp_path / "requests.csv"
    requests_sha = tmp_path / "requests.sha256"
    for path in (protocol, protocol_sha, requests, requests_sha):
        path.write_text(path.name + "\n", encoding="utf-8")
    monkeypatch.setattr(
        stability_module,
        "LOCKED_ANALYSIS_PROTOCOL_SHA256",
        stability_module.sha256_file(protocol),
    )
    monkeypatch.setattr(
        stability_module,
        "LOCKED_GRADIENT_REQUEST_TABLE_SHA256",
        stability_module.sha256_file(requests),
    )

    members, _, batches = _synthetic_four_seed_inputs(tmp_path)
    run_roots: dict[int, Path] = {}
    checkpoint_paths: dict[int, Path] = {}
    receipt_paths: dict[int, Path] = {}
    for member in members:
        member.checkpoint_path.parent.mkdir(parents=True)
        member.checkpoint_path.write_bytes(b"checkpoint")
        member.receipt_path.write_text("{}\n", encoding="utf-8")
        run_roots[member.seed] = member.run_root
        checkpoint_paths[member.seed] = member.checkpoint_path
        receipt_paths[member.seed] = member.receipt_path

    calls: list[str] = []
    temporary_paths: list[Path] = []
    copied_provenance: dict[str, Path] = {}

    monkeypatch.setattr(
        stability_module,
        "verify_locked_gradient_protocol",
        lambda *_args: calls.append("verify_protocol") or {"protocol": "locked"},
    )
    monkeypatch.setattr(
        stability_module,
        "_validate_analysis_protocol_inputs",
        lambda *_args, **_kwargs: calls.append("validate_protocol_inputs"),
    )
    monkeypatch.setattr(
        stability_module,
        "load_prepared_gradient_request_inputs",
        lambda **_kwargs: calls.append("load_gradient_inputs") or (object(),),
    )
    monkeypatch.setattr(
        stability_module,
        "load_and_verify_locked_gradient_requests",
        lambda *_args, **_kwargs: calls.append("verify_requests")
        or tuple(range(24)),
    )
    monkeypatch.setattr(
        stability_module,
        "group_selected_derivative_requests",
        lambda _requests: {alias: (object(),) * 4 for alias in CANCER_ALIASES},
    )
    monkeypatch.setattr(
        stability_module,
        "_locked_request_metadata",
        lambda _requests: {},
    )
    monkeypatch.setattr(
        stability_module,
        "load_prepared_relative_qkv_batches",
        lambda **_kwargs: calls.append("load_batches") or batches,
    )
    monkeypatch.setattr(
        stability_module,
        "validate_four_members",
        lambda *_args, **_kwargs: calls.append("validate_members") or members,
    )
    monkeypatch.setattr(
        stability_module,
        "_analysis_source_provenance",
        lambda: ({"analysis_source_files": {}}, b"dirty diff\n"),
    )
    monkeypatch.setattr(
        stability_module,
        "_preflight_gpu_binding",
        lambda device: (
            calls.append("preflight_gpu") or torch.device(device),
            {
                "cuda_device_order": "PCI_BUS_ID",
                "cuda_visible_devices": "0",
                "logical_device": "cuda:0",
                "selected_physical_gpu": {"index": 0, "uuid": "GPU-synthetic"},
                "compute_applications_before_cuda": [],
                "nvidia_smi_inventory": {"records": []},
            },
        ),
    )
    monkeypatch.setattr(
        stability_module,
        "install_deterministic_cuda_contract",
        lambda _device: calls.append("install_cuda") or torch.device("cuda:0"),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: type("Properties", (), {"name": "synthetic", "total_memory": 1})(),
    )
    monkeypatch.setattr(
        stability_module,
        "_finalize_gpu_binding",
        lambda preflight, **_kwargs: {
            **preflight,
            "torch_device_uuid": "GPU-synthetic",
            "analysis_pid": 123,
            "compute_applications_after_cuda": [
                {"pid": 123, "gpu_uuid": "GPU-synthetic", "process_name": "test"}
            ],
        },
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 7)
    monkeypatch.setattr(stability_module, "set_deterministic_seed", lambda *_a, **_k: None)

    def extract(member: ValidatedMember, *_args: object, **kwargs: object) -> object:
        calls.append(f"extract_{member.seed}")
        temporary_paths.append(Path(kwargs["temporary_dir"]))
        return {"seed": member.seed}

    monkeypatch.setattr(stability_module, "extract_member_compact", extract)
    monkeypatch.setattr(
        stability_module,
        "compute_four_seed_analysis",
        lambda *_args, **_kwargs: FourSeedAnalysisResult(
            report={"schema": "test", "status": "complete", "campaign_id": "test"},
            tables={},
            arrays={},
        ),
    )

    def publish(
        _result: FourSeedAnalysisResult,
        _destination: Path,
        *,
        provenance_files: dict[str, Path],
        provenance_expected_sha256: dict[str, str],
    ) -> dict[str, str]:
        calls.append("publish")
        copied_provenance.update(provenance_files)
        assert set(provenance_files) == set(provenance_expected_sha256)
        return {"manifest_content_sha256": "f" * 64}

    monkeypatch.setattr(stability_module, "write_four_seed_analysis_bundle", publish)
    monkeypatch.setattr(
        stability_module,
        "verify_four_seed_analysis_bundle",
        lambda _path: {"manifest_content_sha256": "f" * 64},
    )
    result = stability_module.run_four_seed_stability_pipeline(
        run_roots=run_roots,
        checkpoint_paths=checkpoint_paths,
        receipt_paths=receipt_paths,
        cohort_dir=cohort,
        graph_dir=graph,
        protocol_path=protocol,
        protocol_sha256_path=protocol_sha,
        request_csv_path=requests,
        request_sha256_path=requests_sha,
        destination=tmp_path / "output",
        device="cuda:0",
    )
    assert calls.index("verify_protocol") < calls.index("install_cuda")
    assert calls.index("verify_requests") < calls.index("install_cuda")
    assert calls.index("validate_members") < calls.index("install_cuda")
    assert calls[-5:] == [
        "extract_0",
        "extract_1",
        "extract_2",
        "extract_3",
        "publish",
    ]
    assert result["status"] == "complete"
    assert len(set(temporary_paths)) == 1
    assert not temporary_paths[0].exists()
    assert {
        f"seed_{seed}_strict_checkpoint_verification.json" for seed in range(4)
    }.issubset(copied_provenance)

    def fail_publish(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic publication failure")

    monkeypatch.setattr(
        stability_module,
        "write_four_seed_analysis_bundle",
        fail_publish,
    )
    with pytest.raises(RuntimeError, match="synthetic publication failure"):
        stability_module.run_four_seed_stability_pipeline(
            run_roots=run_roots,
            checkpoint_paths=checkpoint_paths,
            receipt_paths=receipt_paths,
            cohort_dir=cohort,
            graph_dir=graph,
            protocol_path=protocol,
            protocol_sha256_path=protocol_sha,
            request_csv_path=requests,
            request_sha256_path=requests_sha,
            destination=tmp_path / "failed-output",
            device="cuda:0",
        )
    assert not temporary_paths[-1].exists()


def _complete_graph_batch(alias: str) -> PooledRelativeQKVCoreBatch:
    n_nodes = 16
    pairs = [
        (source, receiver)
        for receiver in range(n_nodes)
        for source in range(n_nodes)
        if source != receiver
    ]
    return PooledRelativeQKVCoreBatch(
        alias=alias,
        target_expression=torch.zeros(n_nodes, 5),
        edge_index=torch.tensor(pairs, dtype=torch.long).T.contiguous(),
        relative_geometry=torch.zeros(len(pairs), 70),
        node_covariates=torch.zeros(n_nodes, 2),
    )


def _synthetic_four_seed_inputs(
    tmp_path: Path,
) -> tuple[
    tuple[ValidatedMember, ...],
    tuple[SeedCompactExtraction, ...],
    tuple[PooledRelativeQKVCoreBatch, ...],
]:
    batches = tuple(_complete_graph_batch(alias) for alias in CANCER_ALIASES)
    rng = np.random.default_rng(20260825)
    base_embedding = rng.normal(size=(len(CANCER_ALIASES) * 16, 6))
    base_attention = rng.uniform(0.01, 0.99, size=(60, 3))
    base_attention = (
        base_attention.reshape(len(CANCER_ALIASES), 10, 3)
        / base_attention.reshape(len(CANCER_ALIASES), 10, 3).sum(
            axis=1, keepdims=True
        )
    ).reshape(60, 3)
    orders = (
        np.asarray([0, 1, 2]),
        np.asarray([2, 0, 1]),
        np.asarray([1, 2, 0]),
        np.asarray([0, 2, 1]),
    )
    fixed_masks = {
        alias: {
            "effective_seed": 100 + index,
            "checksum_sha256": str(index) * 64,
            "masked_entry_count": 500,
        }
        for index, alias in enumerate(CANCER_ALIASES)
    }
    embedding_ids = tuple(
        f"{alias}:node:{node:08d}"
        for alias in CANCER_ALIASES
        for node in range(16)
    )
    fixed_edge_core = np.repeat(np.asarray(CANCER_ALIASES, dtype="U6"), 10)
    fixed_edge_number = np.tile(np.arange(10, dtype=np.int64), len(CANCER_ALIASES))
    fixed_edge_ids = tuple(
        f"{alias}:layer:final:edge:{edge:012d}"
        for alias in CANCER_ALIASES
        for edge in range(10)
    )
    gradient_template: list[dict[str, object]] = []
    for alias_index, alias in enumerate(CANCER_ALIASES):
        for shell_index, shell in enumerate(
            ("(0,50]", "(50,150]", "(150,300]", "(300,500]")
        ):
            gradient_template.append(
                {
                    "request_id": f"rqkv-grad-{alias}-shell-{shell_index:02d}",
                    "core_alias": alias,
                    "shell": shell,
                    "canonical_edge_id": shell_index,
                    "canonical_shell_candidate_index": 0,
                    "shell_candidate_count": 1,
                    "radial_shell_index": shell_index,
                    "distance_um": float(25 + shell_index * 100),
                    "source_node": shell_index,
                    "receiver_node": shell_index + 1,
                    "source_feature_index": shell_index,
                    "source_feature_name": f"source-{shell_index}",
                    "target_feature_index": shell_index + 10,
                    "target_feature_name": f"target-{shell_index}",
                    "attention_head": "mean",
                    "requested_layer": -1,
                    "layer_number": 3,
                    "graph_sha256": str(alias_index + 1) * 64,
                    "fixed_mask_seed": 100 + alias_index,
                    "fixed_mask_sha256": str(alias_index) * 64,
                    "assert_directed_edge": True,
                    "assert_source_feature_observed": True,
                    "assert_target_feature_masked": True,
                    "source_feature_observed": True,
                    "target_feature_masked": True,
                }
            )
    members: list[ValidatedMember] = []
    extractions: list[SeedCompactExtraction] = []
    for seed in range(4):
        completed = 175 + 25 * (seed % 2)
        members.append(
            ValidatedMember(
                seed=seed,
                run_id=f"run-{seed}",
                run_root=tmp_path / f"run-{seed}",
                checkpoint_path=tmp_path / f"run-{seed}/checkpoints/last.ckpt",
                checkpoint_sha256=str(seed + 1) * 64,
                receipt_path=tmp_path / f"receipt-{seed}.json",
                receipt_file_sha256=str(seed + 2) * 64,
                receipt_content_sha256=str(seed + 3) * 64,
                completed_global_epochs=completed,
                loss_curve=np.linspace(0.5, 0.25 + seed * 0.001, completed),
                fixed_metrics={
                    "fit/uniform_per_cell/masked_huber": 0.2 + seed * 0.01,
                    "fit/uniform_per_cell/masked_mae": 0.3 + seed * 0.01,
                    "fit/uniform_per_cell/masked_mse": 0.4 + seed * 0.01,
                    "fit/uniform_per_cell/masked_r2": 0.5 - seed * 0.01,
                },
                fixed_masks=fixed_masks,
                model_state_sha256=str(seed + 4) * 64,
                history_sha256=str(seed + 5) * 64,
                parameter_count=5_003_016,
                model_construction_sha256="f" * 64,
                mask_base_seed=2_026_082_401,
                core_order_seed=2_026_082_402,
                training_schedule_epoch_sha256=("e" * 64,) * completed,
            )
        )
        mutual_paths: dict[str, Path] = {}
        mutual_top: dict[str, np.ndarray] = {}
        for alias_index, alias in enumerate(CANCER_ALIASES):
            values = np.roll(
                np.linspace(0.0, 1.0, 120), seed * 5
            ) + alias_index * 0.001
            path = tmp_path / f"mutual-{seed}-{alias}.npy"
            np.save(path, values.astype(np.float32), allow_pickle=False)
            mutual_paths[alias] = path
            identity = vectorized_mutual_pair_scores(
                np.asarray(batches[alias_index].edge_index),
                np.zeros(batches[alias_index].n_edges, dtype=np.float32),
                n_nodes=batches[alias_index].n_nodes,
            )
            mutual_top[alias] = top_mutual_pair_positions(
                values,
                identity.pair_keys,
            )
        order = orders[seed]
        gradients = tuple(
            {
                "seed": seed,
                **row,
                "attention_value": 0.1,
                "prediction_value": 0.2,
                "d_attention_d_source_feature": (
                    (index + 1) * (seed + 1) * 1e-4
                ),
                "d_prediction_d_source_feature": (
                    (-1.0 if index % 2 else 1.0) * (index + seed + 1) * 1e-3
                ),
            }
            for index, row in enumerate(gradient_template)
        )
        attention = np.asarray(base_attention[:, order], dtype=np.float32)
        content = np.asarray((base_attention * 2.0)[:, order], dtype=np.float32)
        bias = np.asarray((-base_attention * 0.25)[:, order], dtype=np.float32)
        extractions.append(
            SeedCompactExtraction(
                seed=seed,
                completed_global_epochs=completed,
                embedding_ids=embedding_ids,
                embeddings=base_embedding + seed * 0.001,
                fixed_edge_ids=fixed_edge_ids,
                fixed_edge_core=fixed_edge_core,
                fixed_edge_number=fixed_edge_number,
                fixed_edge_source=np.tile(np.arange(10), len(CANCER_ALIASES)),
                fixed_edge_receiver=np.zeros(10 * len(CANCER_ALIASES), dtype=np.int64),
                attention=attention,
                content_logits=content,
                positional_bias=bias,
                combined_logits=content + bias,
                mutual_score_paths=mutual_paths,
                mutual_top_positions=mutual_top,
                gradient_rows=gradients,
                fixed_mask_receipts=fixed_masks,
            )
        )
    return tuple(members), tuple(extractions), batches


def _write_synthetic_locked_request_csv(
    path: Path,
    gradient_rows: tuple[dict[str, object], ...],
) -> None:
    fieldnames = (
        "request_schema",
        "request_id",
        "core_alias",
        "layer",
        "attention_head",
        "canonical_edge_id",
        "canonical_shell_candidate_index",
        "shell_candidate_count",
        "source_node",
        "receiver_node",
        "source_feature",
        "source_feature_index",
        "source_feature_name",
        "target_feature",
        "target_feature_index",
        "target_feature_name",
        "distance_um",
        "radial_shell_index",
        "radial_shell",
        "graph_sha256",
        "fixed_mask_seed",
        "fixed_mask_sha256",
        "assert_directed_edge",
        "assert_shell_membership",
        "assert_source_feature_observed",
        "assert_target_feature_masked",
        "assert_feature_indices_distinct",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in gradient_rows:
            writer.writerow(
                {
                    "request_schema": (
                        "cancer_6core_relative_qkv_locked_gradient_requests_v1"
                    ),
                    "request_id": row["request_id"],
                    "core_alias": row["core_alias"],
                    "layer": row["requested_layer"],
                    "attention_head": row["attention_head"],
                    "canonical_edge_id": row["canonical_edge_id"],
                    "canonical_shell_candidate_index": row[
                        "canonical_shell_candidate_index"
                    ],
                    "shell_candidate_count": row["shell_candidate_count"],
                    "source_node": row["source_node"],
                    "receiver_node": row["receiver_node"],
                    "source_feature": row["source_feature_index"],
                    "source_feature_index": row["source_feature_index"],
                    "source_feature_name": row["source_feature_name"],
                    "target_feature": row["target_feature_index"],
                    "target_feature_index": row["target_feature_index"],
                    "target_feature_name": row["target_feature_name"],
                    "distance_um": row["distance_um"],
                    "radial_shell_index": row["radial_shell_index"],
                    "radial_shell": row["shell"],
                    "graph_sha256": row["graph_sha256"],
                    "fixed_mask_seed": row["fixed_mask_seed"],
                    "fixed_mask_sha256": row["fixed_mask_sha256"],
                    "assert_directed_edge": "true",
                    "assert_shell_membership": "true",
                    "assert_source_feature_observed": "true",
                    "assert_target_feature_masked": "true",
                    "assert_feature_indices_distinct": "true",
                }
            )


def _synthetic_exact_replay_record(
    shape: tuple[int, ...],
    *,
    atol: float,
    rtol: float,
    checksum_character: str,
) -> dict[str, object]:
    checksum = checksum_character * 64
    return {
        "shape": list(shape),
        "dtype": "float32",
        "element_count": int(np.prod(shape, dtype=np.int64)),
        "first_sha256": checksum,
        "second_sha256": checksum,
        "byte_identical": True,
        "within_tolerance": True,
        "failing_element_count": 0,
        "maximum_absolute_difference": 0.0,
        "mean_absolute_difference": 0.0,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _synthetic_strict_receipt(
    member: ValidatedMember,
) -> dict[str, object]:
    per_core: list[dict[str, object]] = []
    for alias_index, alias in enumerate(CANCER_ALIASES):
        selected_edge_count = 60
        channels = {
            name: _synthetic_exact_replay_record(
                (selected_edge_count, 8),
                atol=stability_module.DEFAULT_ATTENTION_ATOL,
                rtol=stability_module.DEFAULT_ATTENTION_RTOL,
                checksum_character="abcdef"[alias_index],
            )
            for name in ("attention", "content", "positional_bias", "combined")
        }
        per_core.append(
            {
                "alias": alias,
                "n_nodes": 16,
                "mask_seed": 100 + alias_index,
                "mask_checksum": str(alias_index) * 64,
                "n_masked_entries": 500,
                "prediction_replay": _synthetic_exact_replay_record(
                    (16, 1000),
                    atol=stability_module.DEFAULT_PREDICTION_ATOL,
                    rtol=stability_module.DEFAULT_PREDICTION_RTOL,
                    checksum_character="abcdef"[alias_index],
                ),
                "selected_attention_replay": {
                    "selected_directed_edges": selected_edge_count,
                    "selected_receivers": [0, 5, 10, 15],
                    "attention_heads": 8,
                    "edge_index_sha256": "a" * 64,
                    "edge_index_reload_sha256": "a" * 64,
                    "maximum_logit_composition_error": 0.0,
                    "maximum_receiver_head_normalization_error": 0.0,
                    "channels": channels,
                },
            }
        )
    receipt: dict[str, object] = {
        "schema": stability_module.VERIFICATION_SCHEMA,
        "status": "passed",
        "campaign_id": stability_module.CAMPAIGN_ID,
        "model_seed": member.seed,
        "run_id": member.run_id,
        "checkpoint": {
            "file_sha256": member.checkpoint_sha256,
            "completed_global_epochs": member.completed_global_epochs,
            "model_state_sha256": member.model_state_sha256,
            "history_sha256": member.history_sha256,
            "independently_reloadable": True,
            "independent_reload_count": 2,
        },
        "plateau_verification": {
            "status": "passed",
            "recomputed_decision": {
                "should_stop": True,
                "model_seed": member.seed,
                "final_epoch": member.completed_global_epochs,
                "validation_or_test_metric": False,
                "checkpoint_selection_metric": False,
            },
        },
        "execution": {
            "deterministic": True,
            "deterministic_warn_only": False,
            "deterministic_seed": member.seed,
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only_enabled": False,
            "cublas_workspace_config": ":4096:8",
        },
        "held_in_fit_replay": {
            "status": "passed",
            "role": "held_in_fit_diagnostic_not_validation_or_test",
            "fixed_masks_identical_across_reloads": True,
            "per_core": per_core,
            "equal_core_metrics": dict(member.fixed_metrics),
        },
        "generalization_claim_supported": False,
        "causal_claim_supported": False,
    }
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)
    return receipt


def _refresh_analysis_manifest(output: Path) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = stability_module._file_manifest(output)
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_compact_four_seed_analysis_aligns_inputs_and_serializes_all_metrics(
    tmp_path: Path,
) -> None:
    members, extractions, batches = _synthetic_four_seed_inputs(tmp_path)
    result = compute_four_seed_analysis(
        members,
        extractions,
        batches,
        protocol_provenance={"protocol_sha256": "a" * 64},
    )
    assert result.report["scope"] == "four_seed_ensemble_spread_seeds_0_1_2_3"
    assert result.report["differing_final_epochs"]["allowed"] is True
    assert result.report["differing_final_epochs"]["epochs_by_seed"] == {
        "0": 175,
        "1": 200,
        "2": 175,
        "3": 200,
    }
    assert result.report["five_seed_campaign_completion_claim_allowed"] is False
    assert result.report["calibrated_biological_confidence_interval"] is False
    assert np.asarray(result.report["embedding_stability"]["linear_cka"]).shape == (
        4,
        4,
    )
    assert result.tables["training_curves"].num_rows == 750
    assert result.tables["fixed_metrics"].num_rows == 4
    assert result.tables["fixed_probe_edges"].num_rows == 240
    assert result.tables["selected_gradients"].num_rows == 96
    assert result.tables["selected_gradient_stability"].num_rows == 48
    mutual = result.tables["mutual_pair_stability"].to_pylist()
    assert len(mutual) >= 600
    assert all(1 <= row["seed_support_count"] <= 4 for row in mutual)
    assert result.arrays["node_embeddings"]["seed_0"].shape == (96, 6)
    assert result.arrays["attention_head_stability"][
        "reference_to_seed_head"
    ].shape == (4, 3)

    misaligned = list(extractions)
    first = misaligned[3]
    misaligned[3] = SeedCompactExtraction(
        **{
            **first.__dict__,
            "fixed_edge_ids": tuple(reversed(first.fixed_edge_ids)),
        }
    )
    with pytest.raises(FourSeedStabilityError, match="not aligned"):
        compute_four_seed_analysis(
            members,
            misaligned,
            batches,
            protocol_provenance={"protocol_sha256": "a" * 64},
        )


def test_complete_small_bundle_passes_strict_staging_and_published_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members, extractions, batches = _synthetic_four_seed_inputs(tmp_path)
    protocol = tmp_path / "analysis_protocol_selected_gradient_stability_v1.yaml"
    requests = tmp_path / "selected_gradient_requests_v1.csv"
    cohort = tmp_path / "cohort_manifest.json"
    graph = tmp_path / "graph_manifest.json"
    protocol.write_text("analysis_protocol_schema: synthetic\n", encoding="utf-8")
    _write_synthetic_locked_request_csv(requests, extractions[0].gradient_rows)
    cohort.write_text("{}\n", encoding="utf-8")
    graph.write_text("{}\n", encoding="utf-8")
    protocol_sha256 = stability_module.sha256_file(protocol)
    request_sha256 = stability_module.sha256_file(requests)
    monkeypatch.setattr(
        stability_module, "LOCKED_ANALYSIS_PROTOCOL_SHA256", protocol_sha256
    )
    monkeypatch.setattr(
        stability_module, "LOCKED_GRADIENT_REQUEST_TABLE_SHA256", request_sha256
    )
    monkeypatch.setattr(stability_module, "LOCKED_NODE_COUNT", 96)
    monkeypatch.setattr(stability_module, "LOCKED_HIDDEN_DIM", 6)
    monkeypatch.setattr(stability_module, "LOCKED_HEAD_COUNT", 3)
    monkeypatch.setattr(stability_module, "RECEIVER_PROBES_PER_CORE", 1)

    protocol_sidecar = tmp_path / "analysis_protocol_selected_gradient_stability_v1.sha256"
    request_sidecar = tmp_path / "selected_gradient_requests_v1.sha256"
    protocol_sidecar.write_text(
        f"{protocol_sha256}  {protocol.name}\n", encoding="ascii"
    )
    request_sidecar.write_text(
        f"{request_sha256}  {requests.name}\n", encoding="ascii"
    )
    protocol_provenance = {
        "analysis_protocol": {"synthetic": True},
        "analysis_protocol_path": protocol.as_posix(),
        "analysis_protocol_file_sha256": protocol_sha256,
        "analysis_protocol_sidecar_file_sha256": stability_module.sha256_file(
            protocol_sidecar
        ),
        "gradient_request_table_path": requests.as_posix(),
        "gradient_request_table_file_sha256": request_sha256,
        "gradient_request_sidecar_file_sha256": stability_module.sha256_file(
            request_sidecar
        ),
        "cohort_manifest_path": cohort.as_posix(),
        "cohort_manifest_file_sha256": stability_module.sha256_file(cohort),
        "graph_manifest_path": graph.as_posix(),
        "graph_manifest_file_sha256": stability_module.sha256_file(graph),
    }
    computed = compute_four_seed_analysis(
        members,
        extractions,
        batches,
        protocol_provenance=protocol_provenance,
    )

    source_paths: dict[str, Path] = {}
    source_records: dict[str, dict[str, object]] = {}
    for index, relative in enumerate(stability_module.ANALYSIS_SOURCE_FILES):
        path = tmp_path / f"source-{index}.py"
        path.write_text(f"# synthetic source {index}\n", encoding="utf-8")
        source_paths[f"analysis_source_{Path(relative).name}"] = path
        source_records[relative] = {
            "sha256": stability_module.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    dirty_diff = tmp_path / "analysis_git_dirty.diff"
    dirty_diff.write_bytes(b"")
    source = {
        "project_root": tmp_path.as_posix(),
        "git_commit": "a" * 40,
        "git_dirty": False,
        "git_status_porcelain_v1": [],
        "tracked_dirty_diff_sha256": stability_module.hashlib.sha256(b"").hexdigest(),
        "analysis_source_files": source_records,
    }
    selected_gpu = {
        "index": 2,
        "uuid": "GPU-synthetic",
        "pci_bus_id": "00000000:02:00.0",
        "name": "synthetic",
        "driver_version": "synthetic",
        "memory_total_mib": 1,
    }
    execution = {
        "started_at_utc": "2026-08-25T00:00:00+00:00",
        "extraction_finished_at_utc": "2026-08-25T00:00:01+00:00",
        "extraction_runtime_seconds": 1.0,
        "analysis_compute_finished_at_utc": "2026-08-25T00:00:02+00:00",
        "analysis_compute_runtime_seconds": 2.0,
        "bundle_report_finalized_at_utc": "2026-08-25T00:00:02+00:00",
        "command_argv": ["synthetic"],
        "working_directory": tmp_path.as_posix(),
        "python_executable": "python",
        "python_version": "synthetic",
        "platform": "synthetic",
        "package_versions": {},
        "source": source,
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": "2",
        "gpu_binding": {
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": "2",
            "logical_device": "cuda:0",
            "selected_physical_gpu": selected_gpu,
            "compute_applications_before_cuda": [],
            "nvidia_smi_inventory": {"records": [selected_gpu]},
            "torch_device_uuid": "GPU-synthetic",
            "analysis_pid": 123,
            "compute_applications_after_cuda": [
                {
                    "pid": 123,
                    "gpu_uuid": "GPU-synthetic",
                    "process_name": "python",
                }
            ],
        },
        "device": "cuda:0",
        "device_name": "synthetic",
        "device_uuid": "GPU-synthetic",
        "device_compute_capability": [8, 0],
        "device_multiprocessor_count": 1,
        "device_total_memory_bytes": 1,
        "torch_version": "synthetic",
        "torch_cuda_version": "synthetic",
        "deterministic_seed": stability_module.ANALYSIS_DETERMINISTIC_SEED,
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "matmul_tf32_enabled": False,
        "cudnn_tf32_enabled": False,
        "attention_and_embedding_replay_amp_enabled": True,
        "attention_and_embedding_replay_amp_dtype": "float16",
        "selected_derivative_replay_amp_enabled": False,
        "selected_derivative_replay_dtype": "float32",
        "receiver_chunk_size": 512,
        "max_edges_per_chunk": 200_000,
        "receiver_wise_softmax_exact": True,
        "neighbor_sampling": False,
        "models_processed_concurrently": 1,
        "cores_staged_on_cuda_concurrently": 1,
        "peak_cuda_memory_allocated_bytes": 1,
        "runtime_scope_note": "synthetic",
    }
    report = dict(computed.report)
    report["execution"] = execution

    provenance: dict[str, Path] = {
        protocol.name: protocol,
        protocol_sidecar.name: protocol_sidecar,
        requests.name: requests,
        request_sidecar.name: request_sidecar,
        "cohort_manifest.json": cohort,
        "graph_manifest.json": graph,
        "analysis_git_dirty.diff": dirty_diff,
        **source_paths,
    }
    for member, report_member in zip(members, report["members"], strict=True):
        receipt = tmp_path / f"strict-receipt-{member.seed}.json"
        receipt_payload = _synthetic_strict_receipt(member)
        receipt.write_text(
            json.dumps(receipt_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_member["strict_receipt_content_sha256"] = receipt_payload[
            "receipt_content_sha256"
        ]
        report_member["strict_receipt_file_sha256"] = stability_module.sha256_file(
            receipt
        )
        provenance[
            f"seed_{member.seed}_strict_checkpoint_verification.json"
        ] = receipt
    execution_path = tmp_path / "analysis_execution_provenance.json"
    execution_path.write_bytes(stability_module.canonical_json_bytes(execution) + b"\n")
    provenance[execution_path.name] = execution_path

    result = FourSeedAnalysisResult(
        report=report,
        tables=computed.tables,
        arrays=computed.arrays,
    )
    output = tmp_path / "strict-bundle"
    manifest = write_four_seed_analysis_bundle(
        result,
        output,
        provenance_files=provenance,
    )
    verified = verify_four_seed_analysis_bundle(output)
    assert verified["valid"] is True
    assert verified["manifest_content_sha256"] == manifest[
        "manifest_content_sha256"
    ]


def _complete_verified_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    test_complete_small_bundle_passes_strict_staging_and_published_verification(
        tmp_path,
        monkeypatch,
    )
    return tmp_path / "strict-bundle"


def test_strict_verifier_rejects_rehashed_derived_attention_npz_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete_verified_bundle(tmp_path, monkeypatch)
    archive_path = output / "arrays/attention_head_stability.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    corrupted = np.full_like(
        arrays["matched_attention_spearman"],
        0.123456789,
    )
    arrays["matched_attention_spearman"] = corrupted
    archive_path.unlink()
    write_deterministic_npz(archive_path, arrays)

    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["attention_head_stability"]["matched_attention_spearman"] = (
        corrupted.tolist()
    )
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _refresh_analysis_manifest(output)

    with pytest.raises(FourSeedStabilityError):
        verify_four_seed_analysis_bundle(output)


def test_strict_verifier_rejects_rehashed_non_top_mutual_support_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete_verified_bundle(tmp_path, monkeypatch)
    table_path = output / "tables/mutual_pair_stability.parquet"
    rows = pq.read_table(table_path).to_pylist()
    core_rows = [
        index for index, row in enumerate(rows) if row["core_alias"] == "CAN-01"
    ]
    supported = [
        index for index in core_rows if rows[index]["seed_0_top100_support"]
    ]
    unsupported = [
        index for index in core_rows if not rows[index]["seed_0_top100_support"]
    ]
    high = max(supported, key=lambda index: rows[index]["seed_0_score"])
    low = min(unsupported, key=lambda index: rows[index]["seed_0_score"])
    assert rows[high]["seed_0_score"] > rows[low]["seed_0_score"]
    rows[high]["seed_0_top100_support"] = False
    rows[high]["seed_support_count"] -= 1
    rows[low]["seed_0_top100_support"] = True
    rows[low]["seed_support_count"] += 1
    pq.write_table(pa.Table.from_pylist(rows), table_path, compression="zstd")

    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    core_support_counts = [
        int(row["seed_support_count"])
        for row in rows
        if row["core_alias"] == "CAN-01"
    ]
    report["mutual_attention_routing_stability"]["per_core"]["CAN-01"][
        "support_count_range"
    ] = [min(core_support_counts), max(core_support_counts)]
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _refresh_analysis_manifest(output)

    with pytest.raises(FourSeedStabilityError):
        verify_four_seed_analysis_bundle(output)


def test_strict_verifier_rejects_rehashed_gradient_identity_not_in_locked_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete_verified_bundle(tmp_path, monkeypatch)
    table_path = output / "tables/selected_gradients.parquet"
    rows = pq.read_table(table_path).to_pylist()
    rows[0]["core_alias"] = "FAKE"
    rows[0]["source_node"] = 999_999
    pq.write_table(pa.Table.from_pylist(rows), table_path, compression="zstd")
    _refresh_analysis_manifest(output)

    with pytest.raises(FourSeedStabilityError):
        verify_four_seed_analysis_bundle(output)


def test_strict_verifier_rejects_rehashed_invalid_nested_checkpoint_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete_verified_bundle(tmp_path, monkeypatch)
    receipt_path = (
        output / "provenance/seed_0_strict_checkpoint_verification.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["held_in_fit_replay"]["per_core"][0]["prediction_replay"][
        "byte_identical"
    ] = False
    receipt.pop("receipt_content_sha256", None)
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["members"][0]["strict_receipt_content_sha256"] = receipt[
        "receipt_content_sha256"
    ]
    report["members"][0]["strict_receipt_file_sha256"] = (
        stability_module.sha256_file(receipt_path)
    )
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _refresh_analysis_manifest(output)

    with pytest.raises(FourSeedStabilityError):
        verify_four_seed_analysis_bundle(output)


@pytest.mark.parametrize("corruption", ("wrong_arrow_type", "invalid_node_domain"))
def test_strict_verifier_rejects_rehashed_invalid_table_types_and_domains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    output = _complete_verified_bundle(tmp_path, monkeypatch)
    if corruption == "wrong_arrow_type":
        table_path = output / "tables/training_curves.parquet"
        rows = pq.read_table(table_path).to_pylist()
        for row in rows:
            row["seed"] = str(row["seed"])
    else:
        table_path = output / "tables/fixed_probe_edges.parquet"
        rows = pq.read_table(table_path).to_pylist()
        rows[0]["source_node"] = -1
    pq.write_table(pa.Table.from_pylist(rows), table_path, compression="zstd")
    _refresh_analysis_manifest(output)

    with pytest.raises(FourSeedStabilityError):
        verify_four_seed_analysis_bundle(output)
