from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

import spatial_benchmark.relative_qkv_gradient_requests as gradient_request_module
from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES
from spatial_benchmark.relative_qkv_gradient_requests import (
    CAMPAIGN_ID,
    EXPECTED_REQUEST_COUNT,
    GRADIENT_PROTOCOL_SCHEMA,
    GRADIENT_REQUEST_SCHEMA,
    GRADIENT_SELECTION_NAMESPACE,
    RADIAL_SHELLS,
    GradientRequestCoreInputs,
    LockedGradientRequestError,
    freeze_locked_gradient_request_csv,
    generate_locked_gradient_requests,
    group_selected_derivative_requests,
    load_and_verify_locked_gradient_requests,
    locked_gradient_request_rows,
    verify_locked_gradient_protocol,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cores() -> tuple[GradientRequestCoreInputs, ...]:
    coordinates = np.asarray(
        [[0.0, 0.0], [25.0, 0.0], [75.0, 0.0], [225.0, 0.0], [400.0, 0.0]],
        dtype=np.float64,
    )
    pairs = sorted(
        (
            (source, receiver)
            for source, receiver in (
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (0, 3),
                (3, 0),
                (0, 4),
                (4, 0),
            )
        ),
        key=lambda pair: (pair[1], pair[0]),
    )
    edges = np.asarray(pairs, dtype=np.int64).T.copy()
    genes = tuple(f"Gene-{index:02d}" for index in range(8))
    mask = np.fromfunction(
        lambda node, gene: (node + gene) % 2 == 0,
        (len(coordinates), len(genes)),
        dtype=int,
    ).astype(np.bool_)
    return tuple(
        GradientRequestCoreInputs(
            alias=alias,
            edge_index=edges,
            coordinates_um=coordinates,
            gene_names=genes,
            fixed_mask=mask,
            fixed_mask_seed=1000 + alias_index,
            fixed_mask_sha256=_sha(f"mask-{alias}"),
            graph_sha256=_sha(f"graph-{alias}"),
        )
        for alias_index, alias in enumerate(CANCER_ALIASES)
    )


def _protocol(request_sha256: str) -> dict[str, object]:
    return {
        "analysis_protocol_schema": GRADIENT_PROTOCOL_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "ensemble": {
            "active_model_seeds": [0, 1, 2, 3],
            "deferred_model_seeds": [4],
        },
        "fixed_probe_extraction": {
            "layer": -1,
            "receiver_probes_per_core": 64,
            "top_edge_fraction": 0.05,
            "mutual_top_edges_per_core_per_seed": 100,
            "empirical_quantiles": [0.05, 0.25, 0.75, 0.95],
        },
        "selected_gradient_requests": {
            "request_schema": GRADIENT_REQUEST_SCHEMA,
            "request_count": 24,
            "requests_per_core": 4,
            "radial_shells": [shell[2] for shell in RADIAL_SHELLS],
            "request_table_sha256": request_sha256,
        },
    }


def _write_sidecar(target: Path, sidecar: Path) -> None:
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    sidecar.write_text(f"{checksum}  {target.name}\n", encoding="ascii")


def test_generates_exact_model_independent_core_shell_requests() -> None:
    cores = _cores()
    first = generate_locked_gradient_requests(cores)
    second = generate_locked_gradient_requests(cores)
    assert first == second
    assert len(first) == EXPECTED_REQUEST_COUNT == 24
    assert [
        (request.core_alias, request.radial_shell_index)
        for request in first
    ] == [
        (alias, shell_index)
        for alias in CANCER_ALIASES
        for shell_index in range(4)
    ]

    by_alias = {core.alias: core for core in cores}
    for request in first:
        core = by_alias[request.core_alias]
        shell = RADIAL_SHELLS[request.radial_shell_index]
        edge_ids = []
        for edge_id in range(core.edge_index.shape[1]):
            source = int(core.edge_index[0, edge_id])
            receiver = int(core.edge_index[1, edge_id])
            distance = float(
                np.linalg.norm(
                    core.coordinates_um[source] - core.coordinates_um[receiver]
                )
            )
            if shell[0] < distance <= shell[1]:
                edge_ids.append(edge_id)
        payload = (
            f"{GRADIENT_SELECTION_NAMESPACE}\0directed-edge\0"
            f"{request.core_alias}\0{shell[2]}"
        ).encode("utf-8")
        expected_rank = int.from_bytes(hashlib.sha256(payload).digest(), "big") % len(
            edge_ids
        )
        assert request.canonical_shell_candidate_index == expected_rank
        assert request.canonical_edge_id == edge_ids[expected_rank]
        assert request.shell_candidate_count == len(edge_ids)
        assert not core.fixed_mask[
            request.source_node, request.source_feature_index
        ]
        assert core.fixed_mask[
            request.receiver_node, request.target_feature_index
        ]
        assert request.source_feature_index != request.target_feature_index
        assert request.source_feature_name == core.gene_names[
            request.source_feature_index
        ]
        assert request.target_feature_name == core.gene_names[
            request.target_feature_index
        ]
        assert request.layer == -1
        assert request.attention_head == "mean"

    rows = locked_gradient_request_rows(first)
    assert all(row["assert_directed_edge"] == "true" for row in rows)
    assert all(row["request_schema"] == GRADIENT_REQUEST_SCHEMA for row in rows)


def test_freeze_and_regeneration_gate_refuse_checksum_or_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cores = _cores()
    requests = generate_locked_gradient_requests(cores)
    csv_path = tmp_path / "requests.csv"
    sha_path = tmp_path / "requests.sha256"
    checksum = freeze_locked_gradient_request_csv(
        requests,
        request_csv_path=csv_path,
        request_sha256_path=sha_path,
    )
    protocol = _protocol(checksum)
    loaded = load_and_verify_locked_gradient_requests(
        csv_path,
        sha_path,
        cores=cores,
        protocol=protocol,
    )
    assert loaded == requests
    grouped = group_selected_derivative_requests(loaded)
    assert tuple(grouped) == tuple(CANCER_ALIASES)
    assert all(len(group) == 4 for group in grouped.values())
    assert all(
        request.layer == -1 and request.attention_head is None
        for group in grouped.values()
        for request in group
    )

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        freeze_locked_gradient_request_csv(
            requests,
            request_csv_path=csv_path,
            request_sha256_path=sha_path,
        )

    changed = csv_path.read_text(encoding="utf-8").replace(
        "rqkv-grad-CAN-01-shell-00",
        "rqkv-grad-CAN-01-shell-XX",
        1,
    )
    csv_path.write_text(changed, encoding="utf-8")
    _write_sidecar(csv_path, sha_path)
    with pytest.raises(LockedGradientRequestError, match="canonical.*regeneration"):
        load_and_verify_locked_gradient_requests(
            csv_path,
            sha_path,
            cores=cores,
        )

    original_link = gradient_request_module.os.link
    link_calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("synthetic second-link failure")
        original_link(source, destination)

    monkeypatch.setattr(gradient_request_module.os, "link", fail_second_link)
    failed_csv = tmp_path / "failed.csv"
    failed_sha = tmp_path / "failed.sha256"
    with pytest.raises(OSError, match="synthetic second-link failure"):
        freeze_locked_gradient_request_csv(
            requests,
            request_csv_path=failed_csv,
            request_sha256_path=failed_sha,
        )
    assert not failed_csv.exists()
    assert not failed_sha.exists()
    assert not list(tmp_path.glob(".gradient-requests-*"))


def test_protocol_checksum_and_locked_settings_are_verified(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.yaml"
    sidecar_path = tmp_path / "protocol.sha256"
    protocol_path.write_text(
        yaml.safe_dump(_protocol("0" * 64), sort_keys=False),
        encoding="utf-8",
    )
    _write_sidecar(protocol_path, sidecar_path)
    loaded = verify_locked_gradient_protocol(protocol_path, sidecar_path)
    assert loaded["ensemble"]["active_model_seeds"] == [0, 1, 2, 3]

    changed = dict(loaded)
    changed["fixed_probe_extraction"] = dict(loaded["fixed_probe_extraction"])
    changed["fixed_probe_extraction"]["receiver_probes_per_core"] = 63
    protocol_path.write_text(
        yaml.safe_dump(changed, sort_keys=False),
        encoding="utf-8",
    )
    _write_sidecar(protocol_path, sidecar_path)
    with pytest.raises(LockedGradientRequestError, match="settings drifted"):
        verify_locked_gradient_protocol(protocol_path, sidecar_path)


def test_rejects_missing_gene_eligibility_and_noncanonical_inputs() -> None:
    cores = list(_cores())
    first = cores[0]
    impossible_mask = np.ones_like(first.fixed_mask)
    cores[0] = GradientRequestCoreInputs(
        alias=first.alias,
        edge_index=first.edge_index,
        coordinates_um=first.coordinates_um,
        gene_names=first.gene_names,
        fixed_mask=impossible_mask,
        fixed_mask_seed=first.fixed_mask_seed,
        fixed_mask_sha256=first.fixed_mask_sha256,
        graph_sha256=first.graph_sha256,
    )
    with pytest.raises(LockedGradientRequestError, match="observed-source-feature"):
        generate_locked_gradient_requests(cores)

    reordered = list(_cores())
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(LockedGradientRequestError, match="exact ordered six"):
        generate_locked_gradient_requests(reordered)
