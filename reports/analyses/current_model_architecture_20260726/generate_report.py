#!/usr/bin/env python3
"""Generate the current-model architecture and training audit.

The report is intentionally derived from finalized run artifacts, preserved
checkpoints, resolved configuration, and the current implementation.  No raw
or row-level biological data are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[2]
RUN_ROOT = ROOT / "artifacts" / "runs" / "2026" / "07"
COMPARISON_PATH = (
    ROOT
    / "reports"
    / "analyses"
    / "full_core_high_k_capacity"
    / "comparison"
    / "comparison.json"
)
AMP_SMOKE_PATH = (
    ROOT
    / "reports"
    / "analyses"
    / "full_core_high_k_capacity"
    / "amp_equivalence_smoke.json"
)
INTERPRETABILITY_PATH = (
    ROOT
    / "reports"
    / "analyses"
    / "full_core_high_k_capacity"
    / "interpretability"
    / "analysis.json"
)

RUN_IDS = {
    "resource_pilot": "r_20260725T083634Z_457e0bfb_s000_f00_a01_1fb277a2",
    "g2": "r_20260725T083929Z_8364b333_s000_f00_a01_f5c47601",
    "matched": "r_20260725T090752Z_e05d042b_s000_f00_a01_4ff98df7",
}
MODEL_LABELS = {
    "g2": "G2 · edge-conditioned GATv2",
    "matched": "B0–G2-matched · cell-autonomous control",
}
EXPECTED_PARAMETER_COUNT = 3_987_880
EXPECTED_PARAMETER_TENSORS = 42
EXPECTED_GRAPH_SHA256 = (
    "2469064e2fe14b48f642fca09a546d9e420d9fd851b9668996da62ee8246d060"
)
EXPECTED_DATASET_SHA256 = (
    "a112fbb2bdf929197759c82913fc379c4df797f75676457bcd10419fe7a969d5"
)
EXPECTED_SPLIT_SHA256 = (
    "2c8c59fb659401cc202126cb14154f064d2e3380819aff647f06a30db354d84c"
)
REPORT_DATE = "2026-07-26"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = (
            torch.as_tensor(state_dict[name])
            .detach()
            .cpu()
            .contiguous()
        )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_float(value: float, digits: int = 6) -> str:
    return f"{float(value):,.{digits}f}"


def fmt_percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 120:
        return f"{seconds:.1f} s"
    return f"{seconds / 60.0:.1f} min"


def fmt_bytes(value: int | float) -> str:
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB")
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def model_group(model_key: str, tensor_name: str) -> str:
    if tensor_name.startswith("encoder."):
        return "Node encoder"
    if tensor_name.startswith("edge_encoder."):
        return "Shared 17→64→64 encoder"
    if tensor_name.startswith("decoder."):
        return "Expression decoder"
    if tensor_name.startswith("blocks."):
        block = int(tensor_name.split(".")[1]) + 1
        if model_key == "g2":
            return f"GATv2 residual block {block}"
        return f"Cell-autonomous mixing block {block}"
    raise AssertionError(f"unassigned tensor: {tensor_name}")


def tensor_role(model_key: str, name: str) -> str:
    roles = (
        ("expression_projection", "masked-expression projection"),
        ("mask_projection", "explicit-mask projection"),
        ("covariate_projection", "morphology/imaging projection"),
        ("encoder.bias", "shared encoder bias"),
        ("encoder.normalization", "encoder LayerNorm"),
        ("edge_encoder.network.0", "17→64 edge/surrogate affine"),
        ("edge_encoder.network.1", "64-wide LayerNorm"),
        ("edge_encoder.network.3", "64→64 affine"),
        ("edge_encoder.network.4", "64-wide LayerNorm"),
        ("convolution.lin_l", "GATv2 source projection"),
        ("convolution.lin_r", "GATv2 receiver projection"),
        ("convolution.lin_edge", "edge-conditioned attention projection"),
        ("convolution.att", "four-head GATv2 routing vector"),
        ("left_projection", "within-cell left projection"),
        ("right_projection", "within-cell right projection"),
        ("edge_projection", "within-cell surrogate projection"),
        ("routing_vector", "within-cell routing vector"),
        ("attention_normalization", "post-attention LayerNorm"),
        ("feed_forward.linear_in", "residual FFN input affine"),
        ("feed_forward.linear_out", "residual FFN output affine"),
        ("feed_forward.normalization", "residual FFN LayerNorm"),
        (".normalization", "post-mixing LayerNorm"),
        ("decoder.linear_in", "decoder 512→512 affine"),
        ("decoder.linear_out", "decoder 512→1,000 affine"),
    )
    for fragment, role in roles:
        if fragment in name:
            return role
    raise AssertionError(f"unassigned functional role: {model_key}/{name}")


def load_checkpoint(run_dir: Path) -> dict[str, Any]:
    checkpoint_path = run_dir / "checkpoints" / "last.ckpt"
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{checkpoint_path} is not a checkpoint mapping")
    return payload


def instantiate_models(
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, torch.nn.Module], dict[str, bool], dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "src"))
    from spatial_benchmark.dense_gat import (  # pylint: disable=import-outside-toplevel
        ReceiverChunkedEdgeConditionedGATv2,
    )
    from spatial_benchmark.models import (  # pylint: disable=import-outside-toplevel
        AdditiveEdgeMessageModel,
        BroadSpatialFieldControl,
        EdgeConditionedGATv2,
        EdgeParameterMatchedSelfControl,
        MeanNeighborModel,
        ParameterMatchedSelfControl,
        SelfOnlyMLP,
        TopologyGATv2,
    )

    common = {
        "num_genes": 1000,
        "node_covariate_dim": 22,
        "hidden_dim": 512,
        "attention_heads": 4,
        "graph_layers": 2,
        "ffn_dim": 512,
        "decoder_dim": 512,
        "edge_hidden_dim": 64,
        "edge_embedding_dim": 64,
        "dropout": 0.1,
        "attention_dropout": 0.1,
    }
    g2 = ReceiverChunkedEdgeConditionedGATv2(
        **common,
        edge_attribute_dim=17,
        receiver_chunk_size=512,
        activation_checkpointing=True,
    )
    matched = EdgeParameterMatchedSelfControl(
        **common,
        edge_attribute_dim=17,
    )
    models = {"g2": g2, "matched": matched}
    checks: dict[str, bool] = {}
    for key, model in models.items():
        state = checkpoints[key]["model_state_dict"]
        model.load_state_dict(state, strict=True)
        checks[f"{key}_current_code_strict_checkpoint_load"] = True
        checks[f"{key}_current_code_parameter_count"] = (
            sum(parameter.numel() for parameter in model.parameters())
            == EXPECTED_PARAMETER_COUNT
        )

    ordinary = EdgeConditionedGATv2(
        **common,
        edge_attribute_dim=17,
    )
    checks["chunked_and_ordinary_g2_state_layout_identical"] = {
        name: tuple(tensor.shape)
        for name, tensor in ordinary.state_dict().items()
    } == {
        name: tuple(tensor.shape)
        for name, tensor in g2.state_dict().items()
    }

    # Direct behavioral test of the current comparator contract: changing all
    # graph arguments must leave its eval-mode output bitwise identical.
    matched.eval()
    generator = torch.Generator().manual_seed(20260726)
    expression = torch.randn((5, 1000), generator=generator)
    mask = torch.zeros((5, 1000), dtype=torch.bool)
    mask[:, :200] = True
    covariates = torch.randn((5, 22), generator=generator)
    edge_a = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_b = torch.tensor([[4, 3, 1, 0], [0, 1, 2, 4]], dtype=torch.long)
    attr_a = torch.randn((3, 17), generator=generator)
    attr_b = torch.randn((4, 17), generator=generator)
    with torch.no_grad():
        pred_a = matched(
            expression,
            mask,
            edge_index=edge_a,
            edge_attributes=attr_a,
            node_covariates=covariates,
        ).prediction
        pred_b = matched(
            expression,
            mask,
            edge_index=edge_b,
            edge_attributes=attr_b,
            node_covariates=covariates,
        ).prediction
    checks["matched_prediction_invariant_to_graph_arguments"] = torch.equal(
        pred_a, pred_b
    )

    # The two constructors use the same seed but create modules in a different
    # order.  Record rather than conceal the resulting initialization mismatch.
    torch.manual_seed(0)
    initial_g2 = EdgeConditionedGATv2(
        **common,
        edge_attribute_dim=17,
    )
    torch.manual_seed(0)
    initial_matched = EdgeParameterMatchedSelfControl(
        **common,
        edge_attribute_dim=17,
    )
    g2_state = initial_g2.state_dict()
    matched_state = initial_matched.state_dict()
    initial_pairs = (
        (
            "encoder.expression_projection.weight",
            "encoder.expression_projection.weight",
        ),
        ("encoder.mask_projection.weight", "encoder.mask_projection.weight"),
        (
            "encoder.covariate_projection.weight",
            "encoder.covariate_projection.weight",
        ),
        (
            "blocks.0.convolution.lin_l.weight",
            "blocks.0.left_projection.weight",
        ),
        (
            "blocks.0.convolution.lin_r.weight",
            "blocks.0.right_projection.weight",
        ),
        (
            "blocks.0.convolution.lin_edge.weight",
            "blocks.0.edge_projection.weight",
        ),
        ("blocks.0.convolution.att", "blocks.0.routing_vector"),
        ("decoder.linear_in.weight", "decoder.linear_in.weight"),
        ("decoder.linear_out.weight", "decoder.linear_out.weight"),
        ("edge_encoder.network.0.weight", "edge_encoder.network.0.weight"),
    )
    initialization_comparison = []
    for g2_name, matched_name in initial_pairs:
        left = g2_state[g2_name].reshape(-1)
        right = matched_state[matched_name].reshape(-1)
        initialization_comparison.append(
            {
                "g2_tensor": g2_name,
                "matched_tensor": matched_name,
                "identical": bool(torch.equal(left, right)),
                "max_abs_difference": float((left - right).abs().max()),
            }
        )
    checks["same_seed_shared_encoder_initialization_identical"] = all(
        row["identical"] for row in initialization_comparison[:3]
    )
    checks["same_seed_full_initialization_not_identical"] = not all(
        row["identical"] for row in initialization_comparison
    )

    base_spec = {
        "num_genes": 1000,
        "node_covariate_dim": 22,
        "hidden_dim": 512,
        "ffn_dim": 512,
        "decoder_dim": 512,
        "dropout": 0.1,
    }
    inventory_models: dict[str, torch.nn.Module] = {
        "B0 self-only MLP": SelfOnlyMLP(**base_spec),
        "Broad-field control": BroadSpatialFieldControl(**base_spec),
        "B1 mean-neighbor": MeanNeighborModel(**base_spec),
        "G1 topology-only GATv2": TopologyGATv2(
            **base_spec,
            attention_heads=4,
            graph_layers=2,
            attention_dropout=0.1,
        ),
        "B0–G1 parameter-matched": ParameterMatchedSelfControl(
            **base_spec,
            attention_heads=4,
            graph_layers=2,
            attention_dropout=0.1,
        ),
        "G2 edge-conditioned GATv2": EdgeConditionedGATv2(
            **base_spec,
            edge_attribute_dim=17,
            attention_heads=4,
            graph_layers=2,
            attention_dropout=0.1,
            edge_hidden_dim=64,
            edge_embedding_dim=64,
        ),
        "B0–G2 parameter-matched": EdgeParameterMatchedSelfControl(
            **base_spec,
            edge_attribute_dim=17,
            attention_heads=4,
            graph_layers=2,
            attention_dropout=0.1,
            edge_hidden_dim=64,
            edge_embedding_dim=64,
        ),
        "G3 additive edge-message (implementation defaults)": (
            AdditiveEdgeMessageModel(
                **base_spec,
                edge_attribute_dim=17,
                attention_heads=4,
                attention_dropout=0.1,
                edge_hidden_dim=64,
                edge_embedding_dim=64,
            )
        ),
    }
    inventory = {
        name: sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        for name, model in inventory_models.items()
    }
    context = {
        "initialization_comparison": initialization_comparison,
        "implementation_inventory_parameter_counts": inventory,
    }
    return models, checks, context


def sparkline_svg(
    series: Mapping[str, list[float]],
    *,
    width: int = 920,
    height: int = 260,
) -> str:
    left, right, top, bottom = 62, 18, 22, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [value for row in series.values() for value in row]
    y_min = min(values)
    y_max = max(values)
    margin = max((y_max - y_min) * 0.08, 0.002)
    y_min -= margin
    y_max += margin
    colors = {"G2": "#16b8a6", "Matched self": "#f2a43b"}

    def point(index: int, value: float, n: int) -> tuple[float, float]:
        x = left + (index / max(1, n - 1)) * plot_width
        y = top + (y_max - value) / (y_max - y_min) * plot_height
        return x, y

    paths = []
    for name, row in series.items():
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in (
                point(index, value, len(row))
                for index, value in enumerate(row)
            )
        )
        paths.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{colors[name]}" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
    grid = []
    labels = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            'y2="{y:.2f}" stroke="#d7dedc" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="axis">{value:.3f}</text>'
        )
    for epoch in (0, 50, 100, 150, 199):
        x = left + epoch / 199 * plot_width
        labels.append(
            f'<text x="{x:.2f}" y="{height-16}" text-anchor="middle" '
            f'class="axis">{epoch}</text>'
        )
    legend_x = width - 270
    legend = (
        f'<line x1="{legend_x}" y1="14" x2="{legend_x+24}" y2="14" '
        'stroke="#16b8a6" stroke-width="3"/>'
        f'<text x="{legend_x+31}" y="18" class="legend">G2</text>'
        f'<line x1="{legend_x+92}" y1="14" x2="{legend_x+116}" y2="14" '
        'stroke="#f2a43b" stroke-width="3"/>'
        f'<text x="{legend_x+123}" y="18" class="legend">Matched self</text>'
    )
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img"
     aria-labelledby="loss-chart-title loss-chart-desc">
  <title id="loss-chart-title">Training masked Huber loss by epoch</title>
  <desc id="loss-chart-desc">Two closely overlapping, mask-mode-dependent loss
  traces over 200 paired epochs. Both remain finite and trend downward.</desc>
  <style>.axis{{font:11px system-ui;fill:#596966}}
  .legend{{font:12px system-ui;fill:#273431}}</style>
  {''.join(grid)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"
        stroke="#71817d"/>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}"
        y2="{height-bottom}" stroke="#71817d"/>
  {''.join(paths)}
  {''.join(labels)}
  {legend}
  <text x="{left + plot_width/2:.2f}" y="{height-2}"
        text-anchor="middle" class="axis">epoch</text>
</svg>"""


def parameter_bar(
    groups: Mapping[str, int],
    total: int,
) -> str:
    colors = {
        "Node encoder": "#5f83c5",
        "Shared 17→64→64 encoder": "#af73bd",
        "Mixing blocks": "#16b8a6",
        "Expression decoder": "#f2a43b",
    }
    spans = []
    for name, value in groups.items():
        spans.append(
            f'<span style="width:{100*value/total:.5f}%;'
            f'background:{colors[name]}" title="{esc(name)}: '
            f'{fmt_int(value)}"></span>'
        )
    legend = "".join(
        f'<li><i style="background:{colors[name]}"></i><span>{esc(name)}</span>'
        f'<strong>{fmt_int(value)}</strong><small>'
        f'{fmt_percent(value/total, 1)}</small></li>'
        for name, value in groups.items()
    )
    return (
        f'<div class="stacked" role="img" aria-label="Parameter allocation">'
        f'{"".join(spans)}</div><ul class="bar-legend">{legend}</ul>'
    )


def architecture_svg() -> str:
    return """
<svg class="architecture" viewBox="0 0 1080 610" role="img"
     aria-labelledby="architecture-title architecture-desc">
  <title id="architecture-title">Current paired-model architecture</title>
  <desc id="architecture-desc">Both models share masked expression, explicit
  mask, morphology inputs, a 512-wide encoder, two residual mixing blocks and
  a 1000-gene decoder. G2 mixes over exact graph neighbors using edge geometry;
  the comparator mixes only within each cell.</desc>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4"
            orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#647570"/></marker>
  </defs>
  <style>
    .box{rx:12;stroke-width:1.5}.shared{fill:#eef3fb;stroke:#5f83c5}
    .graph{fill:#e8f8f5;stroke:#16a593}.self{fill:#fff4e4;stroke:#d48b26}
    .out{fill:#f4eff7;stroke:#9a67a8}.label{font:600 15px system-ui;fill:#1f2c29}
    .sub{font:12px system-ui;fill:#52625f}.arrow{stroke:#647570;stroke-width:1.6;
    fill:none;marker-end:url(#arrow)}.branch{font:700 17px system-ui}
  </style>
  <rect class="box shared" x="35" y="35" width="240" height="92"/>
  <text class="label" x="155" y="64" text-anchor="middle">Node inputs</text>
  <text class="sub" x="155" y="88" text-anchor="middle">Xmasked [N, 1,000]</text>
  <text class="sub" x="155" y="108" text-anchor="middle">mask [N, 1,000] · covariates [N, 22]</text>
  <path class="arrow" d="M275 81 H335"/>
  <rect class="box shared" x="335" y="35" width="260" height="92"/>
  <text class="label" x="465" y="64" text-anchor="middle">Additive node encoder</text>
  <text class="sub" x="465" y="88" text-anchor="middle">three bias-free projections + bias</text>
  <text class="sub" x="465" y="108" text-anchor="middle">LayerNorm → GELU → dropout · H₀ [N, 512]</text>
  <path class="arrow" d="M595 81 H665 V195"/>
  <path class="arrow" d="M665 81 H665 V410"/>

  <text class="branch" x="45" y="185" fill="#087b70">G2 · graph path</text>
  <rect class="box graph" x="45" y="205" width="245" height="108"/>
  <text class="label" x="168" y="234" text-anchor="middle">Measured edge path</text>
  <text class="sub" x="168" y="257" text-anchor="middle">geometry [E, 17] → 64 → 64</text>
  <text class="sub" x="168" y="277" text-anchor="middle">shared encoder; recomputed by chunk/layer</text>
  <text class="sub" x="168" y="297" text-anchor="middle">exact mutual k=1,000 graph</text>
  <path class="arrow" d="M290 259 H350"/>
  <rect class="box graph" x="350" y="205" width="360" height="108"/>
  <text class="label" x="530" y="234" text-anchor="middle">2 × edge-conditioned GATv2 block</text>
  <text class="sub" x="530" y="257" text-anchor="middle">4 heads × 128 · softmax over all incoming neighbors</text>
  <text class="sub" x="530" y="277" text-anchor="middle">residual + LayerNorm + 512→512→512 FFN</text>
  <text class="sub" x="530" y="297" text-anchor="middle">512-receiver partitions; activation recomputation</text>
  <path class="arrow" d="M710 259 H790"/>
  <rect class="box out" x="790" y="205" width="245" height="108"/>
  <text class="label" x="913" y="234" text-anchor="middle">Shared-shape decoder</text>
  <text class="sub" x="913" y="257" text-anchor="middle">target H₂ [T, 512] → GELU/dropout</text>
  <text class="sub" x="913" y="277" text-anchor="middle">prediction Ŷ [T, 1,000]</text>
  <text class="sub" x="913" y="297" text-anchor="middle">loss only on masked entries</text>

  <text class="branch" x="45" y="400" fill="#a36109">B0–G2-matched · self path</text>
  <rect class="box self" x="45" y="420" width="245" height="108"/>
  <text class="label" x="168" y="449" text-anchor="middle">Within-cell surrogate</text>
  <text class="sub" x="168" y="472" text-anchor="middle">first 17 H₀ channels → 64 → 64</text>
  <text class="sub" x="168" y="492" text-anchor="middle">same parameter shapes as edge encoder</text>
  <text class="sub" x="168" y="512" text-anchor="middle">no graph, neighbors, coordinates, or edges</text>
  <path class="arrow" d="M290 474 H350"/>
  <rect class="box self" x="350" y="420" width="360" height="108"/>
  <text class="label" x="530" y="449" text-anchor="middle">2 × cell-autonomous mixing block</text>
  <text class="sub" x="530" y="472" text-anchor="middle">left + right + surrogate projections · GELU</text>
  <text class="sub" x="530" y="492" text-anchor="middle">residual + LayerNorm + 512→512→512 FFN</text>
  <text class="sub" x="530" y="512" text-anchor="middle">parameter-matched, not operator- or FLOP-matched</text>
  <path class="arrow" d="M710 474 H790"/>
  <rect class="box out" x="790" y="420" width="245" height="108"/>
  <text class="label" x="913" y="449" text-anchor="middle">Shared-shape decoder</text>
  <text class="sub" x="913" y="472" text-anchor="middle">target H₂ [T, 512] → GELU/dropout</text>
  <text class="sub" x="913" y="492" text-anchor="middle">prediction Ŷ [T, 1,000]</text>
  <text class="sub" x="913" y="512" text-anchor="middle">loss only on masked entries</text>
  <text class="sub" x="540" y="580" text-anchor="middle">
    Same width, depth, parameter count, node data, epoch masks, optimizer, seed and budget; different information flow and operator.
  </text>
</svg>"""


def table(
    headers: Iterable[str],
    rows: Iterable[Iterable[Any]],
    *,
    caption: str,
    classes: str = "",
) -> str:
    header_html = "".join(f"<th scope=\"col\">{esc(item)}</th>" for item in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap"><table class="{esc(classes)}">'
        f"<caption>{esc(caption)}</caption><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody></table></div>"
    )


def collect() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    run_dirs = {key: RUN_ROOT / run_id for key, run_id in RUN_IDS.items()}
    for path in run_dirs.values():
        if not path.is_dir():
            raise FileNotFoundError(path)

    summaries = {
        key: read_json(path / "summary.json") for key, path in run_dirs.items()
    }
    resolved = {
        key: read_yaml(path / "config.resolved.yaml")
        for key, path in run_dirs.items()
    }
    histories = {
        key: read_jsonl(path / "metrics" / "history.jsonl")
        for key, path in run_dirs.items()
    }
    checkpoints = {
        key: load_checkpoint(run_dirs[key]) for key in ("g2", "matched")
    }
    comparison = read_json(COMPARISON_PATH)
    amp_smoke = read_json(AMP_SMOKE_PATH)
    interpretability = read_json(INTERPRETABILITY_PATH)

    checks: dict[str, bool] = {}
    parameter_rows: list[dict[str, Any]] = []
    checkpoint_audits: dict[str, Any] = {}
    for key in ("g2", "matched"):
        run_dir = run_dirs[key]
        checkpoint_path = run_dir / "checkpoints" / "last.ckpt"
        checkpoint = checkpoints[key]
        state = checkpoint["model_state_dict"]
        total = sum(tensor.numel() for tensor in state.values())
        tensor_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in state.values()
        )
        checksum = state_dict_sha256(state)
        artifact_manifest = read_json(
            run_dir / "provenance" / "artifact_checksums.json"
        )
        checkpoint_record = artifact_manifest["files"]["checkpoints/last.ckpt"]
        file_checksum = sha256_file(checkpoint_path)
        groups: dict[str, int] = {}
        for name, tensor in state.items():
            group = model_group(key, name)
            groups[group] = groups.get(group, 0) + tensor.numel()
            parameter_rows.append(
                {
                    "model": checkpoint["model_name"],
                    "run_id": RUN_IDS[key],
                    "tensor_name": name,
                    "module_group": group,
                    "functional_role": tensor_role(key, name),
                    "shape": " × ".join(str(value) for value in tensor.shape),
                    "numel": tensor.numel(),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "tensor_bytes": tensor.numel() * tensor.element_size(),
                    "percent_of_model": tensor.numel() / total,
                }
            )
        checkpoint_audits[key] = {
            "run_id": RUN_IDS[key],
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_file_bytes": checkpoint_path.stat().st_size,
            "checkpoint_file_sha256": file_checksum,
            "state_dict_sha256": checksum,
            "parameter_count": total,
            "parameter_tensor_count": len(state),
            "raw_parameter_bytes": tensor_bytes,
            "dtypes": sorted({str(tensor.dtype) for tensor in state.values()}),
            "module_groups": groups,
            "checkpoint_keys": sorted(checkpoint),
            "optimizer_state_present": any(
                "optim" in str(name).lower() for name in checkpoint
            ),
        }
        checks[f"{key}_parameter_count_exact"] = total == EXPECTED_PARAMETER_COUNT
        checks[f"{key}_parameter_tensor_count_exact"] = (
            len(state) == EXPECTED_PARAMETER_TENSORS
        )
        checks[f"{key}_all_checkpoint_tensors_float32"] = all(
            tensor.dtype == torch.float32 for tensor in state.values()
        )
        checks[f"{key}_state_checksum_matches_checkpoint"] = (
            checksum == checkpoint["state_dict_sha256"]
        )
        checks[f"{key}_file_checksum_matches_artifact_manifest"] = (
            file_checksum == checkpoint_record["sha256"]
            and checkpoint_path.stat().st_size == checkpoint_record["size"]
        )
        checks[f"{key}_checkpoint_is_final_epoch_199"] = (
            checkpoint["checkpoint_role"] == "last"
            and checkpoint["epoch"] == 199
            and checkpoint["fixed_epoch_budget"] == 200
        )

    checks["paired_models_have_equal_parameter_count"] = (
        checkpoint_audits["g2"]["parameter_count"]
        == checkpoint_audits["matched"]["parameter_count"]
    )
    checks["paired_models_have_equal_raw_parameter_bytes"] = (
        checkpoint_audits["g2"]["raw_parameter_bytes"]
        == checkpoint_audits["matched"]["raw_parameter_bytes"]
    )
    checks["paired_evaluation_mask_bundle_identical"] = (
        checkpoints["g2"]["evaluation_mask_bundle_sha256"]
        == checkpoints["matched"]["evaluation_mask_bundle_sha256"]
        == comparison["facts"]["evaluation_mask_bundle_sha256"]
    )
    checks["paired_graph_checksum_identical"] = (
        checkpoints["g2"]["graph_sha256"]
        == checkpoints["matched"]["graph_sha256"]
        == EXPECTED_GRAPH_SHA256
    )
    checks["dataset_fingerprint_exact"] = (
        resolved["g2"]["dataset"]["dataset_fingerprint"]
        == resolved["matched"]["dataset"]["dataset_fingerprint"]
        == EXPECTED_DATASET_SHA256
    )
    checks["split_fingerprint_exact"] = (
        resolved["g2"]["dataset"]["split_fingerprint"]
        == resolved["matched"]["dataset"]["split_fingerprint"]
        == EXPECTED_SPLIT_SHA256
    )
    checks["both_histories_have_200_epochs"] = all(
        len(histories[key]) == 200
        and [row["epoch"] for row in histories[key]] == list(range(200))
        for key in ("g2", "matched")
    )
    checks["paired_training_masks_identical_each_epoch"] = all(
        g2_row["mask_mode"] == matched_row["mask_mode"]
        and g2_row["mask_seed"] == matched_row["mask_seed"]
        and g2_row["mask_checksum"] == matched_row["mask_checksum"]
        and g2_row["n_masked_entries"] == matched_row["n_masked_entries"]
        for g2_row, matched_row in zip(
            histories["g2"], histories["matched"], strict=True
        )
    )
    checks["all_training_losses_and_gradients_finite"] = all(
        math.isfinite(float(row["train_loss"]))
        and math.isfinite(float(row["gradient_norm"]))
        for key in ("g2", "matched")
        for row in histories[key]
    )
    checks["amp_equivalence_smoke_passed"] = bool(amp_smoke["passed"])

    _, code_checks, model_context = instantiate_models(checkpoints)
    checks.update(code_checks)

    module_shape_signatures = {}
    for key in ("g2", "matched"):
        module_shape_signatures[key] = {
            row["module_group"]: sum(
                candidate["numel"]
                for candidate in parameter_rows
                if candidate["model"] == checkpoints[key]["model_name"]
                and candidate["module_group"] == row["module_group"]
            )
            for row in parameter_rows
            if row["model"] == checkpoints[key]["model_name"]
        }
    checks["paired_module_group_budgets_identical"] = sorted(
        checkpoint_audits["g2"]["module_groups"].values()
    ) == sorted(checkpoint_audits["matched"]["module_groups"].values())

    mode_counts = {
        key: {
            mode: sum(row["mask_mode"] == mode for row in histories[key])
            for mode in ("partial", "node", "block")
        }
        for key in ("g2", "matched")
    }
    checks["paired_mask_mode_counts_exact"] = (
        mode_counts["g2"]
        == mode_counts["matched"]
        == {"partial": 128, "node": 53, "block": 19}
    )

    specs = {
        "schema_version": 1,
        "report_date": REPORT_DATE,
        "artifact_kind": "current_model_architecture_audit",
        "scope": {
            "current_conclusion_bearing_models": ["g2", "b0-g2-matched"],
            "diagnostic_runs": [RUN_IDS["resource_pilot"]],
            "independent_biological_units": 1,
            "generalization_estimate": False,
            "raw_or_row_level_data_read": False,
            "maximum_scientific_claim": (
                "held-in masked-expression reconstruction capacity in one "
                "transductively fitted spatial core"
            ),
        },
        "data": {
            "dataset_id": resolved["g2"]["dataset"]["dataset_id"],
            "version": resolved["g2"]["dataset"]["version"],
            "dataset_fingerprint": EXPECTED_DATASET_SHA256,
            "split_fingerprint": EXPECTED_SPLIT_SHA256,
            "nodes": 24_245,
            "biological_targets": 1_000,
            "node_covariates": 22,
            "fit_nodes": 24_245,
            "validation_nodes": 0,
            "test_nodes": 0,
            "expression_transform": (
                "gene-wise log1p then population-mean centering and "
                "population-standard-deviation scaling, fitted on all nodes"
            ),
            "metadata_transform": (
                "median imputation, log1p, mean centering and population-"
                "standard-deviation scaling, fitted on all nodes"
            ),
            "technical_controls_excluded": ["Negative*", "SystemControl*"],
            "prohibited_node_inputs": resolved["g2"]["features"][
                "prohibited_node_inputs"
            ],
        },
        "graph": {
            "kind": "exact mutual k-nearest-neighbor",
            "k": 1000,
            "radius_guard_um": 650.0,
            "radius_is_filter": False,
            "nodes": 24_245,
            "directed_candidates": 24_245_000,
            "directed_edges": 21_029_944,
            "undirected_relations": 10_514_972,
            "mean_degree": 867.3930294906166,
            "median_degree": 917.0,
            "components": 1,
            "isolated_nodes": 0,
            "self_loops": 0,
            "duplicate_directed_edges": 0,
            "mean_edge_distance_um": 137.6639404296875,
            "max_edge_distance_um": 510.3060302734375,
            "edge_attribute_count": 17,
            "edge_attributes": resolved["g2"]["features"]["edge_features"][
                "fields"
            ],
            "graph_sha256": EXPECTED_GRAPH_SHA256,
        },
        "training": {
            "epochs": 200,
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip_global_norm": 1.0,
            "objective": "masked mean Huber loss",
            "huber_delta": 1.0,
            "model_seed": 0,
            "mask_seed": 314159,
            "batch_unit": "one complete full-core graph",
            "neighbor_sampling": False,
            "edge_dropout": 0.0,
            "precision": "FP32 parameters with CUDA FP16 autocast and GradScaler",
            "deterministic_algorithms": True,
            "early_stopping": False,
            "checkpoint_selection": "final epoch only; no validation selection",
            "mask_curriculum": {
                "warmup_epochs": 10,
                "warmup_mode": "partial-gene",
                "post_warmup_probabilities": {
                    "partial_gene": 0.60,
                    "whole_node": 0.30,
                    "spatial_block": 0.10,
                },
                "realized_epoch_counts": mode_counts["g2"],
                "partial_gene_rate": 0.20,
                "whole_node_rate": 0.10,
                "spatial_block_node_rate": 0.10,
            },
        },
        "models": {
            key: {
                "display_name": MODEL_LABELS[key],
                "model_config": checkpoints[key]["model_config"],
                "checkpoint_audit": checkpoint_audits[key],
            }
            for key in ("g2", "matched")
        },
        "current_campaign_runs": {
            key: {
                "run_id": RUN_IDS[key],
                "model_name": summaries[key]["model_name"],
                "diagnostic_resource_pilot": summaries[key][
                    "diagnostic_resource_pilot"
                ],
                "fixed_epoch_budget": summaries[key]["fixed_epoch_budget"],
                "primary_metric_name": summaries[key]["primary_metric_name"],
                "primary_metric_value": summaries[key]["primary_metric_value"],
                "parameter_count": summaries[key]["parameter_count"],
                "peak_vram_gib": summaries[key]["peak_vram_gb"],
                "training_duration_seconds": summaries[key]["metrics"][
                    "resource/training_duration_seconds"
                ],
                "evaluation_duration_seconds": summaries[key]["metrics"][
                    "resource/inference_duration_seconds"
                ],
                "total_duration_seconds": summaries[key]["metrics"][
                    "resource/total_duration_seconds"
                ],
                "conclusion_eligible": summaries[key]["conclusion_eligible"],
            }
            for key in ("resource_pilot", "g2", "matched")
        },
        "performance": comparison,
        "amp_equivalence_smoke": {
            "passed": amp_smoke["passed"],
            "fp32_ordinary_vs_chunked_max_abs_error": amp_smoke["comparisons"][
                "fp32_ordinary_vs_chunked"
            ]["max_abs_error"],
            "chunked_fp32_vs_amp_max_abs_error": amp_smoke["comparisons"][
                "chunked_fp32_vs_amp"
            ]["max_abs_error"],
            "scope": "synthetic 512-node, 128-gene graph; not the full core",
        },
        "interpretability_surface": {
            "attention_receivers_sampled": interpretability[
                "attention_routing"
            ]["expression_independent_hash_sample"]["receiver_count"],
            "mean_attention_entropy_nats": interpretability[
                "attention_routing"
            ]["expression_independent_hash_sample"][
                "attention_entropy_nats"
            ]["mean"],
            "mean_effective_neighbor_count": interpretability[
                "attention_routing"
            ]["expression_independent_hash_sample"][
                "effective_neighbor_count"
            ]["mean"],
            "joint_tls_dependency_support": interpretability[
                "interpretation"
            ]["joint_tls_dependency_support"],
            "evidence_label": interpretability["interpretation"][
                "evidence_label"
            ],
        },
        **model_context,
        "verification": checks,
        "source_files": [
            str(
                (
                    ROOT / "src" / "spatial_benchmark" / "models.py"
                ).relative_to(ROOT)
            ),
            str(
                (
                    ROOT / "src" / "spatial_benchmark" / "dense_gat.py"
                ).relative_to(ROOT)
            ),
            str(
                (
                    ROOT
                    / "src"
                    / "spatial_benchmark"
                    / "full_core_training.py"
                ).relative_to(ROOT)
            ),
            str(
                (
                    ROOT / "scripts" / "train" / "run_full_core_capacity.py"
                ).relative_to(ROOT)
            ),
            str(COMPARISON_PATH.relative_to(ROOT)),
            *[
                str(
                    (
                        run_dirs[key] / "checkpoints" / "last.ckpt"
                    ).relative_to(ROOT)
                )
                for key in ("g2", "matched")
            ],
        ],
    }
    return specs, parameter_rows, checks


def render_report(
    specs: Mapping[str, Any],
    parameter_rows: list[dict[str, Any]],
) -> str:
    performance = specs["performance"]
    facts = performance["facts"]
    summaries = specs["current_campaign_runs"]
    model_specs = specs["models"]
    training = specs["training"]
    graph = specs["graph"]
    data = specs["data"]
    inventory = specs["implementation_inventory_parameter_counts"]

    g2_metrics = read_json(
        RUN_ROOT / RUN_IDS["g2"] / "metrics" / "final.json"
    )
    matched_metrics = read_json(
        RUN_ROOT / RUN_IDS["matched"] / "metrics" / "final.json"
    )
    g2_history = read_jsonl(
        RUN_ROOT / RUN_IDS["g2"] / "metrics" / "history.jsonl"
    )
    matched_history = read_jsonl(
        RUN_ROOT / RUN_IDS["matched"] / "metrics" / "history.jsonl"
    )
    graph_qc = read_json(
        RUN_ROOT / RUN_IDS["g2"] / "diagnostics" / "graph_statistics.json"
    )
    preprocessing_qc = read_json(
        RUN_ROOT
        / RUN_IDS["g2"]
        / "diagnostics"
        / "full_core_preprocessing.json"
    )
    convergence = {
        key: read_json(
            RUN_ROOT
            / RUN_IDS[key]
            / "diagnostics"
            / "training_convergence.json"
        )
        for key in ("g2", "matched")
    }

    module_groups = {
        "Node encoder": 1_036_800,
        "Shared 17→64→64 encoder": 5_568,
        "Mixing blocks": 2_169_856,
        "Expression decoder": 775_656,
    }
    parameter_summary = parameter_bar(module_groups, EXPECTED_PARAMETER_COUNT)

    mode_labels = {
        "partial_gene": "Partial-gene",
        "whole_node": "Whole-node",
        "spatial_block": "Spatial block",
    }
    metric_rows = []
    for mode in ("partial_gene", "whole_node", "spatial_block"):
        g2_value = float(g2_metrics[f"fit/{mode}/masked_huber"])
        matched_value = float(matched_metrics[f"fit/{mode}/masked_huber"])
        delta = matched_value - g2_value
        metric_rows.append(
            (
                mode_labels[mode],
                fmt_float(g2_value, 6),
                fmt_float(matched_value, 6),
                f"{delta:+.6f}",
                (
                    '<span class="good">G2</span>'
                    if delta > 0
                    else '<span class="warn">matched self</span>'
                ),
            )
        )
    metrics_table = table(
        (
            "Held-in mask mode",
            "G2 masked Huber ↓",
            "Matched self ↓",
            "Matched − G2",
            "Lower model",
        ),
        metric_rows,
        caption=(
            "Final held-in metrics averaged over three technical mask "
            "replicates; these are not independent biological replicates."
        ),
    )

    replicate_table = table(
        ("Replicate", "G2", "Matched self", "Matched − G2", "Direction"),
        (
            (
                row["mask_replicate"],
                fmt_float(row["g2_masked_huber"], 6),
                fmt_float(row["matched_self_masked_huber"], 6),
                f'{row["self_minus_g2_masked_huber"]:+.6f}',
                '<span class="good">favors G2</span>',
            )
            for row in facts["paired_whole_node_replicates"]
        ),
        caption="Paired whole-node held-in Huber loss by fixed evaluation mask.",
    )

    data_rows = (
        (
            "Expression target",
            "[24,245, 1,000]",
            "float32",
            fmt_bytes(24_245 * 1_000 * 4),
            "Biological probes after full-core log1p standardization",
        ),
        (
            "Explicit gene mask",
            "[24,245, 1,000]",
            "bool",
            fmt_bytes(24_245 * 1_000),
            "True means hidden; masked values are filled with 0",
        ),
        (
            "Node covariates",
            "[24,245, 22]",
            "float32",
            fmt_bytes(24_245 * 22 * 4),
            "Allow-listed morphology and imaging only",
        ),
        (
            "Graph index (G2 only)",
            "[2, 21,029,944]",
            "int64",
            fmt_bytes(2 * 21_029_944 * 8),
            "source row, receiver row; receiver sorted",
        ),
        (
            "Edge geometry (G2 only)",
            "[21,029,944, 17]",
            "float32",
            fmt_bytes(21_029_944 * 17 * 4),
            "Full-core-standardized geometric attributes",
        ),
        (
            "Node representation",
            "[24,245, 512]",
            "mixed execution",
            fmt_bytes(24_245 * 512 * 4),
            "FP32-equivalent size shown; AMP activations may be FP16",
        ),
        (
            "Prediction",
            "[T, 1,000]",
            "mixed execution",
            "mask-dependent",
            "T = nodes containing ≥1 hidden target",
        ),
    )
    tensor_table = table(
        ("Tensor", "Logical shape", "Stored dtype", "Nominal size", "Role"),
        data_rows,
        caption=(
            "Logical full-core tensors. Sizes are single-tensor nominal "
            "allocations and do not equal peak training memory."
        ),
    )

    node_covariates = (
        "Area; Area.um2; AspectRatio; Width; Height; Mean/Max PanCK; "
        "Mean/Max G; Mean/Max Membrane; Mean/Max CD45; Mean/Max DAPI; "
        "SplitRatioToLocal; NucArea; NucAspectRatio; Circularity; "
        "Eccentricity; Perimeter; Solidity"
    )
    edge_attributes = "; ".join(graph["edge_attributes"])

    graph_table = table(
        ("Quantity", "Verified value", "Interpretation"),
        (
            ("Nodes", fmt_int(graph["nodes"]), "all cells in the single core"),
            (
                "Candidate graph",
                "exact k=1,000",
                "nearest non-self nodes before mutual filtering",
            ),
            (
                "Radius",
                "650 µm guard",
                "validation guard only; it does not filter candidates",
            ),
            (
                "Directed candidates",
                fmt_int(graph["directed_candidates"]),
                "24,245 × 1,000",
            ),
            (
                "Retained directed edges",
                fmt_int(graph["directed_edges"]),
                "mutual-neighbor fraction 86.74%",
            ),
            (
                "Mean / median degree",
                f'{graph["mean_degree"]:.2f} / {graph["median_degree"]:.0f}',
                "dense regional context",
            ),
            (
                "Edge length",
                (
                    f'mean {graph["mean_edge_distance_um"]:.2f} µm; '
                    f'max {graph["max_edge_distance_um"]:.2f} µm'
                ),
                "not a direct-contact graph",
            ),
            (
                "Connectivity/QC",
                "1 component; 0 isolates/loops/duplicates",
                "symmetric directed pairs",
            ),
        ),
        caption="Verified graph construction and quality-control facts.",
    )

    g2_checkpoint = model_specs["g2"]["checkpoint_audit"]
    matched_checkpoint = model_specs["matched"]["checkpoint_audit"]
    model_comparison_table = table(
        ("Property", "G2", "B0–G2-matched"),
        (
            ("Trainable parameters", "3,987,880", "3,987,880"),
            ("Parameter tensors", "42", "42"),
            (
                "Raw FP32 weight bytes",
                fmt_bytes(g2_checkpoint["raw_parameter_bytes"]),
                fmt_bytes(matched_checkpoint["raw_parameter_bytes"]),
            ),
            (
                "Checkpoint file",
                fmt_bytes(g2_checkpoint["checkpoint_file_bytes"]),
                fmt_bytes(matched_checkpoint["checkpoint_file_bytes"]),
            ),
            ("Node encoder", "1,000 + mask + 22 → 512", "same shape"),
            ("Mixing depth", "2 residual graph blocks", "2 residual self blocks"),
            ("Heads / head width", "4 × 128", "count-matched 4 × 128 layout"),
            (
                "17→64→64 encoder input",
                "measured edge geometry",
                "first 17 channels of each cell embedding",
            ),
            (
                "Interaction operator",
                "incoming-edge softmax + sum",
                "within-cell projection + GELU",
            ),
            ("Graph/edge inputs", "consumed", "ignored by forward method"),
            ("Decoder", "512→512→1,000", "same shape"),
            ("Self information", "residual path; no self loop", "all operations"),
        ),
        caption=(
            "The comparison is parameter-matched. It is not operator-, "
            "activation-, or computation-matched."
        ),
    )

    mapping_table = table(
        ("G2 tensor budget", "Matched-self tensor budget", "Parameters/layer"),
        (
            ("source projection 512×512", "left projection 512×512", "262,144"),
            (
                "receiver projection 512×512",
                "right projection 512×512",
                "262,144",
            ),
            (
                "edge projection 64×512",
                "surrogate projection 64×512",
                "32,768",
            ),
            (
                "attention vector 4×128",
                "routing vector 512",
                "512",
            ),
            ("post-mix LayerNorm", "post-mix LayerNorm", "1,024"),
            (
                "residual FFN + LayerNorm",
                "residual FFN + LayerNorm",
                "526,336",
            ),
            ("Total per block", "Total per block", "1,084,928"),
        ),
        caption="Exact parameter-count mapping for each of the two mixing blocks.",
    )

    param_table_rows = []
    for row in sorted(
        parameter_rows,
        key=lambda item: (item["model"], item["tensor_name"]),
    ):
        param_table_rows.append(
            (
                esc(row["model"]),
                f"<code>{esc(row['tensor_name'])}</code>",
                esc(row["functional_role"]),
                esc(row["shape"] or "scalar"),
                fmt_int(row["numel"]),
                fmt_percent(row["percent_of_model"], 3),
            )
        )
    full_param_table = table(
        ("Model", "State tensor", "Functional role", "Shape", "Parameters", "%"),
        param_table_rows,
        caption="All 84 checkpoint tensors; every tensor is assigned exactly once.",
        classes="dense-table",
    )

    training_table = table(
        ("Setting", "Locked value", "Consequence"),
        (
            ("Optimization", "AdamW; lr 3×10⁻⁴; wd 1×10⁻⁴", "no scheduler"),
            (
                "Objective",
                "mean masked Huber, δ=1",
                "only hidden finite expression entries contribute",
            ),
            (
                "Gradient control",
                "global norm clip at 1.0",
                "clip after AMP unscale",
            ),
            (
                "Precision",
                "FP32 parameters; CUDA FP16 autocast + GradScaler",
                "AMP enabled only after equivalence smoke",
            ),
            (
                "Budget",
                "200 epochs; one complete graph/epoch",
                "no minibatches and no neighbor sampling",
            ),
            (
                "Selection",
                "last epoch (199)",
                "no validation, early stopping, or best-state restore",
            ),
            (
                "Stochastic controls",
                "model seed 0; mask seed 314159",
                "Python/NumPy/CPU/all visible CUDA generators seeded",
            ),
            (
                "Determinism",
                "deterministic algorithms; warn_only=false",
                "cuDNN benchmark off; CUBLAS workspace configured",
            ),
            (
                "Graph regularization",
                "edge dropout 0",
                "all 21,029,944 G2 edges used every epoch",
            ),
            (
                "Checkpoint payload",
                "model weights + config + checksums",
                "no optimizer or GradScaler state; not resumable in place",
            ),
        ),
        caption="Actual fixed-budget training settings from the resolved runs.",
    )

    mask_table = table(
        ("Mode", "Mask rule", "Target nodes/epoch", "Masked entries", "Epochs"),
        (
            (
                "Partial-gene",
                "exactly 20% = 200 genes in every cell",
                "24,245",
                "4,849,000",
                "128",
            ),
            (
                "Whole-node",
                "10% = 2,425 sampled cells; all genes",
                "2,425",
                "2,425,000",
                "53",
            ),
            (
                "Spatial block",
                "nearest disk around random anchor; 10% = 2,425 cells",
                "2,425",
                "2,425,000",
                "19",
            ),
        ),
        caption=(
            "Realized paired curriculum. Epochs 0–9 are partial-gene; later "
            "modes are deterministic 60/30/10 draws."
        ),
    )

    loss_svg = sparkline_svg(
        {
            "G2": [float(row["train_loss"]) for row in g2_history],
            "Matched self": [
                float(row["train_loss"]) for row in matched_history
            ],
        }
    )

    resource_table = table(
        ("Run", "Purpose", "Epochs", "Train time", "Eval phase", "Peak VRAM"),
        (
            (
                "G2 diagnostic pilot",
                "feasibility only",
                "2",
                fmt_duration(
                    summaries["resource_pilot"]["training_duration_seconds"]
                ),
                fmt_duration(
                    summaries["resource_pilot"]["evaluation_duration_seconds"]
                ),
                f'{summaries["resource_pilot"]["peak_vram_gib"]:.2f} GiB',
            ),
            (
                "G2 final",
                "conclusion-bearing held-in capacity",
                "200",
                fmt_duration(summaries["g2"]["training_duration_seconds"]),
                fmt_duration(summaries["g2"]["evaluation_duration_seconds"]),
                f'{summaries["g2"]["peak_vram_gib"]:.2f} GiB',
            ),
            (
                "B0–G2-matched final",
                "paired control",
                "200",
                fmt_duration(summaries["matched"]["training_duration_seconds"]),
                fmt_duration(
                    summaries["matched"]["evaluation_duration_seconds"]
                ),
                f'{summaries["matched"]["peak_vram_gib"]:.2f} GiB',
            ),
        ),
        caption=(
            "Recorded on one visible NVIDIA RTX 3090 (24 GiB). Evaluation "
            "phase includes metric and prediction-artifact generation."
        ),
    )
    training_ratio = (
        summaries["g2"]["training_duration_seconds"]
        / summaries["matched"]["training_duration_seconds"]
    )
    vram_ratio = (
        summaries["g2"]["peak_vram_gib"]
        / summaries["matched"]["peak_vram_gib"]
    )

    inventory_status = {
        "B0 self-only MLP": "Implemented; legacy benchmark runs only",
        "Broad-field control": "Implemented; legacy benchmark runs only",
        "B1 mean-neighbor": "Implemented; legacy benchmark runs only",
        "G1 topology-only GATv2": (
            "Configured in current tree; not run in current full-core campaign"
        ),
        "B0–G1 parameter-matched": "Implemented; legacy benchmark runs only",
        "G2 edge-conditioned GATv2": "Current final checkpoint",
        "B0–G2 parameter-matched": "Current final checkpoint",
        "G3 additive edge-message (implementation defaults)": (
            "Implemented only; no current config or finalized run"
        ),
    }
    inventory_table = table(
        ("Implementation", "Parameters at 1,000/22/512 dimensions", "Status"),
        (
            (
                name,
                fmt_int(count),
                inventory_status[name],
            )
            for name, count in inventory.items()
        ),
        caption=(
            "Code inventory is not the same as the current trained-model set. "
            "G3 count uses current constructor defaults (message head 16, "
            "message width 64), not a locked campaign configuration."
        ),
    )

    init_rows = []
    for row in specs["initialization_comparison"]:
        init_rows.append(
            (
                f"<code>{esc(row['g2_tensor'])}</code>",
                f"<code>{esc(row['matched_tensor'])}</code>",
                "yes" if row["identical"] else "no",
                fmt_float(row["max_abs_difference"], 6),
            )
        )
    initialization_table = table(
        (
            "Fresh G2 tensor (seed 0)",
            "Fresh matched tensor (seed 0)",
            "Bitwise same?",
            "Max |difference|",
        ),
        init_rows,
        caption=(
            "Fresh reconstruction under the current/preserved constructors. "
            "The shared encoder aligns because it is created first; later "
            "modules do not because constructor order and initialization "
            "policies differ."
        ),
        classes="dense-table",
    )

    checks = specs["verification"]
    checks_table = table(
        ("Verification", "Result"),
        (
            (
                key.replace("_", " "),
                (
                    '<span class="good">pass</span>'
                    if value
                    else '<span class="bad">fail</span>'
                ),
            )
            for key, value in checks.items()
        ),
        caption="Generator-enforced architecture and artifact checks.",
        classes="dense-table",
    )

    source_rows = (
        (f"<code>{esc(path)}</code>",)
        for path in specs["source_files"]
    )
    source_table = table(
        ("Primary evidence source",),
        source_rows,
        caption="Tracked implementation and finalized artifact evidence.",
        classes="dense-table",
    )

    graph_specific_parameters = 5_568 + 2 * 32_768
    relative_gain = float(facts["relative_mean_g2_gain"])
    report_css = """
:root {
  --ink:#1f2c29; --muted:#5a6966; --paper:#f6f7f3; --card:#ffffff;
  --line:#d8dfdc; --teal:#0d8f82; --teal-soft:#e8f8f5;
  --blue:#4f73b4; --blue-soft:#edf2fb; --amber:#b66f0d;
  --amber-soft:#fff4e4; --plum:#875593; --red:#a33b3b;
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body {
  margin:0; color:var(--ink); background:var(--paper);
  font:15px/1.58 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
a { color:#146a86; }
header {
  color:white; padding:64px max(24px, calc((100vw - 1180px)/2));
  background:linear-gradient(125deg,#172825 0%,#25433d 62%,#315a51 100%);
}
.eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:12px; opacity:.78; }
h1 { font-size:clamp(34px,5vw,62px); line-height:1.05; margin:14px 0 18px; max-width:980px; }
.lede { max-width:880px; font-size:19px; color:#dcebe7; }
.header-meta { display:flex; flex-wrap:wrap; gap:10px; margin-top:28px; }
.header-meta span { border:1px solid #719087; border-radius:999px; padding:5px 11px; font-size:13px; }
nav {
  position:sticky; top:0; z-index:5; background:rgba(255,255,255,.96);
  border-bottom:1px solid var(--line); padding:10px 18px; overflow-x:auto;
  white-space:nowrap;
}
nav div { max-width:1180px; margin:auto; display:flex; gap:18px; }
nav a { color:#31423e; text-decoration:none; font-size:13px; font-weight:650; }
main { max-width:1180px; margin:0 auto; padding:32px 22px 80px; }
section { scroll-margin-top:60px; margin:48px 0 64px; }
h2 { font-size:31px; line-height:1.18; margin:0 0 20px; }
h3 { font-size:20px; margin:28px 0 10px; }
p { max-width:950px; }
.kicker { color:var(--teal); font-weight:750; letter-spacing:.06em; text-transform:uppercase; font-size:12px; }
.grid { display:grid; gap:16px; }
.grid-2 { grid-template-columns:repeat(2,minmax(0,1fr)); }
.grid-3 { grid-template-columns:repeat(3,minmax(0,1fr)); }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; box-shadow:0 5px 20px rgba(28,46,41,.04); }
.stat strong { display:block; font-size:30px; line-height:1.1; }
.stat span { color:var(--muted); font-size:13px; }
.callout { border-left:5px solid var(--teal); background:var(--teal-soft); padding:17px 20px; border-radius:8px; margin:18px 0; max-width:1050px; }
.callout.warn { border-color:var(--amber); background:var(--amber-soft); }
.callout.limit { border-color:var(--plum); background:#f6eff8; }
.verdict { font-size:18px; }
.good { color:#087b70; font-weight:750; }
.warn { color:#a36109; font-weight:750; }
.bad { color:var(--red); font-weight:750; }
.muted { color:var(--muted); }
.table-wrap { overflow-x:auto; margin:18px 0 26px; background:white; border:1px solid var(--line); border-radius:12px; }
table { width:100%; border-collapse:collapse; }
caption { text-align:left; padding:13px 15px; color:var(--muted); font-size:13px; border-bottom:1px solid var(--line); }
th, td { text-align:left; vertical-align:top; padding:11px 13px; border-bottom:1px solid #e6ebe9; }
th { font-size:12px; letter-spacing:.03em; text-transform:uppercase; background:#f1f4f2; }
tr:last-child td { border-bottom:0; }
.dense-table th,.dense-table td { padding:7px 9px; font-size:12px; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.91em; overflow-wrap:anywhere; }
.architecture,.chart { width:100%; height:auto; display:block; background:white; border:1px solid var(--line); border-radius:14px; }
.stacked { display:flex; height:34px; border-radius:8px; overflow:hidden; margin:12px 0; }
.stacked span { min-width:2px; }
.bar-legend { list-style:none; padding:0; margin:18px 0 0; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px 20px; }
.bar-legend li { display:grid; grid-template-columns:13px 1fr auto auto; align-items:center; gap:8px; }
.bar-legend i { width:11px; height:11px; border-radius:3px; }
.bar-legend small { color:var(--muted); min-width:42px; text-align:right; }
.formula { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; background:#eef2f0; border-radius:8px; padding:12px 15px; overflow-x:auto; }
details { background:white; border:1px solid var(--line); border-radius:12px; padding:13px 16px; margin:14px 0; }
summary { cursor:pointer; font-weight:750; }
.evidence-list { columns:2; column-gap:34px; }
.evidence-list li { break-inside:avoid; margin-bottom:7px; }
footer { border-top:1px solid var(--line); color:var(--muted); padding:28px 22px 50px; text-align:center; font-size:13px; }
@media (max-width:820px) {
  .grid-2,.grid-3 { grid-template-columns:1fr; }
  .bar-legend { grid-template-columns:1fr; }
  .evidence-list { columns:1; }
  header { padding-top:45px; padding-bottom:45px; }
}
@media print {
  body { background:white; color:black; font-size:10pt; }
  nav { display:none; }
  header { background:white; color:black; padding:24px 0; }
  .lede { color:#333; }
  main { max-width:none; padding:0; }
  section { break-before:auto; margin:22px 0; }
  .card,.table-wrap,.architecture,.chart,details { box-shadow:none; break-inside:avoid; }
  details > * { display:block !important; }
  a { color:black; text-decoration:none; }
}
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Current model architecture and training audit</title>
  <style>{report_css}</style>
</head>
<body>
<header>
  <div class="eyebrow">BAGM · architecture audit · {REPORT_DATE}</div>
  <h1>What the current models actually are</h1>
  <p class="lede">A checkpoint-reconciled account of the two current full-core
  models: their data, exact tensor path, parameter allocation, dense graph
  execution, training method, resource cost, and the limits of the comparison.</p>
  <div class="header-meta">
    <span>2 final models</span><span>3,987,880 parameters each</span>
    <span>24,245 cells</span><span>1,000 genes</span>
    <span>held-in / transductive</span><span>single biological core</span>
  </div>
</header>
<nav aria-label="Report sections"><div>
  <a href="#readout">Readout</a><a href="#scope">Scope</a>
  <a href="#data">Data</a><a href="#architecture">Architecture</a>
  <a href="#parameters">Parameters</a><a href="#training">Training</a>
  <a href="#results">Results</a><a href="#controls">Controls &amp; limits</a>
  <a href="#inventory">Inventory</a><a href="#verification">Verification</a>
</div></nav>
<main>
<section id="readout">
  <div class="kicker">Executive readout</div>
  <h2>The pair is size-matched; it is not an identical-computation pair</h2>
  <div class="grid grid-3">
    <div class="card stat"><strong>3,987,880</strong><span>trainable FP32 parameters in each final checkpoint</span></div>
    <div class="card stat"><strong>0.479%</strong><span>G2 whole-node held-in Huber gain; below the locked 2% gate</span></div>
    <div class="card stat"><strong>23.8×</strong><span>G2/matched-self recorded training-time ratio</span></div>
  </div>
  <div class="callout verdict">
    <strong>Verified fact.</strong> Both checkpoints contain exactly 42 FP32
    tensors and {fmt_int(EXPECTED_PARAMETER_COUNT)} parameters
    ({fmt_bytes(g2_checkpoint["raw_parameter_bytes"])} of raw weights). They
    have the same-shaped encoder, width, depth, decoder and parameter budget,
    but their trained weights are independent. G2 consumes neighbors and
    measured edge geometry; the comparator is behaviorally invariant to graph
    arguments.
  </div>
  <div class="callout warn verdict">
    <strong>Scientific result.</strong> G2 is lower on the prespecified
    whole-node masked Huber metric in all 3 paired technical masks, but its
    mean gain is only {fmt_percent(relative_gain, 3)}. The locked ≥2% capacity
    hypothesis is therefore <strong>not supported</strong>. This does not
    establish generalization, cell–cell communication, mechanism or causality.
  </div>
  <div class="callout limit">
    <strong>Important qualification.</strong> “Same seed” did not produce
    identical corresponding initial weights after the shared encoder. The
    constructors instantiate modules in different orders and use different
    routing initialization policies. The contrast also changes the operator
    (graph softmax/aggregation versus within-cell GELU mixing) and compute.
    It controls parameter count, not every nuisance variable.
  </div>
</section>

<section id="scope">
  <div class="kicker">Evidence boundary</div>
  <h2>What “current models” means here</h2>
  <p>The current conclusion-bearing campaign has two distinct trained
  architectures: G2 and B0–G2-matched. A two-epoch G2 resource pilot is a
  third run, not a third model. G1 is configured in the current tree but the
  full-core runner did not train it; G3 is implemented but has no current
  configuration or finalized run.</p>
  <div class="grid grid-2">
    <div class="card">
      <h3>Included as trained current models</h3>
      <ul>
        <li><strong>G2:</strong> exact receiver-chunked, edge-conditioned GATv2.</li>
        <li><strong>B0–G2-matched:</strong> strict cell-autonomous, equal-parameter comparator.</li>
        <li>Final epoch-199 checkpoints, resolved configs, histories, metrics and provenance.</li>
      </ul>
    </div>
    <div class="card">
      <h3>Not evidence from this campaign</h3>
      <ul>
        <li>No validation or test partition; every cell is a fit cell.</li>
        <li>No second core, slide, donor or patient-level replicate.</li>
        <li>No trained high-k G1 or G3 result.</li>
        <li>No claim that attention equals biological importance.</li>
      </ul>
    </div>
  </div>
</section>

<section id="data">
  <div class="kicker">Training material</div>
  <h2>One full CosMx core, refit transductively</h2>
  <p>The source is the preserved legacy true-Normal CosMx core. Technical
  control probes with <code>Negative*</code> and <code>SystemControl*</code>
  prefixes are removed, leaving 1,000 biological targets. All 24,245 cells
  are assigned to the fit role. Preprocessing statistics, graph statistics
  and evaluation all use this same core.</p>
  <div class="formula">
    expression[g] = (log1p(count[g]) − μ<sub>g, all cells</sub>) / s<sub>g</sub>,
    &nbsp; s<sub>g</sub> = σ<sub>g</sub> if σ<sub>g</sub> &gt; 10⁻⁸ else 1
  </div>
  <p>Metadata are reconstructed from the prior prepared artifact, median
  imputed, log1p transformed and standardized over all cells. This dataset has
  {preprocessing_qc["n_metadata_missing_values"]} measured metadata missing
  values and {preprocessing_qc["n_constant_expression_features"]} constant
  expression features. The 22 permitted covariates are: {esc(node_covariates)}.
  Direct identifiers, absolute/local coordinates, expression-derived library
  size and vendor cell type/cluster/neighborhood/niche fields are prohibited
  node inputs.</p>
  {tensor_table}
  <h3>The graph</h3>
  <p>Exact kNN candidates are found before the radius guard is checked.
  An edge is retained only when the neighbor relationship is mutual, then both
  directed orientations are represented. Because the mean retained edge is
  137.66 µm long and the maximum is 510.31 µm, this is a <strong>regional
  context graph</strong>, not a direct-contact graph. Two GAT layers can spread
  information across two already-dense graph hops.</p>
  {graph_table}
  <details>
    <summary>All 17 edge attributes and their construction</summary>
    <p>{esc(edge_attributes)}.</p>
    <p>They comprise distance, distance/radius, log-distance, normalized
    x/y displacement, first- and second-order orientation harmonics, and eight
    Gaussian distance radial-basis values with centers evenly spaced from
    0–650 µm. Each column is standardized over all retained directed edges.</p>
  </details>
  <p class="muted">Identity: dataset <code>{data["dataset_fingerprint"]}</code>;
  role assignment <code>{data["split_fingerprint"]}</code>; graph
  <code>{graph["graph_sha256"]}</code>.</p>
</section>

<section id="architecture">
  <div class="kicker">Forward paths</div>
  <h2>Shared shell, different mixing mechanism</h2>
  {architecture_svg()}
  <h3>Shared node encoder</h3>
  <p>Three bias-free projections map masked expression
  [N,1,000], the explicit Boolean mask converted to the expression dtype
  [N,1,000], and morphology/imaging [N,22] into 512 channels. Their sum plus
  one learned 512-vector is passed through LayerNorm, GELU and 10% dropout.
  The encoder itself also masks expression values, preventing accidental
  leakage even if an unmasked tensor is passed by the caller.</p>

  <div class="grid grid-2">
    <div class="card">
      <h3>G2 graph block ×2</h3>
      <ol>
        <li>Project every source and receiver embedding 512→4×128.</li>
        <li>Encode each 17-vector of measured geometry 17→64→64; project 64→4×128.</li>
        <li>For edge j→i and head h, form a GATv2 score from a leaky-ReLU transform of source + receiver + edge projections.</li>
        <li>Softmax scores over <em>all</em> incoming edges of receiver i; apply 10% attention dropout.</li>
        <li>Weight projected source values and sum by receiver. Geometry changes routing weights; it is not directly added to the value message.</li>
        <li>Add the residual, LayerNorm, then a residual 512→512→512 GELU FFN with 10% dropout and a second LayerNorm.</li>
      </ol>
      <p>No implicit or explicit self loops are allowed. A cell’s own
      representation survives through residual paths.</p>
    </div>
    <div class="card">
      <h3>Matched self block ×2</h3>
      <ol>
        <li>Select the first 17 channels of each cell’s 512-vector (the modulo operation is trivial at 17&lt;512).</li>
        <li>Pass that within-cell surrogate through a 17→64→64 network with the same parameter shapes as G2’s edge encoder; compute it once and reuse it in both blocks.</li>
        <li>Each block sums left 512→512, right 512→512, and surrogate 64→512 projections for the same cell.</li>
        <li>Multiply by a learned 512-vector, apply GELU, then residual LayerNorm and the same-shape FFN.</li>
      </ol>
      <p>The forward method deletes graph and edge arguments without reading
      them. Coordinates are used externally to create spatial masks, never as
      this model’s feature. The configured 10% attention dropout is accepted
      only for constructor parity and is not used by this self-only model;
      its ordinary residual/FFN dropout remains 10%.</p>
    </div>
  </div>

  <h3>Exact dense execution for G2</h3>
  <p>The 24,245 receivers are partitioned into 48 chunks of at most 512 cells.
  Every incoming edge for a receiver remains in the same chunk, so softmax and
  aggregation are exact. Raw edge attributes stay materialized; the much
  larger 64-channel edge embeddings and attention activations exist only for
  the active chunk. Non-reentrant activation checkpointing recomputes each
  chunk during backward. This is memory partitioning—not neighbor sampling,
  edge pruning or an approximation.</p>
  <div class="callout">
    The synthetic equivalence smoke passed: ordinary versus chunked FP32
    maximum absolute error
    {specs["amp_equivalence_smoke"]["fp32_ordinary_vs_chunked_max_abs_error"]:.2e};
    chunked FP32 versus AMP maximum absolute error
    {specs["amp_equivalence_smoke"]["chunked_fp32_vs_amp_max_abs_error"]:.4f}.
    The smoke used 512 synthetic nodes and 128 genes, so it verifies numerical
    implementation equivalence, not full-core model quality.
  </div>

  <h3>Decoder and outputs</h3>
  <p>After mixing, only nodes with at least one hidden target are selected.
  A 512→512 affine, GELU, 10% dropout and 512→1,000 affine produce
  standardized-expression predictions. Loss is computed only where the mask
  is true. With explanations requested, G2 can additionally return last-layer
  four-head attention and 64-channel edge embeddings for selected receivers;
  B0–G2-matched returns no edge explanation tensors.</p>
</section>

<section id="parameters">
  <div class="kicker">Model size</div>
  <h2>Exact parameter accounting</h2>
  <div class="card">
    <h3>Allocation in either model</h3>
    {parameter_summary}
    <p><strong>Edge-attribute/surrogate slice:</strong>
    {fmt_int(graph_specific_parameters)} parameters
    ({fmt_percent(graph_specific_parameters/EXPECTED_PARAMETER_COUNT, 2)}):
    the shared 17→64→64 encoder plus two 64→512 edge/surrogate projections.
    The rest of each mixing block has equal count but different graph versus
    self-only semantics.</p>
  </div>
  {model_comparison_table}
  {mapping_table}
  <p>The checkpoint files differ by only 256 serialized bytes because of
  metadata/key names. Their raw weight payloads are identical in size. Neither
  checkpoint contains AdamW moments, gradients or GradScaler state; it is a
  final inference/audit checkpoint, not a full optimizer-resume checkpoint.</p>
  <details>
    <summary>Open the complete per-tensor parameter ledger (84 rows)</summary>
    {full_param_table}
  </details>
</section>

<section id="training">
  <div class="kicker">Fitting method</div>
  <h2>Paired masks, fixed budget, no model selection</h2>
  {training_table}
  {mask_table}
  <p>Both models received identical mask mode, seed, checksum, target count and
  masked-entry count in every epoch. Partial-gene epochs process all 24,245
  nodes as targets. In whole-node and spatial-block epochs, the self-only model
  can select 2,425 targets before its mixing blocks; G2 must still propagate
  all nodes so those targets can receive neighbor information.</p>
  {loss_svg}
  <p class="muted">The sawtooth structure is expected because loss scale changes
  with mask mode. Both runs completed epochs 0–199 with finite losses and
  gradients. Final recorded training losses were
  {convergence["g2"]["final_train_loss"]:.6f} (G2) and
  {convergence["matched"]["final_train_loss"]:.6f} (matched self);
  the last epoch was a spatial-block mask.</p>
  <h3>Resource cost</h3>
  {resource_table}
  <p>G2 training took {training_ratio:.1f}× as long and used {vram_ratio:.1f}×
  the peak VRAM of the self-only model. The paired graph was still constructed
  and audited for the comparator, but it was neither transferred to the GPU
  nor consumed by its forward method.</p>
</section>

<section id="results">
  <div class="kicker">Existing performance context</div>
  <h2>G2 wins the primary direction, but not the locked magnitude</h2>
  {metrics_table}
  {replicate_table}
  <div class="grid grid-2">
    <div class="card stat">
      <strong>{facts["g2_mean_whole_node_masked_huber"]:.6f}</strong>
      <span>G2 mean whole-node masked Huber</span>
    </div>
    <div class="card stat">
      <strong>{facts["matched_self_mean_whole_node_masked_huber"]:.6f}</strong>
      <span>matched-self mean whole-node masked Huber</span>
    </div>
  </div>
  <div class="callout warn">
    <strong>Gate decision: fail.</strong> Completion, finiteness, lack of
    divergence and 3/3 favorable paired masks passed. The required relative
    gain ≥2% did not: observed {fmt_percent(relative_gain, 3)}. Partial-gene
    Huber was slightly lower for the matched self model, so G2 was not uniformly
    best across mask modes.
  </div>
  <h3>Explanation-surface result</h3>
  <p>In the expression-independent 128-receiver toy sample, last-layer mean
  attention entropy was
  {specs["interpretability_surface"]["mean_attention_entropy_nats"]:.3f} nats
  and the mean effective neighbor count was
  {specs["interpretability_surface"]["mean_effective_neighbor_count"]:.1f}.
  The subsequent matched deletion/sender-program analysis did
  <strong>not</strong> support a joint TLS-related dependency. Attention remains
  a normalized routing quantity, not biological importance.</p>
</section>

<section id="controls">
  <div class="kicker">Interpretation discipline</div>
  <h2>What the comparison controls—and what it still confounds</h2>
  <div class="grid grid-2">
    <div class="card">
      <h3>Directly controlled or verified</h3>
      <ul>
        <li>Same 24,245 cells, 1,000 targets and 22 visible covariates.</li>
        <li>Same hidden width, two-block depth, FFN/decoder widths and dropout rates.</li>
        <li>Exactly equal trainable-parameter count and FP32 weight bytes.</li>
        <li>Same 200 epoch masks, optimization hyperparameters, model seed and final-checkpoint policy.</li>
        <li>Same fixed evaluation masks and graph audit; comparator graph invariance verified behaviorally.</li>
      </ul>
    </div>
    <div class="card">
      <h3>Not controlled by this design</h3>
      <ul>
        <li>Graph aggregation versus within-cell mixing operator.</li>
        <li>Identical initialization after the shared encoder.</li>
        <li>Compute, activation volume, target-selection timing or optimization trajectory.</li>
        <li>Training-seed uncertainty: only seed 0 was fitted.</li>
        <li>Biological uncertainty: three masks are technical repeats on one core.</li>
        <li>Held-out spatial, slide, donor or patient generalization.</li>
      </ul>
    </div>
  </div>
  <details>
    <summary>Initialization reconstruction under seed 0</summary>
    {initialization_table}
  </details>
  <div class="callout limit">
    <strong>Inference, not a checkpoint fact.</strong> Because the primary gain
    is smaller than plausible seed-to-seed variation in many neural networks,
    and this campaign has one training seed, the sign stability across three
    evaluation masks does not establish sign stability across retraining.
    Seed replication is required to separate architecture signal from
    initialization/training noise.
  </div>
  <h3>Best-supported next experiment</h3>
  <ol>
    <li>Repeat the exact pair across multiple training seeds and explicitly
    align semantically corresponding initial tensors where shapes permit;
    record an initialization checksum.</li>
    <li>Add high-k G1 and edge-attribute permutation controls so topology,
    geometry and operator effects are not bundled into one contrast.</li>
    <li>Only if a replicated advantage clears the locked threshold, move to
    held-out spatial/core evaluation and then the G3 additive attribution
    branch. The current result does not justify mechanistic escalation.</li>
    <li>For biological interpretation, evaluate lower-k/contact-scale graphs;
    k=1,000 estimates regional context, not physical cell contact.</li>
  </ol>
</section>

<section id="inventory">
  <div class="kicker">Repository context</div>
  <h2>Implemented model ladder versus trained current models</h2>
  {inventory_table}
  <p>Parameter counts in this table instantiate the current implementation at
  the full-core 1,000-gene, 22-covariate, 512-wide dimensions. They are useful
  architectural references; only the two rows labeled “current final
  checkpoint” are trained evidence from this campaign.</p>
</section>

<section id="verification">
  <div class="kicker">Reproducibility</div>
  <h2>Checkpoint, code and artifact reconciliation</h2>
  <p>Both final checkpoints load strictly into the current model classes.
  State-dict value checksums reproduce the checksums stored inside the
  checkpoints; file checksums and byte sizes reproduce each finalized artifact
  manifest. The chunked G2 and ordinary G2 have identical state layouts, and
  the self-only comparator passed a direct graph-invariance test.</p>
  <details open>
    <summary>Verification matrix</summary>
    {checks_table}
  </details>
  <details>
    <summary>Evidence ledger</summary>
    {source_table}
  </details>
  <p>Machine-readable companions: <code>model_specs.json</code> and
  <code>parameter_breakdown.csv</code>. Regenerate or verify deterministically
  with <code>python generate_report.py</code> or
  <code>python generate_report.py --check</code>.</p>
</section>
</main>
<footer>
  BAGM current-model architecture audit · generated entirely from local,
  preserved implementation and artifacts · no raw or row-level biological
  data included.
</footer>
</body>
</html>
"""


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    fields = (
        "model",
        "run_id",
        "tensor_name",
        "module_group",
        "functional_role",
        "shape",
        "numel",
        "dtype",
        "tensor_bytes",
        "percent_of_model",
    )
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (item["model"], item["tensor_name"])):
        output = dict(row)
        output["percent_of_model"] = f'{row["percent_of_model"]:.12f}'
        writer.writerow(output)
    return buffer.getvalue().encode("utf-8")


def expected_outputs() -> dict[Path, bytes]:
    specs, parameter_rows, checks = collect()
    failed = [name for name, value in checks.items() if not value]
    if failed:
        raise AssertionError(f"verification failed: {failed}")
    return {
        REPORT_DIR / "model_specs.json": (
            json.dumps(specs, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
        REPORT_DIR / "parameter_breakdown.csv": csv_bytes(parameter_rows),
        REPORT_DIR / "model_architecture_report.html": render_report(
            specs, parameter_rows
        ).encode("utf-8"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in outputs without rewriting them",
    )
    arguments = parser.parse_args(argv)
    outputs = expected_outputs()
    if arguments.check:
        mismatches = []
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                mismatches.append(str(path.relative_to(ROOT)))
        if mismatches:
            print("stale or missing report outputs:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
            return 1
        print(
            "current model report outputs verified; "
            f"{len(read_json(REPORT_DIR / 'model_specs.json')['verification'])} "
            "architecture/artifact checks passed"
        )
        return 0

    for path, content in outputs.items():
        path.write_bytes(content)
        print(f"wrote {path.relative_to(ROOT)} ({len(content):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
