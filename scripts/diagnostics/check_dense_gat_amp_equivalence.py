#!/usr/bin/env python3
"""Check ordinary/chunked dense GAT equivalence in FP32 and CUDA AMP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.dense_gat import (  # noqa: E402
    ReceiverChunkedEdgeConditionedGATv2,
)
from spatial_benchmark.models import EdgeConditionedGATv2  # noqa: E402


SEED = 20260725
MODEL_CONFIG: dict[str, int | float] = {
    "num_genes": 128,
    "edge_attribute_dim": 17,
    "node_covariate_dim": 22,
    "hidden_dim": 512,
    "attention_heads": 4,
    "graph_layers": 2,
    "ffn_dim": 512,
    "decoder_dim": 512,
    "edge_hidden_dim": 64,
    "edge_embedding_dim": 64,
    "dropout": 0.0,
    "attention_dropout": 0.0,
}
SYNTHETIC_CONFIG = {
    "n_nodes": 512,
    "incoming_degree": 64,
    "mask_rate": 0.20,
    "receiver_chunk_size": 64,
}
TOLERANCES: dict[str, dict[str, float]] = {
    "fp32_ordinary_vs_chunked": {"atol": 5e-5, "rtol": 5e-5},
    "amp_ordinary_vs_chunked": {"atol": 5e-3, "rtol": 5e-3},
    "chunked_fp32_vs_amp": {"atol": 1e-2, "rtol": 1e-2},
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="One CUDA device, for example cuda:0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional new JSON path; existing paths are never overwritten.",
    )
    return parser


def _synthetic_inputs(
    *,
    seed: int = SEED,
    n_nodes: int = int(SYNTHETIC_CONFIG["n_nodes"]),
    incoming_degree: int = int(SYNTHETIC_CONFIG["incoming_degree"]),
) -> dict[str, Tensor]:
    if n_nodes <= 1:
        raise ValueError("n_nodes must exceed one")
    if incoming_degree <= 0 or incoming_degree >= n_nodes:
        raise ValueError("incoming_degree must be in [1, n_nodes - 1]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    receiver = torch.arange(n_nodes, dtype=torch.long).repeat_interleave(
        incoming_degree
    )
    offset = torch.arange(1, incoming_degree + 1, dtype=torch.long).repeat(
        n_nodes
    )
    source = torch.remainder(receiver + offset, n_nodes)
    edge_index = torch.stack([source, receiver], dim=0)
    if bool((edge_index[1, 1:] < edge_index[1, :-1]).any()):
        raise RuntimeError("Synthetic graph is not receiver-sorted.")
    if bool((edge_index[0] == edge_index[1]).any()):
        raise RuntimeError("Synthetic graph unexpectedly contains a self-loop.")

    num_genes = int(MODEL_CONFIG["num_genes"])
    covariate_dim = int(MODEL_CONFIG["node_covariate_dim"])
    edge_attribute_dim = int(MODEL_CONFIG["edge_attribute_dim"])
    input_expression = torch.randn(
        n_nodes,
        num_genes,
        generator=generator,
        dtype=torch.float32,
    )
    return {
        "input_expression": input_expression,
        "gene_mask": (
            torch.rand(
                n_nodes,
                num_genes,
                generator=generator,
            )
            < float(SYNTHETIC_CONFIG["mask_rate"])
        ),
        "node_covariates": torch.randn(
            n_nodes,
            covariate_dim,
            generator=generator,
            dtype=torch.float32,
        ),
        "edge_index": edge_index,
        "edge_attributes": torch.randn(
            edge_index.shape[1],
            edge_attribute_dim,
            generator=generator,
            dtype=torch.float32,
        ),
        "target_expression": torch.randn(
            n_nodes,
            num_genes,
            generator=generator,
            dtype=torch.float32,
        ),
    }


def _update_tensor_digest(
    digest: Any,
    name: str,
    tensor: Tensor,
) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(
        json.dumps(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(value.numpy().tobytes(order="C"))


def _tensor_collection_sha256(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        _update_tensor_digest(digest, name, values[name])
    return digest.hexdigest()


def _comparison_summary(
    reference: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    if set(reference) != set(candidate):
        raise ValueError("Comparison tensor collections differ.")
    total_error = 0.0
    total_values = 0
    overall_max = 0.0
    overall_max_relative = 0.0
    passed = True
    tensors: dict[str, dict[str, object]] = {}
    for name in sorted(reference):
        expected = reference[name].detach().cpu().to(torch.float64)
        actual = candidate[name].detach().cpu().to(torch.float64)
        if actual.shape != expected.shape:
            raise ValueError(f"Comparison shape differs for {name}.")
        difference = (actual - expected).abs()
        allowed = atol + rtol * expected.abs()
        tensor_passed = bool((difference <= allowed).all())
        relative = difference / expected.abs().clamp_min(1e-12)
        count = difference.numel()
        error_sum = float(difference.sum())
        maximum = float(difference.max()) if count else 0.0
        maximum_relative = float(relative.max()) if count else 0.0
        tensors[name] = {
            "shape": list(expected.shape),
            "max_abs_error": maximum,
            "mean_abs_error": error_sum / count if count else 0.0,
            "max_relative_error": maximum_relative,
            "passed": tensor_passed,
        }
        total_error += error_sum
        total_values += count
        overall_max = max(overall_max, maximum)
        overall_max_relative = max(overall_max_relative, maximum_relative)
        passed = passed and tensor_passed
    return {
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_error": overall_max,
        "mean_abs_error": total_error / total_values if total_values else 0.0,
        "max_relative_error": overall_max_relative,
        "tensors": tensors,
        "passed": passed,
    }


def _initial_state() -> dict[str, Tensor]:
    torch.manual_seed(SEED)
    model = EdgeConditionedGATv2(**MODEL_CONFIG)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    chunked = ReceiverChunkedEdgeConditionedGATv2(
        **MODEL_CONFIG,
        receiver_chunk_size=int(SYNTHETIC_CONFIG["receiver_chunk_size"]),
        activation_checkpointing=True,
    )
    chunked.load_state_dict(state, strict=True)
    if tuple(chunked.state_dict()) != tuple(state):
        raise RuntimeError("Ordinary and receiver-chunked state layouts differ.")
    return state


def _gradient_summary(model: torch.nn.Module) -> dict[str, object]:
    parameter_count = 0
    gradient_count = 0
    finite = True
    nonzero = False
    sum_squared = torch.zeros((), device=next(model.parameters()).device)
    max_abs = torch.zeros_like(sum_squared)
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_count += 1
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            finite = False
            continue
        gradient_count += 1
        finite = finite and bool(torch.isfinite(gradient).all())
        nonzero = nonzero or bool((gradient != 0).any())
        grad_float = gradient.detach().float()
        sum_squared = sum_squared + grad_float.square().sum()
        max_abs = torch.maximum(max_abs, grad_float.abs().max())
    return {
        "parameter_tensor_count": parameter_count,
        "gradient_tensor_count": gradient_count,
        "missing_gradient_parameters": missing,
        "all_finite": finite,
        "any_nonzero": nonzero,
        "global_l2_norm": math.sqrt(float(sum_squared)),
        "max_abs": float(max_abs),
    }


def _run_execution(
    *,
    implementation: str,
    amp: bool,
    state: Mapping[str, Tensor],
    inputs: Mapping[str, Tensor],
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    if implementation == "ordinary":
        model: torch.nn.Module = EdgeConditionedGATv2(**MODEL_CONFIG)
    elif implementation == "receiver_chunked":
        model = ReceiverChunkedEdgeConditionedGATv2(
            **MODEL_CONFIG,
            receiver_chunk_size=int(
                SYNTHETIC_CONFIG["receiver_chunk_size"]
            ),
            activation_checkpointing=True,
        )
    else:
        raise ValueError(f"Unsupported implementation: {implementation}")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp,
    ):
        output = model(
            input_expression=inputs["input_expression"],
            gene_mask=inputs["gene_mask"],
            node_covariates=inputs["node_covariates"],
            edge_index=inputs["edge_index"],
            edge_attributes=inputs["edge_attributes"],
        )
        loss = F.mse_loss(
            output.prediction,
            inputs["target_expression"],
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = _gradient_summary(model)
    torch.cuda.synchronize(device)

    outputs = {
        "prediction": output.prediction.detach().float().cpu(),
        "node_embedding": output.node_embedding.detach().float().cpu(),
    }
    forward_finite = all(
        bool(torch.isfinite(value).all()) for value in outputs.values()
    )
    loss_value = float(loss.detach())
    record: dict[str, object] = {
        "implementation": implementation,
        "precision": "amp_fp16" if amp else "fp32",
        "autocast_enabled": amp,
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "loss": loss_value,
        "finite": {
            "forward_outputs": forward_finite,
            "backward_loss": math.isfinite(loss_value),
            "gradients": gradients["all_finite"],
        },
        "gradients": gradients,
        "peak_vram": {
            "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }
    del output, loss, optimizer, scaler, model
    torch.cuda.empty_cache()
    return outputs, record


def _device_record(device: torch.device) -> dict[str, object]:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "requested": str(device),
        "index": int(index),
        "name": properties.name,
        "compute_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def run_diagnostic(device_name: str = "cuda:0") -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AMP equivalence diagnostic.")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select one CUDA device.")
    if device.index is not None and (
        device.index < 0 or device.index >= torch.cuda.device_count()
    ):
        raise ValueError("--device selects an unavailable CUDA device.")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    cpu_inputs = _synthetic_inputs()
    state = _initial_state()
    input_checksum = _tensor_collection_sha256(cpu_inputs)
    state_checksum = _tensor_collection_sha256(state)
    inputs = {
        name: value.to(device)
        for name, value in cpu_inputs.items()
    }
    executions: dict[str, dict[str, object]] = {}
    output_values: dict[str, dict[str, Tensor]] = {}
    for key, implementation, amp in (
        ("ordinary_fp32", "ordinary", False),
        ("receiver_chunked_fp32", "receiver_chunked", False),
        ("ordinary_amp", "ordinary", True),
        ("receiver_chunked_amp", "receiver_chunked", True),
    ):
        outputs, record = _run_execution(
            implementation=implementation,
            amp=amp,
            state=state,
            inputs=inputs,
            device=device,
        )
        output_values[key] = outputs
        executions[key] = record

    comparison_sources = {
        "fp32_ordinary_vs_chunked": (
            "ordinary_fp32",
            "receiver_chunked_fp32",
        ),
        "amp_ordinary_vs_chunked": (
            "ordinary_amp",
            "receiver_chunked_amp",
        ),
        "chunked_fp32_vs_amp": (
            "receiver_chunked_fp32",
            "receiver_chunked_amp",
        ),
    }
    comparisons = {}
    for name, (reference, candidate) in comparison_sources.items():
        tolerance = TOLERANCES[name]
        comparisons[name] = _comparison_summary(
            output_values[reference],
            output_values[candidate],
            atol=tolerance["atol"],
            rtol=tolerance["rtol"],
        )

    finite_pass = all(
        all(bool(value) for value in execution["finite"].values())
        and bool(execution["gradients"]["any_nonzero"])
        and not execution["gradients"]["missing_gradient_parameters"]
        for execution in executions.values()
    )
    comparison_pass = all(
        bool(comparison["passed"])
        for comparison in comparisons.values()
    )
    graph = cpu_inputs["edge_index"]
    receiver = graph[1]
    graph_contract_pass = bool(
        not (receiver[1:] < receiver[:-1]).any()
        and not (graph[0] == graph[1]).any()
    )
    passed = finite_pass and comparison_pass and graph_contract_pass
    report: dict[str, object] = {
        "schema_version": 1,
        "diagnostic": "dense_gat_amp_equivalence",
        "seed": SEED,
        "device": _device_record(device),
        "model": {
            **MODEL_CONFIG,
            "receiver_chunk_size": int(
                SYNTHETIC_CONFIG["receiver_chunk_size"]
            ),
            "activation_checkpointing": True,
            "state_dict_sha256": state_checksum,
        },
        "synthetic_graph": {
            **SYNTHETIC_CONFIG,
            "n_directed_edges": int(graph.shape[1]),
            "receiver_sorted": True,
            "self_loops": False,
            "input_sha256": input_checksum,
        },
        "shapes": {
            name: list(value.shape)
            for name, value in sorted(cpu_inputs.items())
        },
        "executions": executions,
        "comparisons": comparisons,
        "tolerances": TOLERANCES,
        "checks": {
            "locked_model_shape": (
                MODEL_CONFIG["hidden_dim"] == 512
                and MODEL_CONFIG["graph_layers"] == 2
                and MODEL_CONFIG["attention_heads"] == 4
                and MODEL_CONFIG["edge_embedding_dim"] == 64
                and MODEL_CONFIG["ffn_dim"] == 512
                and MODEL_CONFIG["decoder_dim"] == 512
            ),
            "identical_state_dict": True,
            "receiver_sorted_no_self_loops": graph_contract_pass,
            "finite_forward_backward": finite_pass,
            "comparisons_within_tolerance": comparison_pass,
        },
        "passed": passed,
    }
    report["checks"]["locked_model_shape"] = bool(
        report["checks"]["locked_model_shape"]
    )
    return report


def _emit_report(
    report: Mapping[str, object],
    output: Path | None,
) -> None:
    serialized = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if output is not None:
        destination = output.resolve()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite diagnostic output: {destination}"
            )
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    print(serialized, end="")


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.output is not None:
        destination = arguments.output.resolve()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite diagnostic output: {destination}"
            )
    report = run_diagnostic(arguments.device)
    _emit_report(report, arguments.output)
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
