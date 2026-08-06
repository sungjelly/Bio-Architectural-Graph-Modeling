from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "diagnostics"
    / "check_dense_gat_amp_equivalence.py"
)


def _load_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "check_dense_gat_amp_equivalence",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_synthetic_graph_is_deterministic_receiver_sorted_and_loop_free() -> None:
    module = _load_script()
    first = module._synthetic_inputs(seed=17, n_nodes=12, incoming_degree=3)
    second = module._synthetic_inputs(seed=17, n_nodes=12, incoming_degree=3)

    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name])
    edge_index = first["edge_index"]
    source, receiver = edge_index
    assert edge_index.shape == (2, 36)
    assert torch.equal(
        torch.bincount(receiver, minlength=12),
        torch.full((12,), 3),
    )
    assert bool((receiver[:-1] <= receiver[1:]).all())
    assert not bool((source == receiver).any())
    assert (
        module._tensor_collection_sha256(first)
        == module._tensor_collection_sha256(second)
    )
    assert module.MODEL_CONFIG["hidden_dim"] == 512
    assert module.MODEL_CONFIG["graph_layers"] == 2
    assert module.MODEL_CONFIG["attention_heads"] == 4
    assert module.MODEL_CONFIG["edge_embedding_dim"] == 64
    assert module.MODEL_CONFIG["ffn_dim"] == 512
    assert module.MODEL_CONFIG["decoder_dim"] == 512


def test_comparison_summary_reports_weighted_errors_and_tolerance() -> None:
    module = _load_script()
    reference = {
        "prediction": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        "node_embedding": torch.tensor([[4.0, 5.0]]),
    }
    close = {
        name: value + 1e-4
        for name, value in reference.items()
    }
    passed = module._comparison_summary(
        reference,
        close,
        atol=2e-4,
        rtol=0.0,
    )
    failed = module._comparison_summary(
        reference,
        close,
        atol=1e-6,
        rtol=0.0,
    )

    assert passed["passed"] is True
    assert passed["max_abs_error"] == pytest.approx(1e-4, abs=2e-7)
    assert passed["mean_abs_error"] == pytest.approx(1e-4, abs=2e-7)
    assert failed["passed"] is False
    assert failed["tensors"]["prediction"]["passed"] is False


def test_optional_output_is_json_and_exclusive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    report = {"passed": True, "value": 3}
    assert module._build_parser().parse_args([]).output is None

    destination = tmp_path / "diagnostic.json"
    module._emit_report(report, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert json.loads(capsys.readouterr().out) == report

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._emit_report({"passed": False}, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == report
