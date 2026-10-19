"""Independent review checks; no source or active-run configuration changes."""

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from scripts.train import run_local_gene_residual as runner
from spatial_benchmark.identifiers import scientific_id
from spatial_benchmark.local_gene_residual import LocalGeneResidualCountModel


@pytest.mark.parametrize("change", ["source", "contract", "passed", "status"])
def test_full_run_gate_rejects_pilot_identity_or_correctness_drift(tmp_path, monkeypatch, change) -> None:
    root = tmp_path / "pilot"
    (root / "diagnostics").mkdir(parents=True)
    receipt = {"passed": True, "source_hashes": {"model.py": "original"},
               "frozen_contract_sha256": "contract"}
    if change == "source":
        receipt["source_hashes"] = {"model.py": "changed"}
    elif change == "contract":
        receipt["frozen_contract_sha256"] = "changed"
    elif change == "passed":
        receipt["passed"] = False
    (root / "diagnostics/preflight.json").write_text(json.dumps(receipt))
    registry = SimpleNamespace(get_run=lambda run_id: {
        "status": "failed" if change == "status" else "completed", "artifact_path": str(root)
    })
    monkeypatch.setattr(runner, "verify_run_bundle", lambda path: {"ok": True})
    with pytest.raises(RuntimeError):
        runner.verify_pilot(registry, None, "pilot", {"model.py": "original"},
                            {"metadata": {"frozen_contract_sha256": "contract"}})


def test_scientific_variants_exclude_seed_gpu_and_attempt() -> None:
    first = runner.build_config(SimpleNamespace(
        seed=0, attempt=1, variant="local", profile="full", gpu="0", smoke_run="pilot"
    ))
    second = runner.build_config(SimpleNamespace(
        seed=1, attempt=2, variant="local", profile="full", gpu="1", smoke_run="pilot"
    ))
    self_only = runner.build_config(SimpleNamespace(
        seed=0, attempt=1, variant="self", profile="full", gpu="2", smoke_run="pilot"
    ))
    assert scientific_id(first) == scientific_id(second)
    assert scientific_id(first) != scientific_id(self_only)


def test_atomic_checkpoint_failure_preserves_last_complete_checkpoint(tmp_path, monkeypatch) -> None:
    path = tmp_path / "checkpoints/best.ckpt"
    runner.atomic_checkpoint(path, {"epoch": 2, "model": torch.tensor([3.0])})
    before = path.read_bytes()
    original_save = torch.save

    def broken_save(state, handle):
        handle.write(b"partial checkpoint")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk write failure"):
        runner.atomic_checkpoint(path, {"epoch": 3})
    assert path.read_bytes() == before
    assert torch.load(path, weights_only=True)["epoch"] == 2
    monkeypatch.setattr(torch, "save", original_save)
    runner.atomic_checkpoint(path, {"epoch": 4})
    assert torch.load(path, weights_only=True)["epoch"] == 4
    assert not path.with_suffix(".tmp").exists()


def test_evaluation_pools_by_entries_and_preserves_strata_across_batch_sizes(monkeypatch) -> None:
    genes = 1000
    rng = np.random.default_rng(117)
    cores = []
    masks = {}
    for alias, rate, mask_sizes in (("A", 0.2, [10, 300, 600]), ("B", 2.0, [900, 30])):
        rows = len(mask_sizes)
        counts = rng.poisson(rate, (rows, genes)).astype(np.int32)
        cores.append(SimpleNamespace(
            alias=alias, n_cells=rows, counts=counts,
            expression=np.log1p(counts.astype(np.float32)),
            covariates=np.zeros((rows, 0), dtype=np.float32),
            local_log_mean=np.full((rows, genes), rate, dtype=np.float32),
            local_positive_fraction=np.full((rows, genes), 0.1 * rate, dtype=np.float32),
        ))
        mask = np.zeros((rows, genes), dtype=bool)
        for row, size in enumerate(mask_sizes):
            mask[row, :size] = True
        masks[alias] = SimpleNamespace(mask=mask)
    model = LocalGeneResidualCountModel(genes, covariate_dim=0, hidden_dim=8, dropout=0.0)
    with torch.no_grad():
        model.local_detection_weights.fill_(0.2)
        model.local_count_weights.fill_(0.1)

    def stage_cpu(core):
        return {name: torch.from_numpy(getattr(core, name)) for name in (
            "expression", "counts", "covariates", "local_log_mean", "local_positive_fraction"
        )}

    def forward_cpu(model, tensors, mask, indices):
        return model(tensors["expression"][indices], mask,
                     tensors["covariates"][indices], tensors["local_log_mean"][indices],
                     tensors["local_positive_fraction"][indices])

    monkeypatch.setattr(runner, "stage", stage_cpu)
    monkeypatch.setattr(runner, "forward", forward_cpu)
    # evaluate() otherwise requests a CUDA device for its mask; this proxy only
    # changes placement inside this isolated CPU unit test.
    monkeypatch.setattr(runner, "torch", SimpleNamespace(
        as_tensor=lambda value, **kwargs: torch.as_tensor(value)
    ))
    batch_one = runner.evaluate(model, cores, masks, 1, strata=True)
    batch_all = runner.evaluate(model, cores, masks, 10, strata=True)
    for first, second in zip(batch_one, batch_all):
        if "count_nll" in first:
            for key in first:
                assert first[key] == pytest.approx(second[key], abs=2e-7), key
        else:
            for group in first:
                for key in first[group]:
                    assert first[group][key] == pytest.approx(second[group][key], abs=2e-7), (group, key)
    pooled, per_core, strata = batch_all
    assert pooled["masked_entries"] == 1840
    assert sum(group["masked_entries"] for group in strata.values()) == pooled["masked_entries"]
    assert {key: group["masked_entries"] for key, group in strata.items()} == {
        "1_250": 40, "251_500": 300, "501_750": 600, "751_1000": 900,
    }
    for key in ("raw_count_mae", "count_nll", "detection_brier"):
        weighted = sum(core[key] * core["masked_entries"] for core in per_core.values()) / 1840
        assert pooled[key] == pytest.approx(weighted, abs=2e-7)
    assert not model.training
