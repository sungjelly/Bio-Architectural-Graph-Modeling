"""Synthetic fit-membership, projection and scVI frozen-inference checks."""
from importlib.util import find_spec
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.analysis import so12_cancer_baselines as baseline


def core(values, number=1, role="train", slide=1):
    return SimpleNamespace(alias=f"SO{slide}-C{number:02d}", split=role, slide=slide,
                           n_cells=len(values), target_expression=values, raw_counts=values,
                           coordinates_um=np.zeros((len(values), 2)))


class Unreadable:
    @property
    def target_expression(self):
        raise AssertionError("Held-out features were inspected by PCA fitting")
    split = "test"


def test_pca_fit_is_train_only_and_matches_full_centered_covariance():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(53, 7)) * np.arange(1, 8) + 3
    fitted = baseline.fit_pca_training([core(values[:20]), Unreadable(), core(values[20:])], 4, chunk_rows=9)
    assert int(fitted["training_cells"]) == len(values)
    np.testing.assert_allclose(fitted["mean"], values.mean(axis=0), atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(np.cov(values.T))[::-1]
    np.testing.assert_allclose(fitted["explained_variance"], eigenvalues[:4], rtol=1e-12)
    np.testing.assert_allclose(fitted["components"] @ fitted["components"].T, np.eye(4), atol=1e-12)


def test_pca_projection_never_recenters_the_test_population():
    rng = np.random.default_rng(8)
    train, test = rng.normal(size=(30, 6)), rng.normal(size=(12, 6)) + 10
    fitted = baseline.fit_pca_training([core(train)], 3)
    before = {key: value.copy() for key, value in fitted.items()}
    projected = baseline.project_pca(test, fitted, chunk_rows=5)
    np.testing.assert_allclose(projected, (test - train.mean(0)) @ fitted["components"].T, rtol=1e-6)
    assert np.linalg.norm(projected.mean(0)) > 1
    for key in before:
        np.testing.assert_array_equal(before[key], fitted[key])


def test_scvi_count_input_excludes_tests_and_preserves_explicit_core_roles():
    cores = [core(np.ones((4, 3), np.int32), 1),
             core(np.ones((2, 3), np.int32), 2, "validation"),
             core(np.ones((5, 3), np.int32), 16, "train", 2)]
    counts, slides, indices = baseline.count_inputs(cores, ("a", "b", "c"))
    assert counts.shape == (11, 3) and counts.dtype == np.float32
    assert indices[0].tolist() == [0, 1, 2, 3, 6, 7, 8, 9, 10]
    assert indices[1].tolist() == [4, 5] and len(indices[2]) == 0
    assert set(slides[indices[0]]) == {"1", "2"}
    with pytest.raises(ValueError, match="cannot include test"):
        baseline.count_inputs(cores + [core(np.ones((2, 3)), 15, "test", 2)], ("a", "b", "c"))


def test_scvi_rejects_signed_or_normalized_features():
    for values in (np.ones((4, 3), dtype=np.float32), -np.ones((4, 3), dtype=np.int32)):
        with pytest.raises(ValueError, match="integer raw counts"):
            baseline.count_inputs([core(values), core(values, 16, "train", 2)], ("a", "b", "c"))


def test_validation_selection_retains_best_immutable_state_not_last():
    best = baseline.BestValidationState()
    state = {"weight": torch.tensor([1.0, 2.0])}
    assert best.observe(5.0, 0, state)
    state["weight"].add_(1)
    assert not best.observe(6.0, 1, state)
    assert not best.observe(5.0, 2, state)
    assert best.epoch == 0
    np.testing.assert_array_equal(best.state["weight"].numpy(), [1, 2])
    assert best.observe(4.0, 3, state)
    assert best.epoch == 3
    with pytest.raises(ValueError, match="Nonfinite"):
        best.observe(float("nan"), 4, state)


def test_exact_training_membership_rejects_core_substitution():
    cores = [core(np.ones((1, 2)), number) for number in baseline.TRAIN_CORE_NUMBERS]
    baseline.verify_membership(cores, ("train",), {"train": 21})
    cores[0].alias = "SO2-C15"
    with pytest.raises(ValueError, match="membership"):
        baseline.verify_membership(cores, ("train",), {"train": 21})


@pytest.mark.skipif(find_spec("scvi") is None, reason="isolated scVI environment not on this interpreter")
def test_real_scvi_fixed_split_best_restore_and_frozen_query(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    genes = tuple(f"g{i}" for i in range(60))
    trainval = SimpleNamespace(genes=genes, cores=(
        core(rng.poisson(2, (30, 60)).astype(np.int32), 1),
        core(rng.poisson(2, (10, 60)).astype(np.int32), 2, "validation"),
        core(rng.poisson(3, (30, 60)).astype(np.int32), 16, "train", 2),
        core(rng.poisson(3, (10, 60)).astype(np.int32), 27, "validation", 2)))
    test = SimpleNamespace(genes=genes, cores=(core(rng.poisson(2, (12, 60)).astype(np.int32), 15, "test", 2),))
    events = []
    def load(source, paths, roles):
        events.append(roles)
        if roles == ("test",):
            assert (tmp_path / "baselines/scvi_frozen_before_test.json").is_file()
            return test
        assert roles == ("train", "validation")
        return trainval
    monkeypatch.setattr(baseline, "load_data", load)
    monkeypatch.setattr(baseline, "validate_test_alignment", lambda *args: None)
    monkeypatch.setattr(baseline, "disk_gate", lambda *args: None)
    monkeypatch.setitem(baseline.SCVI_CONFIG, "max_epochs", 2)
    monkeypatch.setitem(baseline.SCVI_CONFIG, "n_epochs_kl_warmup", 1)
    monkeypatch.setitem(baseline.SCVI_CONFIG, "early_stopping_warmup_epochs", 1)
    (tmp_path / "baselines").mkdir()
    embedding, receipt, output_genes = baseline.run_scvi(tmp_path, {}, None, {}, None, "cpu")
    assert embedding.shape == (12, 50) and np.isfinite(embedding).all()
    assert receipt["test_query_adaptation"] is False and receipt["test_used_for_fitting"] is False
    assert receipt["selected_epoch_zero_based"] in (0, 1)
    assert tuple(output_genes) == genes
    assert events == [("train", "validation"), ("test",)]
    assert not list(tmp_path.rglob("*.h5ad"))
    history = [json.loads(line) for line in (tmp_path / "baselines/scvi_training.jsonl").read_text().splitlines()]
    assert len(history) == 2
    assert receipt["validation_elbo"] == min(row["elbo_validation"] for row in history)
