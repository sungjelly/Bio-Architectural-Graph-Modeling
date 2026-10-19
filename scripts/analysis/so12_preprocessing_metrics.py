"""Exploratory diagnostics for joint SO_1/SO_2 preprocessing.

The caller owns fitting preprocessing/PCA on training donors and registering the
experiment. These diagnostics neither fit PCA nor infer a biological cell type.
Vendor-label recovery is circular agreement with expression-derived annotations;
slide predictability and slide mixing are confounded descriptive diagnostics.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SEED = 17012026
MAX_CELLS_PER_CORE = 1000


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _depth_r2(pcs: np.ndarray, depth: np.ndarray) -> float:
    """Variance-weighted PC R² for one core; each PC has its own intercept."""
    if len(pcs) < 3:
        return float("nan")
    y = np.log1p(np.asarray(depth, dtype=np.float64))
    y -= y.mean()
    x = np.asarray(pcs, dtype=np.float64)
    x = x - x.mean(axis=0)
    sum_x2 = np.sum(x * x, axis=0)
    sum_y2 = float(y @ y)
    denom = float(sum_x2.sum())
    if sum_y2 <= 0 or denom <= 0:
        return float("nan")
    # sum(var(PC_j) * corr(PC_j, log1p(depth))**2) / sum(var(PC_j))
    return float(np.clip(np.sum((x.T @ y) ** 2) / (sum_y2 * denom), 0, 1))


def _classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=600,
            solver="lbfgs",
            random_state=SEED,
        ),
    )


def _scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if len(truth) == 0:
        return {"n_cells": 0, "balanced_accuracy": None, "macro_f1": None}
    truth_labels = np.unique(truth)
    return {
        "n_cells": len(truth),
        "n_observed_labels": len(truth_labels),
        # Macro recall over classes present in this evaluation stratum equals
        # balanced accuracy, and handles predicted-only labels without warnings.
        "balanced_accuracy": float(
            recall_score(truth, predicted, labels=truth_labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }


def _fit_with_warnings(model, x, y, context: str, notes: list[str]):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model.fit(x, y)
    for warning in recorded:
        notes.append(f"{context}: {warning.category.__name__}: {warning.message}")
    return model


def _vendor_recovery(
    pcs: np.ndarray,
    md: pd.DataFrame,
    method: str,
    notes: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    results, donor_rows, core_rows = [], [], []
    label_valid = md["vendor_label"].notna().to_numpy(copy=True)
    label_text = md["vendor_label"].fillna("").astype(str).to_numpy()
    label_valid &= label_text != ""
    for slide in sorted(md["slide"].unique(), key=str):
        in_slide = md["slide"].to_numpy() == slide
        tr = in_slide & md["train"].to_numpy() & label_valid
        te = in_slide & ~md["train"].to_numpy() & label_valid
        train_labels = label_text[tr]
        test_labels = label_text[te]
        base = {
            "method": method,
            "slide": str(slide),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "n_train_labels": len(np.unique(train_labels)),
            "n_test_labels": len(np.unique(test_labels)),
            "labels_unseen_in_training": sorted(set(test_labels) - set(train_labels)),
            "n_test_cells_unseen_label": int(np.sum(~np.isin(test_labels, train_labels))),
        }
        if tr.sum() < 3 or te.sum() == 0 or len(np.unique(train_labels)) < 2:
            results.append({**base, "status": "insufficient_training_classes_or_test_cells"})
            continue
        for null in (False, True):
            fit_labels = train_labels.copy()
            if null:
                # Break cell-level correspondence while retaining each training
                # core's annotation composition, a stronger null than global shuffle.
                rng = np.random.default_rng(SEED)
                train_cores = md.loc[tr, "core_alias"].to_numpy()
                for core in sorted(np.unique(train_cores)):
                    idx = np.flatnonzero(train_cores == core)
                    fit_labels[idx] = rng.permutation(fit_labels[idx])
            kind = "within_core_train_label_shuffle" if null else "observed_labels"
            model = _fit_with_warnings(
                _classifier(), pcs[tr], fit_labels, f"{method}/{slide}/{kind}", notes
            )
            prediction = model.predict(pcs[te])
            results.append({**base, "control": kind, "status": "ok", **_scores(test_labels, prediction)})
            test_md = md.loc[te].reset_index(drop=True)
            for column, destination in (("donor_group", donor_rows), ("core_alias", core_rows)):
                for group in sorted(test_md[column].unique(), key=str):
                    subset = (test_md[column].to_numpy() == group)
                    destination.append({
                        "method": method,
                        "slide": str(slide),
                        column: str(group),
                        "control": kind,
                        **_scores(test_labels[subset], prediction[subset]),
                    })
    return results, donor_rows, core_rows


def _slide_diagnostics(pcs: np.ndarray, md: pd.DataFrame, method: str, notes: list[str]) -> dict:
    slides = sorted(md["slide"].unique(), key=str)
    result = {"interpretation": "Confounded by tissue composition and disease; not an integration objective."}
    if len(slides) != 2:
        return {**result, "status": "requires_two_slides"}
    y = (md["slide"].to_numpy() == slides[1]).astype(np.int64)
    tr = md["train"].to_numpy()
    if len(np.unique(y[tr])) == 2 and len(np.unique(y[~tr])) == 2:
        model = _fit_with_warnings(_classifier(), pcs[tr], y[tr], f"{method}/slide_classifier", notes)
        result["heldout_slide_auc"] = float(roc_auc_score(y[~tr], model.predict_proba(pcs[~tr])[:, 1]))
    else:
        result["heldout_slide_auc"] = None
        notes.append(f"{method}: slide classifier needs both slides in train and heldout data.")
    rng = np.random.default_rng(SEED)
    by_slide = [np.flatnonzero(y == value) for value in (0, 1)]
    n_each = min(len(idx) for idx in by_slide)
    if n_each >= 3:
        chosen = np.concatenate([rng.choice(idx, n_each, replace=False) for idx in by_slide])
        selected = pcs[chosen]
        selected_labels = y[chosen]
        # Request one extra candidate, then remove each query's own exact index.
        # Explicit removal is needed with duplicate embeddings and distance ties.
        knn = NearestNeighbors(n_neighbors=6, algorithm="brute", n_jobs=1).fit(selected)
        neighbors = knn.kneighbors(selected, return_distance=False)
        fraction = np.empty(len(chosen), dtype=float)
        for i, row in enumerate(neighbors):
            other = row[row != i][:5]
            fraction[i] = np.mean(selected_labels[other] != selected_labels[i])
        result["pooled_balanced_slide_5nn_cross_slide_fraction"] = float(fraction.mean())
        result["n_cells_per_slide_for_knn"] = n_each
        result["knn_independent_balanced_labels_reference"] = n_each / (2 * n_each - 1)
        result["knn_scope"] = "Balanced-slide pooled train and heldout cells; Euclidean unwhitened PCs."
    else:
        result["pooled_balanced_slide_5nn_cross_slide_fraction"] = None
    return result


def _plot(embeddings: dict, md: pd.DataFrame, depth: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(embeddings)
    fig, axes = plt.subplots(2, len(methods), figsize=(4.4 * len(methods), 8.0), squeeze=False)
    colors = ["#3274a1", "#d28335", "#43a272", "#986db5"]
    for column, method in enumerate(methods):
        pcs = embeddings[method]
        ax = axes[0, column]
        for i, slide in enumerate(sorted(md["slide"].unique(), key=str)):
            selected = np.flatnonzero(md["slide"].to_numpy() == slide)
            if len(selected) > 4000:
                selected = np.random.default_rng(SEED).choice(selected, 4000, replace=False)
            ax.scatter(pcs[selected, 0], pcs[selected, 1], s=2, alpha=.18,
                       color=colors[i % len(colors)], rasterized=True, label=str(slide))
        ax.set(title=method.replace("_", " "), xlabel="PC 1", ylabel="PC 2")
        ax.legend(markerscale=3, frameon=False)
        rows = depth[(depth["method"] == method) & (~depth["train"]) & (depth["population"] == "all_sampled")]
        rows = rows.sort_values("core_alias")
        ax = axes[1, column]
        ax.bar(np.arange(len(rows)), rows["depth_weighted_r2"], color="#56758a")
        ax.set_xticks(np.arange(len(rows)), rows["core_alias"], rotation=65, ha="right", fontsize=8)
        ax.set(ylabel="PC variance associated with log depth (R²)", title="Heldout cores; lower association")
        finite = rows["depth_weighted_r2"].dropna()
        if len(finite):
            ax.axhline(finite.median(), color="#a44438", linestyle="--", linewidth=1,
                       label=f"Median {finite.median():.3f}")
            ax.legend(frameon=False, fontsize=8)
    heldout = depth[(~depth["train"]) & (depth["population"] == "all_sampled")]["depth_weighted_r2"]
    top = max(.05, float(heldout.max()) * 1.12) if heldout.notna().any() else 1.0
    for ax in axes[1]:
        ax.set_ylim(0, top)
    fig.suptitle("Exploratory preprocessing diagnostics\nSlide separation is confounded; label agreement is not independent biological validation.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output_dir / "preprocessing_diagnostics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def evaluate(embeddings: dict[str, np.ndarray], metadata: pd.DataFrame, output_dir: Path) -> dict:
    """Save aggregate diagnostics for aligned, training-fitted PC representations.

    Required metadata: core_alias (or core), slide, donor_group, train, qc_pass,
    vendor_label and depth. All sampled cells are primary; QC-pass cells provide
    a secondary depth diagnostic. No direct donor identifiers may be supplied.
    """
    if not embeddings:
        raise ValueError("At least one embedding is required.")
    md = metadata.copy().reset_index(drop=True)
    if "core_alias" not in md and "core" in md:
        md = md.rename(columns={"core": "core_alias"})
    required = {"core_alias", "slide", "donor_group", "train", "qc_pass", "vendor_label", "depth"}
    if required - set(md):
        raise ValueError(f"Missing metadata fields: {sorted(required - set(md))}")
    if md.empty:
        raise ValueError("Metadata contains no cells.")
    for column in ("core_alias", "slide", "donor_group", "train", "qc_pass", "depth"):
        if md[column].isna().any():
            raise ValueError(f"Missing values in metadata field {column}.")
    for column in ("train", "qc_pass"):
        if not all(isinstance(value, (bool, np.bool_)) for value in md[column].unique()):
            raise ValueError(f"{column} must contain booleans, not strings or integers.")
        md[column] = md[column].astype(bool)
    if not np.isfinite(md["depth"].to_numpy(dtype=float)).all() or (md["depth"] < 0).any():
        raise ValueError("Depth must be finite and nonnegative.")
    for column in ("core_alias", "donor_group"):
        if (md.groupby(column, observed=True)["train"].nunique() > 1).any():
            raise ValueError(f"A {column} spans both training and heldout cells.")
    if (md.groupby("core_alias", observed=True)["slide"].nunique() > 1).any():
        raise ValueError("Core aliases must be unique across slides.")
    raw_embeddings = {}
    for method, matrix in embeddings.items():
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != len(md) or matrix.shape[1] < 2:
            raise ValueError(f"Invalid embedding shape for {method}: {matrix.shape}.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Nonfinite embedding values for {method}.")
        raw_embeddings[method] = matrix
    rng = np.random.default_rng(SEED)
    selected = []
    for _, group in md.groupby("core_alias", sort=True, observed=True):
        idx = group.index.to_numpy()
        selected.extend(rng.choice(idx, MAX_CELLS_PER_CORE, replace=False) if len(idx) > MAX_CELLS_PER_CORE else idx)
    selected = np.sort(np.asarray(selected, dtype=int))
    md = md.iloc[selected].reset_index(drop=True)
    embeddings = {key: value[selected] for key, value in raw_embeddings.items()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    notes, depth_rows, donor_rows, core_label_rows = [], [], [], []
    result: dict[str, Any] = {
        "seed": SEED,
        "n_sampled_cells": len(md),
        "max_cells_per_core": MAX_CELLS_PER_CORE,
        "sampling": "Fixed random per-core sample, identical for all transforms; all sampled cells primary.",
        "n_training_cores": md.loc[md["train"], "core_alias"].nunique(),
        "n_heldout_cores": md.loc[~md["train"], "core_alias"].nunique(),
        "primary_metric": "Median heldout-core variance-weighted PC R² versus log1p biological-probe depth; lower is less depth-associated.",
        "limitations": [
            "Lower depth association is not proof of better biological information or masked-expression prediction.",
            "Vendor annotations derive from expression; recovery is a circular agreement diagnostic, not independent biological validation.",
            "Vendor label namespaces are evaluated separately within slides and are not assumed aligned across slides.",
            "Slide mixing and slide AUC are confounded by tissue composition and disease, and are not optimization objectives.",
            "Paired cores from a donor are not independent biological replicates; per-donor summaries are descriptive.",
            "One deterministic within-training-core label permutation is a diagnostic null, not a permutation significance test.",
        ],
        "methods": {},
        "warnings": notes,
    }
    # Keep BLAS usage bounded across multiple sklearn estimators and distance
    # calculations, regardless of the host machine's advertised CPU count.
    with threadpool_limits(limits=4):
        for method, pcs in embeddings.items():
            method_rows = []
            for core, group in md.groupby("core_alias", sort=True, observed=True):
                for population in ("all_sampled", "qc_pass"):
                    selected_group = group if population == "all_sampled" else group[group["qc_pass"]]
                    idx = selected_group.index.to_numpy()
                    row = {
                        "method": method,
                        "core_alias": str(core),
                        "slide": str(group["slide"].iloc[0]),
                        "donor_group": str(group["donor_group"].iloc[0]),
                        "train": bool(group["train"].iloc[0]),
                        "population": population,
                        "n_cells": len(idx),
                        "depth_weighted_r2": _depth_r2(pcs[idx], md.loc[idx, "depth"].to_numpy()),
                    }
                    method_rows.append(row)
                    depth_rows.append(row)
            frame = pd.DataFrame(method_rows)
            primary = frame[(~frame["train"]) & (frame["population"] == "all_sampled")]
            secondary = frame[(~frame["train"]) & (frame["population"] == "qc_pass")]
            vendor, donors, cores = _vendor_recovery(pcs, md, method, notes)
            donor_rows.extend(donors)
            core_label_rows.extend(cores)
            observed = [row for row in vendor if row.get("control") == "observed_labels"]
            null = [row for row in vendor if row.get("control") == "within_core_train_label_shuffle"]
            result["methods"][method] = {
                "n_components": pcs.shape[1],
                "heldout_core_median_depth_weighted_r2": primary["depth_weighted_r2"].median(),
                "heldout_qc_pass_core_median_depth_weighted_r2": secondary["depth_weighted_r2"].median(),
                "heldout_donor_mean_depth_weighted_r2": primary.groupby("donor_group", observed=True)["depth_weighted_r2"].mean().to_dict(),
                "within_slide_vendor_label_recovery": vendor,
                "mean_across_slides_vendor_balanced_accuracy": np.mean([r["balanced_accuracy"] for r in observed]) if observed else None,
                "mean_across_slides_null_balanced_accuracy": np.mean([r["balanced_accuracy"] for r in null]) if null else None,
                "slide_diagnostics": _slide_diagnostics(pcs, md, method, notes),
            }
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(output_dir / "per_core_depth_association.csv", index=False)
    pd.DataFrame(donor_rows).to_csv(output_dir / "per_donor_vendor_label_recovery.csv", index=False)
    pd.DataFrame(core_label_rows).to_csv(output_dir / "per_core_vendor_label_recovery.csv", index=False)
    _plot(embeddings, md, depth, output_dir)
    result = _json_safe(result)
    (output_dir / "metrics_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
