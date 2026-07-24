from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

warnings.filterwarnings(
    "ignore",
    message=r"savefig\(\) got unexpected keyword argument.*",
    category=MatplotlibDeprecationWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLINICAL = PROJECT_ROOT / "data" / "clinical"
FOV_CORE_MAP_FILE = DATA_CLINICAL / "fov_core_map.csv"
OUT_ROOT = PROJECT_ROOT / "results" / "core_assignment"
FIGURES = OUT_ROOT / "figures"
TABLES = OUT_ROOT / "tables"

SLIDES = ("SO_1", "SO_2")
FILE_PREFIX = {"SO_1": "26040302SO_1", "SO_2": "26040302SO_2"}


def _first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


CLINICAL_WORKBOOK = _first_existing(
    (
        PROJECT_ROOT / "data" / "Gastric Study_2.xlsx",
        DATA_RAW / "Gastric Study_2.xlsx",
    )
)


def _cosmx_raw_path(slide: str, suffix: str) -> Path:
    filename = f"{FILE_PREFIX[slide]}_{suffix}"
    direct = DATA_RAW / filename
    nested = direct / filename
    return nested if nested.is_file() else direct


def _load_fov_core_map() -> pd.DataFrame:
    table = pd.read_csv(FOV_CORE_MAP_FILE)
    required = {"slide", "core_label", "fov"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{FOV_CORE_MAP_FILE} missing columns: {sorted(missing)}")
    table = table[["slide", "core_label", "fov"]].copy()
    table["core_label"] = table["core_label"].astype(int)
    table["fov"] = table["fov"].astype(int)

    duplicated = table.duplicated(["slide", "fov"], keep=False)
    if duplicated.any():
        dupes = table.loc[duplicated, ["slide", "fov"]].drop_duplicates()
        raise ValueError(f"Duplicate FOV assignments:\n{dupes.to_string(index=False)}")
    return table


def _normalize_histology(raw_label: object) -> str:
    label = str(raw_label).strip()
    if "cancer" in label.lower():
        return "Cancer"
    if label in {"HGD", "LGD"}:
        return label
    if label == "정상":
        return "Normal"
    if "주변조직" in label:
        return "AdjacentNormal"
    return label


def _load_label_map() -> pd.DataFrame:
    raw = pd.read_excel(CLINICAL_WORKBOOK, sheet_name=0, header=None)
    labels = raw.iloc[:, :3].copy()
    labels.columns = ["core_label", "donor_id", "raw_tissue_label"]
    labels = labels[pd.to_numeric(labels["core_label"], errors="coerce").notna()]
    labels["core_label"] = labels["core_label"].astype(int)
    labels = labels[labels["core_label"].between(1, 28)].copy()
    labels["donor_id"] = labels["donor_id"].astype(str).str.strip()
    labels["tissue_stage"] = labels["raw_tissue_label"].map(_normalize_histology)

    donor_stage = {}
    for donor, group in labels.groupby("donor_id", sort=False):
        lesion_rows = group[group["tissue_stage"] != "AdjacentNormal"]
        donor_stage[donor] = (
            lesion_rows["tissue_stage"].iloc[0] if len(lesion_rows) else "Unknown"
        )
    labels["donor_stage"] = labels["donor_id"].map(donor_stage)
    labels["tissue_type"] = labels["tissue_stage"].where(
        labels["tissue_stage"].isin(["Normal", "AdjacentNormal"]), "lesion"
    )
    labels["tissue_type"] = labels["tissue_type"].replace(
        {"Normal": "normal", "AdjacentNormal": "normal"}
    )
    return labels[["core_label", "donor_id", "donor_stage", "tissue_type", "tissue_stage"]]


def _assign_cores(slide: str, fov_map: pd.DataFrame) -> pd.DataFrame:
    path = _cosmx_raw_path(slide, "fov_positions_file.csv")
    fovs = pd.read_csv(path)
    x_col = "x_global_mm" if "x_global_mm" in fovs.columns else "x_global_px"
    y_col = "y_global_mm" if "y_global_mm" in fovs.columns else "y_global_px"
    fovs = fovs.rename(columns={x_col: "x", y_col: "y"})[["FOV", "x", "y"]]

    slide_map = fov_map[fov_map["slide"] == slide][["fov", "core_label"]]
    out = fovs.merge(slide_map, left_on="FOV", right_on="fov", how="left")
    return out.drop(columns=["fov"])


def _core_fov_table(fov_map: pd.DataFrame, label_map: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for slide in SLIDES:
        assigned = _assign_cores(slide, fov_map)
        assigned["slide"] = slide
        frames.append(assigned.merge(label_map, on="core_label", how="left"))
    return pd.concat(frames, ignore_index=True)


def _save_assignment_tables(table: pd.DataFrame) -> tuple[Path, Path]:
    TABLES.mkdir(parents=True, exist_ok=True)
    table_path = TABLES / "fov_core_assignments.csv"
    summary_path = TABLES / "core_assignment_summary.csv"
    table.to_csv(table_path, index=False)

    summary = (
        table.groupby(
            ["slide", "core_label", "donor_id", "donor_stage", "tissue_type", "tissue_stage"],
            dropna=False,
        )
        .agg(n_fovs=("FOV", "nunique"), min_fov=("FOV", "min"), max_fov=("FOV", "max"))
        .reset_index()
        .sort_values(["slide", "core_label"], na_position="last")
    )
    summary.to_csv(summary_path, index=False)
    return table_path, summary_path


def _draw_layout(ax, fovs: pd.DataFrame, slide: str, annotate_fovs: bool) -> None:
    assigned = fovs[fovs["core_label"].notna()].copy()
    unassigned = fovs[fovs["core_label"].isna()].copy()
    ax.scatter(
        assigned["x"],
        assigned["y"],
        c=assigned["core_label"].astype(int),
        cmap="tab20",
        norm=Normalize(vmin=1, vmax=28),
        s=62,
        edgecolors="black",
        linewidths=0.25,
    )
    if len(unassigned):
        ax.scatter(
            unassigned["x"],
            unassigned["y"],
            c="#666666",
            marker="x",
            s=85,
            linewidths=1.4,
            label="Unassigned",
        )
    if annotate_fovs:
        for _, row in fovs.iterrows():
            ax.annotate(
                str(int(row["FOV"])),
                (row["x"], row["y"]),
                fontsize=5,
                ha="center",
                va="center",
                color="black",
            )
    for core_label, group in assigned.groupby("core_label"):
        ax.annotate(
            str(int(core_label)),
            (group["x"].mean(), group["y"].mean()),
            fontsize=18,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78, linewidth=0),
        )
    ax.set_title(f"{slide}: FOV -> core label")
    ax.set_xlabel("x_global (mm)")
    ax.set_ylabel("y_global (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if len(unassigned):
        ax.legend(loc="best")


def _save_figures(fov_map: pd.DataFrame, annotate_fovs: bool) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    image_paths = []
    slide_tables = {slide: _assign_cores(slide, fov_map) for slide in SLIDES}

    for slide, fovs in slide_tables.items():
        fig, ax = plt.subplots(figsize=(12, 12))
        _draw_layout(ax, fovs, slide, annotate_fovs)
        out = FIGURES / f"fov_core_layout_{slide}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        image_paths.append(out)

    fig, axes = plt.subplots(1, len(SLIDES), figsize=(12 * len(SLIDES), 12), squeeze=False)
    for ax, slide in zip(axes[0], SLIDES):
        _draw_layout(ax, slide_tables[slide], slide, annotate_fovs)
    mappable = ScalarMappable(norm=Normalize(vmin=1, vmax=28), cmap="tab20")
    mappable.set_array([])
    fig.colorbar(
        mappable,
        ax=list(axes[0]),
        label="core_label",
        ticks=list(range(1, 29)),
        shrink=0.75,
    )
    combined = FIGURES / "fov_core_layout_combined.png"
    fig.savefig(combined, dpi=180, bbox_inches="tight")
    plt.close(fig)
    image_paths.append(combined)
    return image_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CosMx FOV assignments to Gastric Study tissue cores."
    )
    parser.add_argument(
        "--no-fov-labels",
        action="store_true",
        help="Hide individual FOV numbers and show only core labels.",
    )
    args = parser.parse_args()

    fov_map = _load_fov_core_map()
    label_map = _load_label_map()
    table = _core_fov_table(fov_map, label_map)
    table_path, summary_path = _save_assignment_tables(table)
    image_paths = _save_figures(fov_map, annotate_fovs=not args.no_fov_labels)

    print("assignments", table_path)
    print("summary", summary_path)
    for path in image_paths:
        print("figure", path)


if __name__ == "__main__":
    main()
