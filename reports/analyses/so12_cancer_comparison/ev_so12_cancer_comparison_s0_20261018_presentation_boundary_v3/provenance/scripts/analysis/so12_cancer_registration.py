"""Geometry-only registration of the two supplied annotated histology screenshots.

No expression, embedding, cluster label, or cancer overlap enters fitting. Blue
contours are extracted independently only after tissue-silhouette registration.
The result is approximate section alignment, not a same-section pixel truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize

from spatial_benchmark.paths import current_paths

CORES = (15, 23)
COUNTS = {15: 38145, 23: 43462}
# These image-space windows isolate the named central tissue, based on geometry
# alone. They exclude adjacent cores and handwritten blue slide marks.
WINDOWS = {15: (850, 50, 1900, 1440), 23: (800, 50, 1800, 1440)}
BAR_PIXELS = {15: 155, 23: 188}
EXPECTED_LOOPS = {15: 1, 23: 5}
LANDMARK_WINDOWS = {
    15: {"upper_hook": (1500,70,1900,315), "lower_pole": (950,1230,1350,1440)},
    23: {"upper_paired_islands": (1150,70,1510,340), "upper_right_island": (1500,300,1800,570)},
}
REF = "so12_paper_test_leiden/ev_so12_paper_test_hL_s0_20261017"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def window_mask(shape, bounds):
    x0, y0, x1, y1 = bounds
    result = np.zeros(shape, bool)
    result[y0:y1, x0:x1] = True
    return result


def keep_components(mask, minimum):
    labels, _ = ndi.label(mask)
    counts = np.bincount(labels.ravel())
    valid = counts >= minimum
    valid[0] = False
    return valid[labels]


def tissue_mask(rgb, bounds, *, minimum=1800):
    """Pink/purple tissue silhouette; excludes saturated blue annotation strokes."""
    a = np.asarray(rgb).astype(np.int16)
    r, g, b = a.transpose(2, 0, 1)
    tissue = (r - g > 7) & (b - g > 4) & (a.max(axis=2) - a.min(axis=2) > 18)
    tissue &= ~((b > 180) & (r < 100) & (g < 100))
    tissue &= window_mask(r.shape, bounds)
    tissue = ndi.binary_closing(tissue, iterations=4)
    tissue = keep_components(tissue, minimum)
    return ndi.binary_fill_holes(tissue)


def annotation_mask(rgb, bounds, expected_loops, *, minimum_stroke=1000):
    """Extract only saturated-blue loops inside the tissue selection window.

    A two-pixel closing repairs antialiasing gaps; every retained connected
    stroke must enclose a nontrivial region. No contour is inferred from tissue.
    """
    a = np.asarray(rgb)
    stroke = (a[:, :, 2] > 180) & (a[:, :, 0] < 100) & (a[:, :, 1] < 100)
    stroke &= window_mask(stroke.shape, bounds)
    labels, _ = ndi.label(stroke)
    counts = np.bincount(labels.ravel())
    components = np.flatnonzero(counts >= minimum_stroke)
    components = components[components != 0]
    if len(components) != expected_loops:
        raise ValueError(f"Expected {expected_loops} blue contours; found {len(components)}")
    roi = np.zeros(stroke.shape, bool)
    records = []
    for label in components:
        component = labels == label
        closed = ndi.binary_closing(component, iterations=2)
        interior = ndi.binary_fill_holes(closed)
        enclosed = int((interior & ~closed).sum())
        if enclosed <= 100:
            raise ValueError("A blue contour is open or does not enclose a region")
        roi |= interior
        ys, xs = np.nonzero(component)
        records.append({"stroke_pixels": int(component.sum()), "interior_pixels": enclosed,
                        "bounds_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]})
    return roi, stroke, records


def source_silhouette(xy_um, *, spacing_um=10.0, close_um=40.0):
    """Cell-location tissue envelope, independent of cell features and labels."""
    xy = np.asarray(xy_um, np.float64)
    origin = np.floor(xy.min(0) / spacing_um) * spacing_um - 100.0
    points = np.rint((xy-origin) / spacing_um).astype(int)
    shape = (int(points[:, 1].max())+12, int(points[:, 0].max())+12)
    mask = np.zeros(shape, bool)
    mask[points[:, 1], points[:, 0]] = True
    mask = ndi.binary_closing(mask, iterations=max(1, int(close_um / spacing_um)))
    mask = keep_components(mask, 20)
    return ndi.binary_fill_holes(mask), origin, spacing_um


def similarity_matrix(theta_deg, scale_px_per_um, reflected=False):
    theta = np.deg2rad(theta_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mirror = np.diag([-1.0 if reflected else 1.0, 1.0])
    return float(scale_px_per_um) * rotation @ mirror


def warp_source(mask, origin, spacing, matrix, translation, output_shape, *, stride=1):
    """Inverse raster sampling: source xy in µm maps to destination xy pixels."""
    yy, xx = np.indices(output_shape, dtype=np.float64)
    pixels = np.stack((xx.ravel()*stride, yy.ravel()*stride))
    coordinates = np.linalg.solve(matrix, pixels-np.asarray(translation)[:, None])
    coordinates = (coordinates - np.asarray(origin)[:, None])/spacing
    values = ndi.map_coordinates(mask.astype(np.float32), coordinates[::-1], order=1,
                                 mode="constant", cval=0.0)
    return values.reshape(output_shape)


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float(2 * np.count_nonzero(a & b) / max(1, a.sum()+b.sum()))


def fit_geometry(source, origin, spacing, target, nominal_scale, *, stride=6):
    """All rotations and both reflection parities, then bounded similarity fit."""
    sampled = target[::stride, ::stride]
    yx = np.argwhere(source)
    source_center = yx[:, ::-1].mean(0)*spacing + origin
    target_center = np.argwhere(sampled)[:, ::-1].mean(0)*stride
    grid_y, grid_x = np.indices(sampled.shape, dtype=np.float64)
    pixels = np.stack((grid_x.ravel()*stride, grid_y.ravel()*stride))
    floating = source.astype(np.float32)

    def evaluate(parameters, reflected, full=False):
        angle, relative_scale, dx, dy = parameters
        matrix = similarity_matrix(angle, nominal_scale*relative_scale, reflected)
        translation = target_center + np.array([dx, dy])-matrix@source_center
        coordinates = np.linalg.solve(matrix, pixels-translation[:, None])
        coordinates = (coordinates-origin[:, None])/spacing
        warped = ndi.map_coordinates(floating, coordinates[::-1], order=1, mode="constant", cval=0).reshape(sampled.shape)
        score = float(2*(warped*sampled).sum() / max(1, warped.sum()+sampled.sum()))
        return (score, matrix, translation) if full else -score

    candidates = []
    for reflected in (False, True):
        starts = [(evaluate((angle, 1, 0, 0), reflected), angle) for angle in range(-180, 180, 15)]
        # Retain separated local starts, not four nearly identical angles.
        chosen = []
        for _, angle in sorted(starts):
            if all(abs((angle-previous+180) % 360-180) >= 50 for previous in chosen):
                chosen.append(angle)
            if len(chosen) == 3:
                break
        for angle in chosen:
            bounds = [(angle-35, angle+35), (.80, 1.20), (-150, 150), (-150, 150)]
            result = minimize(evaluate, [angle, 1., 0., 0.], args=(reflected,), method="Powell",
                              bounds=bounds, options={"maxiter": 35, "ftol": 2e-5, "xtol": 2e-3})
            score, matrix, translation = evaluate(result.x, reflected, True)
            candidates.append({"silhouette_dice_soft": score, "angle_degrees": float(result.x[0]),
                "relative_scale": float(result.x[1]), "reflected": reflected,
                "matrix_um_to_pixel": matrix.tolist(), "translation_pixel": translation.tolist(),
                "optimizer_success": bool(result.success)})
    return sorted(candidates, key=lambda item: -item["silhouette_dice_soft"])


def boundary_audit(warped, target, um_per_pixel):
    left = warped & ~ndi.binary_erosion(warped)
    right = target & ~ndi.binary_erosion(target)
    distance = np.concatenate((ndi.distance_transform_edt(~right)[left], ndi.distance_transform_edt(~left)[right]))
    return {"silhouette_dice": dice(warped, target), "symmetric_boundary_distance_um":
        {"median": float(np.median(distance)*um_per_pixel), "p90": float(np.quantile(distance,.9)*um_per_pixel),
         "p95": float(np.quantile(distance,.95)*um_per_pixel)}}


def regional_landmark_audit(warped, target, core):
    """Post-fit geometric sanity checks, not held-out landmark validation.

    Windows isolate visually distinctive anatomical subregions without looking
    at cluster assignments. Regional silhouette centroids are reproducible but
    are not point landmarks or a bound on every individual cell's displacement.
    """
    records=[]
    for name,bounds in LANDMARK_WINDOWS[core].items():
        x0,y0,x1,y1=bounds
        centers=[]
        for mask in (target,warped):
            points=np.argwhere(mask[y0:y1,x0:x1])[:,::-1]
            if not len(points):
                raise ValueError("An anatomical check window has no tissue")
            centers.append(points.mean(0)+np.array([x0,y0]))
        records.append({"name":name,"window_xyxy":list(bounds),
            "histology_centroid_xy_pixel":centers[0].tolist(),"transformed_cell_envelope_centroid_xy_pixel":centers[1].tolist(),
            "regional_centroid_residual_um":float(np.linalg.norm(centers[1]-centers[0])*500/BAR_PIXELS[core])})
    return {"interpretation":"Post-fit anatomy correspondence audit; not independent landmark validation or a pixel-error bound.","regions":records}


def rasterize_owners(vertices, global_rows, matrix, translation, shape):
    """Pixel-center polygon rasterization; smallest canonical row wins overlaps."""
    from matplotlib.path import Path as MplPath
    owner = np.full(shape, -1, np.int32)
    overlap = 0
    for row, points in zip(global_rows, vertices, strict=True):
        pixel_xy = np.asarray(points)@np.asarray(matrix).T + np.asarray(translation)
        lo = np.maximum(np.floor(pixel_xy.min(0)-.5).astype(int), 0)
        hi = np.minimum(np.ceil(pixel_xy.max(0)+.5).astype(int), [shape[1], shape[0]])
        if np.any(hi <= lo):
            continue
        yy, xx = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
        inside = MplPath(pixel_xy).contains_points(np.column_stack((xx.ravel()+.5, yy.ravel()+.5))).reshape(xx.shape)
        view = owner[lo[1]:hi[1], lo[0]:hi[0]]
        overlap += int(np.count_nonzero(inside & (view >= 0)))
        view[inside & ((view < 0) | (row < view))] = row
    return owner, overlap


def load_cells(paths):
    reference = paths.report_root / "analyses" / REF
    manifest = json.loads((reference/"extraction_manifest.json").read_text())
    path = reference/"cells.csv.gz"
    if sha(path) != manifest["cells_sha256"]:
        raise ValueError("Canonical test-cell checksum changed")
    cells = pd.read_csv(path, float_precision="round_trip")
    if not np.array_equal(cells.row_index, np.arange(sum(COUNTS.values()))):
        raise ValueError("Canonical cell row order changed")
    if cells.groupby("core_number").size().to_dict() != COUNTS:
        raise ValueError("Unexpected test-core coverage")
    return cells, {"path": str(path), "sha256": sha(path)}


def finalize_review(output, review, paths=None):
    """After anatomical geometry review, generate one shared polygon-owner grid.

    The review must explicitly describe independent anatomical correspondence;
    its exact text is retained. Neither masks nor transforms are refitted here.
    """
    from spatial_benchmark.attention_niche_geometry import load_aligned_cosmx_polygons
    from spatial_benchmark.data import resolve_nested_raw_path
    paths = paths or current_paths()
    output = Path(output).resolve()
    if paths.scratch_root.resolve() not in output.parents or output.name != "registration":
        raise ValueError("Registration writes only active scratch")
    if not isinstance(review, dict) or set(review) != {"15", "23"}:
        raise ValueError("Explicit geometry review is required for both cores")
    for item in review.values():
        if item.get("anatomical_correspondence_confirmed") is not True or not item.get("evidence"):
            raise ValueError("Cannot score without reviewed anatomical correspondence")
    manifest = json.loads((output/"registration_manifest.json").read_text())
    if manifest["status"] != "requires_geometry_review":
        raise ValueError("Registration was already finalized or has an unexpected state")
    cells, cell_record = load_cells(paths)
    if manifest["canonical_cells"] != cell_record:
        raise ValueError("Cell identity changed between fitting and review")
    source = resolve_nested_raw_path(paths.data_root / "raw", "26040302SO_2-polygons.csv")
    upstream = json.loads((paths.report_root/"analyses"/REF/"report/figure_manifest.json").read_text())
    polygon_sha = sha(source)
    if polygon_sha != upstream["raw_polygons"]["sha256"]:
        raise ValueError("Raw polygon source differs from published test maps")
    alignment = load_aligned_cosmx_polygons({"SO_2":source},cells[["slide","fov","cell_ID"]],
        prepared_coordinates_um=cells[["x_um","y_um"]].to_numpy(),centroid_tolerance_um=5.)
    keys = tuple(("SO_2",int(r.fov),int(r.cell_ID)) for r in cells.itertuples())
    if alignment.keys != keys or not alignment.centroid_alignment.aligned:
        raise ValueError("Polygon canonical key/coordinate alignment failed")
    for record in manifest["cores"]:
        core=record["core_number"]
        if sha(record["image"]["path"]) != record["image"]["sha256"]:
            raise ValueError("Clinical image changed")
        with np.load(output/f"core_{core}_geometry.npz",allow_pickle=False) as arrays:
            matrix,translation=arrays["matrix"],arrays["translation"]
            roi,stroke=arrays["roi"],arrays["stroke"]
        if not np.array_equal(matrix,np.asarray(record["best"]["matrix_um_to_pixel"])) or not np.array_equal(translation,np.asarray(record["best"]["translation_pixel"])):
            raise ValueError("Registration transform changed after audit")
        rows=np.flatnonzero(cells.core_number.to_numpy()==core)
        polygons=[alignment.polygons_um[i] for i in rows]
        if any(p.geom_type!="Polygon" or len(p.interiors) for p in polygons):
            raise ValueError("Unsupported cell polygon topology")
        vertices=[np.asarray(p.exterior.coords) for p in polygons]
        owner,overlap=rasterize_owners(vertices,rows,matrix,translation,roi.shape)
        valid=window_mask(roi.shape,WINDOWS[core]) & ~stroke
        radius=50*BAR_PIXELS[core]/500.
        inside_distance=ndi.distance_transform_edt(roi)
        outside_distance=ndi.distance_transform_edt(~roi)
        valid50=valid & (np.where(roi,inside_distance,outside_distance)>radius)
        eroded=inside_distance>radius
        dilated=outside_distance<=radius
        path=output/f"core_{core}_raster.npz"
        np.savez_compressed(path,owner=owner,roi=roi,valid=valid,valid50=valid50,
                            roi_eroded50=eroded,roi_dilated50=dilated)
        represented=np.unique(owner[(owner>=0)&valid])
        record.update({"passed":True,"status":"verified_approximate_section",
            "geometry_review":review[str(core)],"raster_path":str(path.relative_to(output.parent)),
            "raster_sha256":sha(path),"owner_convention":"canonical_global_row_index; -1 uncovered; smallest row wins overlaps",
            "raster_shape":list(owner.shape),"overlapping_polygon_pixel_events":overlap,
            "owned_pixels":int((owner>=0).sum()),"owned_valid_pixels":int(((owner>=0)&valid).sum()),
            "owned_pixels_outside_valid_window_or_on_stroke":int(((owner>=0)&~valid).sum()),
            "n_cells":len(rows),"represented_cells_with_valid_owned_pixel":len(represented),
            "zero_valid_pixel_cells":len(rows)-len(represented),
            "roi_pixels":int((roi&valid).sum()),"roi_covered_pixels":int((roi&valid&(owner>=0)).sum()),
            "boundary_sensitivity_um":50,
            "primary_valid_support":"core selection window excluding actual annotation strokes; pixel-center cell footprint",
            "registration_caveat":"Approximate section alignment; no claim of pixel-exact same-section histological truth."})
        dump(output/f"core_{core}_registration.json",record)
    manifest.update({"status":"verified_approximate_section","passed":True,
                     "raw_polygons":{"path":str(source),"sha256":polygon_sha},
                     "polygon_alignment":{"n_cells":alignment.n_cells,"selected_vertex_rows":alignment.selected_vertex_rows,
                         "centroid_mismatch_count":len(alignment.centroid_alignment.mismatch_indices),
                         "pixel_size_um":alignment.pixel_size_um},
                     "geometry_reviewed":True})
    dump(output/"registration_manifest.json",manifest)
    return manifest


def annotation_review_figure(output):
    """Expose filled source-image masks for visual checking, with no transform."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output)
    manifest=json.loads((output/"registration_manifest.json").read_text())
    figure,axes=plt.subplots(1,2,figsize=(10,7))
    for axis,record in zip(axes,manifest["cores"],strict=True):
        core=record["core_number"]
        image=np.asarray(Image.open(record["image"]["path"]).convert("RGB"))
        with np.load(output/f"core_{core}_geometry.npz",allow_pickle=False) as arrays:
            roi=arrays["roi"]
        rgba=np.zeros((*roi.shape,4),np.float32)
        rgba[roi]=[1,.45,0,.25]
        axis.imshow(image);axis.imshow(rgba)
        x0,y0,x1,y1=WINDOWS[core]
        axis.set_xlim(x0,x1);axis.set_ylim(y1,y0);axis.axis("off")
        axis.set_title(f"Core {core} · {len(record['annotation']['loops'])} source outlines filled")
    figure.suptitle("Annotation extraction check: orange fill; original blue strokes unchanged",fontsize=12)
    figure.tight_layout(rect=(0,0,1,.94));path=output/"annotation_mask_review.png"
    figure.savefig(path,dpi=180,bbox_inches="tight");plt.close(figure)
    return path


def run(output, paths=None):
    """Write bounded geometry audit into this evaluation's active scratch only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = paths or current_paths()
    output = Path(output).resolve()
    if paths.scratch_root.resolve() not in output.parents or output.name != "registration":
        raise ValueError("Registration writes only an active scratch registration directory")
    output.mkdir(parents=True, exist_ok=True)
    cells, cell_record = load_cells(paths)
    records = []
    for core in CORES:
        image_path = paths.data_root / "clinical" / f"Core_{core}.png"
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        target = tissue_mask(rgb, WINDOWS[core])
        selected = cells.loc[cells.core_number.eq(core)]
        source, origin, spacing = source_silhouette(selected[["x_um", "y_um"]].to_numpy())
        candidates = fit_geometry(source, origin, spacing, target, BAR_PIXELS[core]/500.0)
        best = candidates[0]
        matrix, translation = np.asarray(best["matrix_um_to_pixel"]), np.asarray(best["translation_pixel"])
        warped = warp_source(source, origin, spacing, matrix, translation, target.shape) >= .5
        audit = boundary_audit(warped, target, 500/BAR_PIXELS[core])
        audit["regional_landmarks"] = regional_landmark_audit(warped,target,core)
        roi, stroke, loops = annotation_mask(rgb, WINDOWS[core], EXPECTED_LOOPS[core])
        other_parity = max(c["silhouette_dice_soft"] for c in candidates if c["reflected"] != best["reflected"])
        # Automated fit alone never certifies anatomical correspondence. The
        # explicit review receipt is written separately after viewing overlays.
        record = {"core_number": core, "status": "requires_geometry_review", "image": {"path": str(image_path),
            "sha256": sha(image_path), "shape": list(rgb.shape), "scale_bar_um": 500,
            "scale_bar_pixel_length": BAR_PIXELS[core], "scale_endpoint_uncertainty_px": 1},
            "window_xyxy": list(WINDOWS[core]), "best": best, "candidates": candidates,
            "opposite_reflection_dice_gap": best["silhouette_dice_soft"]-other_parity,
            "geometry_audit": audit, "annotation": {"meaning": "user-designated cancer-region outlines",
            "loops": loops, "roi_pixels": int(roi.sum()), "stroke_pixels": int(stroke.sum()),
            "exhaustiveness_verified": False, "same_histological_section_verified": False},
            "registration_features": "pink-purple tissue silhouette and cell-location envelope only; no cancer or cluster overlap"}
        np.savez_compressed(output/f"core_{core}_geometry.npz", tissue=target, warped=warped,
                            roi=roi, stroke=stroke, matrix=matrix, translation=translation)
        figure, axes = plt.subplots(1, 2, figsize=(13, 6))
        axes[0].imshow(rgb)
        axes[0].contour(warped, levels=[.5], colors=["#00a6b2"], linewidths=.65)
        axes[0].set_title(f"Core {core}: cyan = transformed cell-location envelope")
        axes[1].imshow(target, cmap="Greys", vmin=0, vmax=1)
        axes[1].contour(warped, levels=[.5], colors=["#e45c20"], linewidths=.7)
        axes[1].set_title(f"Tissue-only fit: Dice {audit['silhouette_dice']:.3f}")
        x0,y0,x1,y1=WINDOWS[core]
        for axis in axes:
            axis.set_xlim(x0-50,x1+50);axis.set_ylim(y1+20,y0-20);axis.axis("off")
        figure.tight_layout();figure.savefig(output/f"core_{core}_registration_overlay.png",dpi=160);plt.close(figure)
        dump(output/f"core_{core}_registration.json",record)
        records.append(record)
    result={"schema":"so12_cancer_geometry_registration_v1","status":"requires_geometry_review",
            "passed":False,"canonical_cells":cell_record,"cores":records,
            "limitations":["Screenshot outlines are reference regions, not exhaustive cancer-free certification.",
             "Section identity and pixel-exact histology-to-CosMx registration are not documented."]}
    dump(output/"registration_manifest.json",result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.output)
    print(json.dumps({"status":result["status"],"core_dice":{str(r["core_number"]):r["geometry_audit"]["silhouette_dice"] for r in result["cores"]}}))
