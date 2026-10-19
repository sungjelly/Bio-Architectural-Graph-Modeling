"""Fixed-resolution comparison panels using the shared reviewed owner raster."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

if __package__:
    from .so12_cancer_registration import sha, dump, CORES, COUNTS, WINDOWS, BAR_PIXELS
else:
    from so12_cancer_registration import sha, dump, CORES, COUNTS, WINDOWS, BAR_PIXELS
from spatial_benchmark.paths import current_paths

METHODS=("h0","h1","h2","h3","h4","pca","scvi")
PRIMARY=("gt","h4","h0","pca","scvi")
TITLES={"gt":"Annotated histology","h0":"BAGM h0 · encoder","h1":"BAGM h1 · block 1",
        "h2":"BAGM h2 · block 2","h3":"BAGM h3 · block 3","h4":"BAGM h4 · block 4",
        "pca":"PCA · 50 dimensions","scvi":"scVI · 50 dimensions"}


def load_partitions(output):
    output=Path(output)
    manifest=json.loads((output/"clustering_summary.json").read_text())
    selected={}
    for record in manifest["partitions"]:
        if record["resolution"] != 1 or record["seed"] != 0:
            continue
        method=record["method"]
        if method not in METHODS or method in selected:
            raise ValueError("Unexpected or duplicated primary partition")
        item=record["labels"]
        path=Path(item["path"])
        if item.get("relative") is not True or path.is_absolute() or ".." in path.parts:
            raise ValueError("Labels must be output-relative")
        if sha(output/path)!=item["sha256"]:
            raise ValueError("Primary partition checksum changed")
        labels=np.load(output/path,allow_pickle=False)
        if labels.shape!=(sum(COUNTS.values()),) or labels.dtype.kind not in "iu" or labels.min()<0:
            raise ValueError("Labels must cover every canonical test cell")
        selected[method]=(labels,item)
    if set(selected)!=set(METHODS):
        raise ValueError("Seven primary method partitions are required")
    return selected


def palette_for(labels):
    count=int(labels.max())+1
    rgba=plt.get_cmap("turbo")(np.linspace(.04,.96,max(count,2)))[:count]
    return np.rint(rgba[:,:3]*255).astype(np.uint8)


def color_raster(owner, labels, palette):
    if owner.dtype.kind != "i" or owner.min() < -1 or owner.max() >= len(labels):
        raise ValueError("Owner is not a canonical global-row raster")
    rgb=np.full((*owner.shape,3),255,np.uint8)
    occupied=owner>=0
    rgb[occupied]=palette[labels[owner[occupied]]]
    return rgb


def _panel(axis, rgb, roi, core, title, *, annotation=True):
    x0,y0,x1,y1=WINDOWS[core]
    axis.imshow(rgb,interpolation="nearest")
    if annotation:
        # Exactly the same unshifted source-image ROI boundary in every panel.
        axis.contour(roi,levels=[.5],colors=["#1010dd"],linewidths=.5)
    axis.set_xlim(x0,x1);axis.set_ylim(y1,y0);axis.set_aspect("equal");axis.axis("off")
    axis.set_title(title,fontsize=11,pad=8)
    length=BAR_PIXELS[core] # 500 µm in the original screenshot calibration.
    x,y=x0+35,y1-35
    axis.plot([x,x+length],[y,y],color="#111827",linewidth=2.2)
    axis.text(x+length/2,y-12,"500 µm",ha="center",va="bottom",fontsize=8,
              bbox={"facecolor":"white","edgecolor":"none","alpha":.8,"pad":1.})


def _save(figure,directory,basename):
    records={}
    for extension in ("png","pdf"):
        path=directory/f"{basename}.{extension}"
        figure.savefig(path,dpi=220,bbox_inches="tight",facecolor="white")
        records[extension]={"path":str(path.name),"sha256":sha(path),"size_bytes":path.stat().st_size}
    plt.close(figure)
    return records


def render(output,paths=None):
    paths=paths or current_paths()
    output=Path(output).resolve()
    if not output.is_relative_to(paths.scratch_root/"active_runs"):
        raise ValueError("Figures may only write to the active evaluation")
    registration_path=output/"registration/registration_manifest.json"
    registration=json.loads(registration_path.read_text())
    if not registration.get("passed") or registration.get("status")!="verified_approximate_section":
        raise ValueError("Comparison requires reviewed approximate registration")
    if sha(output/"cells.csv.gz")!=registration["canonical_cells"]["sha256"]:
        raise ValueError("Figure cells differ from registration canonical cells")
    selected=load_partitions(output)
    palettes={method:palette_for(values[0]) for method,values in selected.items()}
    figures=output/"figures";figures.mkdir(parents=True,exist_ok=True)
    colors=[]
    for method,(labels,_) in selected.items():
        for cluster,color in enumerate(palettes[method]):
            colors.append({"method":method,"cluster_id":cluster,"color_hex":to_hex(color/255),
                           "cells":int(np.count_nonzero(labels==cluster))})
    pd.DataFrame(colors).to_csv(figures/"cluster_color_keys.csv",index=False)
    records={};images={};rois={}
    for core_record in registration["cores"]:
        core=core_record["core_number"]
        raster=output/core_record["raster_path"]
        if sha(raster)!=core_record["raster_sha256"]:
            raise ValueError("Registration owner raster changed")
        with np.load(raster,allow_pickle=False) as arrays:
            owner,roi=arrays["owner"],arrays["roi"]
        rois[core]=roi
        image_path=Path(core_record["image"]["path"])
        if sha(image_path)!=core_record["image"]["sha256"]:
            raise ValueError("Clinical image changed")
        images[core]={"gt":np.asarray(Image.open(image_path).convert("RGB"))}
        for method,(labels,_) in selected.items():
            images[core][method]=color_raster(owner,labels,palettes[method])
        figure,axes=plt.subplots(1,5,figsize=(18,6.5))
        for axis,method in zip(axes,PRIMARY,strict=True):
            _panel(axis,images[core][method],roi,core,TITLES[method],annotation=method!="gt")
        figure.suptitle(f"Core {core} · {COUNTS[core]:,} cells · Leiden resolution 1, seed 0",fontsize=15,y=.98)
        figure.text(.5,.015,"Blue: supplied region outlines · Approximate section registration · Independent cluster colors across methods",
                    ha="center",fontsize=9)
        figure.subplots_adjust(wspace=.035,top=.88,bottom=.08)
        records[f"core_{core}_five_panel"]=_save(figure,figures,f"core_{core}_five_panel")
    figure,axes=plt.subplots(2,5,figsize=(20,12.8))
    for row,core in enumerate(CORES):
        for column,method in enumerate(PRIMARY):
            _panel(axes[row,column],images[core][method],rois[core],core,
                   f"Core {core} · {TITLES[method]}",annotation=method!="gt")
    figure.text(.5,.012,"Fixed resolution 1, seed 0 · Blue: supplied region outlines · Approximate section alignment · Colors/IDs are method-specific",
                ha="center",fontsize=10)
    figure.subplots_adjust(wspace=.035,hspace=.12,bottom=.05,top=.96)
    records["combined_five_panel"]=_save(figure,figures,"combined_five_panel")
    figure,axes=plt.subplots(2,7,figsize=(26,12.5))
    for row,core in enumerate(CORES):
        for column,method in enumerate(METHODS):
            _panel(axes[row,column],images[core][method],rois[core],core,f"Core {core} · {TITLES[method]}")
    figure.text(.5,.012,"All representations · Fixed resolution 1, seed 0 · Blue: supplied region outlines · Independent cluster colors across methods",
                ha="center",fontsize=10)
    figure.subplots_adjust(wspace=.035,hspace=.12,bottom=.05,top=.96)
    records["all_representations_2x7"]=_save(figure,figures,"all_representations_2x7")
    result={"schema":"so12_cancer_comparison_figures_v1","status":"complete","passed":True,
        "cores":list(CORES),"n_cells":sum(COUNTS.values()),"primary_methods":list(PRIMARY),"all_methods":list(METHODS),
        "resolution":1.,"seed":0,"shared_owner_raster":True,"annotations_shifted":False,
        "scale_bar_um":500,"registration_status":registration["status"],
        "registration_manifest_sha256":sha(registration_path),
        "clustering_manifest_sha256":sha(output/"clustering_summary.json"),
        "partitions":{method:item for method,(_,item) in selected.items()},"figures":records,
        "color_keys":{"path":"figures/cluster_color_keys.csv","sha256":sha(figures/"cluster_color_keys.csv")},
        "interpretation":"Exploratory annotation agreement; outlines are user-designated reference regions, not exhaustive cancer labels. Cell polygons are rasterized in histology coordinates under one tissue-only similarity transform per core."}
    dump(figures/"figure_manifest.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();result=render(args.output)
    print(json.dumps({"status":result["status"],"figures":list(result["figures"])}))
