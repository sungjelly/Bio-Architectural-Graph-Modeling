#!/usr/bin/env python3
"""Build the portable Jacobian biological-validation audit report.

The generator reads aggregate, provenance-checked project artifacts only. It
does not load row-level expression, cell metadata, graphs, or checkpoints.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[2]

COMPARISON_PATH = (
    ROOT
    / "reports/analyses/full_core_g2_count_tokens_multiseed"
    / "comparison/comparison.json"
)
ACTIVE_DIR = (
    ROOT
    / "scratch/active_runs"
    / "posthoc_g2_token_relaxed_categorical_sensitivity_v1"
)
MANIFEST_PATH = ACTIVE_DIR / "analysis_manifest.json"
PROTOCOL_PATH = ACTIVE_DIR / "protocol.json"
PILOT_PATH = ACTIVE_DIR / "resource_pilot.json"
IDENTICAL_REVIEW_PATH = ACTIVE_DIR / "identical_control_review.json"
TLS_PATH = (
    ROOT
    / "reports/analyses/full_core_high_k_capacity"
    / "interpretability/analysis.json"
)
SOURCE_PATH = ROOT / "src/spatial_benchmark/categorical_sensitivity.py"
WORKFLOW_PATH = (
    ROOT / "scripts/analysis/run_g2_token_categorical_sensitivity.py"
)
SOURCES_PATH = REPORT_DIR / "sources.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"{context} is missing {key!r}")
    return mapping[key]


def format_int(value: int | float) -> str:
    return f"{int(value):,}"


def format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}"


def scientific(value: int | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}e}"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def get_path(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return current


def stage_snapshot() -> dict[str, Any]:
    shard_root = ACTIVE_DIR / "shards"
    shard_files = sorted(
        path
        for path in shard_root.rglob("*.json")
        if not path.name.endswith(".sha256.json")
    )
    names = [str(path.relative_to(shard_root)) for path in shard_files]
    counts = {
        "identical": sum("identical" in name for name in names),
        "trained": sum("trained" in name for name in names),
        "random": sum("random" in name for name in names),
    }
    review = IDENTICAL_REVIEW_PATH
    final_dir = (
        ROOT
        / "reports/analyses/full_core_g2_count_tokens_multiseed"
        / "posthoc_relaxed_categorical_sensitivity_v1"
    )
    final_jsons = sorted(final_dir.glob("*.json")) if final_dir.exists() else []
    if final_jsons:
        label = "finalized sensitivity output present"
    elif review.exists():
        label = "identical-control review present; later shards may be pending"
    elif counts["identical"]:
        label = "identical-control shard artifacts present"
    elif PILOT_PATH.exists():
        label = "resource pilot present; identical-control stage pending/running"
    else:
        label = "not initialized"
    return {
        "label": label,
        "shard_file_counts": counts,
        "shard_files": names,
        "identical_control_review_present": review.exists(),
        "final_output_json_files": [path.name for path in final_jsons],
        "note": (
            "This is a filesystem snapshot, not a process monitor. The stage "
            "can advance after report generation."
        ),
    }


def build_evidence() -> dict[str, Any]:
    comparison = load_json(COMPARISON_PATH)
    manifest = load_json(MANIFEST_PATH)
    protocol = load_json(PROTOCOL_PATH)
    pilot = load_json(PILOT_PATH)
    identical_review = (
        load_json(IDENTICAL_REVIEW_PATH)
        if IDENTICAL_REVIEW_PATH.is_file()
        else None
    )
    tls = load_json(TLS_PATH)

    if comparison.get("status") != "complete":
        raise ValueError("token comparison is not complete")
    if manifest.get("analysis_mode") != (
        "post_hoc_exploratory_after_prespecified_gate_failure"
    ):
        raise ValueError("unexpected sensitivity analysis mode")
    if protocol.get("execution", {}).get("full_jacobian_materialized") is not False:
        raise ValueError("protocol no longer says that the Jacobian is absent")
    if set(protocol.get("sufficient_statistics", {})) != {
        "A2",
        "AB",
        "B2",
        "aggregation",
    }:
        raise ValueError("unexpected sensitivity sufficient-statistic schema")
    if not manifest.get("common_identity"):
        raise ValueError("sensitivity manifest lacks common identity")

    source_text = SOURCE_PATH.read_text(encoding="utf-8")
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    required_source_phrases = [
        "without a full categorical VJP",
        "reference_squared_norm",
        "candidate_squared_norm",
        "cross_inner_product",
    ]
    missing = [
        phrase for phrase in required_source_phrases if phrase not in source_text
    ]
    if missing:
        raise ValueError(f"categorical-sensitivity source contract changed: {missing}")
    if '"statistics": {' not in workflow_text:
        raise ValueError("workflow shard payload contract changed")

    identity = manifest["common_identity"]
    n_nodes = int(identity["n_nodes"])
    n_genes = int(identity["n_genes"])
    masks = identity["whole_node_masks"]
    if len(masks) != 3:
        raise ValueError("expected exactly three whole-node masks")
    target_counts = {
        int(row["n_masked"]) // n_genes for row in masks
    }
    if len(target_counts) != 1:
        raise ValueError("whole-node masks have inconsistent target counts")
    n_targets = target_counts.pop()
    output_coordinates = n_targets * n_genes * 4
    effective_input_coordinates = (n_nodes - n_targets) * n_genes * 4
    dense_entries = output_coordinates * effective_input_coordinates
    dense_fp32_bytes = dense_entries * 4

    aggregates = comparison["variant_aggregates"]
    variant_rows: list[dict[str, Any]] = []
    for label, row in aggregates.items():
        metrics = row["metrics"]
        baselines = row["baselines"]
        recalls = row["token_recalls_percent"]
        exact = float(metrics["exact_percent"]["mean"])
        balanced = float(metrics["balanced_percent"]["mean"])
        nonzero = float(metrics["nonzero_percent"]["mean"])
        modal_exact = float(
            baselines["baseline_per_gene_modal_accuracy_percent"]["mean"]
        )
        modal_balanced = float(
            baselines[
                "baseline_per_gene_modal_balanced_accuracy_percent"
            ]["mean"]
        )
        modal_nonzero = float(
            baselines[
                "baseline_per_gene_modal_nonzero_accuracy_percent"
            ]["mean"]
        )
        variant_rows.append(
            {
                "label": label,
                "parameter_count": int(row["parameter_count"]),
                "seed_count": int(metrics["exact_percent"]["n"]),
                "exact_percent": exact,
                "balanced_percent": balanced,
                "nonzero_percent": nonzero,
                "cross_entropy": float(metrics["cross_entropy"]["mean"]),
                "modal_exact_percent": modal_exact,
                "modal_balanced_percent": modal_balanced,
                "modal_nonzero_percent": modal_nonzero,
                "exact_minus_modal_pp": exact - modal_exact,
                "balanced_minus_modal_pp": balanced - modal_balanced,
                "nonzero_minus_modal_pp": nonzero - modal_nonzero,
                "token_recall_percent": {
                    token: float(value["mean"])
                    for token, value in recalls.items()
                },
            }
        )
    variant_rows.sort(key=lambda row: row["parameter_count"])

    pilot_resource = pilot["resource"]
    tls_interpretation = tls["interpretation"]
    deletion = tls["deletion_analysis"]["receiver_program_effect"]
    sender = tls["sender_program_perturbation"]
    sender_effect = sender["receiver_program_effect"]
    tls_evidence = {
        "scope": {
            "same_models_as_token_jacobian": False,
            "run_id": tls["run_id"],
            "model_family_relation": (
                "related continuous-expression full-core G2; not one of the "
                "six tokenized checkpoints"
            ),
            "outcome_conditioned_exploration": bool(
                tls["scope"]["outcome_conditioned_exploration"]
            ),
            "experimental_units": int(tls["scope"]["experimental_units"]),
        },
        "organizer_genes": list(tls["gene_panels"]["organizer"]),
        "receiver_program_genes": list(
            tls["gene_panels"]["receiver_program"]
        ),
        "organizer_score_enrichment_mean": float(
            sender["organizer_score_enrichment"][
                "top_minus_null_score_draw_means"
            ]["mean"]
        ),
        "organizer_score_enrichment_minimum": float(
            sender["organizer_score_enrichment"][
                "minimum_top_minus_null_score_across_draws"
            ]
        ),
        "deletion_top_minus_null_huber_change_mean": float(
            deletion["paired_targeted_minus_null"][
                "program_huber_change_draw_means"
            ]["mean"]
        ),
        "deletion_top_minus_null_prediction_mae_mean": float(
            deletion["paired_targeted_minus_null"][
                "program_prediction_mae_draw_means"
            ]["mean"]
        ),
        "sender_ablation_top_minus_null_huber_change_mean": float(
            sender_effect["paired_targeted_minus_null"][
                "program_huber_change_draw_means"
            ]["mean"]
        ),
        "sender_ablation_top_minus_null_prediction_mae_mean": float(
            sender_effect["paired_targeted_minus_null"][
                "program_prediction_mae_draw_means"
            ]["mean"]
        ),
        "deletion_support": bool(
            tls_interpretation["descriptive_deletion_support"]
        ),
        "sender_program_support": bool(
            tls_interpretation["descriptive_sender_program_support"]
        ),
        "organizer_enrichment_support": bool(
            tls_interpretation[
                "descriptive_organizer_enrichment_support"
            ]
        ),
        "joint_tls_dependency_support": bool(
            tls_interpretation["joint_tls_dependency_support"]
        ),
        "evidence_label": str(tls_interpretation["evidence_label"]),
    }

    gate = comparison["relaxed_jacobian_gate"]
    evidence = {
        "schema_version": 1,
        "artifact_kind": "jacobian_biological_validation_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_classification": (
            "post_hoc_exploratory_artifact_and_claim_audit"
        ),
        "verdict": {
            "entrywise_biological_test": "not_testable_from_current_artifacts",
            "biological_correspondence": "not_established",
            "claim_rejected": (
                "The current evidence does not justify saying that high "
                "Jacobian entries correspond to verified biology."
            ),
            "claim_not_made": (
                "This audit does not establish that no biological "
                "correspondence exists; the required entrywise statistic was "
                "not produced."
            ),
            "reason_codes": [
                "no_materialized_jacobian",
                "no_gene_or_gene_pair_ranking",
                "trace_statistics_are_not_entries",
                "predictive_model_dominated_by_zero_class",
                "single_transductive_core",
                "direct_and_multihop_paths_not_separated",
                "related_tls_positive_control_failed_joint_faithfulness",
            ],
        },
        "jacobian_contract": {
            "estimand": protocol["estimand"],
            "input_coordinates": protocol["input"],
            "output_coordinates": protocol["output"],
            "mask_scope": protocol["mask_scope"],
            "execution": protocol["execution"],
            "retained_sufficient_statistics": protocol[
                "sufficient_statistics"
            ],
            "retained_metrics": protocol["metrics"],
            "maximum_claim": protocol["maximum_claim"],
            "entrywise_axes_retained": False,
            "gene_ranks_retained": False,
            "gene_pair_ranks_retained": False,
            "cell_or_edge_ranks_retained": False,
            "sign_retained_for_biological_pairs": False,
        },
        "dense_matrix_scale": {
            "nodes": n_nodes,
            "genes": n_genes,
            "classes": 4,
            "masked_target_nodes_per_mask": n_targets,
            "observed_source_nodes_per_mask": n_nodes - n_targets,
            "output_coordinates": output_coordinates,
            "effective_input_coordinates": effective_input_coordinates,
            "dense_entries": dense_entries,
            "dense_fp32_bytes": dense_fp32_bytes,
            "dense_fp32_decimal_petabytes": dense_fp32_bytes / 1e15,
            "dense_fp32_pebibytes": dense_fp32_bytes / 2**50,
            "note": (
                "This counts the four represented centered-logit and simplex "
                "coordinates; centering/tangent constraints reduce algebraic "
                "rank but do not create a stored entrywise matrix."
            ),
        },
        "model_fitness": {
            "scope": comparison["scope"],
            "variants": variant_rows,
            "gate": gate,
            "h1_width_gate": comparison["h1_width_gate"],
            "interpretation": (
                "Both trained variants are at or below the per-gene modal "
                "reference on balanced and nonzero accuracy, with zero recall "
                "for tokens 1 and 2. Gradients can still be nonzero, but they "
                "describe a poorly informative held-in classifier."
            ),
        },
        "sensitivity_execution": {
            "analysis_mode": manifest["analysis_mode"],
            "prespecified_gate": manifest["prespecified_gate"],
            "expected_full_graph_vjps": int(
                manifest["execution_contract"]["expected_full_graph_vjps"]
            ),
            "probe_count_per_mask": int(protocol["probes"]["count_per_mask"]),
            "mask_count": int(protocol["mask_scope"]["entry_count"]),
            "bootstrap_replicates": int(protocol["bootstrap"]["replicates"]),
            "pilot": {
                "parameter_count": int(
                    pilot["model_state"]["parameter_count"]
                ),
                "vjp_duration_seconds": float(pilot["vjp_wall_seconds"]),
                "peak_cuda_memory_allocated_bytes": int(
                    pilot["peak_cuda_memory_allocated_bytes"]
                ),
                "tangent_squared_norm": float(
                    pilot["tangent_vjp_squared_norm"]
                ),
                "status": "passed" if pilot["passes"] else "failed",
            },
            "identical_control": (
                {
                    "present": True,
                    "passes": bool(identical_review["passes"]),
                    "shard_count": int(
                        identical_review["identical_shard_count"]
                    ),
                    "probe_count": int(identical_review["probe_count"]),
                    "full_graph_vjp_count": int(
                        identical_review["full_graph_vjp_count"]
                    ),
                    "point": dict(
                        identical_review["estimate"]["point"]
                    ),
                    "intervals": dict(
                        identical_review["estimate"]["intervals"]
                    ),
                    "interval_interpretation": str(
                        identical_review["interval_interpretation"]
                    ),
                }
                if identical_review is not None
                else {
                    "present": False,
                    "passes": None,
                }
            ),
            "stage_snapshot": stage_snapshot(),
        },
        "related_tls_control": tls_evidence,
        "facts": [
            "The prespecified >95% accuracy gate failed for every wider seed.",
            "The subsequent sensitivity run is explicitly post-hoc.",
            "The protocol explicitly sets full_jacobian_materialized to false.",
            "Only A2, B2, and AB trace-probe sufficient statistics are retained.",
            "Those scalars support whole-Jacobian similarity metrics, not entry ranks.",
            (
                "The identical-checkpoint numerical control passed exactly."
                if identical_review is not None
                and identical_review["passes"]
                else "The identical-checkpoint numerical control is pending."
            ),
            "The related TLS control found organizer-marker enrichment but no joint deletion and sender-program support.",
        ],
        "inferences": [
            (
                "A biological overlap analysis cannot be computed without "
                "inventing entry values that are not identifiable from A2/B2/AB."
            ),
            (
                "Because balanced and nonzero accuracy do not beat the modal "
                "reference, any sensitivity ranking would have low biological "
                "credibility even if it were extracted."
            ),
            (
                "The prior TLS result is a counterexample to equating plausible "
                "marker enrichment with faithful model dependence."
            ),
        ],
        "uncertainties": [
            (
                "The global sensitivity run can still establish whether two "
                "trained functions are similar once finalized, but not which "
                "biological relationship explains that similarity."
            ),
            (
                "No conclusion can be drawn about whether a newly defined, "
                "properly controlled program-block Jacobian would show enrichment."
            ),
            (
                "One core provides no patient-level prevalence or replication."
            ),
        ],
        "recommended_next_experiment": {
            "decision": (
                "Do not mine the current trace statistics for biological "
                "pairs. First establish graph-specific predictive value, then "
                "run a separately locked program-block sensitivity study."
            ),
            "ordered_steps": [
                {
                    "step": 1,
                    "name": "repair the predictive prerequisite",
                    "action": (
                        "Use class-aware loss or a count likelihood and compare "
                        "against per-gene modal, same-cell, morphology/context, "
                        "and spatial-smoothing baselines. Require reproducible "
                        "gain on nonzero and balanced metrics."
                    ),
                },
                {
                    "step": 2,
                    "name": "separate the estimand",
                    "action": (
                        "Decompose same-cell, direct one-hop cross-cell, and "
                        "multihop/regional sensitivity. Use contact/paracrine "
                        "graphs rather than treating the k=1000 regional graph "
                        "as direct communication."
                    ),
                },
                {
                    "step": 3,
                    "name": "lock program blocks before looking",
                    "action": (
                        "Define sender and receiver programs, distance bins, "
                        "cell-type pairs, score normalization, top-set cutoffs, "
                        "and the multiple-testing family before extraction. "
                        "Prefer program-level blocks to one-gene perturbations."
                    ),
                },
                {
                    "step": 4,
                    "name": "calibrate attribution",
                    "action": (
                        "Store an axis-labelled program-by-program statistic "
                        "with sign/effect size. Require seed and mask stability, "
                        "identical-model recovery, parameter randomization, "
                        "label/target permutation, and planted-signal recovery."
                    ),
                },
                {
                    "step": 5,
                    "name": "test biological annotation locally",
                    "action": (
                        "Compare locked candidates with downloaded OmniPath and "
                        "CellPhoneDB reference sets using prevalence-, distance-, "
                        "degree-, compartment-, and cell-type-matched nulls. "
                        "Treat overlap as annotation, not verification."
                    ),
                },
                {
                    "step": 6,
                    "name": "test faithfulness and replication",
                    "action": (
                        "Run bounded program deletion/insertion and graph "
                        "interventions, then reproduce the locked candidates in "
                        "independent patients/cores. Seek orthogonal protein or "
                        "imaging support before a mechanism claim."
                    ),
                },
            ],
            "minimum_acceptance": [
                "graph-specific predictive gain over non-graph/context baselines",
                "stable program-block ranks across all three seeds and masks",
                "clear separation from randomized and mechanism-breaking nulls",
                "matched-null enrichment with multiplicity correction",
                "faithful prediction change under a bounded intervention",
                "replication in independent biological units",
            ],
        },
        "privacy": {
            "aggregate_only": True,
            "row_level_data_loaded": False,
            "protected_identifiers_emitted": False,
            "external_upload_of_project_data": False,
        },
        "input_checksums": {
            rel(path): sha256(path)
            for path in [
                COMPARISON_PATH,
                MANIFEST_PATH,
                PROTOCOL_PATH,
                PILOT_PATH,
                *(
                    [IDENTICAL_REVIEW_PATH]
                    if IDENTICAL_REVIEW_PATH.is_file()
                    else []
                ),
                TLS_PATH,
                SOURCE_PATH,
                WORKFLOW_PATH,
                SOURCES_PATH,
            ]
        },
    }
    return evidence


def citation(source_id: str, number: int) -> str:
    return (
        f'<a class="cite" href="#ref-{esc(source_id)}" '
        f'aria-label="Reference {number}">[{number}]</a>'
    )


def performance_table(variants: list[Mapping[str, Any]]) -> str:
    rows = []
    for row in variants:
        width = (
            "Current (512)"
            if int(row["parameter_count"]) < 10_000_000
            else "Wider (1,024)"
        )
        rows.append(
            "<tr>"
            f"<th scope='row'>{esc(width)}</th>"
            f"<td>{format_int(row['parameter_count'])}</td>"
            f"<td>{format_float(row['exact_percent'], 4)}%</td>"
            f"<td>{format_float(row['modal_exact_percent'], 4)}%</td>"
            f"<td class='delta {'pos' if row['exact_minus_modal_pp'] > 0 else 'neg'}'>"
            f"{format_float(row['exact_minus_modal_pp'], 4)} pp</td>"
            f"<td>{format_float(row['balanced_percent'], 4)}%</td>"
            f"<td class='delta neg'>{format_float(row['balanced_minus_modal_pp'], 4)} pp</td>"
            f"<td>{format_float(row['nonzero_percent'], 4)}%</td>"
            f"<td class='delta neg'>{format_float(row['nonzero_minus_modal_pp'], 4)} pp</td>"
            "</tr>"
        )
    return "\n".join(rows)


def recall_bars(variants: list[Mapping[str, Any]]) -> str:
    colors = ["#8fb9ff", "#5de2c2"]
    blocks = []
    for index, row in enumerate(variants):
        name = "Current width" if index == 0 else "Wider"
        recalls = row["token_recall_percent"]
        bars = []
        for token in ["0", "1", "2", "3"]:
            value = float(recalls[token])
            bars.append(
                "<div class='bar-row'>"
                f"<span class='bar-label'>Token {token}</span>"
                "<span class='bar-track'>"
                f"<span class='bar-fill' style='width:{max(value, 0.25):.3f}%;"
                f"background:{colors[index]}'></span>"
                "</span>"
                f"<span class='bar-value'>{value:.3f}%</span>"
                "</div>"
            )
        blocks.append(
            f"<section class='bar-panel' aria-label='{esc(name)} token recall'>"
            f"<h4>{esc(name)}</h4>{''.join(bars)}</section>"
        )
    return "".join(blocks)


def render_report(evidence: Mapping[str, Any], sources: Mapping[str, Any]) -> str:
    variants = evidence["model_fitness"]["variants"]
    scale = evidence["dense_matrix_scale"]
    tls = evidence["related_tls_control"]
    execution = evidence["sensitivity_execution"]
    stage = execution["stage_snapshot"]
    pilot = execution["pilot"]
    identical = execution["identical_control"]
    refs = sources["sources"]
    ref_numbers = {row["id"]: index + 1 for index, row in enumerate(refs)}

    def cite(source_id: str) -> str:
        return citation(source_id, ref_numbers[source_id])

    source_items = []
    for index, source in enumerate(refs, start=1):
        doi = (
            f" DOI: <a href='https://doi.org/{esc(source['doi'])}'>"
            f"{esc(source['doi'])}</a>."
            if source.get("doi")
            else ""
        )
        source_items.append(
            f"<li id='ref-{esc(source['id'])}'>"
            f"<span class='ref-num'>{index}.</span> "
            f"<a href='{esc(source['url'])}'>{esc(source['title'])}</a>. "
            f"{esc(source['authors'])}. {esc(source['venue'])} "
            f"({esc(source['year'])}).{doi} "
            f"<span class='muted'>{esc(source['supports'])}</span>"
            "</li>"
        )

    source_gene_text = ", ".join(tls["organizer_genes"])
    receiver_gene_text = ", ".join(tls["receiver_program_genes"])
    generated = evidence["generated_at_utc"].replace("+00:00", "Z")
    shard_counts = stage["shard_file_counts"]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Jacobian biological-validation audit</title>
  <style>
    :root {{
      --ink:#18222d; --muted:#566575; --paper:#f5f7fa; --card:#ffffff;
      --line:#d7dee7; --navy:#102a43; --blue:#1f66d1; --teal:#087f6c;
      --amber:#9b5c00; --red:#a33a3a; --green:#176b50;
      --shadow:0 8px 28px rgba(16,42,67,.08);
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{
      margin:0; color:var(--ink); background:var(--paper);
      font:16px/1.58 Inter, ui-sans-serif, system-ui, -apple-system,
           BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    a {{ color:var(--blue); }}
    a:hover {{ text-decoration-thickness:2px; }}
    .masthead {{
      color:#fff; background:
        radial-gradient(circle at 85% -10%, rgba(93,226,194,.25), transparent 36%),
        linear-gradient(135deg,#102a43,#173f67 60%,#0e5f5b);
      padding:58px 24px 48px;
    }}
    .wrap {{ width:min(1160px, calc(100% - 36px)); margin:0 auto; }}
    .eyebrow {{
      margin:0 0 12px; color:#a9d9d1; font-size:.77rem;
      font-weight:800; letter-spacing:.12em; text-transform:uppercase;
    }}
    h1 {{ margin:.05em 0 .25em; font-size:clamp(2.15rem,5vw,4.5rem);
          line-height:1.02; letter-spacing:-.045em; max-width:980px; }}
    .dek {{ max-width:850px; color:#dce8f1; font-size:1.15rem; }}
    .meta {{ color:#b8cada; font-size:.86rem; }}
    nav {{
      position:sticky; top:0; z-index:3; background:rgba(255,255,255,.96);
      border-bottom:1px solid var(--line); backdrop-filter:blur(10px);
    }}
    nav .wrap {{ display:flex; gap:20px; overflow:auto; padding:11px 0; }}
    nav a {{ flex:none; color:var(--navy); font-size:.86rem;
             font-weight:700; text-decoration:none; }}
    main {{ padding:30px 0 70px; }}
    section.card {{
      margin:22px 0; padding:30px; background:var(--card);
      border:1px solid var(--line); border-radius:15px; box-shadow:var(--shadow);
    }}
    h2 {{ margin:0 0 16px; color:var(--navy); font-size:1.65rem;
          letter-spacing:-.025em; }}
    h3 {{ margin:27px 0 10px; color:var(--navy); font-size:1.12rem; }}
    h4 {{ margin:0 0 11px; color:var(--navy); }}
    p {{ margin:.7em 0; }}
    .verdict {{
      display:grid; grid-template-columns:1.15fr .85fr; gap:24px;
      border-left:7px solid var(--red)!important;
    }}
    .status {{
      display:inline-flex; padding:5px 10px; border-radius:999px;
      color:#7f2626; background:#fff0f0; border:1px solid #efbbbb;
      font-size:.78rem; font-weight:850; letter-spacing:.05em;
      text-transform:uppercase;
    }}
    .lead {{ font-size:1.28rem; line-height:1.42; color:var(--navy); }}
    .keybox {{
      padding:20px; border-radius:12px; background:#f0f5fa;
      border:1px solid #ccdae7;
    }}
    .keybox p:first-child {{ margin-top:0; }}
    .keybox p:last-child {{ margin-bottom:0; }}
    .fact-grid {{
      display:grid; grid-template-columns:repeat(3,1fr); gap:15px;
      margin-top:18px;
    }}
    .fact {{
      padding:17px; border:1px solid var(--line); border-radius:11px;
      background:#fbfcfd;
    }}
    .fact .label {{
      display:block; color:var(--muted); font-size:.76rem; font-weight:800;
      text-transform:uppercase; letter-spacing:.07em;
    }}
    .fact .value {{
      display:block; margin-top:4px; color:var(--navy); font-size:1.5rem;
      font-weight:850; line-height:1.15;
    }}
    .callout {{
      margin:18px 0; padding:16px 18px; border-radius:10px;
      border-left:5px solid var(--amber); background:#fff8e9;
    }}
    .callout.negative {{ border-color:var(--red); background:#fff2f2; }}
    .callout.info {{ border-color:var(--blue); background:#eef5ff; }}
    .flow {{
      display:grid; grid-template-columns:1fr auto 1fr auto 1fr;
      align-items:stretch; gap:10px; margin:22px 0;
    }}
    .flow-box {{
      padding:17px; border:1px solid var(--line); border-radius:11px;
      background:#f8fafc;
    }}
    .flow-box strong {{ display:block; color:var(--navy); }}
    .arrow {{ align-self:center; color:var(--muted); font-size:1.4rem; }}
    table {{ width:100%; border-collapse:collapse; margin:15px 0 5px;
             font-variant-numeric:tabular-nums; }}
    caption {{ padding:0 0 9px; color:var(--muted); text-align:left;
               font-size:.88rem; }}
    th,td {{ padding:10px 9px; border-bottom:1px solid var(--line);
             text-align:right; vertical-align:top; }}
    thead th {{ color:var(--muted); background:#f4f7fa; font-size:.76rem;
                letter-spacing:.035em; text-transform:uppercase; }}
    th:first-child,td:first-child {{ text-align:left; }}
    tbody tr:hover {{ background:#fafcfe; }}
    .delta.pos {{ color:var(--green); }} .delta.neg {{ color:var(--red); }}
    .scroll {{ overflow-x:auto; }}
    .bar-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    .bar-panel {{ padding:17px; border:1px solid var(--line); border-radius:11px; }}
    .bar-row {{
      display:grid; grid-template-columns:62px 1fr 72px; gap:9px;
      align-items:center; margin:9px 0; font-size:.82rem;
    }}
    .bar-track {{ height:10px; overflow:hidden; background:#e8edf3;
                  border-radius:999px; }}
    .bar-fill {{ display:block; height:100%; border-radius:999px; }}
    .bar-value {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .rubric td:nth-child(2) {{ font-weight:800; }}
    .no {{ color:var(--red); }} .weak {{ color:var(--amber); }}
    .pending {{ color:var(--blue); }} .yes {{ color:var(--green); }}
    .signal-grid {{
      display:grid; grid-template-columns:1fr 1fr; gap:18px; margin:18px 0;
    }}
    .signal {{
      padding:20px; border-radius:12px; border:1px solid var(--line);
      background:#f8fafc;
    }}
    .metric {{ color:var(--navy); font-size:1.55rem; font-weight:850; }}
    .metric small {{ font-size:.78rem; color:var(--muted); font-weight:650; }}
    .tag {{
      display:inline-block; margin:2px 4px 2px 0; padding:3px 7px;
      border-radius:6px; color:#31516e; background:#eaf1f7; font-size:.78rem;
    }}
    ol.steps {{ list-style:none; margin:20px 0; padding:0; counter-reset:step; }}
    ol.steps li {{
      position:relative; margin:0 0 12px; padding:17px 18px 17px 64px;
      border:1px solid var(--line); border-radius:11px; background:#fbfcfd;
    }}
    ol.steps li::before {{
      counter-increment:step; content:counter(step); position:absolute;
      left:18px; top:17px; width:30px; height:30px; display:grid;
      place-items:center; border-radius:50%; color:#fff; background:var(--navy);
      font-weight:850;
    }}
    ol.steps strong {{ color:var(--navy); }}
    .three-col {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
    .three-col ul {{ padding-left:20px; }}
    code {{ padding:.12em .35em; border-radius:4px; background:#edf1f5;
            font:85%/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    details {{ margin:10px 0; border:1px solid var(--line); border-radius:9px;
               background:#fbfcfd; }}
    summary {{ cursor:pointer; padding:12px 15px; color:var(--navy);
               font-weight:750; }}
    details > div {{ padding:0 15px 14px; }}
    .refs {{ padding-left:0; list-style:none; }}
    .refs li {{ margin:0 0 13px; padding-left:2.1em; text-indent:-2.1em; }}
    .ref-num {{ color:var(--muted); font-weight:800; }}
    .cite {{ font-size:.78em; font-weight:800; text-decoration:none; }}
    .muted {{ color:var(--muted); }}
    footer {{ padding:28px 0 50px; color:var(--muted); font-size:.83rem; }}
    @media (max-width:850px) {{
      .verdict,.fact-grid,.bar-grid,.signal-grid,.three-col {{ grid-template-columns:1fr; }}
      .flow {{ grid-template-columns:1fr; }}
      .arrow {{ transform:rotate(90deg); justify-self:center; }}
      section.card {{ padding:22px; }}
    }}
    @media print {{
      body {{ background:#fff; font-size:10pt; }}
      nav {{ display:none; }} .masthead {{ color:#000; background:#fff;
        border-bottom:2px solid #000; padding:20px 0; }}
      .masthead .dek,.masthead .meta,.eyebrow {{ color:#333; }}
      section.card {{ box-shadow:none; break-inside:avoid; margin:10px 0;
        border-color:#aaa; }}
      a {{ color:#000; text-decoration:none; }}
    }}
  </style>
</head>
<body>
  <header class="masthead">
    <div class="wrap">
      <p class="eyebrow">Post-hoc scientific audit · 26 July 2026</p>
      <h1>Do large Jacobian values correspond to verified biology?</h1>
      <p class="dek">For the current tokenized G2 artifacts, that question
      cannot be answered entry by entry: no Jacobian matrix, gene ranking, or
      gene-pair ranking was retained. The available evidence does not support
      a biological correspondence claim.</p>
      <p class="meta">Generated {esc(generated)} · Aggregate-only · One
      transductive CosMx core · Portable single-file report</p>
    </div>
  </header>

  <nav aria-label="Report sections">
    <div class="wrap">
      <a href="#verdict">Verdict</a>
      <a href="#computed">What was computed</a>
      <a href="#fitness">Model fitness</a>
      <a href="#biology">Biological check</a>
      <a href="#limits">Failure modes</a>
      <a href="#next">Next experiment</a>
      <a href="#provenance">Provenance</a>
    </div>
  </nav>

  <main class="wrap">
    <section class="card verdict" id="verdict">
      <div>
        <span class="status">Not testable from current artifacts</span>
        <h2>Bottom line</h2>
        <p class="lead"><strong>No evidence currently establishes that “high
        Jacobian values” correspond to biologically verified signals.</strong>
        This is not a finding of no biology. It is a finding that the required
        entrywise statistic does not exist in the output.</p>
        <p>The active protocol estimates global similarity between two enormous
        Jacobians using random vector–Jacobian products. It retains only
        <code>A2</code>, <code>B2</code>, and <code>AB</code> per probe. Many
        radically different matrices share those same three summaries, so no
        gene or gene pair can be reconstructed from them.</p>
      </div>
      <aside class="keybox" aria-label="Claim boundary">
        <p><strong>Fact:</strong> the protocol explicitly says
        <code>full_jacobian_materialized: false</code>.</p>
        <p><strong>Inference:</strong> biological enrichment cannot be computed
        without inventing non-identifiable entries.</p>
        <p><strong>Uncertainty:</strong> a newly designed program-block
        sensitivity analysis might find supported relationships; this audit
        does not test that future result.</p>
      </aside>
    </section>

    <section class="card" aria-labelledby="evidence-snapshot">
      <h2 id="evidence-snapshot">Evidence snapshot</h2>
      <div class="fact-grid">
        <div class="fact"><span class="label">Dense matrix scale</span>
          <span class="value">{scientific(scale['dense_entries'])}</span>
          effective entries per mask</div>
        <div class="fact"><span class="label">FP32 storage</span>
          <span class="value">{format_float(scale['dense_fp32_decimal_petabytes'],2)} PB</span>
          if materialized densely</div>
        <div class="fact"><span class="label">Stored entry ranks</span>
          <span class="value">0</span>genes, pairs, cells, or edges</div>
        <div class="fact"><span class="label">Biological units</span>
          <span class="value">1 core</span>no patient replication</div>
        <div class="fact"><span class="label">Token 1 / 2 recall</span>
          <span class="value">0% / 0%</span>for both widths</div>
        <div class="fact"><span class="label">Related TLS control</span>
          <span class="value">Failed</span>joint faithfulness criterion</div>
      </div>
    </section>

    <section class="card" id="computed">
      <h2>What the current “Jacobian” analysis actually computes</h2>
      <p>The estimand is a <em>local model-implied relaxed categorical
      sensitivity</em>: centered four-class logits at masked receiver cells are
      differentiated with respect to observed four-channel token indicators,
      then projected onto each token simplex. Token IDs are not treated as
      continuous counts. Under whole-node masking, differentiated inputs are
      other observed nodes; the statistic combines direct graph and two-layer
      multihop routes.</p>

      <div class="flow" role="img" aria-label="Random projection workflow">
        <div class="flow-box"><strong>1 · Output probe</strong>
          Random ±1 vector over {format_int(scale['output_coordinates'])}
          masked-receiver logit coordinates.</div>
        <div class="arrow" aria-hidden="true">→</div>
        <div class="flow-box"><strong>2 · Vector–Jacobian product</strong>
          Compute <code>Jᵀu</code> over
          {format_int(scale['effective_input_coordinates'])} effective observed
          token coordinates.</div>
        <div class="arrow" aria-hidden="true">→</div>
        <div class="flow-box"><strong>3 · Collapse</strong>
          Retain only squared norms and a cross-inner-product:
          <code>A2</code>, <code>B2</code>, <code>AB</code>.</div>
      </div>

      <div class="callout negative">
        <strong>These are trace-level summaries, not high entries.</strong>
        They estimate whole-function cosine, relative discrepancy, and norm
        ratio. They contain no input-gene × output-gene axis, no sign for a
        biological pair, and no source-cell, receiver-cell, edge, distance, or
        pathway decomposition.
      </div>

      <h3>Scale calculation</h3>
      <div class="scroll">
        <table>
          <caption>One whole-node mask; four represented class coordinates.
          Centering/tangent constraints reduce algebraic rank but do not create
          a stored entrywise object.</caption>
          <tbody>
            <tr><th scope="row">Nodes / genes</th>
              <td>{format_int(scale['nodes'])} / {format_int(scale['genes'])}</td></tr>
            <tr><th scope="row">Masked receiver nodes</th>
              <td>{format_int(scale['masked_target_nodes_per_mask'])}</td></tr>
            <tr><th scope="row">Observed source nodes</th>
              <td>{format_int(scale['observed_source_nodes_per_mask'])}</td></tr>
            <tr><th scope="row">Output coordinates</th>
              <td>{format_int(scale['output_coordinates'])}</td></tr>
            <tr><th scope="row">Effective input coordinates</th>
              <td>{format_int(scale['effective_input_coordinates'])}</td></tr>
            <tr><th scope="row">Dense entries</th>
              <td>{format_int(scale['dense_entries'])}</td></tr>
            <tr><th scope="row">Dense FP32 storage</th>
              <td>{format_float(scale['dense_fp32_decimal_petabytes'],3)} PB
              ({format_float(scale['dense_fp32_pebibytes'],3)} PiB)</td></tr>
          </tbody>
        </table>
      </div>

      <h3>Execution status at report generation</h3>
      <p>The run is explicitly
      <code>post_hoc_exploratory_after_prespecified_gate_failure</code>.
      The resource pilot passed: one wider-model VJP took
      {format_float(pilot['vjp_duration_seconds'],3)} s, used
      {format_float(pilot['peak_cuda_memory_allocated_bytes']/2**30,3)} GiB
      peak allocated VRAM, and produced a finite nonzero tangent squared norm
      of {format_float(pilot['tangent_squared_norm'],4)}.</p>
      <p><strong>Filesystem snapshot:</strong> {esc(stage['label'])}.
      Artifact counts — identical: {shard_counts['identical']}, trained:
      {shard_counts['trained']}, randomized: {shard_counts['random']}.
      {esc(stage['note'])}</p>
      {
        (
          "<p><strong>Identical-checkpoint control:</strong> passed across "
          f"{identical['shard_count']} masks and {identical['probe_count']} probes. "
          f"Point cosine = {identical['point']['cosine']:.6f}, relative discrepancy "
          f"= {identical['point']['relative_discrepancy']:.6f}, and norm ratio "
          f"= {identical['point']['norm_ratio']:.6f}. This validates exact numerical "
          "repeatability for identical weights; it does not create entrywise ranks "
          "or biological evidence.</p>"
        )
        if identical['present'] and identical['passes']
        else (
          "<p><strong>Identical-checkpoint control:</strong> not yet available "
          "in the report-generation snapshot.</p>"
        )
      }
      <p>Even a completed run will answer whether model functions are globally
      similar—not which entries are biologically supported.</p>
    </section>

    <section class="card" id="fitness">
      <h2>The predictive prerequisite is weak</h2>
      <p>Both models were fitted and evaluated transductively on one core. The
      headline exact accuracy is dominated by the 91.8% zero class. Relative to
      the all-fit per-gene modal predictor, exact accuracy improves by only
      about 0.04–0.05 percentage points, while balanced and nonzero accuracy are
      worse. Tokens 1 and 2 are never recovered.</p>

      <div class="scroll">
        <table>
          <caption>Means across three model seeds; each run averages three
          technical whole-node masks. “Δ modal” is model minus the all-fit
          per-gene modal baseline.</caption>
          <thead><tr>
            <th>Variant</th><th>Parameters</th><th>Exact</th><th>Modal exact</th>
            <th>Δ modal</th><th>Balanced</th><th>Δ modal</th>
            <th>Nonzero</th><th>Δ modal</th>
          </tr></thead>
          <tbody>{performance_table(variants)}</tbody>
        </table>
      </div>

      <div class="bar-grid" aria-label="Token recall charts">
        {recall_bars(variants)}
      </div>

      <div class="callout">
        <strong>A nonzero gradient is not evidence of useful prediction.</strong>
        Neural networks can have large local derivatives while implementing a
        poor or shortcut-dominated classifier. The prespecified >95% gate failed
        for wider seeds 0–2 ({", ".join(f"{v:.4f}%" for v in evidence['model_fitness']['gate']['wider_seed_values_percent'].values())}),
        and the user-requested continuation is correctly labelled post-hoc.
      </div>
    </section>

    <section class="card" id="biology">
      <h2>Biological cross-check</h2>
      <h3>Current tokenized models: no candidate list exists</h3>
      <p>There is nothing valid to submit to a ligand–receptor or pathway
      reference: the current outputs contain no high genes or high pairs.
      OmniPath preserves literature-curated signaling relationships and
      direction/sign where available {cite('turei2021')}; CellPhoneDB represents
      ligand–receptor complexes and predicts enriched cell-type interactions
      {cite('efremova2020')}. Those resources can annotate a <em>predefined,
      axis-labelled</em> candidate list. They cannot reverse-engineer entries
      from three global norms, and database overlap would still be annotation,
      not verification.</p>

      <h3>Related repository evidence: a useful negative control</h3>
      <p>A different, earlier continuous-expression G2 run tested a prespecified
      TLS-related family. It is not one of the six tokenized checkpoints and is
      not a Jacobian result, so it cannot be substituted for the missing test.
      It does, however, provide a direct counterexample to “biologically
      plausible marker enrichment means the model uses the mechanism.”</p>

      <p><span class="tag">Organizer: {esc(source_gene_text)}</span>
      <span class="tag">Receiver program: {esc(receiver_gene_text)}</span></p>

      <div class="signal-grid">
        <article class="signal">
          <h4>Plausibility / enrichment</h4>
          <div class="metric">{format_float(tls['organizer_score_enrichment_mean'],3)}
            <small>mean top-minus-matched-null organizer score</small></div>
          <p>The minimum across eight matched draws was
          {format_float(tls['organizer_score_enrichment_minimum'],3)}. This
          passed the descriptive organizer-enrichment rule.</p>
        </article>
        <article class="signal">
          <h4>Faithfulness / intervention</h4>
          <div class="metric">{scientific(tls['sender_ablation_top_minus_null_huber_change_mean'],3)}
            <small>sender-ablation Huber contrast</small></div>
          <p>Edge-deletion support: <strong>{str(tls['deletion_support']).lower()}</strong>.
          Sender-program support:
          <strong>{str(tls['sender_program_support']).lower()}</strong>.
          Joint TLS dependency support:
          <strong>{str(tls['joint_tls_dependency_support']).lower()}</strong>.</p>
        </article>
      </div>

      <p>The organizer family itself is biologically credible: experimental
      ectopic-expression work supports CCL19/CCL21 activity in recruitment and
      lymphoid neogenesis {cite('luther2002')}, and genetic double-deficiency
      work supports cooperative CXCR5 and CCR7 roles in lymphoid organ
      organization {cite('ohl2003')}. A recent gastric-cancer multimodal study
      combined single-cell data, multiplex IHC, flow cytometry, and coculture
      assays to support a CXCL13–CXCR5 circuit in mature TLSs
      {cite('wu2025')}.</p>

      <div class="callout negative">
        <strong>Observed result:</strong> biologically established markers were
        enriched among top-routed senders, but the matched deletion and
        organizer-channel interventions did not jointly support a faithful
        TLS-related dependency. The repository’s evidence label is:
        “{esc(tls['evidence_label'])}.” This weakens—not strengthens—the
        assumption that a large model score automatically maps to verified
        biology.
      </div>
    </section>

    <section class="card" id="limits">
      <h2>Why a future high value would still need aggressive controls</h2>
      <div class="three-col">
        <div>
          <h3>Statistic</h3>
          <ul>
            <li>A gradient is local and coordinate-scale dependent.</li>
            <li>Centered logits are not calibrated class probabilities.</li>
            <li>Token 3 collapses all counts ≥3.</li>
            <li>Absolute magnitude discards sign and direction.</li>
            <li>Gene-level values multiply testing and correlated features.</li>
          </ul>
        </div>
        <div>
          <h3>Model and graph</h3>
          <ul>
            <li>Current prediction is dominated by the zero class.</li>
            <li>The k=1000 graph is regional, not a direct-contact graph.</li>
            <li>Two GAT layers mix direct and multihop routes.</li>
            <li>Whole-node masking can be solved by inferred cell identity.</li>
            <li>Segmentation spillover and spatial smoothing remain alternatives.</li>
          </ul>
        </div>
        <div>
          <h3>Biological evidence</h3>
          <ul>
            <li>One core is one biological unit.</li>
            <li>Database overlap is annotation, not replication.</li>
            <li>Preselected markers make lookup circular as validation.</li>
            <li>RNA proximity does not establish protein-level signaling.</li>
            <li>Causality requires controlled perturbation.</li>
          </ul>
        </div>
      </div>

      <p>Randomization checks are particularly important for saliency-style
      methods: Adebayo and colleagues showed that some explanations can remain
      similar after model or data randomization {cite('adebayo2018')}. The active
      global-similarity protocol correctly includes identical and randomized
      controls, but those controls are not yet attached to a gene-pair ranking.</p>

      <div class="scroll">
        <table class="rubric">
          <caption>Evidence ladder for the current claim.</caption>
          <thead><tr><th>Requirement</th><th>Current state</th><th>What it means</th></tr></thead>
          <tbody>
            <tr><th scope="row">Predictive usefulness</th><td class="weak">Weak / failed baselines</td>
              <td>Exact accuracy is a zero-class shortcut; balanced and nonzero metrics trail the modal reference.</td></tr>
            <tr><th scope="row">Global sensitivity exists</th><td class="pending">Post-hoc run</td>
              <td>Can compare whole functions after completion.</td></tr>
            <tr><th scope="row">Entrywise candidate ranking</th><td class="no">Absent</td>
              <td>No genes, pairs, cells, or edges can be called high.</td></tr>
            <tr><th scope="row">Seed/mask stability of candidates</th><td class="no">Not testable</td>
              <td>Global similarity is not candidate-rank stability.</td></tr>
            <tr><th scope="row">Randomization of candidates</th><td class="no">Not testable</td>
              <td>Random controls exist only for whole-Jacobian metrics.</td></tr>
            <tr><th scope="row">External annotation</th><td class="no">No candidate set</td>
              <td>Reference databases cannot be applied without axes.</td></tr>
            <tr><th scope="row">Faithfulness</th><td class="weak">Related TLS control negative</td>
              <td>Marker enrichment did not survive joint intervention criteria.</td></tr>
            <tr><th scope="row">Independent replication</th><td class="no">Absent</td>
              <td>One core cannot establish patient prevalence.</td></tr>
            <tr><th scope="row">Causal mechanism</th><td class="no">Absent</td>
              <td>No biological perturbation was performed.</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="card" id="next">
      <h2>Recommended next experiment</h2>
      <p class="lead">Do not reinterpret the current trace estimates as a
      biological matrix. The next discriminating experiment is a locked,
      program-level, direct-cross-cell sensitivity study—but only after the
      model demonstrates graph-specific predictive gain on informative
      metrics.</p>

      <ol class="steps">
        {''.join(
          f"<li><strong>{esc(step['name'].title())}.</strong> {esc(step['action'])}</li>"
          for step in evidence['recommended_next_experiment']['ordered_steps']
        )}
      </ol>

      <h3>Proposed conclusion object</h3>
      <div class="keybox">
        <code>source cell type + source program → receiver cell type + target
        program + distance bin + signed effect + uncertainty + seed stability
        + null calibration + faithfulness + patient prevalence</code>
      </div>

      <h3>Minimum go criteria</h3>
      <ul>
        {''.join(
          f"<li>{esc(item)}</li>"
          for item in evidence['recommended_next_experiment']['minimum_acceptance']
        )}
      </ul>
      <p>OmniPath or CellPhoneDB overlap should be reported as one annotation
      dimension. The platform can measure RNA and protein spatially at
      subcellular resolution {cite('he2022')}, but this dataset’s RNA-only
      model output is not itself orthogonal protein support.</p>
    </section>

    <section class="card" id="provenance">
      <h2>Provenance, reproducibility, and sources</h2>
      <p>This report reads aggregate JSON, protocol, and source-contract files
      only. It does not load raw expression, cell metadata, graph arrays, or
      checkpoints. No project data or result list was sent to an external
      service. External literature was selected to define the evidence rubric
      and annotate the pre-existing TLS positive-control family.</p>

      <details>
        <summary>Audited local inputs and SHA-256 checksums</summary>
        <div><table><tbody>
          {''.join(
            f"<tr><th scope='row'><code>{esc(path)}</code></th>"
            f"<td><code>{esc(digest)}</code></td></tr>"
            for path,digest in evidence['input_checksums'].items()
          )}
        </tbody></table></div>
      </details>

      <details>
        <summary>Fact, inference, and uncertainty register</summary>
        <div class="three-col">
          <div><h3>Facts</h3><ul>{''.join(f"<li>{esc(x)}</li>" for x in evidence['facts'])}</ul></div>
          <div><h3>Inferences</h3><ul>{''.join(f"<li>{esc(x)}</li>" for x in evidence['inferences'])}</ul></div>
          <div><h3>Uncertainties</h3><ul>{''.join(f"<li>{esc(x)}</li>" for x in evidence['uncertainties'])}</ul></div>
        </div>
      </details>

      <h3>Primary and resource literature</h3>
      <ol class="refs">{''.join(source_items)}</ol>
      <p class="muted">Literature access date: {esc(sources['accessed_utc'])}.
      {esc(sources['policy'])}</p>
    </section>
  </main>

  <footer class="wrap">
    <p><strong>Maximum defensible conclusion:</strong> The current artifacts do
    not permit an entrywise biological correspondence test, and the weak
    predictive prerequisite plus a negative related faithfulness control argue
    against biological interpretation. A properly controlled future
    program-block analysis remains an open experiment.</p>
  </footer>
</body>
</html>
"""


def main() -> None:
    evidence = build_evidence()
    sources = load_json(SOURCES_PATH)
    if not isinstance(sources, Mapping) or not isinstance(
        sources.get("sources"), list
    ):
        raise ValueError("sources.json has an invalid structure")
    dump_json(REPORT_DIR / "evidence.json", evidence)
    report = render_report(evidence, sources)
    (REPORT_DIR / "report.html").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "generated",
                "report": str(REPORT_DIR / "report.html"),
                "evidence": str(REPORT_DIR / "evidence.json"),
                "report_bytes": len(report.encode("utf-8")),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
