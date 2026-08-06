#!/usr/bin/env python3
"""Extract, rank, annotate, and verify the finalized GeneMAE Jacobian pairs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spatial_benchmark.jacobian_pair_analysis import (  # noqa: E402
    JacobianPairAnalysisError,
    file_sha256,
    load_jacobian_artifact,
    profile_spearman_matrices,
    rank_mutual_sensitivity_pairs,
    rank_profile_pairs,
)
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402


CAMPAIGN_ID = "cmp_20260802_genemae_jacobian_pair_annotation"
UPSTREAM_RUN_ID = "r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a"
EXPECTED_INPUT_SHA256 = (
    "6a5bdbcc17c397a84696062a249cd56f83b17be3a6cb3b80e8772606c1b9cf33"
)
TOP_PAIR_COUNT = 10
EXPECTED_GENE_COUNT = 39
CONTRACT_RELATIVE = Path(
    "experiments/campaigns/"
    "cmp_20260802_genemae_jacobian_pair_annotation/"
    "frozen_task_contract.yaml"
)
EVIDENCE_RELATIVE = Path(
    "experiments/campaigns/"
    "cmp_20260802_genemae_jacobian_pair_annotation/"
    "biological_evidence.yaml"
)
INPUT_RELATIVE = Path(
    "runs/2026/07/"
    "r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a/"
    "interpretation/report.json"
)
OUTPUT_RELATIVE = Path("analyses/genemae_jacobian_gene_pairs_20260802")


class EvidenceError(ValueError):
    """Raised when manual biological evidence is incomplete or inconsistent."""


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _load_yaml_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise EvidenceError(f"{label} must contain a mapping")
    return payload


def _load_biological_evidence(
    path: Path, top_pairs: Sequence[Mapping[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    payload = _load_yaml_mapping(path, label="biological evidence")
    if payload.get("schema_version") != 1:
        raise EvidenceError("biological evidence schema_version must be 1")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise EvidenceError("biological evidence requires a non-empty sources list")
    sources: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping):
            raise EvidenceError(f"sources[{index}] must be a mapping")
        source = dict(raw)
        source_id = source.get("id")
        title = source.get("title")
        url = source.get("url")
        if not isinstance(source_id, str) or not source_id:
            raise EvidenceError(f"sources[{index}].id must be a non-empty string")
        if source_id in source_ids:
            raise EvidenceError(f"Duplicate source id: {source_id}")
        if not isinstance(title, str) or not title:
            raise EvidenceError(f"sources[{index}].title must be non-empty")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise EvidenceError(f"sources[{index}].url must use https://")
        source_ids.add(source_id)
        sources.append(source)

    raw_rows = payload.get("pair_evidence")
    if not isinstance(raw_rows, list):
        raise EvidenceError("pair_evidence must be a list")
    required_pairs = {
        _canonical_pair(str(row["gene_a"]), str(row["gene_b"]))
        for row in top_pairs
    }
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    required_text = (
        "classification",
        "summary",
        "direct_relationship",
        "shared_program_or_pathway",
        "coexpression_or_cell_state",
        "perturbational_evidence",
        "gastric_specific_evidence",
        "limitations",
    )
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise EvidenceError(f"pair_evidence[{index}] must be a mapping")
        row = dict(raw)
        raw_genes = row.get("genes")
        if (
            not isinstance(raw_genes, list)
            or len(raw_genes) != 2
            or any(not isinstance(gene, str) for gene in raw_genes)
        ):
            raise EvidenceError(
                f"pair_evidence[{index}].genes must contain two strings"
            )
        pair = _canonical_pair(raw_genes[0], raw_genes[1])
        if pair in observed:
            raise EvidenceError(f"Duplicate pair evidence for {pair}")
        if pair not in required_pairs:
            raise EvidenceError(f"Evidence contains non-ranked pair {pair}")
        for key in required_text:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise EvidenceError(
                    f"pair_evidence[{index}].{key} must be non-empty"
                )
        citations = row.get("source_ids")
        if not isinstance(citations, list) or not citations:
            raise EvidenceError(
                f"pair_evidence[{index}] requires source_ids; use a documented "
                "search source for a no-evidence result"
            )
        unknown = [item for item in citations if item not in source_ids]
        if unknown:
            raise EvidenceError(
                f"pair_evidence[{index}] has unknown source ids: {unknown}"
            )
        row["source_ids"] = list(citations)
        observed[pair] = row
    missing = sorted(required_pairs - set(observed))
    if missing:
        raise EvidenceError(f"Missing biological evidence for pairs: {missing}")
    return observed, sources


def _write_matrix_csv(path: Path, genes: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target_gene/source_gene", *genes])
        for gene, row in zip(genes, matrix, strict=True):
            writer.writerow([gene, *(format(float(value), ".17g") for value in row)])


def _write_dict_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _pair_sources(
    evidence: Mapping[str, Any], source_by_id: Mapping[str, Mapping[str, Any]]
) -> str:
    return "; ".join(
        f"[{source_by_id[source_id]['title']}]({source_by_id[source_id]['url']})"
        for source_id in evidence["source_ids"]
    )


def _markdown_report(
    *,
    artifact_note: str,
    top_pairs: Sequence[Mapping[str, Any]],
    evidence_by_pair: Mapping[tuple[str, str], Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    receiver_top: Sequence[Mapping[str, Any]],
    source_top: Sequence[Mapping[str, Any]],
) -> str:
    source_by_id = {str(source["id"]): source for source in sources}
    lines = [
        "# GeneMAE Jacobian-pair annotation",
        "",
        "## Result",
        "",
        "A verified 39-by-39 signed gene-to-gene sensitivity matrix was "
        "extracted from the seven-seed GeneMAE ensemble. The leading pairs "
        "are dominated by collagen/ECM and simple-epithelial keratin genes. "
        "Six have direct structural or biochemical evidence, three have "
        "shared-program or indirect functional evidence, and ACTA2-PSCA has "
        "only a weak unvalidated co-capture. This is not independent "
        "validation: the 39 genes were biology-selected, the analysis is "
        "held-in, and the upstream graph-gradient structure gate passed in "
        "0/10 cores.",
        "",
        "The primary pair score is `0.5*(abs(J[i,j]) + abs(J[j,i]))`. It is "
        "an unsigned mutual model-sensitivity magnitude, not a gene-expression "
        "correlation coefficient. Rows of `J` are targets and columns are "
        "sources.",
        "",
        "## Top ten mutual sensitivities and biological annotation",
        "",
        "| Rank | Pair | Mutual | Directed J | Evidence | Biological annotation |",
        "|---:|---|---:|---|---|---|",
    ]
    for row in top_pairs:
        pair = _canonical_pair(str(row["gene_a"]), str(row["gene_b"]))
        evidence = evidence_by_pair[pair]
        directed = (
            f"{row['gene_a']}<-{row['gene_b']} "
            f"{float(row['j_gene_a_target_gene_b_source']):.5f}; "
            f"{row['gene_b']}<-{row['gene_a']} "
            f"{float(row['j_gene_b_target_gene_a_source']):.5f}"
        )
        citation_text = _pair_sources(evidence, source_by_id)
        lines.append(
            f"| {row['rank']} | `{row['gene_a']}–{row['gene_b']}` | "
            f"{float(row['mutual_absolute_sensitivity']):.5f} | {directed} | "
            f"{evidence['classification']} | {evidence['summary']} "
            f"{citation_text} |"
        )
    lines.extend(
        [
            "",
            "`Direct` in the evidence column refers to an independently "
            "documented physical/biochemical relationship between the gene "
            "products. `Shared program` is weaker: it means a common matrix, "
            "epithelial, or cell-state program without a demonstrated direct "
            "pairwise interaction. Literature support does not validate the "
            "model's sign, direction, graph dependence, or tissue mechanism.",
            "",
            "## Literal Jacobian-profile correlations",
            "",
            "A separate Spearman analysis asked whether two target rows respond "
            "similarly across other source genes, or whether two source columns "
            "have similar downstream target profiles. Both pair coordinates "
            "were excluded. These are profile similarities, not direct edges.",
            "",
            "### Receiver-row profiles",
            "",
            "| Rank | Pair | Spearman rho |",
            "|---:|---|---:|",
        ]
    )
    for row in receiver_top:
        lines.append(
            f"| {row['rank']} | `{row['gene_a']}–{row['gene_b']}` | "
            f"{float(row['spearman_rho']):.4f} |"
        )
    lines.extend(
        [
            "",
            "### Source-column profiles",
            "",
            "| Rank | Pair | Spearman rho |",
            "|---:|---|---:|",
        ]
    )
    for row in source_top:
        lines.append(
            f"| {row['rank']} | `{row['gene_a']}–{row['gene_b']}` | "
            f"{float(row['spearman_rho']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Only `COL1A1–COL3A1` appears in both top-ten profile lists. "
            "Several other high profile correlations cross canonical cell "
            "lineages, which is evidence that aggregate scale, cell-state "
            "mixtures, or low-rank model structure can create high rho without "
            "a direct biological relationship. These 741 post-hoc pairwise "
            "profile comparisons are descriptive and have no inferential "
            "p-values.",
            "",
            "## Interpretation limits",
            "",
            f"- Upstream matrix note: {artifact_note}",
            "- The ensemble was selected for held-in partial-gene reconstruction "
            "(Huber 0.348210), not patient-held-out graph prediction.",
            "- Its global graph-use improvement was 1.09%, below the frozen 2% "
            "gate; only KRT8 passed target-specific graph-use eligibility.",
            "- The upstream graph-gradient structure null failed in every core: "
            "the locked-pair statistic was stronger after node-label "
            "permutation in 10/10 cores.",
            "- The matrix uses unmasked inputs and mixes the node-wise branch, "
            "same-cell graph self-loops, and cross-cell paths up to four hops.",
            "- It covers 39 preselected markers rather than the full 1,000-gene "
            "panel; favorable literature overlap is therefore circular.",
            "- No independent cohort, orthogonal measurement, or controlled "
            "perturbation tests these model-derived pairs.",
            "",
            "## Maximum defensible claim",
            "",
            "The listed pairs are literature-annotated, held-in aggregate "
            "GeneMAE model-implied sensitivities. They are not biologically "
            "verified model relationships, graph-dependent communication "
            "edges, mechanisms, or causal effects.",
            "",
            "Complete matrices and rankings are provided as CSV files in this "
            "directory; `provenance.json` and `verification.json` bind them to "
            "the immutable upstream artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _output_checksums(output: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: file_sha256(output / name) for name in names}


def run_analysis(
    *,
    paths: ProjectPaths,
    input_path: Path,
    output_path: Path,
    evidence_path: Path,
    expected_input_sha256: str,
) -> dict[str, Any]:
    artifact = load_jacobian_artifact(
        input_path,
        expected_sha256=expected_input_sha256,
        expected_gene_count=EXPECTED_GENE_COUNT,
    )
    top_pairs = rank_mutual_sensitivity_pairs(
        artifact.genes,
        artifact.signed_directed,
        artifact.symmetric_absolute,
        top_n=TOP_PAIR_COUNT,
    )
    receiver_matrix, source_matrix = profile_spearman_matrices(
        artifact.genes, artifact.signed_directed
    )
    receiver_top = rank_profile_pairs(
        artifact.genes, receiver_matrix, top_n=TOP_PAIR_COUNT
    )
    source_top = rank_profile_pairs(
        artifact.genes, source_matrix, top_n=TOP_PAIR_COUNT
    )
    evidence_by_pair, sources = _load_biological_evidence(
        evidence_path, top_pairs
    )

    with input_path.open("r", encoding="utf-8") as stream:
        upstream = json.load(stream)
    graph_null = upstream.get("graph_gradient_null", {})
    if graph_null.get("passing_core_count") != 0:
        raise JacobianPairAnalysisError(
            "Expected the frozen adverse graph-null result of 0 passing cores"
        )
    if upstream.get("claim_verdict") != "biological_mechanism_not_validated":
        raise JacobianPairAnalysisError(
            "Unexpected upstream biological claim verdict"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    _write_matrix_csv(
        output_path / "signed_directed_jacobian.csv",
        artifact.genes,
        artifact.signed_directed,
    )
    _write_matrix_csv(
        output_path / "symmetric_absolute_jacobian.csv",
        artifact.genes,
        artifact.symmetric_absolute,
    )
    _write_matrix_csv(
        output_path / "receiver_profile_spearman.csv",
        artifact.genes,
        receiver_matrix,
    )
    _write_matrix_csv(
        output_path / "source_profile_spearman.csv",
        artifact.genes,
        source_matrix,
    )
    pair_fields = (
        "rank",
        "gene_a",
        "gene_b",
        "mutual_absolute_sensitivity",
        "j_gene_a_target_gene_b_source",
        "j_gene_b_target_gene_a_source",
        "dominant_absolute_direction",
        "receiver_profile_spearman",
        "source_profile_spearman",
    )
    positions = {gene: index for index, gene in enumerate(artifact.genes)}
    for row in top_pairs:
        first = positions[str(row["gene_a"])]
        second = positions[str(row["gene_b"])]
        row["receiver_profile_spearman"] = float(
            receiver_matrix[first, second]
        )
        row["source_profile_spearman"] = float(source_matrix[first, second])
    _write_dict_csv(
        output_path / "top_mutual_sensitivity_pairs.csv",
        top_pairs,
        pair_fields,
    )
    profile_fields = ("rank", "gene_a", "gene_b", "spearman_rho")
    _write_dict_csv(
        output_path / "top_receiver_profile_pairs.csv",
        receiver_top,
        profile_fields,
    )
    _write_dict_csv(
        output_path / "top_source_profile_pairs.csv",
        source_top,
        profile_fields,
    )

    biological_rows: list[dict[str, Any]] = []
    for pair_row in top_pairs:
        pair = _canonical_pair(
            str(pair_row["gene_a"]), str(pair_row["gene_b"])
        )
        evidence = evidence_by_pair[pair]
        biological_rows.append(
            {
                "rank": pair_row["rank"],
                "gene_a": pair_row["gene_a"],
                "gene_b": pair_row["gene_b"],
                "classification": evidence["classification"],
                "summary": evidence["summary"],
                "direct_relationship": evidence["direct_relationship"],
                "shared_program_or_pathway": evidence[
                    "shared_program_or_pathway"
                ],
                "coexpression_or_cell_state": evidence[
                    "coexpression_or_cell_state"
                ],
                "perturbational_evidence": evidence[
                    "perturbational_evidence"
                ],
                "gastric_specific_evidence": evidence[
                    "gastric_specific_evidence"
                ],
                "limitations": evidence["limitations"],
                "source_ids": ";".join(evidence["source_ids"]),
            }
        )
    biological_fields = tuple(biological_rows[0])
    _write_dict_csv(
        output_path / "biological_evidence.csv",
        biological_rows,
        biological_fields,
    )
    (output_path / "sources.json").write_text(
        json.dumps({"schema_version": 1, "sources": sources}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_path / "report.md").write_text(
        _markdown_report(
            artifact_note=artifact.adaptation_note,
            top_pairs=top_pairs,
            evidence_by_pair=evidence_by_pair,
            sources=sources,
            receiver_top=receiver_top,
            source_top=source_top,
        ),
        encoding="utf-8",
    )
    contract_path = paths.project_root / CONTRACT_RELATIVE
    script_path = Path(__file__).resolve()
    module_path = paths.project_root / "src/spatial_benchmark/jacobian_pair_analysis.py"
    evidence_source_sha256 = file_sha256(evidence_path)
    provenance = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "upstream_run_id": UPSTREAM_RUN_ID,
        "input": {
            "path": _relative_or_absolute(input_path, paths.project_root),
            "sha256": artifact.sha256,
            "gene_count": len(artifact.genes),
            "shape": list(artifact.signed_directed.shape),
            "signed_matrix_role": "target_rows_source_columns",
            "published_transform": artifact.published_transform,
        },
        "contract": {
            "path": CONTRACT_RELATIVE.as_posix(),
            "sha256": file_sha256(contract_path),
            "exploratory": True,
        },
        "biological_evidence_input": {
            "path": _relative_or_absolute(evidence_path, paths.project_root),
            "sha256": evidence_source_sha256,
        },
        "code": {
            _relative_or_absolute(script_path, paths.project_root): file_sha256(
                script_path
            ),
            _relative_or_absolute(module_path, paths.project_root): file_sha256(
                module_path
            ),
        },
        "ranking": {
            "primary": "0.5*(abs(J[i,j])+abs(J[j,i]))",
            "top_n": TOP_PAIR_COUNT,
            "diagonal_excluded": True,
            "profile_correlations_exclude_pair_coordinates": True,
        },
        "upstream_adverse_evidence": {
            "claim_verdict": upstream["claim_verdict"],
            "graph_gradient_null_passing_cores": graph_null[
                "passing_core_count"
            ],
            "graph_gradient_null_total_cores": len(
                graph_null.get("core_rows", [])
            ),
            "eligible_targets": upstream.get("eligible_targets", []),
        },
    }
    (output_path / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    verified_names = (
        "signed_directed_jacobian.csv",
        "symmetric_absolute_jacobian.csv",
        "receiver_profile_spearman.csv",
        "source_profile_spearman.csv",
        "top_mutual_sensitivity_pairs.csv",
        "top_receiver_profile_pairs.csv",
        "top_source_profile_pairs.csv",
        "biological_evidence.csv",
        "sources.json",
        "report.md",
        "provenance.json",
    )
    verification = {
        "schema_version": 1,
        "status": "passed",
        "checks": {
            "input_checksum_match": artifact.sha256
            == expected_input_sha256,
            "gene_count_39": len(artifact.genes) == EXPECTED_GENE_COUNT,
            "matrix_shapes_39_by_39": artifact.signed_directed.shape
            == (EXPECTED_GENE_COUNT, EXPECTED_GENE_COUNT),
            "all_matrix_values_finite": bool(
                np.isfinite(artifact.signed_directed).all()
                and np.isfinite(artifact.symmetric_absolute).all()
                and np.isfinite(receiver_matrix).all()
                and np.isfinite(source_matrix).all()
            ),
            "symmetric_transform_reproduced": bool(
                np.allclose(
                    artifact.symmetric_absolute,
                    0.5
                    * (
                        np.abs(artifact.signed_directed)
                        + np.abs(artifact.signed_directed.T)
                    ),
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            "top_ten_unique_pairs": len(top_pairs) == TOP_PAIR_COUNT
            and len(
                {
                    _canonical_pair(str(row["gene_a"]), str(row["gene_b"]))
                    for row in top_pairs
                }
            )
            == TOP_PAIR_COUNT,
            "all_top_pairs_have_evidence": len(evidence_by_pair)
            == TOP_PAIR_COUNT,
            "upstream_negative_claim_retained": upstream["claim_verdict"]
            == "biological_mechanism_not_validated",
            "upstream_graph_null_retained": graph_null["passing_core_count"]
            == 0,
        },
        "output_sha256": _output_checksums(output_path, verified_names),
    }
    if not all(verification["checks"].values()):
        raise JacobianPairAnalysisError(
            f"Analysis verification failed: {verification['checks']}"
        )
    (output_path / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "campaign_id": CAMPAIGN_ID,
        "output": str(output_path),
        "top_pairs": top_pairs,
        "verification": verification["checks"],
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_existing_output(
    *, paths: ProjectPaths, input_path: Path, output_path: Path
) -> dict[str, Any]:
    verification_path = output_path / "verification.json"
    with verification_path.open("r", encoding="utf-8") as stream:
        verification = json.load(stream)
    if not isinstance(verification, Mapping):
        raise JacobianPairAnalysisError("verification.json must be a mapping")
    if verification.get("status") != "passed":
        raise JacobianPairAnalysisError("verification status is not passed")
    checksums = verification.get("output_sha256")
    if not isinstance(checksums, Mapping) or not checksums:
        raise JacobianPairAnalysisError("verification output checksums are missing")
    mismatches = {
        name: {"expected": expected, "observed": file_sha256(output_path / name)}
        for name, expected in checksums.items()
        if file_sha256(output_path / name) != expected
    }
    if mismatches:
        raise JacobianPairAnalysisError(f"Output checksum mismatch: {mismatches}")
    if file_sha256(input_path) != EXPECTED_INPUT_SHA256:
        raise JacobianPairAnalysisError("Upstream input checksum changed")
    rows = _read_csv_rows(output_path / "top_mutual_sensitivity_pairs.csv")
    if len(rows) != TOP_PAIR_COUNT:
        raise JacobianPairAnalysisError("Top-pair CSV does not contain ten rows")
    if [int(row["rank"]) for row in rows] != list(
        range(1, TOP_PAIR_COUNT + 1)
    ):
        raise JacobianPairAnalysisError("Top-pair ranks are not contiguous")
    return {
        "status": "passed",
        "verified_file_count": len(checksums),
        "top_pair_count": len(rows),
        "input_sha256": EXPECTED_INPUT_SHA256,
        "output": _relative_or_absolute(output_path, paths.project_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument(
        "--expected-input-sha256", default=EXPECTED_INPUT_SHA256
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = (
        ProjectPaths.from_environment({"BAGM_ROOT": str(arguments.root)})
        if arguments.root is not None
        else current_paths(anchor=__file__)
    )
    paths.validate()
    input_path = (
        arguments.input.resolve(strict=False)
        if arguments.input is not None
        else paths.artifact_root / INPUT_RELATIVE
    )
    output_path = (
        arguments.output.resolve(strict=False)
        if arguments.output is not None
        else paths.report_root / OUTPUT_RELATIVE
    )
    evidence_path = (
        arguments.evidence.resolve(strict=False)
        if arguments.evidence is not None
        else paths.project_root / EVIDENCE_RELATIVE
    )
    result = (
        verify_existing_output(
            paths=paths, input_path=input_path, output_path=output_path
        )
        if arguments.verify_only
        else run_analysis(
            paths=paths,
            input_path=input_path,
            output_path=output_path,
            evidence_path=evidence_path,
            expected_input_sha256=arguments.expected_input_sha256,
        )
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
