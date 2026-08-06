#!/usr/bin/env python3
"""Verify the Jacobian biological-validation report and its provenance."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AuditHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.external_asset_urls: list[str] = []
        self.reference_links: set[str] = set()
        self.has_title = False
        self.has_main = False
        self.has_h1 = False
        self.has_table_caption = False
        self.open_tags: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = {key: value for key, value in attrs}
        element_id = attributes.get("id")
        if element_id:
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids.add(element_id)
        if tag == "title":
            self.has_title = True
        elif tag == "main":
            self.has_main = True
        elif tag == "h1":
            self.has_h1 = True
        elif tag == "caption":
            self.has_table_caption = True
        if tag in {"img", "script", "iframe", "video", "audio", "source"}:
            source = attributes.get("src")
            if source and not source.startswith("data:"):
                self.external_asset_urls.append(source)
        if tag == "link":
            href = attributes.get("href")
            rel = (attributes.get("rel") or "").lower()
            if href and ("stylesheet" in rel or "icon" in rel):
                self.external_asset_urls.append(href)
        if tag == "a":
            href = attributes.get("href") or ""
            if href.startswith("#ref-"):
                self.reference_links.add(href[1:])


def all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return True


def main() -> None:
    required = [
        REPORT_DIR / "README.md",
        REPORT_DIR / "sources.json",
        REPORT_DIR / "generate_report.py",
        REPORT_DIR / "evidence.json",
        REPORT_DIR / "report.html",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing report artifacts: {missing}")

    evidence = json.loads(
        (REPORT_DIR / "evidence.json").read_text(encoding="utf-8")
    )
    sources = json.loads(
        (REPORT_DIR / "sources.json").read_text(encoding="utf-8")
    )
    report_text = (REPORT_DIR / "report.html").read_text(encoding="utf-8")

    checks: dict[str, bool] = {}
    checks["evidence_schema"] = (
        evidence.get("schema_version") == 1
        and evidence.get("artifact_kind")
        == "jacobian_biological_validation_audit"
    )
    checks["verdict_is_limited"] = (
        evidence.get("verdict", {}).get("entrywise_biological_test")
        == "not_testable_from_current_artifacts"
        and evidence.get("verdict", {}).get("biological_correspondence")
        == "not_established"
        and "does not establish that no biological correspondence exists"
        in evidence.get("verdict", {}).get("claim_not_made", "")
    )
    checks["no_entrywise_claim"] = (
        evidence.get("jacobian_contract", {}).get("entrywise_axes_retained")
        is False
        and evidence.get("jacobian_contract", {}).get("gene_ranks_retained")
        is False
        and evidence.get("jacobian_contract", {}).get(
            "gene_pair_ranks_retained"
        )
        is False
    )
    checks["matrix_arithmetic"] = (
        evidence["dense_matrix_scale"]["dense_entries"]
        == evidence["dense_matrix_scale"]["output_coordinates"]
        * evidence["dense_matrix_scale"]["effective_input_coordinates"]
        and evidence["dense_matrix_scale"]["dense_fp32_bytes"]
        == 4 * evidence["dense_matrix_scale"]["dense_entries"]
    )
    checks["finite_numeric_payload"] = all_finite(evidence)
    checks["aggregate_privacy"] = (
        evidence.get("privacy", {}).get("aggregate_only") is True
        and evidence.get("privacy", {}).get("row_level_data_loaded") is False
        and evidence.get("privacy", {}).get("protected_identifiers_emitted")
        is False
        and evidence.get("privacy", {}).get("external_upload_of_project_data")
        is False
    )

    input_checksum_checks = {}
    for relative_path, expected in evidence.get("input_checksums", {}).items():
        path = ROOT / relative_path
        input_checksum_checks[relative_path] = (
            path.is_file() and sha256(path) == expected
        )
    checks["all_input_checksums_match"] = bool(input_checksum_checks) and all(
        input_checksum_checks.values()
    )

    parser = AuditHTMLParser()
    parser.feed(report_text)
    parser.close()
    required_ids = {
        "verdict",
        "computed",
        "fitness",
        "biology",
        "limits",
        "next",
        "provenance",
    }
    source_ids = {f"ref-{row['id']}" for row in sources["sources"]}
    checks["html_semantics"] = (
        parser.has_title
        and parser.has_main
        and parser.has_h1
        and parser.has_table_caption
        and required_ids.issubset(parser.ids)
        and not parser.duplicate_ids
    )
    checks["portable_no_external_assets"] = not parser.external_asset_urls
    checks["all_source_anchors_present"] = source_ids.issubset(parser.ids)
    checks["all_citations_resolve"] = parser.reference_links.issubset(
        parser.ids
    ) and bool(parser.reference_links)
    checks["critical_text_present"] = all(
        phrase in report_text
        for phrase in [
            "Not testable from current artifacts",
            "no Jacobian matrix, gene ranking, or",
            "trace-level summaries, not high entries",
            "This is not a finding of no biology",
            "Joint TLS dependency support",
            "Do not reinterpret the current trace estimates",
        ]
    )
    checks["reasonable_report_size"] = (
        25_000 <= len(report_text.encode("utf-8")) <= 2_000_000
    )

    valid = all(checks.values())
    payload = {
        "schema_version": 1,
        "artifact_kind": "jacobian_biological_validation_report_verification",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "checks": checks,
        "input_checksum_checks": input_checksum_checks,
        "report": {
            "path": str((REPORT_DIR / "report.html").relative_to(ROOT)),
            "bytes": (REPORT_DIR / "report.html").stat().st_size,
            "sha256": sha256(REPORT_DIR / "report.html"),
            "external_asset_urls": parser.external_asset_urls,
            "ids": sorted(parser.ids),
        },
        "artifact_checksums": {
            str(path.relative_to(ROOT)): sha256(path) for path in required
        },
    }
    (REPORT_DIR / "verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    if not valid:
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"report verification failed: {failed}")


if __name__ == "__main__":
    main()
