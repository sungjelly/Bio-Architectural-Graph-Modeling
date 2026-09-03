# Curated BAGM Results

This directory is the human-facing index of important BAGM conclusions. It is
not the raw output directory and not a second experiment registry.

Canonical execution bundles remain under `artifacts/runs/`. Detailed cross-run
analyses and numeric source data remain under `reports/`. A result record here
summarizes what the evidence supports and points back to those sources by
immutable ID and checksum.

## Browse

Start with [`CATALOG.md`](CATALOG.md). Result records are ordered as:

```text
<experiment_type>/<method_family>/<result_id>/
```

Examples of experiment types include `predictive_benchmark`, `ablation`,
`stability_audit`, `faithfulness_audit`, `null_calibration`,
`niche_discovery`, and `external_validation`. The complete controlled list is
in `configs/schema/result_record_v1.yaml`.

Each result directory contains:

- `result.yaml`: authoritative structured conclusion and provenance;
- `README.md`: concise human-readable question, result, evidence, limitations,
  and reproduction path;
- `figures/`, `tables/`, and `attachments/`: optional curated payloads declared
  with size and SHA-256 in `result.yaml`.

The preserved `core_assignment/` directory predates this contract and is not a
curated conclusion record unless it is explicitly migrated through a new
validated result.

## Create a result

From `/workspace/BAGM`:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py create \
  --result-id res_example_graph_gain \
  --title "Example graph-specific predictive gain" \
  --experiment-type predictive_benchmark \
  --method-family relative_qkv \
  --lifecycle-stage validation_confirmation \
  --study-axis graph_specific_gain \
  --campaign-id cmp_example
```

The command creates a draft with visible `TODO` fields. A draft may be used to
assemble evidence, but it cannot be marked `verified` until every required
statement and evidence dimension is complete, sources are checksum-bound, and
the human-readable README has no placeholders.

## Validate and catalog

```bash
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog --check
```

Use `--verify-sources` and `--verify-payloads` when the referenced local files
are mounted. Structural validation remains portable when large ignored
payloads or externally rooted artifact stores are unavailable.

The detailed contract, promotion criteria, correction policy, and evidence
boundaries are in [`docs/result_management.md`](../docs/result_management.md).
