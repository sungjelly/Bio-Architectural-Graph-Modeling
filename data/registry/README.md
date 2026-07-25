# Dataset and Split Registry

These tracked YAML files contain non-identifying metadata only. They describe
immutable local inputs and split contracts; they are not copies of the data.
Protected paths resolve locally and must never be uploaded.

`datasets.yaml` records dataset versions, aggregate counts, schemas, and
deterministic fingerprints. `splits.yaml` records the unit, method, seed,
constraints, and fingerprint for each split. Direct patient, donor, cell, FOV,
and restricted clinical identifiers are prohibited. A hash of a direct
identifier is also prohibited unless it is generated with an untracked secret
and cannot be reversed by dictionary matching.

Fingerprint rules are deterministic:

1. Normalize every path relative to the declared protected root.
2. Sort records lexicographically by normalized path or declared input role.
3. Include the algorithm version, byte size, and SHA-256 for every file.
4. Serialize as sorted compact JSON and hash the UTF-8 bytes with SHA-256.
5. For a split, include ordered assignment keys, split labels, split method,
   seed, grouping unit, and version. Never write those assignment keys into the
   tracked registry.

The full-snapshot fingerprint records a deterministic sorted manifest of the
protected local raw files; the protected normal-core fingerprints were
verified from the archived preparation manifest. Recalculate fingerprints
through the project fingerprint utility before a conclusion-bearing run. A
changed source, preprocessing version, assignment, or fingerprint requires a
new version or split ID rather than an in-place registry rewrite.

See `docs/data_management.md` for access, immutability, leakage, and backup
rules.
