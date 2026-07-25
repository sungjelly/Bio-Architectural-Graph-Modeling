# Infrastructure implementation contract

Objective: provide deterministic project paths, configuration composition,
identifiers, data/split fingerprints, prediction validation, and immutable run
finalization without changing scientific model behavior.

This is operational infrastructure, not a biological experiment. It has no
scientific estimand or biological claim. The principal alternatives considered
are accidental dependence on the working directory, configuration ambiguity,
identifier collisions, protected-identifier leakage, and partially finalized
runs. Acceptance requires deterministic unit fixtures, strict failure on those
conditions, preservation of failed runs, and no GPU or training execution.

Verification:

```bash
PYTHONPATH=src pytest -q tests/unit/infrastructure
```
