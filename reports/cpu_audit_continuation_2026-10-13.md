# Read-only real-data audit continuation — 2026-10-13

The passive gate now rejects nonfinite or nonmonotone timestamps, noninteger
episode IDs, and discontiguous repeated episodes before fitting lag hypotheses.
The adapter validates every shard's required trajectory columns before loading
any data; a missing column no longer falls back to reading image/blob columns.

The Hub driver requires an immutable 40-character revision label and a harvest
`source_sha256` matching the local trajectory content. Harvest entries are keyed
by full Hub repository ID; a basename match is insufficient. These checks bind
local content to the supplied harvest, not to a verified remote revision. Reports
mark remote revision binding and licensing `UNVERIFIED`, even when a license is
declared in the harvest. A hash is not license evidence.

The driver rejects CLI work inside the dataset or repository source tree, reused
work/output directories, existing evidence and dangling output symlinks. Dataset
directory aliases and normalized paths are resolved before writes; aggregate and
dataset output collisions and overlapping work/report destinations are rejected.
JSON evidence uses exclusive creation, and direct CLI corroboration also requires
a fresh work directory. Missing/failed datasets remain in the requested
denominator and produce exit 2, rather than a successful empty audit.

`results/cpu_availability_2026-10-13.json` records the fresh local availability
check: 29 requested Hub datasets, 29 missing, 0 audited. The historical 29-dataset
Hub JSON and the six-dataset baseline are preserved; they are not fresh inference
results. No raw trajectory Parquet was found at the checked local `data` path or
historical WSL `/tmp/hub_audit/data` path. This is a bounded path check, not an
exhaustive filesystem search. Licensing, real nuisance-inference outcomes,
trajectory conversion effects and real-data CLI/pybind agreement remain unrun.
Converters continue to be blocked in Hub reports.

Focused reproduction uses the task-local environment at
`../execution/CPU-2026-10-13/env` (relative to the repositories' parent directory):

```powershell
python -m pytest -q tests/test_real_audit_integrity.py -p no:dash
python experiments/run_hub_audit.py --data-root data --harvest PATH_TO_HARVEST.json --no-cli `
  --out-dir NEW_OUTPUT_DIRECTORY --audit-json NEW_REPORT.json
```

The second command needs a harvest JSON (an empty object suffices for the
missing-input check). Use fresh outputs. Checks on synthetic boundary cases
validate rejection behavior; they do not establish identification quality on
unavailable real trajectories.
