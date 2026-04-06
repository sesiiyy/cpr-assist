Frozen baseline v1 — model binaries (optional local copy)

Expected after running (from repo root):

  python experiments/ct_depth_tabular/code/freeze_ct_depth_baseline.py

Files:
  ridge.joblib         — champion Pipeline (preprocess + MultiOutput Ridge)
  mean_baseline.joblib — reference baseline only (per_sex means)
  config_resolved.yaml — copy of the YAML used for the source run

These paths are gitignored by default so large binaries are not committed unless your policy allows it.
See ../FROZEN_BASELINE.json and ../README.md.
