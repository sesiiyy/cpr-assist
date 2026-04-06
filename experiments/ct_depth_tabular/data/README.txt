Generated tabular outputs from extract_manifest.py:

  ct_depth_manifest.csv   — one row per paired case (sex, age, age_sq, male_age, depth band in cm)
  extraction_report.json — pairing skips, parse errors, unpaired keys

These files are gitignored by default. Run from repo root:

  python experiments/ct_depth_tabular/code/extract_manifest.py --config experiments/ct_depth_tabular/config/default.yaml

Optional: copy the CSV to datasets/ct_data/ for project-wide use.

Specification: ../../docs/CT_DEPTH_TABULAR_TRACK_A.md
