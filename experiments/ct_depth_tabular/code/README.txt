extract_manifest.py — build ct_depth_manifest.csv from data/ct_data_original/*.mrk.json
train_compare.py         — sklearn + xgboost baselines → runs/<utc>/
train_pytorch_tabular.py — PyTorch MLPs (cuda/xpu/cpu), same split → runs/<utc>_pytorch_tabular/
aggregate_ct_depth_runs.py — scan runs/*/metrics.json → aggregated/ + SETTLED_MODEL.md
freeze_ct_depth_baseline.py — copy champion joblibs → ../frozen/models/

Run from repository root. Delivery freeze: ../frozen/ — overview: ../EXPERIMENT.md — full spec: ../../docs/CT_DEPTH_TABULAR_TRACK_A.md
