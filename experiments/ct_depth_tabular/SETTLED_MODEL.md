# Settled / frozen model — CT depth tabular (Track A placeholder)

**Delivery record:** **`frozen/FROZEN_BASELINE.json`** (`freeze_id`: **`ct_depth_tabular_v1`**). Human-oriented notes: **`frozen/README.md`**.

## Champion (locked for v1)

| Field | Value |
|-------|--------|
| **Model** | sklearn **`ridge`** — `Pipeline` with `ColumnTransformer` + `MultiOutputRegressor(Ridge(alpha=1.0))` |
| **Inputs** | **`sex`**, **`age_years`** (v1 recipe: no engineered columns; see `config/frozen_baseline_v1.yaml`) |
| **Outputs** | **`depth_min_cm`**, **`depth_max_cm`** |
| **Reference run** | `20260404_055703_utc` |
| **Test mean MAE** | **0.05401 cm** (average of per-target MAEs on held-out test, **n_test = 31**) |
| **Band inversions (test)** | **0** |

**Mean baseline** (`per_sex`) is **not** the delivered regressor; it remains for comparison only.

## How this was chosen

`aggregate_ct_depth_runs.py` ranks all entries in `runs/*/metrics.json` by **`test_mae_mean_cm`**, excludes **`mean_baseline`** from the “settled” pick, then breaks ties by fewer **`test_pred_band_inversions`**, then newest **`run_id`**. Full table: **`aggregated/comparison_all_runs.md`**.

## Reproduce and export joblibs

```powershell
python experiments/ct_depth_tabular/code/extract_manifest.py --config experiments/ct_depth_tabular/config/frozen_baseline_v1.yaml
python experiments/ct_depth_tabular/code/train_compare.py --config experiments/ct_depth_tabular/config/frozen_baseline_v1.yaml
python experiments/ct_depth_tabular/code/freeze_ct_depth_baseline.py
```

That copies **`ridge.joblib`**, **`mean_baseline.joblib`**, and **`config_resolved.yaml`** into **`frozen/models/`** (see **`frozen/models/README.txt`**).

**Note:** `config/default.yaml` may differ from v1 (e.g. engineered features). For **delivery parity** with this freeze, always use **`frozen_baseline_v1.yaml`**.
