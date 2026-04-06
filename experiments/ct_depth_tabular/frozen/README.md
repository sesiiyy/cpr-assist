# Frozen Track A tabular baseline (delivery package)

This folder **locks the delivered outcome** of the offline CT depth tabular experiment: which model won, on what data split metrics, and how to reproduce it.

| File | Purpose |
|------|---------|
| **`FROZEN_BASELINE.json`** | Machine-readable freeze record (champion, metrics snapshot, reproduction commands). |
| **`models/README.txt`** | Where `ridge.joblib` / `mean_baseline.joblib` land after the freeze script. |
| **`../config/frozen_baseline_v1.yaml`** | Pinned config for v1 (**sex + age_years** only; matches the settled run). |

## Champion (v1)

- **Model:** sklearn **`ridge`** — `MultiOutputRegressor(Ridge(alpha=1.0))` inside a `Pipeline` with scaled **`age_years`** and one-hot **`sex`**.
- **Reference run:** `20260404_055703_utc` (see `FROZEN_BASELINE.json` for embedded test metrics).
- **Not chosen for delivery:** `mean_baseline` (comparison only); trees/MLP scored worse on this split (see `aggregated/comparison_all_runs.md`).

## Populate `models/` (joblib)

After a successful `train_compare.py` run that still exists under `runs/<id>/models/`:

```powershell
python experiments/ct_depth_tabular/code/freeze_ct_depth_baseline.py
```

Defaults read **`aggregated/settled_model.json`** (or **`FROZEN_BASELINE.json`**) for `run_id` and copy **`ridge.joblib`**, **`mean_baseline.joblib`**, and **`config_resolved.yaml`** into **`frozen/models/`**.

## Main CPR pipeline (`src/`)

After **`freeze_ct_depth_baseline.py`** has populated **`models/ridge.joblib`**, set patient demographics in **`configs/default.yaml`** under **`track_a.frozen_tabular_v1.patient`** (`sex`: **`male`** | **`female`**, **`age_years`**). Every **`load_config()`** call applies the Ridge model and sets **`track_a.target_lower_cm`** / **`target_upper_cm`** / **`confidence`** for fusion and logging. See **`src/track_a/frozen_tabular.py`**.

## Bumping the freeze

Change **`freeze_id`** and **`FROZEN_BASELINE.json`** fields together; add **`frozen_baseline_v2.yaml`** if training recipe changes. Do not edit v1 JSON after delivery—add a new version.
