# CT depth band — tabular extraction and 5-model comparison

**Full reference (data format, algorithms, metrics, artifacts):** [`docs/CT_DEPTH_TABULAR_TRACK_A.md`](../../docs/CT_DEPTH_TABULAR_TRACK_A.md)

## Frozen delivery (v1)

The **deliverable baseline** for this experiment is **locked** under **`frozen/`**:

- **`frozen/FROZEN_BASELINE.json`** — champion (**sklearn ridge**), metric snapshot, reproduction commands.
- **`config/frozen_baseline_v1.yaml`** — pinned training recipe (**sex + age_years** only; matches the settled run).
- **`SETTLED_MODEL.md`** — short summary; **`frozen/README.md`** — delivery checklist.
- **`code/freeze_ct_depth_baseline.py`** — copy **`ridge.joblib`** / **`mean_baseline.joblib`** from a run into **`frozen/models/`**.

Exploratory training may use **`config/default.yaml`** (e.g. engineered features, PyTorch); **v1 delivery** must be reproduced with **`frozen_baseline_v1.yaml`** unless you intentionally bump **`freeze_id`** and document a new version.

## Purpose

Build a **tabular** dataset from 3D Slicer **`.mrk.json`** markups under `data/ct_data_original/` (Male/Female folders), then compare **five predictors** of **`depth_min_cm`** and **`depth_max_cm`** from **sex** and **age** only. No CT images are used.

This supports a **Track A placeholder** until an image-based model exists. It does **not** integrate with `src/` yet; fusion still expects `target_lower_cm` / `target_upper_cm` from config or a future loader.

## Data source

- **Input:** Line markup files named with `minimum depth` / `maximum depth` (case/spacing variants tolerated). **`measurements[].length`** is in **mm**.
- **Sex:** folder name `Male` / `Female`.
- **Age:** parsed from the filename stem (leading digits before `y`/`Y`); ambiguous stems keep full `case_key` for audit.

**Privacy:** manifests list relative paths to markup files — avoid committing CSVs if your policy treats paths as sensitive.

## Configuration

Single file: [`config/default.yaml`](config/default.yaml).

| Section | Purpose |
|---------|---------|
| `ct_original_root`, `extract.*`, `output.*` | Extraction paths and pairing rules |
| `train.*` | Manifest path, split fractions, seed, `use_engineered_features` (`age_sq`, `male_age`), sklearn hyperparams, `tree_backend` |
| `pytorch_tabular.*` | Batch size, epochs, patience, LR, `device`, `mc_dropout_samples`, `head` (`plain` or `ordered`), `architectures` |
| `runs_dir` | Where timestamped run folders are created |

## Reproduce

From repository root (`cpr/`):

```powershell
# 1) Build manifest + extraction report
python experiments/ct_depth_tabular/code/extract_manifest.py `
  --config experiments/ct_depth_tabular/config/default.yaml

# 2) Train & compare models (writes runs/<utc>/)
python experiments/ct_depth_tabular/code/train_compare.py `
  --config experiments/ct_depth_tabular/config/default.yaml

# 3) Optional — PyTorch MLP sweep (same CSV & split; no images). Device: CUDA → XPU → CPU
python experiments/ct_depth_tabular/code/train_pytorch_tabular.py `
  --config experiments/ct_depth_tabular/config/default.yaml

# 4) Aggregate all runs — ranked test MAE + recommended model (see SETTLED_MODEL.md)
python experiments/ct_depth_tabular/code/aggregate_ct_depth_runs.py

# 5) (Delivery) Copy champion joblibs into frozen/models/ after train_compare
python experiments/ct_depth_tabular/code/freeze_ct_depth_baseline.py
```

**Ordered band head:** PyTorch models with `head: ordered` predict `max = min + softplus(δ)` so the band never inverts. **MC dropout** (`mc_dropout_samples` > 0): test metrics use the mean of stochastic forward passes; `test_mc_dropout_std_mean_cm` summarizes spread (requires `dropout` > 0).

Tune MLP depth/dropout under `pytorch_tabular` in the YAML. Set `pytorch_tabular.device: cpu` to force CPU.

Optional: `tree_backend: histgb` in YAML to avoid the `xgboost` dependency (sklearn only).

## Outputs

| Artifact | Location |
|----------|----------|
| Paired cases CSV | `experiments/ct_depth_tabular/data/ct_depth_manifest.csv` (gitignored) |
| Parse/pair report | `experiments/ct_depth_tabular/data/extraction_report.json` (gitignored) |
| Metrics + table | `runs/<run_id>/metrics.json`, `comparison_table.md` (sklearn) or `comparison_table_pytorch.md` (torch) |
| Fitted models | `experiments/ct_depth_tabular/runs/<run_id>/models/*.joblib` |
| PyTorch runs | `.../runs/<run_id>_pytorch_tabular/models/*.pt` + `preprocessor.joblib` |
| **Frozen v1 (delivery)** | `frozen/FROZEN_BASELINE.json`, `config/frozen_baseline_v1.yaml`, `frozen/models/*.joblib` (after freeze script) |

## Models

1. **Mean baseline** — train-set mean(s) of both targets (global or per sex).
2. **Ridge** — `MultiOutputRegressor(Ridge)`.
3. **RandomForestRegressor** — native multi-output.
4. **XGBRegressor** or **HistGradientBoostingRegressor** — `MultiOutputRegressor` wrapper (`tree_backend` in YAML).
5. **MLPRegressor** — `MultiOutputRegressor` for two heads.

**PyTorch (optional):** several **MLP** configs (AdamW + early stopping); optional **ordered** head and **MC dropout** uncertainty on the test split; same targets, **no images**.

## Limitations

- **Not clinically validated** — research / engineering baseline only.
- **Age from filename** is fragile; verify `extraction_report.json` for parse warnings.
- **Tabular-only** — no appearance; runtime RGB→band is a separate future model.

## Optional project copy

Copy `ct_depth_manifest.csv` to `datasets/ct_data/` if you want a single project-level table (create folder if missing; `datasets/` may be gitignored at repo root).
