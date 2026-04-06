# Experiment: CT depth tabular — Track A, frozen Ridge v1

**Task:** Regress **depth_min_cm** and **depth_max_cm** from tabular features (manifest from CT case metadata). **No images** in this experiment.

## At a glance

This line of work is **Track A (tabular only)**: predict a **recommended compression depth band** as two numbers, **minimum and maximum depth in centimetres**, from **demographics and case metadata** extracted into a manifest — **no chest images** and no video. It is intended as a **simple, interpretable baseline** and a placeholder until an imaging-based model exists.

**What ran:** On **204** manifest rows, a fixed **train / validation / test** split (**142 / 31 / 31**) was used for every model. Several **sklearn** pipelines were compared: a **per-sex mean** baseline (predict the average band seen in training for that sex), **Ridge** regression, **random forest**, **XGBoost**, and a small **sklearn MLP**. A **separate run** trained a few **PyTorch MLPs** on the **same split** for an extra comparison. Metrics are **MAE and RMSE** on each boundary (cm) and a **band inversion** count (cases where the predicted maximum depth is below the predicted minimum — undesirable).

**Headline outcome:** The **Ridge** pipeline (scaled age + one-hot sex, then **Ridge** per target) was **chosen for frozen v1** (`ct_depth_tabular_v1`, training run **`20260404_055703_utc`**). On the **held-out test** set (**31** rows), its **average of the two MAEs** (min and max depth) is about **0.054 cm**, with **no** band inversions. That edges out the **per-sex mean baseline** (about **0.0545 cm** mean MAE on the same test), while tree models and the default sklearn MLP were **clearly worse** on mean MAE in this table. **Interpretation:** errors on the order of **hundredths of a cm** on this tiny test set are **not** clinical proof of safety — the dataset is small and tabular-only; the value here is **relative ranking** and a **reproducible** frozen recipe.

**This folder:** Full tables below; **trained `.joblib` files are not shipped** here.

---

## 1. Frozen delivery (v1)

| Item | Value |
|------|--------|
| freeze_id | `ct_depth_tabular_v1` |
| frozen_at_utc | 2026-04-04 |
| Settled model key | `ridge` |
| Runner | sklearn |
| Implementation | `sklearn.pipeline.Pipeline`: `ColumnTransformer` (StandardScaler on `age_years`, OneHotEncoder on `sex`) + `MultiOutputRegressor(Ridge(alpha=1.0))` |
| Input features (v1) | `sex`, `age_years` only |
| Targets | `depth_min_cm`, `depth_max_cm` |
| Source training run_id | `20260404_055703_utc` |
| Frozen exports | `ridge.joblib` (delivered regressor), `mean_baseline.joblib` (reference only) — **not shipped** in this share |

---

## 2. How the settled model was chosen

Among all run/model entries from the training comparison:

1. Minimize **test_mae_mean_cm** (mean of MAE on `depth_min_cm` and `depth_max_cm` on the held-out test set).
2. Exclude **mean_baseline** from being selected as the settled regressor.
3. Tie-break: fewer **test_pred_band_inversions** (predictions where predicted max is less than predicted min).
4. Tie-break: newer **run_id** (stable sort).

**Settled result:**

| Field | Value |
|-------|--------|
| run_id | `20260404_055703_utc` |
| runner | sklearn |
| model | ridge |
| test_mae_mean_cm | 0.05401 |
| tie_break_note | Lowest test_mae_mean_cm; then fewer band inversions; then newest run_id (stable sort). |
| excluded_from_settled | mean_baseline |

---

## 3. Data split (reference sklearn run `20260404_055703_utc`)

| Split | n |
|------:|--:|
| Total | 204 |
| Train | 142 |
| Val | 31 |
| Test | 31 |
| mean_baseline_mode | per_sex |
| tree_backend (sklearn compare) | xgboost |
| random_state (stated in frozen record) | 42 |

---

## 4. Model definitions (sklearn comparison in source project)

Shared preprocessing: ColumnTransformer-style pipeline on tabular columns. For **frozen v1**, inputs are **sex** + **age_years** only (no engineered columns in that freeze).

| model key | Definition |
|-----------|------------|
| **mean_baseline** | No fitted regressor: predicts **per-sex** mean of `(depth_min_cm, depth_max_cm)` from training rows (mode `per_sex`). Used as reference; **not** eligible as settled model. |
| **ridge** | `MultiOutputRegressor(Ridge(alpha))` with **alpha = 1.0** (default from config), after ColumnTransformer. **Settled / frozen v1.** |
| **random_forest** | `RandomForestRegressor` (default **200** trees, optional `max_depth`, `random_state=42`, `n_jobs=-1`) wrapped in `MultiOutputRegressor`. |
| **xgboost** | `XGBRegressor` per target: default **300** estimators, **max_depth 4**, **learning_rate 0.05**, `random_state=42`, wrapped in `MultiOutputRegressor`. |
| **mlp** | `sklearn.neural_network.MLPRegressor`: default hidden **(64, 32)**, **max_iter 2000**, **early_stopping=True**, **validation_fraction 0.1**, wrapped in `MultiOutputRegressor`. |

---

## 5. Sklearn run — validation metrics (all models, run `20260404_055703_utc`)

| Model | MAE min | MAE max | RMSE min | RMSE max | mean MAE | band inv |
|-------|--------:|--------:|---------:|---------:|---------:|---------:|
| mean_baseline | 0.03383 | 0.04576 | 0.04068 | 0.05533 | 0.03980 | 0 |
| ridge | 0.03235 | 0.04542 | 0.03915 | 0.05467 | 0.03888 | 0 |
| random_forest | 0.04354 | 0.07042 | 0.05924 | 0.12045 | 0.05698 | 0 |
| xgboost | 0.04353 | 0.06621 | 0.06845 | 0.12446 | 0.05487 | 0 |
| mlp | 0.04288 | 0.06982 | 0.05566 | 0.08331 | 0.05635 | 0 |

(Units: cm for MAE/RMSE; **band inv** = count of rows where predicted max is less than predicted min.)

---

## 6. Sklearn run — test metrics (all models, run `20260404_055703_utc`)

| Model | MAE min | MAE max | RMSE min | RMSE max | mean MAE | band inv |
|-------|--------:|--------:|---------:|---------:|---------:|---------:|
| mean_baseline | 0.03344 | 0.07549 | 0.04076 | 0.18477 | 0.05446 | 0 |
| ridge | 0.03200 | 0.07602 | 0.03944 | 0.18271 | 0.05401 | 0 |
| random_forest | 0.03724 | 0.10593 | 0.04741 | 0.21664 | 0.07158 | 0 |
| xgboost | 0.03779 | 0.09733 | 0.04725 | 0.21789 | 0.06756 | 0 |
| mlp | 0.03965 | 0.09573 | 0.04695 | 0.18769 | 0.06769 | 0 |

---

## 7. Aggregate ranking (settled sklearn run + PyTorch run)

**Sklearn** models below are from **`20260404_055703_utc` only** (frozen Ridge source run). A second sklearn run on the same recipe duplicated these test numbers and is omitted here.

**PyTorch** models are from **`20260404_064815_utc_pytorch_tabular`**. Sorted by **test mean MAE (cm)** ascending.

| # | run_id | runner | model | test mean MAE (cm) | band inv |
|---|--------|--------|-------|-------------------:|---------:|
| 1 | 20260404_055703_utc | sklearn | ridge | 0.05401 | 0 |
| 2 | 20260404_064815_utc_pytorch_tabular | pytorch | mlp_shallow | 0.05430 | 0 |
| 3 | 20260404_055703_utc | sklearn | mean_baseline | 0.05446 | 0 |
| 4 | 20260404_064815_utc_pytorch_tabular | pytorch | mean_baseline | 0.05446 | 0 |
| 5 | 20260404_064815_utc_pytorch_tabular | pytorch | mlp_deep | 0.05486 | 0 |
| 6 | 20260404_064815_utc_pytorch_tabular | pytorch | mlp_wide | 0.06042 | 0 |
| 7 | 20260404_055703_utc | sklearn | xgboost | 0.06756 | 0 |
| 8 | 20260404_055703_utc | sklearn | mlp | 0.06769 | 0 |
| 9 | 20260404_055703_utc | sklearn | random_forest | 0.07158 | 0 |
| 10 | 20260404_064815_utc_pytorch_tabular | pytorch | mlp_medium | 0.07621 | 0 |

---

## 8. PyTorch MLP run (`20260404_064815_utc_pytorch_tabular`)

**Device:** xpu. **Split:** same counts as sklearn (204 total, 142 / 31 / 31). **Architecture:** fully connected MLP (ReLU, dropout) with two scalar outputs for min/max depth (cm).

### 8.1 Architecture by model key

| Model | hidden layers | dropout |
|-------|---------------|--------:|
| mean_baseline | — | — |
| mlp_shallow | [64, 32] | 0.10 |
| mlp_medium | [128, 64] | 0.15 |
| mlp_deep | [256, 128, 64] | 0.20 |
| mlp_wide | [512, 256] | 0.25 |

### 8.2 PyTorch — validation

| Model | MAE min | MAE max | RMSE min | RMSE max | mean MAE | band inv |
|-------|--------:|--------:|---------:|---------:|---------:|---------:|
| mean_baseline | 0.03383 | 0.04576 | 0.04068 | 0.05533 | 0.03980 | 0 |
| mlp_shallow | 0.04465 | 0.05098 | 0.05426 | 0.06086 | 0.04782 | 0 |
| mlp_medium | 0.03794 | 0.07901 | 0.04677 | 0.09344 | 0.05848 | 0 |
| mlp_deep | 0.04723 | 0.03981 | 0.05658 | 0.05121 | 0.04352 | 0 |
| mlp_wide | 0.05699 | 0.05062 | 0.07199 | 0.06085 | 0.05381 | 0 |

### 8.3 PyTorch — test (n_test = 31)

| Model | MAE min | MAE max | RMSE min | RMSE max | mean MAE | band inv |
|-------|--------:|--------:|---------:|---------:|---------:|---------:|
| mean_baseline | 0.03344 | 0.07549 | 0.04076 | 0.18477 | 0.05446 | 0 |
| mlp_shallow | 0.03962 | 0.06899 | 0.05112 | 0.17578 | 0.05430 | 0 |
| mlp_medium | 0.04922 | 0.10321 | 0.05564 | 0.18411 | 0.07621 | 0 |
| mlp_deep | 0.04025 | 0.06947 | 0.04740 | 0.18522 | 0.05486 | 0 |
| mlp_wide | 0.04658 | 0.07426 | 0.06254 | 0.17705 | 0.06042 | 0 |

---

## 9. Ridge evaluation snapshot (frozen record, test + val)

| Split | MAE min | MAE max | RMSE min | RMSE max | mean MAE | band inv |
|-------|--------:|--------:|---------:|---------:|---------:|---------:|
| test | 0.032 | 0.07602 | 0.03944 | 0.18271 | 0.05401 | 0 |
| val | 0.03235 | 0.04542 | 0.03915 | 0.05467 | 0.03888 | 0 |

---

## 10. Limitations (frozen baseline text)

- Not clinically validated; tabular-only (no images).
- Age parsed from filename stems; verify extraction in the training data pipeline when reproducing.
- A future imaging-based Track A model may replace this contract for a production imaging path.
