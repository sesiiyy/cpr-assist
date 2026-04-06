# CT depth tabular — combined ranking (test mean MAE)

**Sklearn** rows come from the **single settled training run** `20260404_055703_utc` (same run as frozen Ridge v1). An earlier sklearn run on the same recipe was omitted because it duplicated these test metrics.

**PyTorch** rows come from `20260404_064815_utc_pytorch_tabular` (same manifest split, different code path).

Sorted by **test mean MAE (cm)** ascending.

| run_id | runner | model | test mean MAE (cm) | band inv | MC std mean (cm) |
|---|---|---|---:|---|---:|
| `20260404_055703_utc` | sklearn | ridge | 0.05401 | 0 | — |
| `20260404_064815_utc_pytorch_tabular` | pytorch | mlp_shallow | 0.05430 | 0 | — |
| `20260404_055703_utc` | sklearn | mean_baseline | 0.05446 | 0 | — |
| `20260404_064815_utc_pytorch_tabular` | pytorch | mean_baseline | 0.05446 | 0 | — |
| `20260404_064815_utc_pytorch_tabular` | pytorch | mlp_deep | 0.05486 | 0 | — |
| `20260404_064815_utc_pytorch_tabular` | pytorch | mlp_wide | 0.06042 | 0 | — |
| `20260404_055703_utc` | sklearn | xgboost | 0.06756 | 0 | — |
| `20260404_055703_utc` | sklearn | mlp | 0.06769 | 0 | — |
| `20260404_055703_utc` | sklearn | random_forest | 0.07158 | 0 | — |
| `20260404_064815_utc_pytorch_tabular` | pytorch | mlp_medium | 0.07621 | 0 | — |
