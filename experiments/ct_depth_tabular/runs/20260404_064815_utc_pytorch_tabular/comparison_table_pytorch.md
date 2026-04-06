# CT depth tabular — PyTorch MLP comparison

Run `20260404_064815_utc_pytorch_tabular` | device `xpu` | test n = 31

| Model | MAE min (cm) | MAE max (cm) | RMSE min | RMSE max | mean MAE | band inversions |
|---|---:|---:|---:|---:|---:|---:|
| mean_baseline | 0.03344 | 0.07549 | 0.04076 | 0.18477 | 0.05446 | 0 |
| mlp_shallow | 0.03962 | 0.06899 | 0.05112 | 0.17578 | 0.0543 | 0 |
| mlp_medium | 0.04922 | 0.10321 | 0.05564 | 0.18411 | 0.07621 | 0 |
| mlp_deep | 0.04025 | 0.06947 | 0.0474 | 0.18522 | 0.05486 | 0 |
| mlp_wide | 0.04658 | 0.07426 | 0.06254 | 0.17705 | 0.06042 | 0 |
