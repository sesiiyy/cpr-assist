#!/usr/bin/env python3
"""
Train five tabular baselines on ct_depth_manifest.csv; write metrics, comparison table, joblib models.

Usage (repo root)::

  python experiments/ct_depth_tabular/code/train_compare.py \\
    --config experiments/ct_depth_tabular/config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import tabular_common as tabc  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae_min = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_max = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    rmse_min = _rmse(y_true[:, 0], y_pred[:, 0])
    rmse_max = _rmse(y_true[:, 1], y_pred[:, 1])
    inv = int(np.sum(y_pred[:, 1] < y_pred[:, 0]))
    return {
        "test_mae_depth_min_cm": round(mae_min, 5),
        "test_mae_depth_max_cm": round(mae_max, 5),
        "test_rmse_depth_min_cm": round(rmse_min, 5),
        "test_rmse_depth_max_cm": round(rmse_max, 5),
        "test_mae_mean_cm": round((mae_min + mae_max) / 2, 5),
        "test_pred_band_inversions": inv,
    }


def predict_mean_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    mode: str,
) -> np.ndarray:
    if mode == "global":
        mu = y_train.mean(axis=0)
        return np.tile(mu, (len(X_test), 1))
    if mode == "per_sex":
        out = np.zeros((len(X_test), 2), dtype=np.float64)
        df_tr = X_train.copy()
        df_tr["_ymin"] = y_train[:, 0]
        df_tr["_ymax"] = y_train[:, 1]
        stats = df_tr.groupby("sex")[["_ymin", "_ymax"]].mean()
        global_mu = y_train.mean(axis=0)
        for i, sx in enumerate(X_test["sex"].values):
            if sx in stats.index:
                out[i, 0] = stats.loc[sx, "_ymin"]
                out[i, 1] = stats.loc[sx, "_ymax"]
            else:
                out[i] = global_mu
        return out
    raise ValueError(f"Unknown mean_baseline mode: {mode}")


def split_three_way(
    df: pd.DataFrame,
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    group_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train_indices, val_indices, test_indices as positions into df (iloc)."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1"
    n = len(df)
    idx = np.arange(n)

    if group_col and not df[group_col].duplicated().any():
        group_col = None

    if group_col and df[group_col].nunique() < n:
        groups = df[group_col].values
        unique = np.unique(groups)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(unique))
        u = unique[order]
        n_g = len(u)
        n_test = max(1, int(round(test_frac * n_g)))
        n_val = max(1, int(round(val_frac * n_g)))
        n_train = n_g - n_test - n_val
        if n_train < 1:
            n_train = 1
            n_val = max(1, n_g - n_test - n_train)
        test_set = set(u[:n_test])
        val_set = set(u[n_test : n_test + n_val])
        train_set = set(u[n_test + n_val :])
        mask_train = np.array([g in train_set for g in groups])
        mask_val = np.array([g in val_set for g in groups])
        mask_test = np.array([g in test_set for g in groups])
        return idx[mask_train], idx[mask_val], idx[mask_test]

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, random_state=seed, shuffle=True
    )
    rel_val = val_frac / (train_frac + val_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=rel_val, random_state=seed, shuffle=True
    )
    return idx_train, idx_val, idx_test


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline | None  # None for mean baseline


def build_models(cfg: dict[str, Any]) -> list[ModelSpec]:
    t = cfg.get("train", {}) or {}
    tree_backend = str(t.get("tree_backend", "xgboost")).lower()
    use_eng = bool(t.get("use_engineered_features", True))

    def new_prep():
        return tabc.build_preprocessor(use_eng)

    specs: list[ModelSpec] = []

    specs.append(ModelSpec("mean_baseline", None))

    ridge = MultiOutputRegressor(Ridge(alpha=float(t.get("ridge_alpha", 1.0))))
    specs.append(
        ModelSpec(
            "ridge",
            Pipeline([("prep", new_prep()), ("reg", ridge)]),
        )
    )

    rf = RandomForestRegressor(
        n_estimators=int(t.get("rf_n_estimators", 200)),
        max_depth=t.get("rf_max_depth"),
        random_state=int(t.get("random_state", 42)),
        n_jobs=-1,
    )
    specs.append(ModelSpec("random_forest", Pipeline([("prep", new_prep()), ("reg", rf)])))

    if tree_backend == "histgb":
        hgb = HistGradientBoostingRegressor(
            max_iter=int(t.get("xgb_n_estimators", 300)),
            max_depth=t.get("xgb_max_depth"),
            learning_rate=float(t.get("xgb_learning_rate", 0.05)),
            random_state=int(t.get("random_state", 42)),
        )
        specs.append(
            ModelSpec(
                "hist_gradient_boosting",
                Pipeline([("prep", new_prep()), ("reg", MultiOutputRegressor(hgb))]),
            )
        )
    else:
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "tree_backend is xgboost but xgboost is not installed. "
                "pip install xgboost or set train.tree_backend: histgb in YAML."
            ) from e
        xgb = XGBRegressor(
            n_estimators=int(t.get("xgb_n_estimators", 300)),
            max_depth=int(t.get("xgb_max_depth", 4)),
            learning_rate=float(t.get("xgb_learning_rate", 0.05)),
            random_state=int(t.get("random_state", 42)),
            n_jobs=-1,
            verbosity=0,
        )
        specs.append(
            ModelSpec(
                "xgboost",
                Pipeline([("prep", new_prep()), ("reg", MultiOutputRegressor(xgb))]),
            )
        )

    hidden = t.get("mlp_hidden", [64, 32])
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(int(x) for x in hidden),
        max_iter=int(t.get("mlp_max_iter", 2000)),
        random_state=int(t.get("random_state", 42)),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    specs.append(
        ModelSpec(
            "mlp",
            Pipeline([("prep", new_prep()), ("reg", MultiOutputRegressor(mlp))]),
        )
    )

    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/compare CT depth tabular models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_CODE_DIR.parent / "config" / "default.yaml",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load data and split only.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("ct_depth_train")

    cfg = load_config(args.config)
    t = cfg.get("train", {}) or {}
    manifest_path = _REPO_ROOT / str(t.get("manifest_csv", "experiments/ct_depth_tabular/data/ct_depth_manifest.csv"))
    if not manifest_path.is_file():
        log.error("Manifest missing: %s — run extract_manifest.py first", manifest_path)
        return 1

    df = pd.read_csv(manifest_path)
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    if bool(cfg.get("extract", {}).get("drop_missing_age", False)):
        df = df.dropna(subset=["age_years"])
    df = df.dropna(subset=["age_years", "depth_min_cm", "depth_max_cm", "sex"])
    df["sex"] = df["sex"].astype(str).str.lower()
    df = tabc.ensure_engineered_features(df)

    use_eng = bool(t.get("use_engineered_features", True))
    x_cols = tabc.x_feature_columns(use_eng)
    y = df[["depth_min_cm", "depth_max_cm"]].values.astype(np.float64)
    X = df[x_cols].copy()

    seed = int(t.get("random_state", 42))
    train_frac = float(t.get("train_frac", 0.70))
    val_frac = float(t.get("val_frac", 0.15))
    test_frac = float(t.get("test_frac", 0.15))
    group_col = "case_key" if bool(t.get("group_split_by_case_key", False)) else None

    idx_train, idx_val, idx_test = split_three_way(
        df, y, train_frac, val_frac, test_frac, seed, group_col
    )

    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    log.info(
        "rows=%d train=%d val=%d test=%d manifest=%s",
        len(df),
        len(idx_train),
        len(idx_val),
        len(idx_test),
        manifest_path.relative_to(_REPO_ROOT),
    )

    if args.dry_run:
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    runs_root = _REPO_ROOT / str(cfg.get("runs_dir", "experiments/ct_depth_tabular/runs"))
    run_dir = runs_root / run_id
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)

    shutil.copy2(args.config, run_dir / "config_resolved.yaml")

    mean_mode = str(t.get("mean_baseline", "per_sex")).lower()
    specs = build_models(cfg)
    log.info(
        "run_dir=%s rows=%d train=%d val=%d test=%d models=%d",
        run_dir.relative_to(_REPO_ROOT),
        len(df),
        len(idx_train),
        len(idx_val),
        len(idx_test),
        len(specs),
    )

    results: dict[str, Any] = {
        "run_id": run_id,
        "runner": "sklearn",
        "use_engineered_features": bool(t.get("use_engineered_features", True)),
        "n_total": len(df),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
        "mean_baseline_mode": mean_mode,
        "tree_backend": str(t.get("tree_backend", "xgboost")),
        "models": {},
    }

    md_lines = [
        "# CT depth tabular — model comparison",
        "",
        f"Run `{run_id}` | test n = {len(idx_test)}",
        "",
        "| Model | MAE min (cm) | MAE max (cm) | RMSE min | RMSE max | mean MAE | band inversions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for spec in specs:
        if spec.name == "mean_baseline":
            y_pred_test = predict_mean_baseline(X_train, y_train, X_test, mean_mode)
            y_pred_val = predict_mean_baseline(X_train, y_train, X_val, mean_mode)
            joblib.dump({"type": "mean_baseline", "mode": mean_mode}, models_dir / "mean_baseline.joblib")
        else:
            assert spec.pipeline is not None
            spec.pipeline.fit(X_train, y_train)
            y_pred_val = spec.pipeline.predict(X_val)
            y_pred_test = spec.pipeline.predict(X_test)
            joblib.dump(spec.pipeline, models_dir / f"{spec.name}.joblib")

        val_blk = _metrics_block(y_val, y_pred_val)
        test_blk = _metrics_block(y_test, y_pred_test)
        results["models"][spec.name] = {"val": val_blk, "test": test_blk}

        te = test_blk
        md_lines.append(
            f"| {spec.name} | {te['test_mae_depth_min_cm']} | {te['test_mae_depth_max_cm']} | "
            f"{te['test_rmse_depth_min_cm']} | {te['test_rmse_depth_max_cm']} | "
            f"{te['test_mae_mean_cm']} | {te['test_pred_band_inversions']} |"
        )

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "comparison_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    log.info("Wrote %s", run_dir.relative_to(_REPO_ROOT))
    print(json.dumps({"run_dir": str(run_dir.relative_to(_REPO_ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
