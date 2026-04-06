#!/usr/bin/env python3
"""
Train several PyTorch MLPs on the same tabular manifest (sex + age → min/max depth cm).
Uses the same train/val/test split as train_compare.py. Device order: CUDA → XPU → CPU.

Usage (repo root)::

  python experiments/ct_depth_tabular/code/train_pytorch_tabular.py \\
    --config experiments/ct_depth_tabular/config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import tabular_common as tabc  # noqa: E402
import train_compare as tc  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def resolve_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularMLPOrdered(nn.Module):
    """Predict min and max with max = min + softplus(δ) so max >= min."""

    def __init__(self, in_dim: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head_min = nn.Linear(d, 1)
        self.head_delta = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        min_p = self.head_min(z).squeeze(-1)
        delta = F.softplus(self.head_delta(z).squeeze(-1))
        max_p = min_p + delta
        return torch.stack([min_p, max_p], dim=1)


def build_torch_mlp(in_dim: int, hidden: list[int], dropout: float, head: str) -> nn.Module:
    h = str(head or "plain").lower()
    if h == "ordered":
        return TabularMLPOrdered(in_dim, hidden, dropout)
    return TabularMLP(in_dim, hidden, dropout)


def mc_dropout_predict(
    model: nn.Module,
    Xt: np.ndarray,
    device: torch.device,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-mode forward passes; return mean (n,2) and std (n,2)."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    model.train()
    xt = torch.from_numpy(Xt).to(device)
    outs: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_samples):
            outs.append(model(xt))
    stack = torch.stack(outs, dim=0)
    mean = stack.mean(dim=0).cpu().numpy()
    std = stack.std(dim=0, unbiased=False).cpu().numpy()
    return mean, std


def train_one_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    device: torch.device,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    log: logging.Logger,
) -> int:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    bad = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        sync_device(device)

        model.eval()
        with torch.no_grad():
            pv = model(X_val_t.to(device))
            val_loss = float(loss_fn(pv, y_val_t.to(device)).item())
        sync_device(device)

        val_mae = float(np.mean(np.abs(pv.cpu().numpy() - y_val_t.cpu().numpy())))

        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        if (epoch + 1) % 100 == 0:
            log.info("epoch %d train_mse=%.6f val_mae_mean=%.6f", epoch + 1, total / max(n, 1), val_mae)

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info("best_epoch=%d best_val_mae_mean=%.6f", best_epoch, best_val)
    return best_epoch


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch tabular depth band models (no images).")
    parser.add_argument(
        "--config",
        type=Path,
        default=_CODE_DIR.parent / "config" / "default.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("ct_depth_torch")

    cfg = load_config(args.config)
    t = cfg.get("train", {}) or {}
    pt = cfg.get("pytorch_tabular", {}) or {}

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

    y = df[["depth_min_cm", "depth_max_cm"]].values.astype(np.float32)
    X = df[x_cols].copy()

    seed = int(t.get("random_state", 42))
    seed_all(seed)

    idx_train, idx_val, idx_test = tc.split_three_way(
        df,
        y.astype(np.float64),
        float(t.get("train_frac", 0.70)),
        float(t.get("val_frac", 0.15)),
        float(t.get("test_frac", 0.15)),
        seed,
        "case_key" if bool(t.get("group_split_by_case_key", False)) else None,
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

    prep = tabc.build_preprocessor(use_eng)
    Xt_train = prep.fit_transform(X_train).astype(np.float32)
    Xt_val = prep.transform(X_val).astype(np.float32)
    Xt_test = prep.transform(X_test).astype(np.float32)

    device = resolve_device(pt.get("device"))
    log.info("device=%s", device)

    if args.dry_run:
        return 0

    suffix = str(pt.get("run_id_suffix", "_pytorch_tabular"))
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc") + suffix
    runs_root = _REPO_ROOT / str(cfg.get("runs_dir", "experiments/ct_depth_tabular/runs"))
    run_dir = runs_root / run_id
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(prep, models_dir / "preprocessor.joblib")
    shutil.copy2(args.config, run_dir / "config_resolved.yaml")

    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)

    batch_size = int(pt.get("batch_size", 32))
    max_epochs = int(pt.get("max_epochs", 500))
    patience = int(pt.get("patience", 45))
    lr = float(pt.get("lr", 1e-3))
    wd = float(pt.get("weight_decay", 1e-4))

    X_val_tensor = torch.from_numpy(Xt_val)
    y_val_tensor = torch.from_numpy(y_val)

    archs = pt.get("architectures") or [
        {"name": "mlp_default", "hidden": [128, 64], "dropout": 0.15, "head": "plain"},
    ]
    mc_samples = int(pt.get("mc_dropout_samples", 0))

    results: dict[str, Any] = {
        "run_id": run_id,
        "runner": "pytorch",
        "device": str(device),
        "use_engineered_features": use_eng,
        "mc_dropout_samples": mc_samples,
        "n_total": len(df),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
        "models": {},
    }

    mc_note = f" MC dropout samples (test uncertainty) = {mc_samples}." if mc_samples > 0 else ""
    md_lines = [
        "# CT depth tabular — PyTorch MLP comparison",
        "",
        f"Run `{run_id}` | device `{device}` | test n = {len(idx_test)}.{mc_note}",
        "",
        "| Model | head | MAE min (cm) | MAE max (cm) | RMSE min | RMSE max | mean MAE | band inversions | MC std mean (cm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    mean_mode = str(t.get("mean_baseline", "per_sex")).lower()
    if bool(pt.get("include_mean_baseline", True)):
        y_pred_test = tc.predict_mean_baseline(X_train, y_train.astype(np.float64), X_test, mean_mode)
        y_pred_val = tc.predict_mean_baseline(X_train, y_train.astype(np.float64), X_val, mean_mode)
        test_blk = tc._metrics_block(y_test.astype(np.float64), y_pred_test)
        results["models"]["mean_baseline"] = {
            "val": tc._metrics_block(y_val.astype(np.float64), y_pred_val),
            "test": test_blk,
        }
        md_lines.append(
            f"| mean_baseline | — | {test_blk['test_mae_depth_min_cm']} | {test_blk['test_mae_depth_max_cm']} | "
            f"{test_blk['test_rmse_depth_min_cm']} | {test_blk['test_rmse_depth_max_cm']} | "
            f"{test_blk['test_mae_mean_cm']} | {test_blk['test_pred_band_inversions']} | — |"
        )

    in_dim = Xt_train.shape[1]

    for arch in archs:
        name = str(arch.get("name", "mlp"))
        hidden = [int(x) for x in arch.get("hidden", [128, 64])]
        dropout = float(arch.get("dropout", 0.15))
        head = str(arch.get("head", "plain")).lower()
        log.info("training %s head=%s hidden=%s dropout=%s", name, head, hidden, dropout)

        ds = TensorDataset(torch.from_numpy(Xt_train), torch.from_numpy(y_train))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) > batch_size)

        model = build_torch_mlp(in_dim, hidden, dropout, head).to(device)
        train_one_mlp(
            model,
            loader,
            X_val_tensor,
            y_val_tensor,
            device,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            weight_decay=wd,
            log=log,
        )

        model.eval()
        with torch.no_grad():
            pred_val = model(torch.from_numpy(Xt_val).to(device)).cpu().numpy()
            pred_test_det = model(torch.from_numpy(Xt_test).to(device)).cpu().numpy()
        sync_device(device)

        val_blk = tc._metrics_block(y_val.astype(np.float64), pred_val)
        test_blk = tc._metrics_block(y_test.astype(np.float64), pred_test_det)
        mc_std_cell = "—"
        if mc_samples > 0 and dropout > 0:
            pred_test_mc, std_test = mc_dropout_predict(model, Xt_test, device, mc_samples)
            test_blk = tc._metrics_block(y_test.astype(np.float64), pred_test_mc)
            test_blk["test_mc_dropout_std_mean_cm"] = round(float(np.mean(std_test)), 6)
            mc_std_cell = str(test_blk["test_mc_dropout_std_mean_cm"])
        elif mc_samples > 0 and dropout <= 0:
            log.warning("mc_dropout_samples=%d but dropout=0 for %s — skipping MC", mc_samples, name)

        results["models"][name] = {
            "val": val_blk,
            "test": test_blk,
            "hidden": hidden,
            "dropout": dropout,
            "head": head,
        }

        md_lines.append(
            f"| {name} | {head} | {test_blk['test_mae_depth_min_cm']} | {test_blk['test_mae_depth_max_cm']} | "
            f"{test_blk['test_rmse_depth_min_cm']} | {test_blk['test_rmse_depth_max_cm']} | "
            f"{test_blk['test_mae_mean_cm']} | {test_blk['test_pred_band_inversions']} | {mc_std_cell} |"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "in_dim": in_dim,
                "hidden": hidden,
                "dropout": dropout,
                "head": head,
                "name": name,
                "use_engineered_features": use_eng,
            },
            models_dir / f"{name}.pt",
        )

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "comparison_table_pytorch.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    log.info("Wrote %s", run_dir.relative_to(_REPO_ROOT))
    print(json.dumps({"run_dir": str(run_dir.relative_to(_REPO_ROOT)), "device": str(device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
