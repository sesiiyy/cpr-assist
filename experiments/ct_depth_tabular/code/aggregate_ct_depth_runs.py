#!/usr/bin/env python3
"""
Scan experiments/ct_depth_tabular/runs/*/metrics.json and rank models by test mean MAE.

Writes aggregated/comparison_all_runs.md and aggregated/settled_model.json, plus SETTLED_MODEL.md summary.

Usage (repo root)::

  python experiments/ct_depth_tabular/code/aggregate_ct_depth_runs.py
  python experiments/ct_depth_tabular/code/aggregate_ct_depth_runs.py --runs-dir experiments/ct_depth_tabular/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
_EXP_ROOT = _CODE_DIR.parent
_DEFAULT_RUNS = _EXP_ROOT / "runs"
_DEFAULT_OUT = _EXP_ROOT / "aggregated"

# Baselines shown in tables but not chosen as the production regressor
_EXCLUDE_FROM_SETTLED = frozenset({"mean_baseline"})


def _runner_from_metrics(data: dict[str, Any], parent_name: str) -> str:
    r = data.get("runner")
    if isinstance(r, str) and r:
        return r
    if "_pytorch" in parent_name or parent_name.endswith("_tabular"):
        return "pytorch"
    return "sklearn"


def collect_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        if not metrics_path.is_file():
            continue
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(data.get("run_id", metrics_path.parent.name))
        parent = metrics_path.parent.name
        runner = _runner_from_metrics(data, parent)
        models = data.get("models")
        if not isinstance(models, dict):
            continue
        for model_name, block in models.items():
            if not isinstance(block, dict):
                continue
            test = block.get("test")
            if not isinstance(test, dict):
                continue
            mae = test.get("test_mae_mean_cm")
            if mae is None:
                continue
            inv = test.get("test_pred_band_inversions", "")
            mc_std = test.get("test_mc_dropout_std_mean_cm")
            rows.append(
                {
                    "run_id": run_id,
                    "runner": runner,
                    "model": str(model_name),
                    "test_mae_mean_cm": float(mae),
                    "test_pred_band_inversions": inv,
                    "test_mc_dropout_std_mean_cm": mc_std,
                    "metrics_path": str(metrics_path.relative_to(_REPO_ROOT)).replace("\\", "/"),
                }
            )
    return rows


def _inversion_key(r: dict[str, Any]) -> int:
    v = r.get("test_pred_band_inversions")
    try:
        return int(v)
    except (TypeError, ValueError):
        return 10**9


def pick_settled(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [r for r in rows if r["model"] not in _EXCLUDE_FROM_SETTLED]
    if not candidates:
        return None
    # Stable sort: newest run_id first, then by error so ties prefer the latest training run.
    candidates.sort(key=lambda r: r["run_id"], reverse=True)
    candidates.sort(key=lambda r: (r["test_mae_mean_cm"], _inversion_key(r)))
    best = candidates[0]
    return {
        "run_id": best["run_id"],
        "runner": best["runner"],
        "model": best["model"],
        "test_mae_mean_cm": best["test_mae_mean_cm"],
        "metrics_path": best["metrics_path"],
        "tie_break_note": "Lowest test_mae_mean_cm; then fewer band inversions; then newest run_id (stable sort).",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate CT depth tabular run metrics.")
    p.add_argument("--runs-dir", type=Path, default=_DEFAULT_RUNS)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args()
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else _REPO_ROOT / args.runs_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(runs_dir)
    settled = pick_settled(rows)

    md_lines = [
        "# CT depth tabular — all runs (test mean MAE)",
        "",
        f"Scanned `{runs_dir.relative_to(_REPO_ROOT)}` — {len(rows)} model entries from `metrics.json` files.",
        "",
        "| run_id | runner | model | test mean MAE (cm) | band inv | MC std mean (cm) | metrics.json |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["test_mae_mean_cm"], x["run_id"], x["model"])):
        mc = r.get("test_mc_dropout_std_mean_cm")
        mc_s = "—" if mc is None else str(mc)
        inv = r.get("test_pred_band_inversions", "")
        md_lines.append(
            f"| `{r['run_id']}` | {r['runner']} | {r['model']} | {r['test_mae_mean_cm']:.5f} | {inv} | {mc_s} | `{r['metrics_path']}` |"
        )
    md_lines.append("")

    (out_dir / "comparison_all_runs.md").write_text("\n".join(md_lines), encoding="utf-8")

    settled_payload: dict[str, Any] = {
        "runs_dir": str(runs_dir.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "n_entries": len(rows),
        "settled": settled,
        "excluded_from_settled": sorted(_EXCLUDE_FROM_SETTLED),
    }
    (out_dir / "settled_model.json").write_text(json.dumps(settled_payload, indent=2), encoding="utf-8")

    if settled:
        smd = [
            "# Settled model (CT depth tabular)",
            "",
            "Chosen by `aggregate_ct_depth_runs.py`: lowest `test_mae_mean_cm` on the held-out test split, excluding `mean_baseline`.",
            "",
            f"- **Runner:** {settled['runner']}",
            f"- **Model key:** `{settled['model']}`",
            f"- **Run:** `{settled['run_id']}`",
            f"- **Test mean MAE (cm):** {settled['test_mae_mean_cm']:.5f}",
            f"- **metrics.json:** `{settled['metrics_path']}`",
            "",
            "Re-train or load checkpoints from that run directory under `experiments/ct_depth_tabular/runs/`.",
            "",
            f"Full table: `aggregated/comparison_all_runs.md`.",
        ]
    else:
        smd = [
            "# Settled model (CT depth tabular)",
            "",
            "No metrics found. Run `train_compare.py` and/or `train_pytorch_tabular.py`, then re-run `aggregate_ct_depth_runs.py`.",
        ]
    (_EXP_ROOT / "SETTLED_MODEL.md").write_text("\n".join(smd) + "\n", encoding="utf-8")

    print(json.dumps({"n_entries": len(rows), "settled": settled, "out": str(out_dir.relative_to(_REPO_ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
