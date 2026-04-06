#!/usr/bin/env python3
"""
Copy champion sklearn artifacts from a training run into experiments/ct_depth_tabular/frozen/models/.

Defaults: run_id and model name from aggregated/settled_model.json, falling back to frozen/FROZEN_BASELINE.json.

Usage (repo root)::

  python experiments/ct_depth_tabular/code/freeze_ct_depth_baseline.py
  python experiments/ct_depth_tabular/code/freeze_ct_depth_baseline.py --run-id 20260404_055703_utc
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
_EXP = _CODE_DIR.parent
_FROZEN = _EXP / "frozen"
_AGG = _EXP / "aggregated"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def resolve_run_and_model() -> tuple[str, str]:
    settled = _load_json(_AGG / "settled_model.json")
    if settled and isinstance(settled.get("settled"), dict):
        s = settled["settled"]
        rid = s.get("run_id")
        mk = s.get("model")
        if isinstance(rid, str) and isinstance(mk, str):
            return rid, mk
    frozen = _load_json(_FROZEN / "FROZEN_BASELINE.json")
    if frozen and isinstance(frozen.get("champion"), dict):
        c = frozen["champion"]
        rid = c.get("source_run_id")
        mk = c.get("model_key")
        if isinstance(rid, str) and isinstance(mk, str):
            return rid, mk
    raise SystemExit("Could not read run_id/model from aggregated/settled_model.json or frozen/FROZEN_BASELINE.json")


def main() -> int:
    p = argparse.ArgumentParser(description="Copy CT depth tabular champion joblibs into frozen/models/.")
    p.add_argument("--run-id", type=str, default=None, help="Override training run directory name under runs/")
    p.add_argument("--model-key", type=str, default=None, help="Champion model filename stem (default: ridge)")
    args = p.parse_args()

    run_id, model_key = resolve_run_and_model()
    if args.run_id:
        run_id = args.run_id
    if args.model_key:
        model_key = args.model_key

    run_dir = _EXP / "runs" / run_id
    src_models = run_dir / "models"
    dst = _FROZEN / "models"
    dst.mkdir(parents=True, exist_ok=True)

    if not src_models.is_dir():
        print("Source models directory missing: %s — train train_compare.py first." % src_models, file=sys.stderr)
        return 1

    copied: list[str] = []
    for name in (f"{model_key}.joblib", "mean_baseline.joblib"):
        src = src_models / name
        if not src.is_file():
            print("Missing %s — skip" % src, file=sys.stderr)
            continue
        shutil.copy2(src, dst / name)
        copied.append(str((dst / name).relative_to(_REPO_ROOT)))

    cfg = run_dir / "config_resolved.yaml"
    if cfg.is_file():
        shutil.copy2(cfg, dst / "config_resolved.yaml")
        copied.append(str((dst / "config_resolved.yaml").relative_to(_REPO_ROOT)))

    if not copied:
        print("No files copied.", file=sys.stderr)
        return 1

    print(json.dumps({"run_id": run_id, "model_key": model_key, "copied": copied}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
