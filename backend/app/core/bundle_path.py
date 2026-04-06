"""
CPR ML bundle on ``sys.path`` (``src/``, ``cpr_api``).

Default root: ``<cpr-assist>/cpr_ml`` (sibling of ``backend/``). Override with env
``CPR_ML_ROOT``. Validates ``src/config.py`` before injecting paths.

Import this module before any ``src.*`` or ``cpr_api.*`` import (see ``app.main``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _assist_root() -> Path:
    # .../backend/app/core/bundle_path.py -> parents[3] = cpr-assist
    return Path(__file__).resolve().parents[3]


def resolve_cpr_ml_root() -> Path | None:
    raw = os.environ.get("CPR_ML_ROOT", "").strip()
    if raw:
        root = Path(raw).resolve()
    else:
        root = _assist_root() / "cpr_ml"
    if (root / "src" / "config.py").is_file():
        return root
    return None


def ensure_cpr_ml_paths() -> Path | None:
    """Prepend bundle root and ``api/`` to ``sys.path`` when the bundle is present."""
    root = resolve_cpr_ml_root()
    if root is None:
        return None
    api_dir = root / "api"
    for p in (root, api_dir):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return root


CPR_ML_ROOT: Path | None = ensure_cpr_ml_paths()


def is_vision_bundle_available() -> bool:
    return CPR_ML_ROOT is not None
