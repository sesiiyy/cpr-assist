"""Session document → merged CPR YAML config (Track A band + patient overrides)."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from app.core import bundle_path  # noqa: F401 — sys.path before cpr_api
from app.core.config import settings

from cpr_api.config_merge import build_merged_config
from cpr_api.schemas import PatientDemographics


def _config_path_for_load() -> Path | str | None:
    if settings.cpr_config_path:
        return Path(settings.cpr_config_path)
    return None


def patient_demographics_from_session_doc(doc: dict[str, Any]) -> PatientDemographics | None:
    p = doc.get("patient") or {}
    age = p.get("age")
    raw_g = str(p.get("gender") or "").strip().lower()
    sex: str | None = None
    if raw_g in ("male", "m", "man"):
        sex = "male"
    elif raw_g in ("female", "f", "woman"):
        sex = "female"
    age_years = float(age) if age is not None else None
    if sex is None and age_years is None:
        return None
    return PatientDemographics(sex=sex, age_years=age_years)


def apply_force_device_to_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = (settings.cpr_force_device or "").strip()
    if not raw:
        return cfg
    out = deepcopy(cfg)
    tb = out.setdefault("track_b", {})
    s0 = tb.setdefault("s0_rgb_classifier", {})
    s0["device"] = raw
    return out


def build_session_merged_config(doc: dict[str, Any]) -> dict[str, Any]:
    if not bundle_path.is_vision_bundle_available():
        raise RuntimeError("CPR ML bundle missing: expected cpr_ml/src/config.py under CPR_ML_ROOT")
    patient = patient_demographics_from_session_doc(doc)
    cfg = build_merged_config(_config_path_for_load(), patient)
    return apply_force_device_to_cfg(cfg)
