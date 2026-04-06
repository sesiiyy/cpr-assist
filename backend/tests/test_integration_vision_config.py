"""Optional smoke: merged YAML config from vendored cpr_ml."""
from __future__ import annotations

import pytest

from app.core import bundle_path
from app.core.config import settings
from app.services.cpr_config import build_session_merged_config


pytestmark = pytest.mark.integration


def test_build_session_merged_config(monkeypatch) -> None:
    if not bundle_path.is_vision_bundle_available():
        pytest.skip("cpr_ml bundle or vision deps missing")
    monkeypatch.setattr(settings, "cpr_config_path", None)
    cfg = build_session_merged_config({"patient": {"age": 40, "gender": "male"}})
    assert isinstance(cfg, dict)
    assert "track_a" in cfg
