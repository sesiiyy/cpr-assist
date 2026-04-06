"""Apply deployment GPU/device preferences before ML models load."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def apply_cpr_gpu_policy(force_device: str | None) -> None:
    """
    If ``force_device`` is set (or env ``CPR_FORCE_DEVICE``), record it for downstream
    config merge (S0 ``device`` string). Env is set so future upstream code can read it too.
    """
    raw = (force_device or os.environ.get("CPR_FORCE_DEVICE") or "").strip()
    if not raw:
        return
    os.environ["CPR_FORCE_DEVICE"] = raw
    logger.info("CPR device policy: CPR_FORCE_DEVICE=%s", raw)
