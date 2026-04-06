"""
Session vision: scene gate (readiness) + live compression coaching (session harness + fusion).

Uses vendored CPR ML under ``cpr_ml/``. Vision dependencies (``opencv-python``, ``numpy``)
must be present on the host.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.schemas.ml import CompressionInferenceResponse, SceneInferenceResponse

logger = logging.getLogger(__name__)


def _decode_frame_b64(b64: str) -> Any:
    raw = base64.b64decode(b64, validate=False)
    if len(raw) > settings.cpr_max_image_bytes:
        raise ValueError("frame exceeds configured max image size (CPR_MAX_IMAGE_BYTES)")
    arr = np.frombuffer(raw, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("could not decode image (use JPEG or PNG)")
    return im


def _tick_to_compression(tick: dict[str, Any]) -> CompressionInferenceResponse:
    """
    Wrap the tick result from run_session_tick into our pydantic response.
    Completely based on cpr_ml output.
    """
    return CompressionInferenceResponse(
        readiness=tick.get("readiness", {}),
        track_a=tick.get("track_a", {}),
        track_b_b1=tick.get("track_b_b1"),
        track_b_s0_rgb=tick.get("track_b_s0_rgb"),
        fusion_v1=tick.get("fusion_v1", {}),
        frame_index=tick.get("frame_index", 0),
    )


def infer_scene(
    frames: list[str],
    frame_mime_type: str,
    *,
    session_id: str,
    user_id: str,
    session_doc: dict[str, Any],
) -> SceneInferenceResponse:
    """
    Run readiness check on the last frame.
    Completely based on cpr_ml.src.readiness.process_frame output.
    """
    from src.readiness import process_frame
    from app.services.cpr_config import build_session_merged_config

    if not frames:
        raise ValueError("at least one frame is required")

    try:
        bgr = _decode_frame_b64(frames[-1])
        cfg = build_session_merged_config(session_doc)
        fr = process_frame(bgr, config=cfg)
        rj = fr.to_jsonable()
        return SceneInferenceResponse(**rj)
    except ValueError:
        raise
    except Exception:
        logger.exception("infer_scene failed session_id=%s", session_id)
        raise


def infer_compression(
    frames: list[str],
    frame_mime_type: str,
    *,
    session_id: str,
    user_id: str,
) -> CompressionInferenceResponse:
    """
    Run live session tick.
    Completely based on cpr_api.session_harness.run_session_tick output.
    """
    from cpr_api.session_harness import run_session_tick
    from app.services import harness_registry

    state = harness_registry.get(session_id)
    if state is None:
        raise ValueError("session harness missing or expired — start the session again")

    if not frames:
        raise ValueError("at least one frame is required")

    try:
        bgr = _decode_frame_b64(frames[-1])
        tick = run_session_tick(state, bgr)
        harness_registry.touch(session_id)
        return _tick_to_compression(tick)
    except ValueError:
        raise
    except Exception:
        logger.exception("infer_compression failed session_id=%s", session_id)
        raise


def register_session_harness(session_id: str, session_doc: dict[str, Any]) -> None:
    """Build and store harness after a successful scene check."""
    from cpr_api.session_harness import build_session_harness_state
    from app.services import harness_registry
    from app.services.cpr_config import build_session_merged_config

    cfg = build_session_merged_config(session_doc)
    harness_registry.put(session_id, build_session_harness_state(cfg))
