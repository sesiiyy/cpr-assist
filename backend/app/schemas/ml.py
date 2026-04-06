from typing import Any
from pydantic import BaseModel, Field


class SceneInferenceResponse(BaseModel):
    """
    Directly mirrors cpr_ml.src.readiness.types.FrameReadinessResult.to_jsonable().
    """
    timestamp: float
    state: str
    patient_detected: bool
    caregiver_detected: bool
    patient_horizontal_ok: bool
    side_view_ok: bool
    alignment_ok: bool
    frame_ready: bool
    stable_ready: bool
    patient_chest_roi: list[int] | None = None
    caregiver_upper_roi: list[int] | None = None
    prompt: str
    readiness_score: float
    meta: dict[str, Any] = Field(default_factory=dict)


class CompressionInferenceResponse(BaseModel):
    """
    Directly mirrors cpr_api.session_harness.run_session_tick() output.
    """
    readiness: dict[str, Any]
    track_a: dict[str, Any]
    track_b_b1: dict[str, Any] | None = None
    track_b_s0_rgb: dict[str, Any] | None = None
    fusion_v1: dict[str, Any]
    frame_index: int
