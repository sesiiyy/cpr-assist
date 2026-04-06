from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.ml import CompressionInferenceResponse, SceneInferenceResponse


SessionStatus = Literal["created", "running", "paused", "completed"]


class SessionCreateRequest(BaseModel):
    patient_age: int | None = None
    patient_gender: str | None = None


class SessionResponse(BaseModel):
    id: str
    session_id: str
    user_id: str
    status: SessionStatus
    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    mode: Literal["live"]


class SessionStartRequest(BaseModel):
    frames: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Scene check still(s); base64-encoded image bytes (typically one frame).",
    )
    frame_mime_type: str = Field(default="image/jpeg")


class SessionStartResponse(BaseModel):
    session: SessionResponse
    scene: SceneInferenceResponse


class SessionLiveInferenceRequest(BaseModel):
    frames: list[str] = Field(
        min_length=1,
        max_length=120,
        description="Chronological stills (oldest first). Each item is base64-encoded image bytes.",
    )
    frame_mime_type: str = Field(default="image/jpeg")
    timestamp_ms: int = Field(ge=0, description="Client clock for this sample.")


class SessionLiveInferenceResponse(CompressionInferenceResponse):
    issued_at: datetime
    timestamp_ms: int


class PromptEvent(BaseModel):
    session_id: str
    prompt: str
    issued_at: datetime
    reason: str


class SessionSummaryResponse(BaseModel):
    session_id: str
    avg_rate: float
    avg_depth_mm: float | None = None
    technique_errors: int | None = None
    active_compression_time_s: float
    quality_score: float
    prompt_timeline: list[PromptEvent]


class SessionSummaryListItem(BaseModel):
    session_id: str
    avg_rate: float
    avg_depth_mm: float | None = None
    quality_score: float
    active_compression_time_s: float
    ended_at: datetime | None = None
