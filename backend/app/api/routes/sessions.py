import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pymongo import ReturnDocument

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.mongo import get_collection
from app.schemas.session import (
    SessionCreateRequest,
    SessionLiveInferenceRequest,
    SessionLiveInferenceResponse,
    SessionResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionSummaryListItem,
    SessionSummaryResponse,
)
from app.services.audit_service import log_audit
from app.services.harness_registry import release as harness_release
from app.services.inference import infer_compression, infer_scene, register_session_harness

router = APIRouter()


def _to_session_response(doc: dict) -> SessionResponse:
    return SessionResponse(
        id=doc["_id"],
        session_id=doc["session_id"],
        user_id=doc["user_id"],
        status=doc["status"],
        created_at=doc["created_at"],
        started_at=doc.get("started_at"),
        ended_at=doc.get("ended_at"),
        mode="live",
    )


@router.post("", response_model=SessionResponse)
def create_session(payload: SessionCreateRequest, user: dict = Depends(get_current_user)) -> SessionResponse:
    sessions = get_collection("cpr_sessions")
    _id = str(uuid4())
    now = datetime.now(UTC)
    doc = {
        "_id": _id,
        "session_id": f"CPR-{now.strftime('%Y%m%d')}-{_id[:8]}",
        "user_id": user["_id"],
        "institution_id": user.get("institution_id"),
        "status": "created",
        "mode": "live",
        "patient": {"age": payload.patient_age, "gender": payload.patient_gender},
        "created_at": now,
    }
    sessions.insert_one(doc)
    log_audit("session.create", user["_id"], {"session_id": doc["session_id"]})
    return _to_session_response(doc)


@router.post("/{session_id}/start", response_model=SessionStartResponse)
async def start_session(
    session_id: str,
    payload: SessionStartRequest,
    user: dict = Depends(get_current_user),
) -> SessionStartResponse:
    sessions = get_collection("cpr_sessions")
    doc = sessions.find_one({"session_id": session_id, "user_id": user["_id"]})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    if doc.get("status") != "created":
        raise HTTPException(status_code=400, detail="Session already started")

    try:
        scene = await asyncio.wait_for(
            asyncio.to_thread(
                infer_scene,
                payload.frames,
                payload.frame_mime_type,
                session_id=session_id,
                user_id=user["_id"],
                session_doc=doc,
            ),
            timeout=settings.inference_timeout_sec,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="scene inference timed out") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not scene.frame_ready:
        raise HTTPException(status_code=422, detail={"warnings": [scene.prompt] if scene.prompt else ["scene_not_ready"]})

    register_session_harness(session_id, doc)

    now = datetime.now(UTC)
    res = sessions.find_one_and_update(
        {"session_id": session_id, "user_id": user["_id"], "status": "created"},
        {"$set": {"status": "running", "started_at": now}},
        return_document=ReturnDocument.AFTER,
    )
    if not res:
        raise HTTPException(status_code=409, detail="Session cannot be started")
    log_audit("session.start", user["_id"], {"session_id": session_id})
    return SessionStartResponse(session=_to_session_response(res), scene=scene)


@router.post("/{session_id}/pause", response_model=SessionResponse)
def pause_session(session_id: str, user: dict = Depends(get_current_user)) -> SessionResponse:
    sessions = get_collection("cpr_sessions")
    res = sessions.find_one_and_update(
        {"session_id": session_id, "user_id": user["_id"]},
        {"$set": {"status": "paused"}},
        return_document=ReturnDocument.AFTER,
    )
    if not res:
        raise HTTPException(status_code=404, detail="Session not found")
    log_audit("session.pause", user["_id"], {"session_id": session_id})
    return _to_session_response(res)


@router.post("/{session_id}/resume", response_model=SessionResponse)
def resume_session(session_id: str, user: dict = Depends(get_current_user)) -> SessionResponse:
    sessions = get_collection("cpr_sessions")
    res = sessions.find_one_and_update(
        {"session_id": session_id, "user_id": user["_id"]},
        {"$set": {"status": "running"}},
        return_document=ReturnDocument.AFTER,
    )
    if not res:
        raise HTTPException(status_code=404, detail="Session not found")
    log_audit("session.resume", user["_id"], {"session_id": session_id})
    return _to_session_response(res)


@router.post("/{session_id}/live-inference", response_model=SessionLiveInferenceResponse)
async def live_inference(
    session_id: str,
    payload: SessionLiveInferenceRequest,
    user: dict = Depends(get_current_user),
) -> SessionLiveInferenceResponse:
    sessions = get_collection("cpr_sessions")
    sess = sessions.find_one({"session_id": session_id, "user_id": user["_id"]})
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess.get("status") != "running":
        raise HTTPException(status_code=400, detail="Session not running")

    try:
        inf = await asyncio.wait_for(
            asyncio.to_thread(
                infer_compression,
                payload.frames,
                payload.frame_mime_type,
                session_id=session_id,
                user_id=user["_id"],
            ),
            timeout=settings.inference_timeout_sec,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="live inference timed out") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    issued_at = datetime.now(UTC)

    tb1 = inf.track_b_b1 or {}
    last_cycle = tb1.get("last_cycle") or {}
    ta = inf.track_a or {}
    fusion = inf.fusion_v1 or {}

    est_rate = float(tb1.get("cpm_nt") or 0.0)
    est_depth_cm = float(last_cycle.get("depth_cm") or 0.0)
    min_depth_mm = float(ta.get("target_lower_cm") or 5.0) * 10.0
    max_depth_mm = float(ta.get("target_upper_cm") or 6.0) * 10.0

    get_collection("session_metrics").insert_one(
        {
            "session_id": session_id,
            "user_id": user["_id"],
            "timestamp_ms": payload.timestamp_ms,
            "estimated_rate": est_rate,
            "estimated_depth_mm": est_depth_cm * 10.0,
            "min_depth_mm": min_depth_mm,
            "max_depth_mm": max_depth_mm,
            "paused_seconds": 0.0,
            "created_at": issued_at,
        }
    )

    prompt = fusion.get("primary_issue") if fusion.get("cue_ready") else None
    if prompt:
        get_collection("session_events").insert_one(
            {
                "session_id": session_id,
                "user_id": user["_id"],
                "prompt": prompt,
                "reason": fusion.get("primary_issue", "unknown"),
                "b2_technique_flagged": fusion.get("b2_technique_flagged", False),
                "b2_class_name": fusion.get("b2_class_name"),
                "issued_at": issued_at,
            }
        )

    return SessionLiveInferenceResponse(
        **inf.model_dump(),
        issued_at=issued_at,
        timestamp_ms=payload.timestamp_ms,
    )


@router.post("/{session_id}/stop", response_model=SessionResponse)
def stop_session(session_id: str, user: dict = Depends(get_current_user)) -> SessionResponse:
    sessions = get_collection("cpr_sessions")
    now = datetime.now(UTC)
    res = sessions.find_one_and_update(
        {"session_id": session_id, "user_id": user["_id"]},
        {"$set": {"status": "completed", "ended_at": now}},
        return_document=ReturnDocument.AFTER,
    )
    if not res:
        raise HTTPException(status_code=404, detail="Session not found")

    harness_release(session_id)

    metrics = list(get_collection("session_metrics").find({"session_id": session_id, "user_id": user["_id"]}))
    events = list(get_collection("session_events").find({"session_id": session_id, "user_id": user["_id"]}))

    rate_metrics = [m.get("estimated_rate", 0) for m in metrics if m.get("estimated_rate") is not None]
    avg_rate = sum(rate_metrics) / max(1, len(rate_metrics))

    depth_metrics = [m.get("estimated_depth_mm", 0) for m in metrics if m.get("estimated_depth_mm") is not None]
    avg_depth = sum(depth_metrics) / max(1, len(depth_metrics))

    technique_errors = sum(1 for e in events if e.get("b2_technique_flagged"))
    
    depth_penalty = sum(1 for d in depth_metrics if d < 50 or d > 60) * 0.5
    rate_penalty = sum(1 for r in rate_metrics if r < 100 or r > 120) * 0.5
    
    quality_score = max(0.0, 100.0 - (technique_errors * 5.0) - depth_penalty - rate_penalty)

    pause_duration_s = sum(float(m.get("paused_seconds", 0) or 0) for m in metrics)
    active_compression_time_s = max(0.0, len(metrics) * 0.333 - pause_duration_s) # 333ms per frame
    
    get_collection("session_summaries").update_one(
        {"session_id": session_id, "user_id": user["_id"]},
        {
            "$set": {
                "session_id": session_id,
                "user_id": user["_id"],
                "ended_at": now,
                "avg_rate": round(avg_rate, 2),
                "avg_depth_mm": round(avg_depth, 2),
                "technique_errors": technique_errors,
                "active_compression_time_s": round(active_compression_time_s, 2),
                "quality_score": round(quality_score, 2),
                "prompt_timeline": [
                    {
                        "session_id": e["session_id"],
                        "prompt": e["prompt"],
                        "reason": e["reason"],
                        "issued_at": e["issued_at"],
                        "b2_class_name": e.get("b2_class_name"),
                    }
                    for e in events
                ],
            }
        },
        upsert=True,
    )
    log_audit("session.stop", user["_id"], {"session_id": session_id})
    return _to_session_response(res)


@router.get("/summaries", response_model=list[SessionSummaryListItem])
def list_session_summaries(user: dict = Depends(get_current_user)) -> list[SessionSummaryListItem]:
    summaries = get_collection("session_summaries")
    sessions = get_collection("cpr_sessions")
    docs = list(summaries.find({"user_id": user["_id"]}, {"_id": 0}).sort("ended_at", -1).limit(100))
    items: list[SessionSummaryListItem] = []
    for doc in docs:
        sess = sessions.find_one({"session_id": doc["session_id"], "user_id": user["_id"]}, {"ended_at": 1})
        ended_at = doc.get("ended_at") or (sess.get("ended_at") if sess else None)
        items.append(
            SessionSummaryListItem(
                session_id=doc["session_id"],
                quality_score=float(doc.get("quality_score", 0)),
                avg_rate=float(doc.get("avg_rate", 0)),
                avg_depth_mm=float(doc.get("avg_depth_mm", 0)) if doc.get("avg_depth_mm") else None,
                active_compression_time_s=float(doc.get("active_compression_time_s", 0)),
                ended_at=ended_at,
            )
        )
    return items


@router.get("/{session_id}/summary", response_model=SessionSummaryResponse)
def session_summary(session_id: str, user: dict = Depends(get_current_user)) -> SessionSummaryResponse:
    summaries = get_collection("session_summaries")
    summary = summaries.find_one({"session_id": session_id, "user_id": user["_id"]})
    if not summary:
        summary = {
            "session_id": session_id,
            "avg_rate": 0.0,
            "active_compression_time_s": 0.0,
            "quality_score": 0.0,
            "prompt_timeline": [],
        }
    return SessionSummaryResponse(**summary)
