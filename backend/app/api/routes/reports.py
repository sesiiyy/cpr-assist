from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, require_roles
from app.db.mongo import get_collection
from app.services.audit_service import log_audit

router = APIRouter()


@router.get("/instructor-dashboard")
def instructor_dashboard(user: dict = Depends(require_roles("instructor", "admin"))) -> dict:
    summaries = get_collection("session_summaries")
    
    agg = list(summaries.aggregate([
        {
            "$group": {
                "_id": None,
                "avg_quality_score": {"$avg": "$quality_score"},
                "avg_rate": {"$avg": "$avg_rate"},
                "avg_depth_mm": {"$avg": "$avg_depth_mm"},
                "total_technique_errors": {"$sum": "$technique_errors"},
                "total_sessions": {"$sum": 1}
            }
        }
    ]))
    
    stats = agg[0] if agg else {}
    
    return {
        "avg_quality_score": round(float(stats.get("avg_quality_score", 0)), 2),
        "avg_rate": round(float(stats.get("avg_rate", 0)), 2),
        "avg_depth_mm": round(float(stats.get("avg_depth_mm", 0)), 2),
        "total_technique_errors": int(stats.get("total_technique_errors", 0)),
        "total_sessions": int(stats.get("total_sessions", 0)),
        "viewer": user["_id"],
    }


@router.post("/export")
def export_report(session_id: str, user: dict = Depends(require_roles("admin", "instructor"))) -> dict:
    job_id = str(uuid4())
    doc = {
        "job_id": job_id,
        "session_id": session_id,
        "created_by": user["_id"],
        "status": "completed",
        "anonymized": True,
        "created_at": datetime.now(UTC),
    }
    get_collection("export_jobs").insert_one(doc)
    log_audit("report.export", user["_id"], {"session_id": session_id, "job_id": job_id})
    return doc


@router.get("/export/{job_id}")
def get_export(job_id: str, user: dict = Depends(get_current_user)) -> dict:
    job = get_collection("export_jobs").find_one({"job_id": job_id}, {"_id": 0})
    return job or {"error": "not_found"}
