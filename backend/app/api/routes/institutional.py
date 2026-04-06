from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import require_roles
from app.db.mongo import get_collection
from app.services.audit_service import log_audit

router = APIRouter()


@router.get("/audit/logs")
def audit_logs(user: dict = Depends(require_roles("admin"))):
    logs = list(get_collection("audit_logs").find({}, {"_id": 0}).sort("created_at", -1).limit(200))
    return {"items": logs, "viewer": user["_id"]}


@router.get("/institutions/{institution_id}/stats")
def institution_stats(institution_id: str, user: dict = Depends(require_roles("admin", "instructor"))):
    if user.get("role") != "admin":
        uid_inst = user.get("institution_id")
        if not uid_inst or uid_inst != institution_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot access another institution's statistics",
            )
    sessions = get_collection("cpr_sessions")
    count = sessions.count_documents({"institution_id": institution_id})
    log_audit("institution.stats_view", user["_id"], {"institution_id": institution_id})
    return {"institution_id": institution_id, "session_count": count, "viewer": user["_id"]}
