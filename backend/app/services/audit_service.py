from datetime import UTC, datetime

from app.db.mongo import get_collection


def log_audit(action: str, actor_id: str | None, details: dict | None = None) -> None:
    audit = get_collection("audit_logs")
    audit.insert_one(
        {
            "action": action,
            "actor_id": actor_id,
            "details": details or {},
            "created_at": datetime.now(UTC),
        }
    )
