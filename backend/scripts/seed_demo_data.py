from datetime import UTC, datetime
from uuid import uuid4

from app.core.security import hash_password
from app.db.mongo import get_collection


def run() -> None:
    users = get_collection("users")
    institutions = get_collection("institutions")
    existing_inst = institutions.find_one({"name": "Demo Hospital"})
    if existing_inst:
        inst_id = existing_inst["_id"]
    else:
        inst_id = str(uuid4())
        institutions.insert_one({"_id": inst_id, "name": "Demo Hospital", "created_at": datetime.now(UTC)})

    for email, role in [
        ("caregiver@example.com", "caregiver"),
        ("instructor@example.com", "instructor"),
        ("admin@example.com", "admin"),
    ]:
        existing = users.find_one({"email": email})
        if existing:
            users.update_one(
                {"email": email},
                {
                    "$set": {
                        "email": email,
                        "password_hash": hash_password("password123"),
                        "role": role,
                        "institution_id": inst_id,
                    }
                },
            )
        else:
            user_id = str(uuid4())
            users.insert_one(
                {
                    "_id": user_id,
                    "email": email,
                    "password_hash": hash_password("password123"),
                    "role": role,
                    "institution_id": inst_id,
                    "created_at": datetime.now(UTC),
                }
            )
    print("Seeded demo users and institution.")


if __name__ == "__main__":
    run()
