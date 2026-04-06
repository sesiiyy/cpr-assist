from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError, jwt

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.db.mongo import get_collection
from app.schemas.auth import LoginRequest, RefreshRequest, TokenResponse, UserResponse
from app.services.audit_service import log_audit

router = APIRouter()


@router.post("/register", response_model=UserResponse)
def register(payload: LoginRequest) -> UserResponse:
    users = get_collection("users")
    existing = users.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid4())
    doc = {
        "_id": user_id,
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "role": "caregiver",
        "institution_id": None,
        "created_at": datetime.now(UTC),
    }
    users.insert_one(doc)
    log_audit("auth.register", user_id)
    return UserResponse(
        id=doc["_id"],
        email=doc["email"],
        role=doc["role"],
        institution_id=doc["institution_id"],
        created_at=doc["created_at"],
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    users = get_collection("users")
    user = users.find_one({"email": payload.email})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    access_token = create_access_token(user["_id"])
    refresh_token = create_refresh_token(user["_id"])
    get_collection("refresh_tokens").insert_one(
        {"token": refresh_token, "user_id": user["_id"], "created_at": datetime.now(UTC)}
    )
    log_audit("auth.login", user["_id"])
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest) -> TokenResponse:
    try:
        token_payload = jwt.decode(
            payload.refresh_token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid refresh token") from exc
    if token_payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")
    tokens = get_collection("refresh_tokens")
    token_doc = tokens.find_one({"token": payload.refresh_token})
    if not token_doc:
        raise HTTPException(status_code=401, detail="Refresh token not found")
    user_id = token_payload["sub"]
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)
    tokens.delete_one({"token": payload.refresh_token})
    tokens.insert_one({"token": refresh_token, "user_id": user_id, "created_at": datetime.now(UTC)})
    log_audit("auth.refresh", user_id)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/logout")
def logout(payload: RefreshRequest, user: dict = Depends(get_current_user)) -> dict[str, str]:
    get_collection("refresh_tokens").delete_one({"token": payload.refresh_token, "user_id": user["_id"]})
    log_audit("auth.logout", user["_id"])
    return {"status": "ok"}


@router.post("/logout-refresh")
def logout_refresh(payload: RefreshRequest) -> dict[str, str]:
    try:
        token_payload = jwt.decode(
            payload.refresh_token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid refresh token") from exc
    if token_payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")
    tokens = get_collection("refresh_tokens")
    token_doc = tokens.find_one({"token": payload.refresh_token})
    if not token_doc:
        raise HTTPException(status_code=401, detail="Refresh token not found")
    user_id = token_payload["sub"]
    tokens.delete_one({"token": payload.refresh_token})
    log_audit("auth.logout", user_id)
    return {"status": "ok"}


@router.get("/me", response_model=UserResponse)
def me(user: dict = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=user["_id"],
        email=user["email"],
        role=user["role"],
        institution_id=user.get("institution_id"),
        created_at=user["created_at"],
    )
