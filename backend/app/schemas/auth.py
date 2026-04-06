from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr


Role = Literal["caregiver", "instructor", "admin"]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    role: Role
    institution_id: str | None = None
    created_at: datetime
