from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str
    jwt_secret: str
    jwt_algorithm: str
    access_token_expire_minutes: int
    refresh_token_expire_days: int
    mongodb_uri: str
    mongodb_db: str

    inference_timeout_sec: float = Field(default=120.0, ge=5.0, le=600.0)
    cpr_config_path: str | None = Field(default=None, description="Optional YAML base; else bundle default.yaml")
    cpr_harness_ttl_seconds: float = Field(default=3600.0, ge=60.0, le=86400.0 * 7)
    cpr_force_device: str | None = Field(default=None, description="Torch device string, e.g. cpu or cuda:0")
    cpr_max_image_bytes: int = Field(default=20 * 1024 * 1024, ge=256 * 1024, le=80 * 1024 * 1024)


settings = Settings()
