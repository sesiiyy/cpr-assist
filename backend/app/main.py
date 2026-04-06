from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.core import bundle_path  # noqa: F401 — CPR_ML_ROOT on sys.path before routers
from app.core.config import settings
from app.core.gpu_policy import apply_cpr_gpu_policy

from app.api.router import api_router
from app.services.harness_registry import start_ttl_sweeper, stop_ttl_sweeper


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    apply_cpr_gpu_policy(settings.cpr_force_device)
    start_ttl_sweeper()
    yield
    stop_ttl_sweeper()


app = FastAPI(title="CPR Assist API", version="0.1.0", lifespan=_lifespan)
app.include_router(api_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
