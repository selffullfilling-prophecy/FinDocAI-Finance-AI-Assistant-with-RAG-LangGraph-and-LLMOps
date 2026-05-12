from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check() -> dict:
    settings = get_settings()

    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.env,
    }