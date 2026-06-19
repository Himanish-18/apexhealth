"""
Healthcare Knowledge Navigator — API Routes.

Defines all FastAPI route handlers. Business-logic routes will be
added in subsequent phases; this module provides foundational
health and status endpoints.
"""

from fastapi import APIRouter, status
from pydantic import BaseModel

from configs.constants import PROJECT_META


# ── Response Schemas ─────────────────────────────────────────────────

class StatusResponse(BaseModel):
    """Schema for the root status endpoint."""

    status: str
    project: str


class HealthResponse(BaseModel):
    """Schema for the health check endpoint."""

    status: str


# ── Router ───────────────────────────────────────────────────────────

router = APIRouter()


@router.get(
    "/",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Status",
    description="Returns the current service status and project name.",
    tags=["System"],
)
async def root() -> StatusResponse:
    """Return the service status and project identifier."""
    return StatusResponse(
        status="running",
        project=PROJECT_META.name,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Lightweight health check for load balancers and orchestrators.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Return a simple health status."""
    return HealthResponse(status="healthy")
