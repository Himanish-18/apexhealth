"""
Healthcare Knowledge Navigator — FastAPI Application Entry Point.

Initializes the FastAPI application with lifespan management, middleware,
and route registration. This is the ASGI entry point used by Uvicorn.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import setup_logging, get_logger
from backend.routes import router
from configs.constants import PROJECT_META
from configs.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Startup:
        - Loads settings and configures logging.
        - Creates required data directories.
        - Logs boot diagnostics.

    Shutdown:
        - Cleans up resources gracefully.
    """
    # ── Startup ──────────────────────────────────────────────────────
    settings = get_settings()
    setup_logging(settings)
    logger = get_logger("main")

    # Ensure data directories exist
    for directory in (
        settings.raw_xml_dir,
        settings.parsed_json_dir,
        settings.chunks_dir,
        settings.logs_dir,
        settings.models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  %s v%s", PROJECT_META.name, PROJECT_META.version)
    logger.info("=" * 60)
    logger.info("Qdrant endpoint  : %s", settings.qdrant_url)
    logger.info("Embedding model  : %s", settings.embedding_model)
    logger.info("Reranker model   : %s", settings.reranker_model)
    logger.info("Collection       : %s", settings.collection_name)
    logger.info("Top-K / Final-K  : %d / %d", settings.top_k, settings.final_k)
    logger.info("Log level        : %s", settings.log_level)
    logger.info("Application startup complete.")

    yield  # ← Application is running

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Application shutdown initiated. Cleaning up resources...")
    logger.info("Shutdown complete.")


# ── Application Factory ─────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        FastAPI: A fully-configured application ready to serve.
    """
    application = FastAPI(
        title=PROJECT_META.name,
        description=PROJECT_META.description,
        version=PROJECT_META.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS Middleware ──────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register Routes ──────────────────────────────────────────────
    application.include_router(router)

    return application


# ── ASGI Entry Point ─────────────────────────────────────────────────
app: FastAPI = create_app()
