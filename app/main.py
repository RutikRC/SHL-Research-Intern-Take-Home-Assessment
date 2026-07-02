"""
FastAPI application entry point.

Creates and configures the ASGI application, registers routers,
installs middleware, and defines global exception handlers.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.admin import router as admin_router
from app.api.health import router as health_router
from app.api.routes import router as chat_router
from app.core.config import get_settings
from app.core.logging_ import configure_logging, get_logger
from app.database.session import Base, engine
from app.database import models  # noqa: F401 – registers ORM models on Base.metadata
from app.utils.helpers import generate_request_id

settings = get_settings()
logger = get_logger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, Any]:
    """Handle application startup and shutdown events."""
    configure_logging(settings.LOG_LEVEL)
    logger.info(
        "app_startup",
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        log_level=settings.LOG_LEVEL,
    )
    await _check_database_connection()
    await _create_tables()
    yield
    await engine.dispose()
    logger.info("app_shutdown")


async def _check_database_connection() -> None:
    """Attempt a lightweight database connection and log the result."""
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            row = result.scalar_one()
        logger.info(
            "database_connected",
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            db=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            server_responded=True,
            ping_result=row,
        )
    except Exception as exc:
        logger.warning(
            "database_unreachable",
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            db=settings.POSTGRES_DB,
            error=str(exc),
            hint="The /chat endpoint will fail at runtime if the database stays down.",
        )


async def _create_tables() -> None:
    """Create all ORM-mapped tables if they do not already exist.

    Uses SQLAlchemy's ``create_all`` which is a no-op for existing tables,
    making it safe to call on every startup without migrations.
    Also enables the pgvector extension and creates an IVFFLAT index
    on the assessment_embeddings table for cosine similarity search.
    """
    try:
        async with engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # Create all ORM-mapped tables
            await conn.run_sync(Base.metadata.create_all)
            # Create IVFFLAT index for cosine similarity if it doesn't exist
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_assessment_embeddings_vector "
                    "ON assessment_embeddings "
                    "USING ivfflat (embedding vector_cosine_ops) "
                    "WITH (lists = 100)"
                )
            )
        logger.info("database_tables_created", tables=list(Base.metadata.tables.keys()))
    except Exception as exc:
        logger.warning(
            "database_tables_creation_failed",
            error=str(exc),
            hint="The /chat endpoint may fail if required tables are missing.",
        )


# ── Application factory ─────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any) -> JSONResponse:  # noqa: ANN401
    """Log every request with method, path, status, and execution time."""
    request_id = request.headers.get("X-Request-ID", generate_request_id())
    start_time = time.monotonic()

    response: JSONResponse = await call_next(request)  # type: ignore[assignment]

    elapsed = time.monotonic() - start_time
    logger.info(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        elapsed_seconds=round(elapsed, 4),
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ── Routers ─────────────────────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(admin_router)


# ── Global exception handlers ───────────────────────────────────────────────

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle known HTTP exceptions with a clean JSON body."""
    logger.warning(
        "http_exception",
        path=str(request.url),
        status_code=exc.status_code,
        detail=exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic / FastAPI validation errors."""
    logger.warning(
        "validation_error",
        path=str(request.url),
        errors=exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Request validation failed"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return a safe 500 response."""
    logger.exception(
        "unhandled_exception",
        path=str(request.url),
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )
