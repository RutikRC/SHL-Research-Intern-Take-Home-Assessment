"""
Service layer for catalog ingestion.

Downloads the SHL catalog JSON, validates, normalises, and upserts records
into PostgreSQL.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import httpx
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging_ import get_logger
from app.repositories.catalog_repository import CatalogRepository

logger = get_logger(__name__)


class CatalogService:
    """Business logic for downloading, validating, and persisting the SHL catalog."""

    def __init__(self, repository: CatalogRepository | None = None) -> None:
        self._repository = repository or CatalogRepository()
        self._settings = get_settings()

    # ── Public interface ─────────────────────────────────────────────────────

    async def import_catalog(self, session: AsyncSession) -> dict[str, Any]:
        """Run the full catalog import pipeline.

        Steps:
            1. Fetch raw JSON from the SHL API.
            2. Validate the top-level response structure.
            3. Normalise each record.
            4. Upsert into PostgreSQL.
            5. Log a summary.

        Args:
            session: An active async database session.

        Returns:
            A dict with ``success``, ``total``, ``inserted``, and ``updated``.
        """
        start = time.monotonic()
        logger.info("catalog_import_started")

        raw_data = await self._fetch_catalog()
        total = self._validate_catalog(raw_data)
        inserted = 0
        updated = 0

        for record in raw_data:
            normalised = self._normalise_record(record)
            is_insert = await self._repository.upsert_assessment(
                session, normalised,
            )
            if is_insert:
                inserted += 1
            else:
                updated += 1

        await session.commit()

        elapsed = round(time.monotonic() - start, 4)
        logger.info(
            "catalog_import_completed",
            total=total,
            inserted=inserted,
            updated=updated,
            elapsed_seconds=elapsed,
        )

        return {
            "success": True,
            "total": total,
            "inserted": inserted,
            "updated": updated,
        }

    async def import_catalog_local(self, session: AsyncSession) -> dict[str, Any]:
        """Import catalog from the local data/catalogue.json file.

        Steps:
            1. Read and parse the local JSON file.
            2. Validate the top-level response structure.
            3. Normalise each record.
            4. Upsert into PostgreSQL.
            5. Log a summary.

        Args:
            session: An active async database session.

        Returns:
            A dict with ``success``, ``total``, ``inserted``, and ``updated``.
        """
        import json

        start = time.monotonic()
        logger.info("catalog_import_local_started", path=self._settings.CATALOG_LOCAL_PATH)

        try:
            with open(self._settings.CATALOG_LOCAL_PATH, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Local catalog file not found at {self._settings.CATALOG_LOCAL_PATH}",
            )
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Invalid JSON in local catalog file: {exc}",
            )

        total = self._validate_catalog(raw_data)
        inserted = 0
        updated = 0

        for record in raw_data:
            normalised = self._normalise_record(record)
            is_insert = await self._repository.upsert_assessment(
                session, normalised,
            )
            if is_insert:
                inserted += 1
            else:
                updated += 1

        await session.commit()

        elapsed = round(time.monotonic() - start, 4)
        logger.info(
            "catalog_import_local_completed",
            total=total,
            inserted=inserted,
            updated=updated,
            elapsed_seconds=elapsed,
        )

        return {
            "success": True,
            "total": total,
            "inserted": inserted,
            "updated": updated,
        }

    async def count_assessments(self, session: AsyncSession) -> int:
        """Return the current number of assessments in the database."""
        return await self._repository.count(session)

    # ── Fetching ─────────────────────────────────────────────────────────────

    async def _fetch_catalog(self) -> list[dict[str, Any]]:
        """Download the SHL catalog JSON with retry logic."""
        url = self._settings.CATALOG_API_URL
        timeout = self._settings.HTTP_REQUEST_TIMEOUT_SECONDS
        max_retries = self._settings.HTTP_MAX_RETRIES
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                logger.info(
                    "catalog_downloaded",
                    url=url, status_code=response.status_code, attempt=attempt,
                )
                # Use strict=False to tolerate invalid control characters in the API response.
                return json.loads(response.text, strict=False)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                logger.warning(
                    "catalog_http_error",
                    status_code=exc.response.status_code, attempt=attempt, url=url,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "catalog_network_error",
                    error=str(exc), attempt=attempt, url=url,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Invalid JSON received from catalog API: {exc}",
                ) from exc

        raise HTTPException(
            status_code=502,
            detail=(
                f"Failed to download catalog after {max_retries} attempts: "
                f"{last_exc}" if last_exc else "Unknown error"
            ),
        )

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_catalog(self, data: Any) -> int:
        """Validate the top-level catalog response."""
        if not isinstance(data, list):
            raise HTTPException(
                status_code=502,
                detail=f"Expected a JSON array, got {type(data).__name__}",
            )
        if len(data) == 0:
            raise HTTPException(
                status_code=502, detail="Catalog response is an empty array",
            )
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=502,
                    detail=f"Item at index {i} is not a JSON object",
                )
            if "entity_id" not in item or not item["entity_id"]:
                raise HTTPException(
                    status_code=502,
                    detail=f"Item at index {i} is missing a non-empty 'entity_id'",
                )
            if "name" not in item or not item["name"]:
                raise HTTPException(
                    status_code=502,
                    detail=f"Item at index {i} is missing a non-empty 'name'",
                )
        return len(data)

    # ── Normalisation ────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_record(record: dict) -> dict[str, Any]:
        """Normalise a single raw catalog record into a clean dict for upsert."""
        remote_raw = record.get("remote", "no")
        if isinstance(remote_raw, bool):
            remote = remote_raw
        elif isinstance(remote_raw, str):
            remote = remote_raw.strip().lower() in ("yes", "true", "1", "y")
        else:
            remote = False

        adaptive_raw = record.get("adaptive", "no")
        if isinstance(adaptive_raw, bool):
            adaptive = adaptive_raw
        elif isinstance(adaptive_raw, str):
            adaptive = adaptive_raw.strip().lower() in ("yes", "true", "1", "y")
        else:
            adaptive = False

        scraped_at: datetime | None = None
        scraped_raw = record.get("scraped_at")
        if scraped_raw and isinstance(scraped_raw, str):
            scraped_stripped = scraped_raw.strip()
            if scraped_stripped:
                try:
                    scraped_at = datetime.fromisoformat(scraped_stripped)
                except (ValueError, TypeError):
                    pass

        return {
            "entity_id": str(record.get("entity_id", "")).strip(),
            "name": str(record.get("name", "")).strip(),
            "url": str(record.get("link", "")).strip(),
            "description": str(record.get("description", "")).strip(),
            "duration": str(record.get("duration", "")).strip() or None,
            "remote": remote,
            "adaptive": adaptive,
            "job_levels": [
                str(jl).strip() for jl in (record.get("job_levels") or [])
            ],
            "languages": [
                str(lang).strip() for lang in (record.get("languages") or [])
            ],
            "keys": [
                str(k).strip() for k in (record.get("keys") or [])
            ],
            "scraped_at": scraped_at,
        }
