"""
Unit tests for catalog normalisation and validation logic.

No external dependencies (no database, no HTTP).
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.services.catalog_service import CatalogService


SAMPLE_RECORD = {
    "entity_id": "4302",
    "name": "Global Skills Development Report",
    "link": "https://www.shl.com/products/product-catalog/view/test/",
    "scraped_at": "2026-06-30T12:00:00Z",
    "job_levels": ["Manager"],
    "languages": ["en"],
    "duration": "30 minutes",
    "remote": "yes",
    "adaptive": "no",
    "description": "A report.",
    "keys": ["skills"],
}


class TestNormalisation:
    """Verify _normalise_record produces correct output."""

    @staticmethod
    def _normalise(record: dict) -> dict:
        return CatalogService._normalise_record(record)

    def test_remote_yes_to_true(self) -> None:
        result = self._normalise({"remote": "yes", "entity_id": "x", "name": "x"})
        assert result["remote"] is True

    def test_remote_no_to_false(self) -> None:
        result = self._normalise({"remote": "no", "entity_id": "x", "name": "x"})
        assert result["remote"] is False

    def test_adaptive_yes_to_true(self) -> None:
        result = self._normalise({"adaptive": "yes", "entity_id": "x", "name": "x"})
        assert result["adaptive"] is True

    def test_adaptive_no_to_false(self) -> None:
        result = self._normalise({"adaptive": "no", "entity_id": "x", "name": "x"})
        assert result["adaptive"] is False

    def test_bool_remote_passthrough(self) -> None:
        result = self._normalise({"remote": True, "entity_id": "x", "name": "x"})
        assert result["remote"] is True

    def test_null_arrays_become_empty(self) -> None:
        result = self._normalise({
            "job_levels": None, "languages": None, "keys": None,
            "entity_id": "x", "name": "x",
        })
        assert result["job_levels"] == []
        assert result["languages"] == []
        assert result["keys"] == []

    def test_empty_duration_is_none(self) -> None:
        result = self._normalise({"duration": "", "entity_id": "x", "name": "x"})
        assert result["duration"] is None

    def test_strips_whitespace(self) -> None:
        result = self._normalise({
            "entity_id": "  abc  ", "name": "  Test  ", "link": "  http://x.com  ",
        })
        assert result["entity_id"] == "abc"
        assert result["name"] == "Test"
        assert result["url"] == "http://x.com"

    def test_parses_scraped_at(self) -> None:
        result = self._normalise(SAMPLE_RECORD)
        assert result["scraped_at"] is not None
        assert result["scraped_at"].isoformat() == "2026-06-30T12:00:00+00:00"

    def test_missing_scraped_at_none(self) -> None:
        result = self._normalise({"entity_id": "x", "name": "x"})
        assert result["scraped_at"] is None

    def test_link_mapped_to_url(self) -> None:
        result = self._normalise(SAMPLE_RECORD)
        assert result["url"] == SAMPLE_RECORD["link"]


class TestValidation:
    """Verify _validate_catalog rejects bad data."""

    @staticmethod
    def _valid_data() -> list:
        return [SAMPLE_RECORD]

    def test_valid_list_returns_count(self) -> None:
        svc = CatalogService()
        assert svc._validate_catalog(self._valid_data()) == 1

    def test_non_list_raises_502(self) -> None:
        svc = CatalogService()
        with pytest.raises(HTTPException) as exc:
            svc._validate_catalog({"not": "a list"})
        assert exc.value.status_code == 502

    def test_empty_list_raises_502(self) -> None:
        svc = CatalogService()
        with pytest.raises(HTTPException) as exc:
            svc._validate_catalog([])
        assert exc.value.status_code == 502

    def test_missing_entity_id_raises(self) -> None:
        svc = CatalogService()
        with pytest.raises(HTTPException):
            svc._validate_catalog([{"name": "no id"}])

    def test_missing_name_raises(self) -> None:
        svc = CatalogService()
        with pytest.raises(HTTPException):
            svc._validate_catalog([{"entity_id": "1"}])
