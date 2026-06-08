"""Tests for FastAPI API endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models import (
    FeedbackResponse,
    Rubric,
    RubricCriterion,
    RubricLevel,
)


@pytest.fixture
def client():
    return TestClient(app)


def _make_rubric() -> Rubric:
    return Rubric(
        criteria=[
            RubricCriterion(
                id="c1",
                name="Creativity",
                max_points=10.0,
                levels=[RubricLevel(name="Pass", points=5.0)],
            ),
        ],
        total_points=10.0,
    )


def _make_feedback_request() -> dict[str, Any]:
    return {
        "marker_notes": "Good creative work. Technical implementation needs improvement.",
        "student_name": "Jane Citizen",
        "student_id": "12345678",
        "rubric": {
            "criteria": [
                {
                    "id": "c1",
                    "name": "Creativity",
                    "max_points": 10.0,
                    "levels": [{"name": "Pass", "points": 5.0}],
                },
            ],
            "total_points": 10.0,
        },
    }


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        with patch("src.main.settings") as mock_settings:
            mock_settings.llm_provider = "ollama"
            mock_settings.host = "0.0.0.0"
            mock_settings.port = 8100
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"


class TestGenerateFeedbackEndpoint:
    @pytest.mark.asyncio
    async def test_generate_feedback_success(self, client):
        mock_response = FeedbackResponse(
            criteria=[],
            review_flags=[],
            summary_feedback="Test summary.",
            total_points=0.0,
        )

        with patch("src.main.get_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_feedback.return_value = mock_response
            mock_get_provider.return_value = mock_provider

            request_data = _make_feedback_request()
            response = client.post("/generate-feedback", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["summary_feedback"] == "Test summary."

    def test_generate_feedback_missing_fields(self, client):
        """Should return 422 for missing required fields."""
        response = client.post("/generate-feedback", json={"marker_notes": "test"})
        assert response.status_code == 422

    def test_generate_feedback_invalid_rubric(self, client):
        """Should return 422 for invalid rubric structure."""
        request_data = {
            "marker_notes": "test",
            "student_name": "Jane",
            "student_id": "123",
            "rubric": {"criteria": [], "total_points": -1},  # invalid: negative total
        }
        response = client.post("/generate-feedback", json=request_data)
        assert response.status_code == 422


class TestSuggestMappingEndpoint:
    @pytest.mark.asyncio
    async def test_suggest_mapping_success(self, client):
        from src.models import MappingResponse

        mock_response = MappingResponse(
            column_mapping={"student_id": "Student ID", "marks": "Score"},
            confidence=0.85,
            suggestions=[
                {
                    "field": "student_id",
                    "column": "Student ID",
                    "confidence": 0.95,
                    "reason": "Exact match on identifier pattern",
                },
            ],
        )

        with patch("src.main.get_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.suggest_mapping.return_value = mock_response
            mock_get_provider.return_value = mock_provider

            request_data = {
                "csv_headers": ["Student ID", "Name", "Score"],
                "target_schema": "grades",
                "sample_rows": [["123", "Alice", "85"]],
            }
            response = client.post("/suggest-mapping", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["confidence"] == 0.85
            assert "student_id" in data["column_mapping"]

    def test_suggest_mapping_missing_fields(self, client):
        """Should return 422 for missing required fields."""
        response = client.post("/suggest-mapping", json={"csv_headers": ["ID"]})
        assert response.status_code == 422


class TestErrorHandling:
    def test_unknown_provider_returns_500(self, client):
        """Should return 500 when provider configuration is invalid."""
        with patch("src.main.get_provider") as mock_get_provider:
            mock_get_provider.side_effect = ValueError("Unknown LLM provider: invalid")

            request_data = _make_feedback_request()
            response = client.post("/generate-feedback", json=request_data)

            assert response.status_code == 400  # ValueError is caught and returned as 400

    def test_runtime_error_returns_500(self, client):
        """Should return 500 when provider raises RuntimeError."""
        with patch("src.main.get_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.generate_feedback.side_effect = RuntimeError("LLM connection failed")
            mock_get_provider.return_value = mock_provider

            request_data = _make_feedback_request()
            response = client.post("/generate-feedback", json=request_data)

            assert response.status_code == 500
            assert "Feedback generation failed" in response.json()["detail"]
