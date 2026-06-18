"""Tests for LLM provider abstraction."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from src.models import (
    FeedbackRequest,
    MappingRequest,
    Rubric,
    RubricCriterion,
    RubricLevel,
)
from src.providers import LLMProvider, get_provider


class TestProviderFactory:
    def test_default_provider_is_ollama(self):
        with patch("src.providers.settings") as mock_settings:
            mock_settings.llm_provider = "ollama"
            mock_settings.ollama_model = "qwen2.5:14b"
            provider = get_provider()
            assert provider.__class__.__name__ == "OllamaProvider"

    def test_openai_provider_factory(self):
        with patch("src.providers.settings") as mock_settings:
            mock_settings.llm_provider = "openai"
            mock_settings.openai_base_url = "http://example.com/v1"
            mock_settings.openai_api_key = "test-key"
            mock_settings.openai_model = "gpt-4"
            mock_settings.ollama_model = "qwen2.5:14b"
            provider = get_provider()
            assert provider.__class__.__name__ == "OpenAICompatibleProvider"

    def test_invalid_provider_raises(self):
        with patch("src.providers.settings") as mock_settings:
            mock_settings.llm_provider = "invalid"
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_provider()


class TestProviderInterface:
    """Verify the abstract LLMProvider contract is correctly implemented."""

    @pytest.mark.asyncio
    async def test_generate_feedback_calls_pipeline(self):
        """Ensure generate_feedback calls run_feedback_pipeline."""
        # Create a concrete implementation for testing
        call_order = []

        class TestProvider(LLMProvider):
            async def chat(
                self,
                messages: list[dict[str, str]],
                schema: dict[str, Any] | None = None,
                options: dict[str, Any] | None = None,
            ) -> str:
                temp = options.get("temperature", 0.0) if options else 0.0
                system_content = messages[0].get("content", "") if messages else ""
                call_order.append(("chat", temp, "editing assistant" in system_content, "summary paragraph" in system_content, schema is not None))

                # Pass 1: unpack (temperature 0.3, no schema)
                if temp == 0.3 and schema is None and "editing assistant" in system_content:
                    return "The student performed adequately."

                # Pass 4: summary (temperature 0.3, summary system prompt)
                if temp == 0.3 and "summary paragraph" in system_content:
                    return "Test."

                # Pass 2: audit (temperature 0.0, with schema)
                if temp == 0.0 and schema is not None and "AuditOutput" in str(schema):
                    return json.dumps({"review_flags": []})

                # Pass 3: compile (temperature 0.0, with schema)
                if temp == 0.0 and schema is not None and "CompileOutput" in str(schema):
                    return json.dumps({"criteria": []})

                return "{}"

            async def generate_feedback(self, request: FeedbackRequest):
                from src.pipeline import run_feedback_pipeline
                return await run_feedback_pipeline(request, self.chat, "test-model")

            async def suggest_mapping(self, request: MappingRequest):
                from src.pipeline import run_mapping_suggestion
                return await run_mapping_suggestion(request, self.chat)

            async def embed_text(self, text: str, model: str | None = None) -> list[float]:
                return [0.1, 0.2, 0.3]

        provider = TestProvider()
        rubric = Rubric(
            criteria=[
                RubricCriterion(
                    id="c1",
                    name="Test Criterion",
                    max_points=10.0,
                    levels=[RubricLevel(name="Pass", points=5.0)],
                ),
            ],
            total_points=10.0,
        )
        request = FeedbackRequest(
            marker_notes="Test notes.",
            student_name="Test Student",
            student_id="TS123",
            rubric=rubric,
        )

        response = await provider.generate_feedback(request)
        assert response.summary_feedback == "Test."
        assert len(call_order) == 4  # All 4 passes should be called

    @pytest.mark.asyncio
    async def test_suggest_mapping_calls_pipeline(self):
        """Ensure suggest_mapping calls run_mapping_suggestion."""

        class TestProvider(LLMProvider):
            async def chat(
                self,
                messages: list[dict[str, str]],
                schema: dict[str, Any] | None = None,
                options: dict[str, Any] | None = None,
            ) -> str:
                return json.dumps({
                    "column_mapping": {"student_id": "ID"},
                    "confidence": 0.9,
                    "suggestions": [{"field": "student_id", "column": "ID", "confidence": 0.9, "reason": "Match"}],
                })

            async def generate_feedback(self, request: FeedbackRequest):
                pass

            async def suggest_mapping(self, request: MappingRequest):
                from src.pipeline import run_mapping_suggestion
                return await run_mapping_suggestion(request, self.chat)

            async def embed_text(self, text: str, model: str | None = None) -> list[float]:
                return [0.1, 0.2, 0.3]

        provider = TestProvider()
        request = MappingRequest(csv_headers=["ID", "Name"], target_schema="students")
        response = await provider.suggest_mapping(request)

        assert response.confidence == 0.9
        assert response.column_mapping == {"student_id": "ID"}
