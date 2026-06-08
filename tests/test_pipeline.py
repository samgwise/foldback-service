"""Tests for the multi-pass feedback pipeline with mocked LLM calls."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from src.models import (
    FeedbackRequest,
    FeedbackResponse,
    MappingRequest,
    MappingResponse,
    Rubric,
    RubricCriterion,
    RubricLevel,
)
from src.pipeline import run_feedback_pipeline, run_mapping_suggestion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rubric() -> Rubric:
    return Rubric(
        criteria=[
            RubricCriterion(
                id="c1",
                name="Creativity",
                description="Demonstrates original thinking.",
                max_points=10.0,
                levels=[
                    RubricLevel(name="Pass", description="Adequate", points=5.0),
                    RubricLevel(name="Distinction", description="Excellent", points=10.0),
                ],
            ),
            RubricCriterion(
                id="c2",
                name="Technical Skill",
                description="Quality of implementation.",
                max_points=10.0,
                levels=[
                    RubricLevel(name="Pass", points=5.0),
                    RubricLevel(name="Distinction", points=10.0),
                ],
            ),
        ],
        total_points=20.0,
    )


def _make_request(notes: str = "Good creative effort. Technical side is a bit rough.") -> FeedbackRequest:
    return FeedbackRequest(
        marker_notes=notes,
        student_name="Jane Citizen",
        student_id="12345678",
        rubric=_make_rubric(),
        assignment_brief="Create a radiophonic production.",
        few_shot_examples=None,
        model=None,
    )


def _make_mock_chat(
    unpack_text: str = "The student demonstrated good creative effort. The technical side was somewhat rough.",
    audit_flags: list[dict[str, str]] | None = None,
    compile_criteria: list[dict[str, Any]] | None = None,
    summary_text: str = "Overall, a solid effort with room for technical improvement.",
) -> Callable[[list[dict[str, str]], dict[str, Any] | None, dict[str, Any] | None], Awaitable[str]]:
    """Return a mock chat function that responds with canned data for each pipeline pass."""
    call_count = 0

    async def mock_chat(
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        nonlocal call_count
        call_count += 1
        temp = options.get("temperature", 0.0) if options else 0.0
        system_content = messages[0].get("content", "") if messages else ""

        # Pass 1: unpack (free-form text, temperature 0.3, system mentions "editing assistant")
        if temp == 0.3 and "editing assistant" in system_content and schema is None:
            return unpack_text

        # Pass 4: summary (temperature 0.3, system mentions "academic tutor" and "summary paragraph")
        if temp == 0.3 and "summary paragraph" in system_content:
            return summary_text

        # Pass 2: audit (temperature 0.0, audit schema)
        if temp == 0.0 and schema is not None and "AuditOutput" in str(schema):
            flags = audit_flags or []
            return json.dumps({"review_flags": flags})

        # Pass 3: compile (temperature 0.0, compile schema)
        if temp == 0.0 and schema is not None and "CompileOutput" in str(schema):
            criteria = compile_criteria or [
                {
                    "criterion_id": "c1",
                    "points": 8.0,
                    "max_points": 10.0,
                    "level_selected": "Distinction",
                    "feedback": "You demonstrated strong creative effort.",
                },
                {
                    "criterion_id": "c2",
                    "points": 5.0,
                    "max_points": 10.0,
                    "level_selected": "Pass",
                    "feedback": "Your technical implementation needs improvement.",
                },
            ]
            return json.dumps({"criteria": criteria})

        # Default: mapping or unknown
        return "{}"

    return mock_chat


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestFeedbackPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_returns_feedback_response(self):
        request = _make_request()
        mock_chat = _make_mock_chat()
        response = await run_feedback_pipeline(request, mock_chat, "qwen2.5:14b")

        assert isinstance(response, FeedbackResponse)
        assert len(response.criteria) == 2
        assert response.summary_feedback == "Overall, a solid effort with room for technical improvement."
        assert response.total_points == 13.0

    @pytest.mark.asyncio
    async def test_pipeline_passes_review_flags(self):
        request = _make_request()
        mock_chat = _make_mock_chat(
            audit_flags=[
                {
                    "flag_type": "Vague Feedback",
                    "target_criteria": "Technical Skill",
                    "issue_description": "The marker did not specify which technical aspects need work.",
                },
            ],
        )
        response = await run_feedback_pipeline(request, mock_chat, "qwen2.5:14b")

        assert len(response.review_flags) == 1
        assert response.review_flags[0].flag_type == "Vague Feedback"

    @pytest.mark.asyncio
    async def test_pipeline_filters_blacklisted_criteria(self):
        request = _make_request()
        mock_chat = _make_mock_chat(
            compile_criteria=[
                {
                    "criterion_id": "c1",
                    "points": 8.0,
                    "max_points": 10.0,
                    "level_selected": "Distinction",
                    "feedback": "Good.",
                },
                {
                    "criterion_id": "total",
                    "points": 8.0,
                    "max_points": 10.0,
                    "feedback": "Hallucinated row.",
                },
            ],
        )
        response = await run_feedback_pipeline(request, mock_chat, "qwen2.5:14b")

        assert len(response.criteria) == 1
        assert response.criteria[0].criterion_id == "c1"

    @pytest.mark.asyncio
    async def test_pipeline_filters_blacklisted_flags(self):
        request = _make_request()
        mock_chat = _make_mock_chat(
            audit_flags=[
                {
                    "flag_type": "Missing Information",
                    "target_criteria": "Summary",
                    "issue_description": "Should be filtered out.",
                },
                {
                    "flag_type": "Missing Information",
                    "target_criteria": "Global",
                    "issue_description": "Should remain.",
                },
            ],
        )
        response = await run_feedback_pipeline(request, mock_chat, "qwen2.5:14b")

        assert len(response.review_flags) == 1
        assert response.review_flags[0].target_criteria == "Global"


# ---------------------------------------------------------------------------
# Mapping suggestion tests
# ---------------------------------------------------------------------------

class TestMappingSuggestion:
    @pytest.mark.asyncio
    async def test_suggest_mapping_returns_valid_response(self):
        request = MappingRequest(
            csv_headers=["Student ID", "Name", "Score"],
            target_schema="grades",
            sample_rows=[["123", "Alice", "85"]],
        )

        async def mock_chat(messages, schema=None, options=None):
            return json.dumps({
                "column_mapping": {"student_id": "Student ID", "marks": "Score"},
                "confidence": 0.85,
                "suggestions": [
                    {
                        "field": "student_id",
                        "column": "Student ID",
                        "confidence": 0.95,
                        "reason": "Exact match",
                    },
                ],
            })

        response = await run_mapping_suggestion(request, mock_chat)

        assert isinstance(response, MappingResponse)
        assert response.confidence == 0.85
        assert len(response.suggestions) == 1
        assert response.suggestions[0].field == "student_id"
