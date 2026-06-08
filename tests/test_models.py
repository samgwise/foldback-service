"""Tests for Pydantic model validation and blacklist filtering."""

import pytest
from pydantic import ValidationError

from src.models import (
    CRITERIA_BLACKLIST,
    CriterionAssessment,
    FeedbackResponse,
    MappingRequest,
    MappingResponse,
    MappingSuggestion,
    ReviewFlag,
    Rubric,
    RubricCriterion,
    RubricLevel,
    _is_blacklisted,
)

# ---------------------------------------------------------------------------
# Blacklist tests
# ---------------------------------------------------------------------------

class TestBlacklist:
    def test_blacklisted_terms_are_detected(self):
        for term in CRITERIA_BLACKLIST:
            assert _is_blacklisted(term) is True

    def test_non_blacklisted_terms_pass(self):
        assert _is_blacklisted("Creativity") is False
        assert _is_blacklisted("Technical Skill") is False

    def test_blacklist_is_case_insensitive(self):
        assert _is_blacklisted("Total Score") is True
        assert _is_blacklisted("TOTAL SCORE") is True

    def test_blacklist_ignores_whitespace(self):
        assert _is_blacklisted("  total  ") is True


# ---------------------------------------------------------------------------
# Rubric model tests
# ---------------------------------------------------------------------------

class TestRubricModels:
    def _make_criterion(self, id: str = "c1", name: str = "Creativity", max_points: float = 10.0) -> RubricCriterion:
        return RubricCriterion(
            id=id,
            name=name,
            max_points=max_points,
            levels=[RubricLevel(name="Pass", points=5.0), RubricLevel(name="Distinction", points=10.0)],
        )

    def test_valid_rubric(self):
        rubric = Rubric(
            criteria=[self._make_criterion()],
            total_points=10.0,
        )
        assert rubric.total_points == 10.0

    def test_rubric_rejects_blacklisted_criterion_id(self):
        with pytest.raises(ValidationError, match="blacklisted"):
            self._make_criterion(id="total")

    def test_rubric_rejects_blacklisted_criterion_name(self):
        with pytest.raises(ValidationError, match="blacklisted"):
            self._make_criterion(name="Review Flags")


# ---------------------------------------------------------------------------
# CriterionAssessment tests
# ---------------------------------------------------------------------------

class TestCriterionAssessment:
    def test_valid_assessment(self):
        assessment = CriterionAssessment(
            criterion_id="c1",
            points=8.0,
            max_points=10.0,
            level_selected="Distinction",
            feedback="Good work on the presentation.",
        )
        assert assessment.points == 8.0

    def test_blacklisted_criterion_id_rejected(self):
        with pytest.raises(ValidationError, match="blacklisted"):
            CriterionAssessment(
                criterion_id="total score",
                points=0.0,
                max_points=10.0,
                feedback="Should not pass",
            )


# ---------------------------------------------------------------------------
# ReviewFlag tests
# ---------------------------------------------------------------------------

class TestReviewFlag:
    def test_valid_flag(self):
        flag = ReviewFlag(
            flag_type="Vague Feedback",
            target_criteria="Creativity",
            issue_description="The feedback lacks specific examples.",
        )
        assert flag.flag_type == "Vague Feedback"

    def test_invalid_flag_type_rejected(self):
        with pytest.raises(ValidationError):
            ReviewFlag(
                flag_type="Invalid Type",
                target_criteria="c1",
                issue_description="Bad type",
            )


# ---------------------------------------------------------------------------
# FeedbackResponse tests
# ---------------------------------------------------------------------------

class TestFeedbackResponse:
    def test_response_filters_blacklisted_criteria(self):
        # Build criteria list directly to bypass CriterionAssessment-level blacklist validator
        # (the FeedbackResponse validator filters after construction)
        valid = CriterionAssessment(criterion_id="c1", points=8.0, max_points=10.0, feedback="Good.")
        # Use model_construct to bypass the field-level validator for the blacklisted one
        blacklisted = CriterionAssessment.model_construct(
            criterion_id="total", points=8.0, max_points=10.0, feedback="Bad."
        )
        response = FeedbackResponse(
            criteria=[valid, blacklisted],
            review_flags=[],
            summary_feedback="Well done.",
            total_points=8.0,
        )
        # After validation, blacklisted criteria should be filtered out
        assert len(response.criteria) == 1
        assert response.criteria[0].criterion_id == "c1"

    def test_response_allows_global_flag(self):
        response = FeedbackResponse(
            criteria=[
                CriterionAssessment(criterion_id="c1", points=8.0, max_points=10.0, feedback="Good."),
            ],
            review_flags=[
                ReviewFlag(
                    flag_type="Missing Information",
                    target_criteria="Global",
                    issue_description="No feedback for c1.",
                ),
            ],
            summary_feedback="Well done.",
            total_points=8.0,
        )
        assert len(response.review_flags) == 1
        assert response.review_flags[0].target_criteria == "Global"


# ---------------------------------------------------------------------------
# MappingRequest / MappingResponse tests
# ---------------------------------------------------------------------------

class TestMappingModels:
    def test_valid_mapping_request(self):
        request = MappingRequest(
            csv_headers=["Student ID", "Name", "Score"],
            target_schema="grades",
            sample_rows=[["123", "Alice", "85"]],
        )
        assert request.target_schema == "grades"

    def test_valid_mapping_response(self):
        response = MappingResponse(
            column_mapping={"student_id": "Student ID", "marks": "Score"},
            confidence=0.9,
            suggestions=[
                MappingSuggestion(
                    field="student_id",
                    column="Student ID",
                    confidence=0.95,
                    reason="Exact match",
                ),
            ],
        )
        assert response.confidence == 0.9
