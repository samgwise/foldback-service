"""Pydantic models for the Foldback Service API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Blacklist constants
# ---------------------------------------------------------------------------
# Criteria names that indicate hallucinated meta-rows and must be filtered out.
CRITERIA_BLACKLIST: set[str] = {
    "total",
    "total score",
    "total summary",
    "review flags",
    "review flag",
    "summary",
    "summary_feedback",
    "summary feedback",
    "total_points",
    "total points",
    "breakdown",
}


def _is_blacklisted(name: str) -> bool:
    """Return True if the criterion name matches a blacklisted meta-row."""
    return name.lower().strip() in CRITERIA_BLACKLIST


# ---------------------------------------------------------------------------
# Rubric structures (mirrors assessments.rubric JSON in the Tauri app)
# ---------------------------------------------------------------------------
class RubricLevel(BaseModel):
    name: str = Field(description="Name of the achievement level (e.g., 'High Distinction').")
    description: str | None = Field(default=None, description="Descriptor for this level.")
    points: float = Field(description="Points awarded when this level is selected.")


class RubricCriterion(BaseModel):
    id: str = Field(description="Our internal criterion identifier (e.g., 'c1', 'c2').")
    name: str = Field(description="Display name of the criterion.")
    description: str | None = Field(default=None, description="Detailed description of what the criterion assesses.")
    max_points: float = Field(description="Maximum points achievable for this criterion.")
    levels: list[RubricLevel] = Field(
        default_factory=list,
        description="Ordered list of achievement levels from lowest to highest.",
    )

    @field_validator("id")
    @classmethod
    def _id_not_blacklisted(cls, v: str) -> str:
        if _is_blacklisted(v):
            raise ValueError(f"Criterion id '{v}' is blacklisted")
        return v

    @field_validator("name")
    @classmethod
    def _name_not_blacklisted(cls, v: str) -> str:
        if _is_blacklisted(v):
            raise ValueError(f"Criterion name '{v}' is blacklisted")
        return v


class Rubric(BaseModel):
    criteria: list[RubricCriterion] = Field(description="List of marking criteria.")
    total_points: float = Field(description="Sum of all criterion max_points.")

    @field_validator("total_points")
    @classmethod
    def _total_points_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("total_points must be non-negative")
        return v


# ---------------------------------------------------------------------------
# Feedback generation request / response
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    marker_notes: str = Field(description="Rough notes written by the marker.")
    student_name: str = Field(description="Full name of the student.")
    student_id: str = Field(description="Student identifier (e.g., student number).")
    rubric: Rubric = Field(description="Rubric structure with criteria and levels.")
    assignment_brief: str | None = Field(default=None, description="Optional assignment brief / instructions.")
    few_shot_examples: str | None = Field(default=None, description="Optional historical marking examples.")
    precedents: list["PrecedentExample"] = Field(default_factory=list, description="Historical grading precedents for case-law-aware compilation.")
    model: str | None = Field(default=None, description="Optional model name override for this request.")


class PrecedentExample(BaseModel):
    """A historical grading precedent used as case-law reference during compilation."""
    model_config = ConfigDict(populate_by_name=True)
    massaged_notes: str = Field(
        description="Sanitised/combined notes from the historical grading case.",
        alias="massagedNotes",
    )
    criterion_assessments: list[dict] = Field(
        default_factory=list,
        description="Historical CriterionAssessment outputs (JSON dicts with criterion_id, points, feedback, etc.).",
        alias="criterionAssessments",
    )


class EmbeddingRequest(BaseModel):
    """Request to generate a text embedding vector."""
    text: str = Field(description="Input text to embed.")
    model: str | None = Field(default=None, description="Optional embedding model name override.")


class EmbeddingResponse(BaseModel):
    """Response containing an embedding vector."""
    embedding: list[float] = Field(description="The embedding vector as a list of floats.")
    dimension: int = Field(description="Dimensionality of the embedding vector.")


class CriterionAssessment(BaseModel):
    criterion_id: str = Field(description="Matches the rubric criterion id.")
    points: float = Field(description="Points awarded for this criterion. Use a continuous scale from 0 to max_points; interpolate between rubric level anchors rather than snapping to a single level's point value.")
    max_points: float = Field(description="Maximum points available for this criterion.")
    level_selected: str | None = Field(default=None, description="The name of the rubric level closest to the awarded score (reference only — does not constrain the points value).")
    feedback: str = Field(description="Student-facing feedback for this criterion.")

    @field_validator("criterion_id")
    @classmethod
    def _criterion_id_not_blacklisted(cls, v: str) -> str:
        if _is_blacklisted(v):
            raise ValueError(f"Criterion id '{v}' is blacklisted")
        return v


class ReviewFlag(BaseModel):
    flag_type: Literal[
        "Vague Feedback",
        "Missing Information",
        "Guideline Mismatch",
        "Contradictory Statement",
        "Text Conflict",
    ] = Field(description="Category of the detected issue.")
    target_criteria: str = Field(description="Rubric criterion this flag relates to, or 'Global'.")
    issue_description: str = Field(description="Explanation of the issue for human review.")


class FeedbackResponse(BaseModel):
    criteria: list[CriterionAssessment] = Field(
        description="Structured criterion-level assessments mapped to the rubric.",
    )
    review_flags: list[ReviewFlag] = Field(
        default_factory=list,
        description="Flags raised during auditing of the marker notes.",
    )
    summary_feedback: str = Field(description="Polished overall summary for the student.")
    total_points: float = Field(description="Sum of awarded points across all criteria.")
    grounding_warnings: list[dict[str, str]] = Field(
        default_factory=list,
        description="Warnings for criteria where feedback could not be fully grounded in marker notes.",
    )

    @field_validator("criteria")
    @classmethod
    def _filter_blacklisted_criteria(cls, v: list[CriterionAssessment]) -> list[CriterionAssessment]:
        return [c for c in v if not _is_blacklisted(c.criterion_id)]

    @field_validator("review_flags")
    @classmethod
    def _filter_blacklisted_flags(cls, v: list[ReviewFlag]) -> list[ReviewFlag]:
        return [f for f in v if not _is_blacklisted(f.target_criteria) or f.target_criteria == "Global"]


# ---------------------------------------------------------------------------
# Column mapping suggestion request / response
# ---------------------------------------------------------------------------
class MappingRequest(BaseModel):
    csv_headers: list[str] = Field(description="Column headers from the uploaded CSV.")
    target_schema: str = Field(description="Target table name (e.g., 'students', 'enrolments', 'grades').")
    sample_rows: list[list[str]] = Field(
        default_factory=list,
        description="First few rows of the CSV to aid inference.",
    )


class MappingSuggestion(BaseModel):
    field: str = Field(description="Internal field name.")
    column: str = Field(description="Best-matching CSV header.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0.")
    reason: str = Field(description="Explanation of the match.")


class MappingResponse(BaseModel):
    column_mapping: dict[str, str] = Field(description="Map of internal field to external CSV header.")
    confidence: float = Field(description="Overall mapping confidence (0.0–1.0).")
    suggestions: list[MappingSuggestion] = Field(
        default_factory=list,
        description="Per-field mapping suggestions with confidence scores.",
    )


# ---------------------------------------------------------------------------
# Internal pipeline artefacts (not exposed directly via the API)
# ---------------------------------------------------------------------------
class UnpackOutput(BaseModel):
    """Pass 1 — sanitised narrative text."""
    sanitized_notes: str = Field(description="Cleaned, coherent narrative derived from raw marker notes.")


class AuditOutput(BaseModel):
    """Pass 2 — quality flags."""
    review_flags: list[ReviewFlag] = Field(default_factory=list)


class CompileOutput(BaseModel):
    """Pass 3 — structured criterion assessments."""
    criteria: list[CriterionAssessment] = Field(description="Criterion-level scores and feedback.")


class UngroundedItem(BaseModel):
    """Pass 3.5 — grounding verification finding."""
    criterion_id: str = Field(description="Matches the rubric criterion id.")
    original_feedback: str = Field(description="The original feedback text that contains ungrounded claims.")
    ungrounded_phrases: list[str] = Field(description="Exact phrases from the feedback not present in marker notes.")
    reason: str = Field(description="Explanation of why the feedback is ungrounded.")


class GroundingOutput(BaseModel):
    """Pass 3.5 — grounding verification result."""
    ungrounded_items: list[UngroundedItem] = Field(
        default_factory=list,
        description="Criteria with feedback not grounded in marker notes.",
    )


class RefinedCriterion(BaseModel):
    """A single criterion assessment refined for grounding."""
    criterion_id: str = Field(description="Matches the rubric criterion id.")
    points: float = Field(description="Points awarded for this criterion.")
    max_points: float = Field(description="Maximum points available for this criterion.")
    level_selected: str | None = Field(default=None, description="The rubric level closest to the awarded score.")
    feedback: str = Field(description="Student-facing feedback, grounded in marker notes.")


class RefinementOutput(BaseModel):
    """Pass 3.6 — refined criterion assessments after grounding correction."""
    refined_criteria: list[RefinedCriterion] = Field(
        default_factory=list,
        description="Refined criterion assessments with ungrounded content removed.",
    )


class SummaryOutput(BaseModel):
    """Pass 4 — overall summary paragraph."""
    summary_feedback: str = Field(description="Polished, student-facing summary.")
