"""Multi-pass LLM pipeline for feedback generation and mapping suggestion."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from src.models import (
    AuditOutput,
    CompileOutput,
    FeedbackRequest,
    FeedbackResponse,
    MappingRequest,
    MappingResponse,
    MappingSuggestion,
    SummaryOutput,
    UnpackOutput,
    _is_blacklisted,
)

logger = logging.getLogger(__name__)

# Type alias for the chat function signature used by providers
ChatFn = Callable[[list[dict[str, str]], dict[str, Any] | None, dict[str, Any] | None], Awaitable[str]]


# ---------------------------------------------------------------------------
# Pass 1 — Unpack / Sanitisation
# ---------------------------------------------------------------------------
_UNPACK_SYSTEM = """You are an objective editing assistant. Your single task is to rewrite messy, abbreviated grading notes into a clean, smooth, descriptive narrative paragraph.

CRITICAL CLEANING RULES:
1. If there are rhetorical questions, rewrite them into flat statements of fact.
2. Expand all shorthand, typos, and fragments into complete sentences.
3. Remove any personal notes, emotional rants, or meta-commentary.
4. Output ONLY the polished, sanitised narrative text. Do not write JSON, titles, or introductions."""


async def _pass_unpack(marker_notes: str, chat: ChatFn, model: str) -> UnpackOutput:
    response = await chat(
        messages=[
            {"role": "system", "content": _UNPACK_SYSTEM},
            {"role": "user", "content": f"Please sanitise this text:\n{marker_notes}"},
        ],
        options={"temperature": 0.3, "model": model},
    )
    return UnpackOutput(sanitized_notes=response.strip())


# ---------------------------------------------------------------------------
# Pass 2 — Audit (quality flags)
# ---------------------------------------------------------------------------
_AUDIT_SYSTEM = """You are an internal quality assurance auditor. Look at the grading notes, the rubric, and the assignment brief.
Identify if there are any missing details, vague metrics, or total score guideline mismatches.
Output your findings strictly using the ReviewFlag schema structure."""


async def _pass_audit(
    sanitized_notes: str,
    rubric_json: str,
    assignment_brief: str | None,
    chat: ChatFn,
    model: str,
) -> AuditOutput:
    brief_text = assignment_brief or "No specific assignment brief guidelines provided."
    payload = f"""[CRITERIA DEFAULTS]
{rubric_json}

[ASSIGNMENT BRIEF]
{brief_text}

[TARGET GRADING NOTES]
Notes: {sanitized_notes}"""

    schema = AuditOutput.model_json_schema()
    response = await chat(
        messages=[
            {"role": "system", "content": _AUDIT_SYSTEM},
            {"role": "user", "content": payload},
        ],
        schema=schema,
        options={"temperature": 0.0, "model": model},
    )
    data = json.loads(response)
    # Filter out blacklisted flags
    flags = [f for f in data.get("review_flags", []) if not _is_blacklisted(f.get("target_criteria", "")) or f.get("target_criteria") == "Global"]
    return AuditOutput(review_flags=flags)


# ---------------------------------------------------------------------------
# Pass 3 — Compile (criterion assessments)
# ---------------------------------------------------------------------------
_COMPILE_SYSTEM = """You are an expert academic tutor speaking directly to a student. Your job is to translate raw grading notes into clear, encouraging, second-person ("You") feedback criteria blocks.

CRITICAL RESTRICTIONS:
1. Speak directly to the student: Use "You", "Your project", "Your presentation".
2. Zero Meta-Commentary: Never say "The notes state", "The professor adjusted", or "This criteria shows".
3. Strict Domain Isolation: Only evaluate items explicitly mentioned in the rubric categories. Do not create pseudo-criteria rows like "Total Score".

GRADIENT SCORING PROTOCOL:
Treat rubric levels as anchor points on a continuous scale — NOT as discrete buckets.
- Compare the student's work quality against the level descriptors.
- Interpolate the score between the two closest level anchors.
  Example: if the work falls between "Credit" (8/12) and "Distinction" (10/12) with slightly more credit-like qualities, you might award 8.5/12 or 9/12.
- The "points" field should reflect this interpolated value, not just snap to a level's exact point value.
- You may award any value from 0 up to max_points, not limited to the predefined level points.
- If the notes mention partial credit or a range, interpolate accordingly.
- Use "level_selected" to indicate which named level is the closest reference point, but do not restrict your point value to that level's points.

ZERO-DATA SCORING PROTOCOL:
If the raw grading notes do not mention a specific rubric category or fail to provide a clear deduction value, you must:
- Award the FULL maximum points possible for that category by default.
- Write a supportive, general acknowledgement in the feedback (e.g., "Your implementation meets standard expectations for this milestone.").
- Do not try to guess a penalty if the notes are silent on that section."""


async def _pass_compile(
    sanitized_notes: str,
    rubric_json: str,
    few_shot_examples: str | None,
    chat: ChatFn,
    model: str,
) -> CompileOutput:
    examples_text = few_shot_examples or "No historical examples provided. Rely entirely on the rubric guidelines below."
    payload = f"""[HISTORICAL BENCHMARKS]
{examples_text}

[TARGET RUBRIC CRITERIA EXPECTED]
{rubric_json}

[RAW GRADING FRAGMENTS TO TRANSLATE]
{sanitized_notes}"""

    schema = CompileOutput.model_json_schema()
    response = await chat(
        messages=[
            {"role": "system", "content": _COMPILE_SYSTEM},
            {"role": "user", "content": payload},
        ],
        schema=schema,
        options={"temperature": 0.0, "model": model},
    )
    data = json.loads(response)
    criteria = data.get("criteria", [])
    # Filter out blacklisted criteria rows
    criteria = [c for c in criteria if not _is_blacklisted(c.get("criterion_id", ""))]
    return CompileOutput(criteria=criteria)


# ---------------------------------------------------------------------------
# Pass 4 — Summary Generation
# ---------------------------------------------------------------------------
_SUMMARY_SYSTEM = """You are an expert academic tutor. Write a polished, objective summary paragraph for the student based on their criterion-level feedback.
Be encouraging but honest. Use second-person ("You", "Your"). Do not mention rubric categories by id.
Keep the summary under 200 words."""


async def _pass_summary(
    criteria: list[dict[str, Any]],
    chat: ChatFn,
    model: str,
) -> SummaryOutput:
    criteria_json = json.dumps(criteria, indent=2)
    response = await chat(
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": f"Write a summary for these criterion assessments:\n{criteria_json}"},
        ],
        options={"temperature": 0.3, "model": model},
    )
    return SummaryOutput(summary_feedback=response.strip())


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------
async def run_feedback_pipeline(
    request: FeedbackRequest,
    chat: ChatFn,
    model: str,
) -> FeedbackResponse:
    """Execute the 4-pass feedback pipeline and return the final response."""
    rubric_json = request.rubric.model_dump_json(indent=2)

    # Pass 1: Unpack
    unpack_result = await _pass_unpack(request.marker_notes, chat, model)
    logger.debug("Pass 1 complete: unpacked notes")

    # Pass 2: Audit
    audit_result = await _pass_audit(
        unpack_result.sanitized_notes,
        rubric_json,
        request.assignment_brief,
        chat,
        model,
    )
    logger.debug("Pass 2 complete: %d flags detected", len(audit_result.review_flags))

    # Pass 3: Compile
    compile_result = await _pass_compile(
        unpack_result.sanitized_notes,
        rubric_json,
        request.few_shot_examples,
        chat,
        model,
    )
    logger.debug("Pass 3 complete: %d criteria compiled", len(compile_result.criteria))

    # Pass 4: Summary
    summary_result = await _pass_summary(
        [c.model_dump() for c in compile_result.criteria],
        chat,
        model,
    )
    logger.debug("Pass 4 complete: summary generated")

    total_points = sum(c.points for c in compile_result.criteria)

    return FeedbackResponse(
        criteria=compile_result.criteria,
        review_flags=audit_result.review_flags,
        summary_feedback=summary_result.summary_feedback,
        total_points=total_points,
    )


# ---------------------------------------------------------------------------
# Mapping suggestion pipeline
# ---------------------------------------------------------------------------
_MAPPING_SYSTEM = """You are a data integration assistant. Given CSV headers and a target database schema name, propose a mapping from internal field names to the external CSV column names.
Also provide a confidence score and a brief reason for each mapping.
Return valid JSON matching the MappingResponse schema."""


async def run_mapping_suggestion(
    request: MappingRequest,
    chat: ChatFn,
) -> MappingResponse:
    """Analyse CSV headers and propose a column mapping."""
    schema = MappingResponse.model_json_schema()
    payload = f"""Target schema: {request.target_schema}
CSV headers: {json.dumps(request.csv_headers)}
Sample rows: {json.dumps(request.sample_rows)}

Propose a column mapping from internal fields to the CSV headers above."""

    response = await chat(
        messages=[
            {"role": "system", "content": _MAPPING_SYSTEM},
            {"role": "user", "content": payload},
        ],
        schema=schema,
        options={"temperature": 0.0},
    )
    data = json.loads(response)
    return MappingResponse(
        column_mapping=data.get("column_mapping", {}),
        confidence=data.get("confidence", 0.0),
        suggestions=[MappingSuggestion(**s) for s in data.get("suggestions", [])],
    )
