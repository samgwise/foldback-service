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

_UNPACK_SYSTEM_WITH_PRECEDENTS = """You are an objective editing assistant. Your task is to rewrite messy, abbreviated grading notes into a clean, smooth, descriptive narrative paragraph.

CRITICAL CLEANING RULES:
1. If there are rhetorical questions, rewrite them into flat statements of fact.
2. Expand all shorthand, typos, and fragments into complete sentences.
3. Remove any personal notes, emotional rants, or meta-commentary.
4. PRESERVE all specific, actionable details from the marker notes — do not over-sanitise or remove concrete examples, specific feedback points, or detailed observations.
5. Output ONLY the polished, sanitised narrative text. Do not write JSON, titles, or introductions.

RUBRIC CONTEXT:
The rubric structure is provided below for terminology awareness. Use it to understand the domain language, but do not let it dominate your sanitisation decisions. The marker notes take priority.

HISTORICAL PRECEDENTS:
Precedent examples are provided to guide your understanding of how similar notes have been sanitised previously. Use them to inform your approach to ambiguous or shorthand text, but apply judgement based on the current content."""


async def _pass_unpack(
    marker_notes: str,
    chat: ChatFn,
    model: str,
    precedents: list[dict[str, Any]] | None = None,
) -> UnpackOutput:
    if precedents:
        # Build precedent block for unpack pass
        precedent_lines = []
        for i, p in enumerate(precedents, 1):
            precedent_lines.append(f"### PRECEDENT {i}")
            precedent_lines.append(f"[PAST MARKER NOTES]:\n{p.get('massaged_notes', '')}")
            if p.get('criterion_assessments'):
                precedent_lines.append(f"[PAST CRITERION SCORES]:\n{json.dumps(p.get('criterion_assessments', []), indent=2)}")
            precedent_lines.append("")
        precedents_block = "\n".join(precedent_lines)
        rubric_context = f"[HISTORICAL PRECEDENTS FOR REFERENCE]:\n{precedents_block}"
    else:
        rubric_context = ""

    system_prompt = _UNPACK_SYSTEM_WITH_PRECEDENTS if precedents else _UNPACK_SYSTEM

    payload = f"""Please sanitise this text:\n{marker_notes}

{rubric_context}""".strip()

    response = await chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
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

_AUDIT_SYSTEM_WITH_PRECEDENTS = """You are an internal quality assurance auditor. Review the grading notes against the rubric, assignment brief, and historical grading precedents.

CRITICAL AUDIT INSTRUCTIONS:
1. Identify missing details, vague metrics, or total score guideline mismatches.
2. Check for divergence from established precedent patterns — flag when the current grading approach differs from historical precedents without clear justification.
3. Flag when marker notes contradict established precedent interpretations — if a precedent shows consistent treatment of a specific criterion, but current notes diverge, raise a flag.
4. Output your findings strictly using the ReviewFlag schema structure."""


async def _pass_audit(
    sanitized_notes: str,
    rubric_json: str,
    assignment_brief: str | None,
    chat: ChatFn,
    model: str,
    precedents: list[dict[str, Any]] | None = None,
) -> AuditOutput:
    brief_text = assignment_brief or "No specific assignment brief guidelines provided."

    # Build precedent block if available
    precedent_text = ""
    if precedents:
        system_prompt = _AUDIT_SYSTEM_WITH_PRECEDENTS
        precedent_lines = []
        for i, p in enumerate(precedents, 1):
            precedent_lines.append(f"### PRECEDENT {i}")
            precedent_lines.append(f"[PAST MARKER NOTES]:\n{p.get('massaged_notes', '')}")
            if p.get('criterion_assessments'):
                precedent_lines.append(f"[PAST CRITERION SCORES]:\n{json.dumps(p.get('criterion_assessments', []), indent=2)}")
            precedent_lines.append("")
        precedents_block = "\n".join(precedent_lines)
        precedent_text = f"\n\n[HISTORICAL GRADING PRECEDENTS FOR AUDIT REFERENCE]:\n{precedents_block}"
    else:
        system_prompt = _AUDIT_SYSTEM

    payload = f"""[CRITERIA DEFAULTS]
{rubric_json}

[ASSIGNMENT BRIEF]
{brief_text}

[TARGET GRADING NOTES]
Notes: {sanitized_notes}{precedent_text}"""

    schema = AuditOutput.model_json_schema()
    response = await chat(
        messages=[
            {"role": "system", "content": system_prompt},
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
_COMPILE_SYSTEM_WITH_PRECEDENTS = """You are a deterministic assessment mapping agent. Your sole purpose is to map raw, massaged assignment notes into a structured JSON rubric schema.

CRITICAL OPERATIONAL CONSTRAINT:
The rubric provided below may be broad, open-ended, or in markdown format rather than structured JSON. You MUST interpret its ambiguities PRIMARILY through the lens of the historical grading precedents provided in the "HISTORICAL ASSESSMENT PRECEDENTS" section. Precedents are your authoritative reference for how similar work has been graded. If a characteristic in the new assignment closely matches a historical example, apply the same evaluation standard and grade mapping. When the rubric and precedents appear to conflict, defer to the precedents as the established interpretation.

SPECIFICITY REQUIREMENT:
Extract and preserve every specific detail, example, and actionable point from the marker notes. Students need concrete feedback to improve. If the notes mention particular strengths, weaknesses, or areas for improvement, include them in your feedback — do not generalise or omit specifics just because they don't map perfectly to rubric categories.

RUBRIC FORMAT NOTE:
The rubric may be provided as structured JSON OR as markdown text. If it's markdown, parse the structure to identify criterion names and descriptors. Do not fail or hallucinate if the format differs from the JSON schema you expect.

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
- Use precedent scores as calibration references when interpolating between rubric anchors.
- Use "level_selected" to indicate which named level is the closest reference point, but do not restrict your point value to that level's points.

ZERO-DATA SCORING PROTOCOL:
If the raw grading notes do not mention a specific rubric category or fail to provide a clear deduction value:
- Check historical precedents for typical scoring patterns in that category.
- If precedents show deductions are common for similar work, apply a moderate penalty with precedent-informed feedback.
- If precedents show consistent full marks for similar work, award full points with supportive feedback.
- Never guess a penalty without precedent support.
- Write feedback that references what precedents typically look like for that criterion."""

_COMPILE_SYSTEM_COLD_START = """You are an expert academic tutor speaking directly to a student. Your job is to translate raw grading notes into clear, encouraging, second-person ("You") feedback criteria blocks.

CRITICAL OPERATIONAL CONSTRAINT:
No historical precedents are available. Grade strictly against the rubric criteria provided below. Do not invent criteria or scores beyond what the rubric defines.

RUBRIC FORMAT NOTE:
The rubric may be provided as structured JSON OR as markdown text. If it's markdown, parse the structure to identify criterion names and descriptors. Do not fail or hallucinate if the format differs from the JSON schema you expect.

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
    precedents: list[dict[str, Any]] | None = None,
) -> CompileOutput:
    if precedents:
        # Build precedent block with criterion assessments
        precedent_lines = []
        for i, p in enumerate(precedents, 1):
            precedent_lines.append(f"### PRECEDENT {i}")
            precedent_lines.append(f"[PAST MARKER NOTES]:\n{p.get('massaged_notes', '')}")
            if p.get('criterion_assessments'):
                precedent_lines.append(f"[PAST CRITERION ASSESSMENTS]:\n{json.dumps(p.get('criterion_assessments', []), indent=2)}")
            precedent_lines.append("")
        precedents_block = "\n".join(precedent_lines)
        system_prompt = _COMPILE_SYSTEM_WITH_PRECEDENTS
        examples_text = few_shot_examples or ""
        payload = f"""[HISTORICAL ASSESSMENT PRECEDENTS (GOLD STANDARD EXAMPLES)]:
{precedents_block}

[TARGET RUBRIC CRITERIA EXPECTED]
{rubric_json}

[RAW GRADING FRAGMENTS TO TRANSLATE]
{sanitized_notes}"""
    else:
        system_prompt = _COMPILE_SYSTEM_COLD_START
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
            {"role": "system", "content": system_prompt},
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

    # Precedents for early passes (Pass 1 & 2)
    precedents_for_early = [p.model_dump() for p in request.precedents] if request.precedents else None

    # Pass 1: Unpack (precedent-aware)
    unpack_result = await _pass_unpack(
        request.marker_notes,
        chat,
        model,
        precedents=precedents_for_early,
    )
    logger.debug("Pass 1 complete: unpacked notes (precedents=%s)", "yes" if precedents_for_early else "no")

    # Pass 2: Audit (precedent-aware)
    audit_result = await _pass_audit(
        unpack_result.sanitized_notes,
        rubric_json,
        request.assignment_brief,
        chat,
        model,
        precedents=precedents_for_early,
    )
    logger.debug("Pass 2 complete: %d flags detected (precedents=%s)", len(audit_result.review_flags), "yes" if precedents_for_early else "no")

    # Pass 3: Compile (precedent-aware)
    precedents = [p.model_dump() for p in request.precedents] if request.precedents else None
    compile_result = await _pass_compile(
        unpack_result.sanitized_notes,
        rubric_json,
        request.few_shot_examples,
        chat,
        model,
        precedents=precedents,
    )
    logger.debug("Pass 3 complete: %d criteria compiled (precedents=%s)", len(compile_result.criteria), "yes" if precedents else "no")

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
