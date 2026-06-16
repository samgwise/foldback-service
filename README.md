# Foldback Service: AI-Assisted Feedback Generation

A FastAPI microservice that receives rough marker notes, rubric context, and optional historical marking precedents, then returns structured, student-facing feedback using a multi-pass LLM pipeline. Designed to sit alongside the [student-management-app](https://github.com/.../student-management-app) (Tauri 2 + Leptos) as a separate process.

Reuses the proven multi-pass architecture from the [foldback](https://github.com/.../foldback) prototype.

## Architecture

```
foldback-service/
├── pyproject.toml         # Dependencies (FastAPI, uvicorn, pydantic, httpx, ollama)
├── .env.example           # Environment variable template
├── src/
│   ├── __init__.py
│   ├── main.py            # FastAPI app with /generate-feedback, /embeddings, /suggest-mapping, /health
│   ├── models.py          # Pydantic request/response schemas + blacklist validation
│   ├── config.py          # Environment-based configuration
│   ├── providers.py       # LLM provider abstraction (Ollama, OpenAI-compatible)
│   └── pipeline.py        # Multi-pass processing logic (unpack → audit → compile → summary)
└── tests/
    ├── test_models.py     # Schema validation, blacklist filtering
    ├── test_pipeline.py   # Multi-pass logic with mocked LLM
    ├── test_providers.py  # Provider abstraction interface
    └── test_api.py        # Endpoint tests with TestClient
```

## The Multi-Pass Pipeline

Rather than throwing raw text at an LLM and hoping for valid JSON, the service splits data processing into distinct stages:

1. **Pass 1 — Unpack/Sanitisation** (temperature: 0.3, free-form text): Cleans messy grading notes, removes rhetorical questions, expands fragments into complete sentences. Output is a polished narrative.
2. **Pass 2 — Audit** (temperature: 0.0, strict JSON): Checks the sanitized text against the rubric and assignment brief. Generates `ReviewFlag` objects for vague, missing, or contradictory content.
3. **Pass 3 — Compile** (temperature: 0.0, strict Pydantic JSON): Maps sanitized text to individual rubric criteria. If historical precedents are supplied, they are treated as gold-standard examples for resolving rubric ambiguity and maintaining consistent grade mapping. If no precedents are supplied, the service uses cold start rubric-only grading. Applies the **Zero-Data Scoring Protocol** — if a criterion is unmentioned, full points are awarded by default with supportive feedback.
4. **Pass 4 — Summary Generation** (temperature: 0.3): Creates a polished, student-facing summary paragraph from all criterion assessments.

## Marking Precedents and RAG

The service supports precedent-aware feedback generation as part of the student-management app's RAG pipeline:

1. The app builds a precedent query from marker notes and voice transcript text.
2. The app calls `POST /embeddings` to generate an embedding for that query.
3. The app performs hybrid SQLite search over historical `grading_records` using FTS5 keyword matching plus vector similarity.
4. The retrieved precedents are sent to `POST /generate-feedback` in the `precedents` field.
5. Pass 3 injects those precedents into the compile prompt as historical assessment precedents.

Precedents are advisory historical examples but are presented to the LLM as immutable case law for consistent rubric interpretation. When the list is empty, the prompt explicitly enters cold start mode and grades strictly against the rubric.

## LLM Provider Abstraction

The service supports two back-ends via a common `LLMProvider` interface:

| Provider | Package | Description |
|----------|---------|-------------|
| `OllamaProvider` | `ollama` | Connects to a local Ollama server (default: `localhost:11434`). Supports model override per request. |
| `OpenAICompatibleProvider` | `httpx` | Calls any OpenAI-compatible REST API (including Ollama's `/v1` endpoint). Configurable via environment variables. |

Both providers reuse the same pipeline implementation — only the `chat()` call differs.

Both providers also implement `embed_text()` for `POST /embeddings`. Ollama uses its embeddings endpoint, while the OpenAI-compatible provider uses an embeddings API compatible with OpenAI-style clients.

## Safety & Validation

- **Blacklist filter**: Deterministic Python-level validation gates intercept and discard hallucinated meta-rows (e.g. "Total Score", "Review Flags", "Summary") before they corrupt final results.
- **Pydantic validation**: All endpoint inputs and outputs are validated against strict schemas. Invalid `ReviewFlag` types and blacklisted criterion IDs are rejected at construction.
- **Timeout handling**: LLM calls are wrapped with configurable timeouts (default: 120 seconds).
- **Error responses**: Provider errors are caught and returned as meaningful HTTP 400/500 responses with diagnostic details in logs.
- **Request logging**: All API requests and responses are logged with timing information via FastAPI middleware.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package and workflow manager)
- [Ollama](https://ollama.com/) running locally with your target model pulled:
  ```bash
  ollama pull qwen2.5:14b
  ```

### 2. Installation

```bash
cd foldback-service
uv sync
```

### 3. Configuration

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FOLDBACK_LLM_PROVIDER` | `ollama` | Provider to use: `ollama` or `openai` |
| `FOLDBACK_OLLAMA_MODEL` | `qwen2.5:14b` | Ollama model name |
| `FOLDBACK_OPENAI_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible API base URL |
| `FOLDBACK_OPENAI_API_KEY` | _(empty)_ | API key for OpenAI-compatible provider |
| `FOLDBACK_OPENAI_MODEL` | _(empty)_ | Model name for OpenAI-compatible provider |
| `FOLDBACK_EMBEDDING_MODEL` | `nomic-embed-text` | Default embedding model for `POST /embeddings` |
| `FOLDBACK_NUM_CTX` | `16384` | Ollama context window size |
| `FOLDBACK_NUM_PREDICT` | `1024` | Ollama max tokens to generate |
| `FOLDBACK_PORT` | `8100` | Service port |
| `FOLDBACK_HOST` | `0.0.0.0` | Service bind address |
| `FOLDBACK_LOG_LEVEL` | `INFO` | Logging level |
| `FOLDBACK_REQUEST_TIMEOUT` | `120` | LLM request timeout in seconds |

### 4. Running the Service

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8100 --reload
```

Or with environment variables:

```bash
FOLDBACK_PORT=8100 uv run uvicorn src.main:app --reload
```

### 5. API Documentation

Once running, interactive API docs are available at:

- Swagger UI: `http://localhost:8100/docs`
- ReDoc: `http://localhost:8100/redoc`

## API Endpoints

### POST /generate-feedback

Generates structured feedback from marker notes and rubric context.

**Request body:**

```json
{
  "marker_notes": "Good creative effort. Technical side is a bit rough — audio quality needs work.",
  "student_name": "Jane Citizen",
  "student_id": "12345678",
  "rubric": {
    "criteria": [
      {
        "id": "c1",
        "name": "Creativity",
        "description": "Demonstrates original thinking and creative approach.",
        "max_points": 10.0,
        "levels": [
          {"name": "Pass", "description": "Adequate creativity", "points": 5.0},
          {"name": "Distinction", "description": "Excellent creativity", "points": 10.0}
        ]
      }
    ],
    "total_points": 10.0
  },
  "assignment_brief": "Create a 4-minute radiophonic production...",
  "few_shot_examples": null,
  "precedents": [
    {
      "massaged_notes": "The work showed strong creative development but had inconsistent audio mixing.",
      "criterion_assessments": [
        {
          "criterion_id": "c1",
          "points": 8.0,
          "level_selected": "Distinction",
          "feedback": "You demonstrated strong creative development."
        }
      ]
    }
  ],
  "model": null
}
```

**Response (precedents omitted for brevity; when present, they inform rubric interpretation):**

```json
{
  "criteria": [
    {
      "criterion_id": "c1",
      "points": 8.0,
      "max_points": 10.0,
      "level_selected": "Distinction",
      "feedback": "You demonstrated strong creative effort in your production."
    }
  ],
  "review_flags": [
    {
      "flag_type": "Vague Feedback",
      "target_criteria": "Technical Skill",
      "issue_description": "The marker noted audio quality issues but did not specify which technical aspects need work."
    }
  ],
  "summary_feedback": "Overall, your creative approach was strong and you demonstrated good original thinking. Focus on improving the technical aspects of your audio production for next time.",
  "total_points": 8.0
}
```

### POST /embeddings

Generates an embedding vector for text. The student-management app uses this endpoint to embed marker notes before hybrid precedent search.

**Request body:**

```json
{
  "text": "Strong creative concept but inconsistent audio mix.",
  "model": null
}
```

**Response:**

```json
{
  "embedding": [0.0123, -0.0456, 0.0789],
  "dimension": 3
}
```

### POST /suggest-mapping

Suggests a CSV column mapping for a target database schema.

**Request body:**

```json
{
  "csv_headers": ["Student ID", "Name", "Score"],
  "target_schema": "grades",
  "sample_rows": [["12345678", "Jane Citizen", "85"]]
}
```

**Response:**

```json
{
  "column_mapping": {
    "student_id": "Student ID",
    "marks": "Score"
  },
  "confidence": 0.85,
  "suggestions": [
    {
      "field": "student_id",
      "column": "Student ID",
      "confidence": 0.95,
      "reason": "Exact match on identifier pattern"
    }
  ]
}
```

### GET /health

Basic health-check endpoint.

**Response:**

```json
{
  "status": "ok",
  "provider": "ollama"
}
```

## Testing

Run the full test suite:

```bash
uv run pytest
```

This covers:
- Schema validation and blacklist filtering (`test_models.py`)
- Multi-pass pipeline logic with mocked LLM calls (`test_pipeline.py`)
- Provider abstraction interface and factory (`test_providers.py`)
- FastAPI endpoint responses and error handling (`test_api.py`)

## Integration with student-management-app

The Tauri app communicates with this service via HTTP. Typical flow:

1. Marker enters rough notes in the grade form UI
2. Tauri app sends POST `/generate-feedback` with the notes and rubric context
3. Service runs the 4-pass pipeline and returns structured feedback
4. Tauri app populates `criterion_grades` rows with the returned `criteria` and stores `review_flags` for coordinator review
5. Student-facing `public_comments` are generated from the `summary_feedback` field

## Design Decisions

- **Separate microservice**: Keeps the LLM pipeline decoupled from the Tauri/Rust/Leptos stack, allowing independent development and deployment.
- **Provider abstraction**: Both Ollama and OpenAI-compatible providers share the same pipeline code, making it easy to switch back-ends without changing business logic.
- **Multi-pass architecture**: Proven in the foldback prototype to handle messy, abbreviated grading notes reliably. Each pass has a specific role and temperature setting to prevent instruction drift.
- **Blacklist filtering**: Applied at both the model level (Pydantic validators) and pipeline level (post-processing) to ensure hallucinated meta-rows never reach the client.
- **Australian English**: All system prompts and feedback text use Australian spelling conventions (e.g., "sanitise", "acknowledgement").
