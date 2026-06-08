"""FastAPI application for the Foldback Service."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from src.config import settings
from src.models import FeedbackRequest, FeedbackResponse, MappingRequest, MappingResponse
from src.providers import get_provider
from src.task_manager import TaskManager, TaskType

logger = logging.getLogger(__name__)

# Lazily-loaded STT service singleton
_stt_service: object | None = None

# Task manager singleton
_task_manager: TaskManager | None = None


def _get_stt_service():
    """Return the singleton TranscriptionService, loading it on first call."""
    global _stt_service
    if _stt_service is None:
        from src.stt import TranscriptionService

        _stt_service = TranscriptionService(
            model_name=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
        )
        _stt_service.load_model()
    return _stt_service


def _get_task_manager() -> TaskManager:
    """Return the singleton TaskManager, creating it on first call."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(
            batch_size_llm=settings.batch_size_llm,
            batch_size_stt=settings.batch_size_stt,
            idle_unload_seconds=settings.idle_unload_seconds,
        )
        # Wire services
        stt = _get_stt_service()
        _task_manager.set_services(get_provider(), stt)
    return _task_manager


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Foldback Service starting on %s:%s (provider=%s)", settings.host, settings.port, settings.llm_provider)
    # Start task manager background loop
    tm = _get_task_manager()
    await tm.start()
    yield
    logger.info("Foldback Service shutting down")
    await tm.stop()
    global _stt_service
    if _stt_service is not None:
        _stt_service.unload_model()
        _stt_service = None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Foldback Service",
    description="AI-Assisted Feedback Generation microservice for student management",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info("%s %s -> %s (%.2fs)", request.method, request.url.path, response.status_code, duration)
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.exception("Runtime error during request processing")
    return JSONResponse(status_code=500, content={"detail": "Internal processing error. Check logs for details."})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/generate-feedback", response_model=FeedbackResponse)
async def generate_feedback(request_body: FeedbackRequest, async_mode: bool = Query(False, alias="async")):
    """Generate structured feedback from marker notes and rubric context.

    Pass ``?async=true`` to enqueue the request in the task manager instead of
    processing synchronously.
    """
    if async_mode:
        tm = _get_task_manager()
        payload = {"operation": "generate_feedback", "request": request_body.model_dump()}
        task_id = await tm.submit(TaskType.LLM, payload)
        return {"task_id": task_id, "status": "queued"}

    try:
        provider = get_provider()
        response = await provider.generate_feedback(request_body)
        return response
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Feedback generation failed")
        raise HTTPException(status_code=500, detail=f"Feedback generation failed: {exc}") from exc


@app.post("/suggest-mapping", response_model=MappingResponse)
async def suggest_mapping(request_body: MappingRequest):
    """Suggest a CSV column mapping for the target schema."""
    try:
        provider = get_provider()
        response = await provider.suggest_mapping(request_body)
        return response
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Mapping suggestion failed")
        raise HTTPException(status_code=500, detail=f"Mapping suggestion failed: {exc}") from exc


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), async_mode: bool = Query(False, alias="async")):
    """Transcribe an uploaded audio file and return the plain transcript.

    Pass ``?async=true`` to enqueue the request in the task manager instead of
    processing synchronously.
    """
    # Validate content type loosely (allow empty or any audio/*)
    if audio.content_type and not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an audio file.")

    tmp_path = None
    try:
        # Save uploaded audio to a temporary file so whisperx can read it.
        suffix = os.path.splitext(audio.filename or ".wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        if async_mode:
            tm = _get_task_manager()
            task_id = await tm.submit(TaskType.STT, {"audio_path": tmp_path})
            return {"task_id": task_id, "status": "queued"}

        # Run transcription in a thread pool so the event loop isn't blocked.
        stt = _get_stt_service()
        loop = asyncio.get_running_loop()
        transcript: str = await loop.run_in_executor(None, stt.transcribe, tmp_path)

        return {"transcript": transcript}
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        await audio.close()
        if tmp_path and os.path.exists(tmp_path) and not async_mode:
            os.unlink(tmp_path)


@app.post("/tasks")
async def create_task(task_type: str, payload: dict):
    """Enqueue a task in the task manager."""
    tm = _get_task_manager()
    try:
        ttype = TaskType(task_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task type: {task_type}. Must be 'llm' or 'stt'.")
    task_id = await tm.submit(ttype, payload)
    return {"task_id": task_id, "status": "queued"}


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a queued or processed task."""
    tm = _get_task_manager()
    status = await tm.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@app.get("/resources")
async def get_resources():
    """Get current resource usage and queue state."""
    tm = _get_task_manager()
    return await tm.get_resources_async()


@app.get("/health")
async def health_check():
    """Basic health-check endpoint."""
    return {"status": "ok", "provider": settings.llm_provider}
