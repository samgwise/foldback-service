"""Task manager for orchestrating LLM and STT workloads in the foldback-service.

Manages a queue of tasks, swaps GPU-loaded models between LLM and STT to
minimise VRAM contention, and processes tasks in batches by type.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    LLM = "llm"
    STT = "stt"


@dataclass
class Task:
    task_id: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    payload: Any = None
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    priority: int = 0


class TaskQueue:
    """Thread-safe task queue with lookups by task_id."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, Task] = {}
        self._pending_order: list[str] = []

    async def add_task(self, task_type: TaskType, payload: Any, priority: int = 0) -> Task:
        task = Task(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority,
        )
        async with self._lock:
            self._tasks[task.task_id] = task
            # Insert by priority (higher first), then FIFO
            inserted = False
            for i, existing_id in enumerate(self._pending_order):
                existing = self._tasks[existing_id]
                if priority > existing.priority:
                    self._pending_order.insert(i, task.task_id)
                    inserted = True
                    break
            if not inserted:
                self._pending_order.append(task.task_id)
        logger.info("Task %s added (type=%s, priority=%d)", task.task_id, task_type.value, priority)
        return task

    async def get_task(self, task_id: str) -> Task | None:
        async with self._lock:
            return self._tasks.get(task_id)

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if status == TaskStatus.PROCESSING:
                task.started_at = time.time()
                if task.task_id in self._pending_order:
                    self._pending_order.remove(task.task_id)
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.completed_at = time.time()
                if task.task_id in self._pending_order:
                    self._pending_order.remove(task.task_id)

    async def get_next(self, task_type: TaskType) -> Task | None:
        async with self._lock:
            for tid in self._pending_order:
                task = self._tasks[tid]
                if task.status == TaskStatus.PENDING and task.task_type == task_type:
                    task.status = TaskStatus.QUEUED
                    return task
            return None

    async def count_pending(self, task_type: TaskType | None = None) -> int:
        async with self._lock:
            return sum(
                1
                for t in self._tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.QUEUED)
                and (task_type is None or t.task_type == task_type)
            )

    async def list_all(self) -> list[Task]:
        async with self._lock:
            return list(self._tasks.values())


class ResourceManager:
    """Tracks which model is loaded and manages VRAM."""

    def __init__(self, idle_unload_seconds: int = 30) -> None:
        self.loaded_model: str | None = None  # "llm" or "stt" or None
        self.idle_since: float | None = None
        self.idle_unload_seconds = idle_unload_seconds
        self._stt_service: Any | None = None
        self._llm_provider: Any | None = None

    def set_stt_service(self, svc: Any) -> None:
        self._stt_service = svc

    def set_llm_provider(self, provider: Any) -> None:
        self._llm_provider = provider

    def can_process(self, task_type: TaskType) -> bool:
        return self.loaded_model in (None, task_type.value)

    async def load_model(self, task_type: TaskType) -> bool:
        """Load the requested model, unloading the other if necessary."""
        if self.loaded_model == task_type.value:
            self.idle_since = None
            return True

        loop = asyncio.get_running_loop()

        # Unload existing model
        if self.loaded_model == "llm" and self._llm_provider is not None:
            logger.info("Unloading LLM model...")
            pass
        elif self.loaded_model == "stt" and self._stt_service is not None:
            logger.info("Unloading STT model...")
            await loop.run_in_executor(None, self._stt_service.unload_model)

        self.loaded_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load new model
        if task_type == TaskType.STT and self._stt_service is not None:
            logger.info("Loading STT model...")
            await loop.run_in_executor(None, self._stt_service.load_model)
            self.loaded_model = "stt"
        elif task_type == TaskType.LLM:
            logger.info("LLM model ready (stateless HTTP client).")
            self.loaded_model = "llm"

        self.idle_since = None
        return True

    async def unload_all(self) -> None:
        loop = asyncio.get_running_loop()
        if self.loaded_model == "stt" and self._stt_service is not None:
            await loop.run_in_executor(None, self._stt_service.unload_model)
        self.loaded_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.idle_since = None
        logger.info("All models unloaded.")

    def mark_idle(self) -> None:
        if self.idle_since is None:
            self.idle_since = time.time()

    def should_unload_idle(self) -> bool:
        if self.idle_since is None or self.loaded_model is None:
            return False
        return (time.time() - self.idle_since) > self.idle_unload_seconds

    def get_vram_info(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {"device": "cpu", "vram_used_mb": 0, "vram_total_mb": 0}
        try:
            vram_used = torch.cuda.memory_allocated() / 1024 / 1024
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            return {
                "device": torch.cuda.get_device_name(0),
                "vram_used_mb": round(vram_used, 1),
                "vram_total_mb": round(vram_total, 1),
            }
        except Exception:
            return {"device": "unknown", "vram_used_mb": 0, "vram_total_mb": 0}


class TaskManager:
    """Orchestrates task queue processing and resource management."""

    def __init__(
        self,
        batch_size_llm: int = 1,
        batch_size_stt: int = 1,
        idle_unload_seconds: int = 30,
    ) -> None:
        self.queue = TaskQueue()
        self.resources = ResourceManager(idle_unload_seconds=idle_unload_seconds)
        self.batch_size_llm = batch_size_llm
        self.batch_size_stt = batch_size_stt
        self._running = False
        self._task = None

    def set_services(self, llm_provider: Any, stt_service: Any) -> None:
        self.resources.set_llm_provider(llm_provider)
        self.resources.set_stt_service(stt_service)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Task manager started.")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.resources.unload_all()
        logger.info("Task manager stopped.")

    async def submit(self, task_type: TaskType, payload: Any, priority: int = 0) -> str:
        task = await self.queue.add_task(task_type, payload, priority)
        return task.task_id

    async def get_status(self, task_id: str) -> dict[str, Any] | None:
        task = await self.queue.get_task(task_id)
        if task is None:
            return None
        return {
            "task_id": task.task_id,
            "type": task.task_type.value,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
        }

    async def get_resources_async(self) -> dict[str, Any]:
        vram = self.resources.get_vram_info()
        return {
            "loaded_model": self.resources.loaded_model,
            "queue_depth": await self.queue.count_pending(),
            "pending_llm": await self.queue.count_pending(TaskType.LLM),
            "pending_stt": await self.queue.count_pending(TaskType.STT),
            "vram": vram,
        }

    async def _process_loop(self) -> None:
        while self._running:
            try:
                await self._process_one_cycle()
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in task manager process loop")
                await asyncio.sleep(1.0)

    async def _process_one_cycle(self) -> None:
        # Check idle unload
        if self.resources.should_unload_idle():
            logger.info("Idle timeout reached; unloading models.")
            await self.resources.unload_all()

        # Determine which task type to process based on queue and current loaded model
        pending_llm = await self.queue.count_pending(TaskType.LLM)
        pending_stt = await self.queue.count_pending(TaskType.STT)

        if pending_llm == 0 and pending_stt == 0:
            self.resources.mark_idle()
            return

        # Prefer keeping the currently loaded model to avoid swap overhead
        if self.resources.loaded_model == "llm" and pending_llm > 0:
            await self._process_batch(TaskType.LLM, self.batch_size_llm)
        elif self.resources.loaded_model == "stt" and pending_stt > 0:
            await self._process_batch(TaskType.STT, self.batch_size_stt)
        elif pending_llm > 0:
            await self._process_batch(TaskType.LLM, self.batch_size_llm)
        elif pending_stt > 0:
            await self._process_batch(TaskType.STT, self.batch_size_stt)

    async def _process_batch(self, task_type: TaskType, batch_size: int) -> None:
        await self.resources.load_model(task_type)
        processed = 0

        while processed < batch_size and self._running:
            task = await self.queue.get_next(task_type)
            if task is None:
                break

            await self.queue.update_status(task.task_id, TaskStatus.PROCESSING)
            logger.info("Processing task %s (type=%s)", task.task_id, task_type.value)

            try:
                if task_type == TaskType.LLM:
                    result = await self._run_llm_task(task.payload)
                elif task_type == TaskType.STT:
                    result = await self._run_stt_task(task.payload)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

                await self.queue.update_status(task.task_id, TaskStatus.COMPLETED, result=result)
                logger.info("Task %s completed.", task.task_id)
            except Exception as exc:
                await self.queue.update_status(task.task_id, TaskStatus.FAILED, error=str(exc))
                logger.exception("Task %s failed.", task.task_id)

            processed += 1

        # After a batch, check if there are pending tasks of the other type
        # If so, switch after processing a few more of the current type (simple strategy)
        other_type = TaskType.STT if task_type == TaskType.LLM else TaskType.LLM
        other_pending = await self.queue.count_pending(other_type)
        if other_pending > 0 and processed >= batch_size:
            logger.info("Switching to %s tasks after %s batch.", other_type.value, task_type.value)

    async def _run_llm_task(self, payload: Any) -> Any:
        """Run an LLM task via the existing provider."""
        from src.models import FeedbackRequest, MappingRequest

        provider = self.resources._llm_provider
        if provider is None:
            raise RuntimeError("LLM provider not configured.")
        operation = payload.get("operation", "generate_feedback")
        if operation == "generate_feedback":
            request = FeedbackRequest(**payload.get("request", payload))
            return (await provider.generate_feedback(request)).model_dump()
        elif operation == "suggest_mapping":
            request = MappingRequest(**payload.get("request", payload))
            return (await provider.suggest_mapping(request)).model_dump()
        else:
            raise ValueError(f"Unknown LLM operation: {operation}")

    async def _run_stt_task(self, payload: Any) -> Any:
        """Run an STT task via the transcription service."""
        if self.resources._stt_service is None:
            raise RuntimeError("STT service not configured.")
        audio_path = payload["audio_path"]
        try:
            loop = asyncio.get_running_loop()
            transcript = await loop.run_in_executor(None, self.resources._stt_service.transcribe, audio_path)
            return {"transcript": transcript}
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
