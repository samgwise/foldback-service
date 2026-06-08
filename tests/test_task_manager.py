"""Tests for the task manager and queue.

This file can be run directly with ``python tests/test_task_manager.py``
or via pytest when available.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

# Provide a mock torch module so src.task_manager can import without torch installed.
_fake_torch = MagicMock()
_fake_torch.cuda.is_available.return_value = False
sys.modules["torch"] = _fake_torch

# Provide a mock pydantic module so src.models can import.
_pydantic = MagicMock()

def _mock_base_model_init(self, **kwargs):
    pass

def _mock_base_model_dump(self):
    return {}

_pydantic.BaseModel = type("BaseModel", (), {
    "__init__": _mock_base_model_init,
    "model_dump": _mock_base_model_dump,
})
_pydantic.Field = MagicMock()
_pydantic.field_validator = lambda *a, **k: lambda f: f
sys.modules["pydantic"] = _pydantic

from src.task_manager import ResourceManager, TaskManager, TaskQueue, TaskStatus, TaskType

try:
    import pytest
except ImportError:
    # Minimal fallback so the file can be imported and run directly.
    class _PytestMock:
        class mark:
            @staticmethod
            def asyncio(f):
                return f
        @staticmethod
        def fixture(f):
            return f
    pytest = _PytestMock()  # type: ignore[assignment]


def _async_return(value):
    """Return a coroutine that resolves to *value*."""
    async def _inner(*args, **kwargs):
        return value
    return _inner


class TestTaskQueue:
    @pytest.mark.asyncio
    async def test_add_and_get_task(self):
        q = TaskQueue()
        task = await q.add_task(TaskType.LLM, {"op": "test"}, priority=5)
        assert task.task_type == TaskType.LLM
        assert task.priority == 5
        assert task.status == TaskStatus.PENDING

        retrieved = await q.get_task(task.task_id)
        assert retrieved is not None
        assert retrieved.task_id == task.task_id

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        q = TaskQueue()
        t1 = await q.add_task(TaskType.LLM, {}, priority=1)
        t2 = await q.add_task(TaskType.LLM, {}, priority=5)
        t3 = await q.add_task(TaskType.LLM, {}, priority=3)

        next_task = await q.get_next(TaskType.LLM)
        assert next_task is not None
        assert next_task.task_id == t2.task_id  # highest priority first

    @pytest.mark.asyncio
    async def test_update_status(self):
        q = TaskQueue()
        task = await q.add_task(TaskType.STT, {"audio_path": "/tmp/x.wav"})
        await q.update_status(task.task_id, TaskStatus.PROCESSING)
        updated = await q.get_task(task.task_id)
        assert updated.status == TaskStatus.PROCESSING
        assert updated.started_at is not None

        await q.update_status(task.task_id, TaskStatus.COMPLETED, result={"transcript": "hello"})
        assert updated.status == TaskStatus.COMPLETED
        assert updated.result == {"transcript": "hello"}
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_count_pending(self):
        q = TaskQueue()
        await q.add_task(TaskType.LLM, {}, priority=0)
        await q.add_task(TaskType.LLM, {}, priority=0)
        await q.add_task(TaskType.STT, {}, priority=0)
        assert await q.count_pending() == 3
        assert await q.count_pending(TaskType.LLM) == 2
        assert await q.count_pending(TaskType.STT) == 1

    @pytest.mark.asyncio
    async def test_list_all(self):
        q = TaskQueue()
        t1 = await q.add_task(TaskType.LLM, {})
        t2 = await q.add_task(TaskType.STT, {})
        all_tasks = await q.list_all()
        assert len(all_tasks) == 2
        assert {t.task_id for t in all_tasks} == {t1.task_id, t2.task_id}


class TestResourceManager:
    @pytest.mark.asyncio
    async def test_can_process_idle(self):
        rm = ResourceManager()
        assert rm.can_process(TaskType.LLM)
        assert rm.can_process(TaskType.STT)

    @pytest.mark.asyncio
    async def test_load_llm(self):
        rm = ResourceManager()
        await rm.load_model(TaskType.LLM)
        assert rm.loaded_model == "llm"
        assert rm.can_process(TaskType.LLM)
        assert not rm.can_process(TaskType.STT)

    @pytest.mark.asyncio
    async def test_unload_all(self):
        rm = ResourceManager()
        await rm.load_model(TaskType.LLM)
        await rm.unload_all()
        assert rm.loaded_model is None
        assert rm.can_process(TaskType.LLM)
        assert rm.can_process(TaskType.STT)

    @pytest.mark.asyncio
    async def test_idle_unload(self):
        rm = ResourceManager(idle_unload_seconds=0)
        await rm.load_model(TaskType.LLM)
        rm.mark_idle()
        await asyncio.sleep(0.1)
        assert rm.should_unload_idle()

    @pytest.mark.asyncio
    async def test_load_stt_with_mock_service(self):
        rm = ResourceManager()
        mock_stt = MagicMock()
        rm.set_stt_service(mock_stt)
        await rm.load_model(TaskType.STT)
        assert rm.loaded_model == "stt"
        mock_stt.load_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload_stt_calls_unload(self):
        rm = ResourceManager()
        mock_stt = MagicMock()
        rm.set_stt_service(mock_stt)
        await rm.load_model(TaskType.STT)
        await rm.unload_all()
        mock_stt.unload_model.assert_called_once()


class TestTaskManager:
    @pytest.mark.asyncio
    async def test_submit_and_get_status(self):
        tm = TaskManager()
        tid = await tm.submit(TaskType.LLM, {"operation": "generate_feedback", "request": {}})
        status = await tm.get_status(tid)
        assert status is not None
        assert status["type"] == "llm"
        assert status["status"] == "pending"

    @pytest.mark.asyncio
    async def test_resources_async(self):
        tm = TaskManager()
        await tm.submit(TaskType.LLM, {})
        res = await tm.get_resources_async()
        assert res["loaded_model"] is None
        assert res["pending_llm"] == 1
        assert res["pending_stt"] == 0
        assert "queue_depth" in res
        assert "vram" in res

    @pytest.mark.asyncio
    async def test_start_stop(self):
        tm = TaskManager()
        await tm.start()
        assert tm._running is True
        await tm.stop()
        assert tm._running is False

    @pytest.mark.asyncio
    async def test_process_one_cycle_no_tasks(self):
        tm = TaskManager()
        await tm.start()
        await tm._process_one_cycle()
        # No tasks means mark_idle and return
        assert tm.resources.idle_since is not None
        await tm.stop()

    @pytest.mark.asyncio
    async def test_process_llm_batch_mock(self):
        tm = TaskManager(batch_size_llm=1)
        tm._running = True  # Avoid background loop race; process batch directly
        mock_provider = MagicMock()
        mock_provider.generate_feedback = _async_return(
            MagicMock(model_dump=lambda: {"summary": "ok"})
        )
        tm.resources.set_llm_provider(mock_provider)
        valid_request = {
            "operation": "generate_feedback",
            "request": {
                "marker_notes": "test",
                "student_name": "Alice",
                "student_id": "123",
                "rubric": {
                    "criteria": [
                        {"id": "c1", "name": "Creativity", "max_points": 10.0, "levels": [{"name": "Pass", "points": 5.0}]}
                    ],
                    "total_points": 10.0
                }
            }
        }
        tid = await tm.submit(TaskType.LLM, valid_request)
        await tm._process_batch(TaskType.LLM, 1)
        await tm.stop()

        status = await tm.get_status(tid)
        assert status["status"] == "completed"
        assert status["result"] == {"summary": "ok"}

    @pytest.mark.asyncio
    async def test_process_stt_batch_mock(self):
        tm = TaskManager(batch_size_stt=1)
        tm._running = True
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "hello world"
        tm.resources.set_stt_service(mock_stt)

        tid = await tm.submit(TaskType.STT, {"audio_path": "/tmp/fake.wav"})
        await tm._process_batch(TaskType.STT, 1)
        await tm.stop()

        status = await tm.get_status(tid)
        assert status["status"] == "completed"
        assert status["result"] == {"transcript": "hello world"}

    @pytest.mark.asyncio
    async def test_failed_task(self):
        tm = TaskManager(batch_size_llm=1)
        tm._running = True
        mock_provider = MagicMock()
        async def _raise(*args, **kwargs):
            raise RuntimeError("boom")
        mock_provider.generate_feedback = _raise
        tm.resources.set_llm_provider(mock_provider)
        valid_request = {
            "operation": "generate_feedback",
            "request": {
                "marker_notes": "test",
                "student_name": "Alice",
                "student_id": "123",
                "rubric": {
                    "criteria": [
                        {"id": "c1", "name": "Creativity", "max_points": 10.0, "levels": [{"name": "Pass", "points": 5.0}]}
                    ],
                    "total_points": 10.0
                }
            }
        }
        tid = await tm.submit(TaskType.LLM, valid_request)
        await tm._process_batch(TaskType.LLM, 1)
        await tm.stop()

        status = await tm.get_status(tid)
        assert status["status"] == "failed"
        assert "boom" in status["error"]


async def _run_all() -> None:
    """Run all test methods manually when pytest is not installed."""
    import traceback
    classes = (TestTaskQueue, TestResourceManager, TestTaskManager)
    total = 0
    passed = 0
    for cls in classes:
        for name in dir(cls):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                await getattr(cls(), name)()
                print(f"  PASS: {cls.__name__}.{name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL: {cls.__name__}.{name} — {exc}")
                traceback.print_exc()
    print(f"\nResults: {passed}/{total} passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_run_all())
