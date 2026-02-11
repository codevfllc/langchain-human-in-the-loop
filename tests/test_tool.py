from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from codevf.models.task import TaskResponse
from langchain_human_in_the_loop import AttachmentInput, HumanInTheLoop


class DummyClient:
    def __init__(self) -> None:
        self.tasks = MagicMock()


def _task(payload: dict) -> TaskResponse:
    return TaskResponse.from_payload(payload)


def test_hitl_returns_message_on_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DummyClient()
    client.tasks.create.return_value = _task({
        "id": "task_123",
        "status": "pending",
        "mode": "standard",
        "maxCredits": 20,
        "createdAt": "2026-01-01T00:00:00Z",
    })
    client.tasks.retrieve.side_effect = [
        _task({
            "id": "task_123",
            "status": "pending",
            "mode": "standard",
            "maxCredits": 20,
            "createdAt": "2026-01-01T00:00:00Z",
        }),
        _task({
            "id": "task_123",
            "status": "completed",
            "mode": "standard",
            "maxCredits": 20,
            "createdAt": "2026-01-01T00:00:00Z",
            "result": {"message": "All good", "deliverables": []},
        }),
    ]

    monkeypatch.setattr("langchain_human_in_the_loop.tool.time.sleep", lambda _: None)

    hitl = HumanInTheLoop(project_id=1, client=client)
    result = hitl.invoke("Review this function for errors.")

    assert result == {"status": "approved", "output": "All good"}


def test_default_timeout_uses_max_credit_formula() -> None:
    client = DummyClient()

    hitl = HumanInTheLoop(project_id=1, max_credits=20, client=client)

    assert hitl.timeout == 340.0


def test_default_timeout_requires_max_credits() -> None:
    client = DummyClient()

    with pytest.raises(
        ValueError,
        match="max_credits configuration is required to compute the default invoke timeout",
    ):
        HumanInTheLoop(project_id=1, max_credits=None, client=client)  # type: ignore[arg-type]


def test_hitl_maps_attachments() -> None:
    client = DummyClient()
    client.tasks.create.return_value = _task({
        "id": "task_456",
        "status": "completed",
        "mode": "standard",
        "maxCredits": 20,
        "createdAt": "2026-01-01T00:00:00Z",
        "result": {"message": "Done", "deliverables": []},
    })
    client.tasks.retrieve.return_value = client.tasks.create.return_value

    hitl = HumanInTheLoop(project_id=1, client=client)
    hitl.invoke(
        "Review this file.",
        attachments=[
            AttachmentInput(
                file_name="app.py",
                mime_type="text/x-python",
                content="print('hi')",
            )
        ],
    )

    client.tasks.create.assert_called_once()
    _, kwargs = client.tasks.create.call_args
    assert kwargs["attachments"] == [
        {"fileName": "app.py", "mimeType": "text/x-python", "content": "print('hi')"}
    ]


def test_hitl_timeout(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    client = DummyClient()
    client.tasks.create.return_value = _task({
        "id": "task_999",
        "status": "pending",
        "mode": "standard",
        "maxCredits": 20,
        "createdAt": "2026-01-01T00:00:00Z",
    })
    client.tasks.retrieve.return_value = client.tasks.create.return_value

    times = iter([0.0, 10.0, 20.0, 30.0])
    monkeypatch.setattr("langchain_human_in_the_loop.tool.time.monotonic", lambda: next(times))
    monkeypatch.setattr("langchain_human_in_the_loop.tool.time.sleep", lambda _: None)

    hitl = HumanInTheLoop(project_id=1, client=client, timeout=5)
    caplog.set_level(logging.INFO, logger="langchain_human_in_the_loop.tool")

    with pytest.raises(TimeoutError, match="--timeout"):
        hitl.invoke("Review this function for errors.")
    assert "Invoke timeout: 5s" in caplog.text
    assert "Invoke timed out after 10s (configured timeout: 5s)." in caplog.text


def test_hitl_timeout_minus_one_waits_without_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DummyClient()
    client.tasks.create.return_value = _task({
        "id": "task_infinite",
        "status": "pending",
        "mode": "standard",
        "maxCredits": 20,
        "createdAt": "2026-01-01T00:00:00Z",
    })
    client.tasks.retrieve.side_effect = [
        _task({
            "id": "task_infinite",
            "status": "pending",
            "mode": "standard",
            "maxCredits": 20,
            "createdAt": "2026-01-01T00:00:00Z",
        }),
        _task({
            "id": "task_infinite",
            "status": "pending",
            "mode": "standard",
            "maxCredits": 20,
            "createdAt": "2026-01-01T00:00:00Z",
        }),
        _task({
            "id": "task_infinite",
            "status": "completed",
            "mode": "standard",
            "maxCredits": 20,
            "createdAt": "2026-01-01T00:00:00Z",
            "result": {"message": "Done", "deliverables": []},
        }),
    ]

    times = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr("langchain_human_in_the_loop.tool.time.monotonic", lambda: next(times))
    monkeypatch.setattr("langchain_human_in_the_loop.tool.time.sleep", lambda _: None)

    hitl = HumanInTheLoop(project_id=1, client=client, timeout=-1)
    result = hitl.invoke("Review this function for errors.")

    assert hitl.timeout is None
    assert result == {"status": "approved", "output": "Done"}


@pytest.mark.asyncio
async def test_hitl_async_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DummyClient()
    hitl = HumanInTheLoop(project_id=1, client=client)

    monkeypatch.setattr(hitl, "_run", lambda prompt, attachments=None: {"status": "approved", "output": "ok"})

    result = await hitl.ainvoke("Test async.")
    assert result == {"status": "approved", "output": "ok"}
