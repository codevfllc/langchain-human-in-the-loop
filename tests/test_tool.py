from __future__ import annotations

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


def test_hitl_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
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
    with pytest.raises(TimeoutError):
        hitl.invoke("Review this function for errors.")


@pytest.mark.asyncio
async def test_hitl_async_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DummyClient()
    hitl = HumanInTheLoop(project_id=1, client=client)

    monkeypatch.setattr(hitl, "_run", lambda prompt, attachments=None: {"status": "approved", "output": "ok"})

    result = await hitl.ainvoke("Test async.")
    assert result == {"status": "approved", "output": "ok"}
