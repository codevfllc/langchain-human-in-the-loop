from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ConfigDict

from codevf import CodeVFClient
from codevf.models.task import ServiceMode, TaskResponse

DEFAULT_TIMEOUT_PER_CREDIT_SECONDS = 2
DEFAULT_TIMEOUT_BUFFER_SECONDS = 300
INFINITE_TIMEOUT_SENTINEL = -1

logger = logging.getLogger(__name__)


class AttachmentInput(BaseModel):
    file_name: str = Field(..., alias="fileName")
    mime_type: str = Field(..., alias="mimeType")
    content: Optional[str] = None
    base64: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    def to_mapping(self) -> Dict[str, str]:
        payload = {"fileName": self.file_name, "mimeType": self.mime_type}
        if self.content is not None:
            payload["content"] = self.content
        if self.base64 is not None:
            payload["base64"] = self.base64
        return payload


class HumanInTheLoopInput(BaseModel):
    prompt: str = Field(..., description="Natural-language request for CodeVF.")
    attachments: Optional[List[AttachmentInput]] = Field(
        default=None,
        description="Optional files/logs to attach. Each item needs fileName/file_name, mimeType/mime_type, and content or base64.",
    )
    tag_id: Optional[int] = Field(
        default=None,
        alias="tagId",
        description="Optional expertise tag ID from GET /tags (Engineer, Vibe Coder, General Purpose).",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


@runtime_checkable
class TasksClientProtocol(Protocol):
    def create(
        self,
        prompt: str,
        max_credits: int,
        project_id: int,
        *,
        mode: ServiceMode | str,
        metadata: Optional[Dict[str, Any]],
        attachments: Optional[Iterable[Dict[str, Any]]],
        tag_id: Optional[int],
    ) -> TaskResponse: ...

    def retrieve(self, task_id: str) -> TaskResponse: ...


@runtime_checkable
class CodeVFClientProtocol(Protocol):
    tasks: TasksClientProtocol


class HumanInTheLoop:
    def __init__(
        self,
        *,
        project_id: int,
        max_credits: int = 50,
        mode: ServiceMode | str = ServiceMode.STANDARD,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        tag_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        client: Optional[CodeVFClient] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if client is None:
            client = CodeVFClient(api_key=api_key, base_url=base_url)
        self.client = client
        self.project_id = project_id
        self.max_credits = max_credits
        self.mode = mode
        self.poll_interval = poll_interval
        self.timeout = _resolve_timeout_seconds(timeout=timeout, max_credits=max_credits)
        self.tag_id = tag_id
        self.metadata = metadata

    def invoke(
        self,
        prompt: str,
        *,
        attachments: Optional[List[Union[AttachmentInput, Dict[str, Any]]]] = None,
        tag_id: Optional[int] = None,
    ) -> Dict[str, str]:
        return self._run(prompt, attachments, tag_id)

    async def ainvoke(
        self,
        prompt: str,
        *,
        attachments: Optional[List[Union[AttachmentInput, Dict[str, Any]]]] = None,
        tag_id: Optional[int] = None,
    ) -> Dict[str, str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, prompt, attachments, tag_id)

    def _run(
        self,
        prompt: str,
        attachments: Optional[List[Union[AttachmentInput, Dict[str, Any]]]] = None,
        tag_id: Optional[int] = None,
    ) -> Dict[str, str]:
        attachment_payload = _normalize_attachments(attachments)
        effective_tag_id = self.tag_id if tag_id is None else tag_id
        logger.info("Invoke timeout: %s", _format_timeout_for_log(self.timeout))

        task = self.client.tasks.create(
            prompt=prompt,
            max_credits=self.max_credits,
            project_id=self.project_id,
            mode=self.mode,
            metadata=self.metadata,
            attachments=attachment_payload,
            tag_id=effective_tag_id,
        )

        start = time.monotonic()
        while True:
            current = self.client.tasks.retrieve(task.id)
            status = current.status.lower()
            if status in {"completed", "failed", "canceled", "cancelled", "expired"}:
                return _format_hitl_result(current)

            elapsed = time.monotonic() - start
            if self.timeout is not None and elapsed > self.timeout:
                logger.error(
                    "Invoke timed out after %s (configured timeout: %s).",
                    _format_elapsed_time(elapsed),
                    _format_timeout_for_log(self.timeout),
                )
                raise TimeoutError(
                    f"Invoke timed out after {_format_elapsed_time(elapsed)} while waiting for "
                    f"CodeVF task '{task.id}' (configured timeout: {_format_timeout_for_log(self.timeout)}). "
                    "Increase the timeout with --timeout <seconds> or disable it with --timeout -1."
                )

            time.sleep(self.poll_interval)

    def as_langchain_tool(self) -> StructuredTool:
        def _run_tool(
            prompt: str,
            attachments: Optional[List[Dict[str, Any]]] = None,
            tag_id: Optional[int] = None,
        ) -> str:
            result = self.invoke(prompt, attachments=attachments, tag_id=tag_id)
            return result["output"]

        return StructuredTool.from_function(
            func=_run_tool,
            name="codevf_review",
            description="Send a request to CodeVF for human code review, debugging, or verification.",
            args_schema=HumanInTheLoopInput,
        )

def _normalize_attachments(
    attachments: Optional[List[Union[AttachmentInput, Dict[str, Any]]]],
) -> Optional[List[Dict[str, Any]]]:
    if not attachments:
        return None

    normalized: List[Dict[str, Any]] = []
    for item in attachments:
        if isinstance(item, AttachmentInput):
            normalized.append(item.to_mapping())
        elif isinstance(item, dict):
            if "file_name" in item or "mime_type" in item:
                payload = {}
                if "file_name" in item:
                    payload["fileName"] = item["file_name"]
                if "mime_type" in item:
                    payload["mimeType"] = item["mime_type"]
                if "fileName" in item:
                    payload["fileName"] = item["fileName"]
                if "mimeType" in item:
                    payload["mimeType"] = item["mimeType"]
                if "content" in item:
                    payload["content"] = item["content"]
                if "base64" in item:
                    payload["base64"] = item["base64"]
                normalized.append(payload)
            else:
                normalized.append(item)
        else:
            raise TypeError("Attachments must be AttachmentInput or dict objects.")

    return normalized


def _format_hitl_result(task: TaskResponse) -> Dict[str, str]:
    status = task.status.lower()
    output = _extract_output(task)
    if status == "completed":
        return {"status": "approved", "output": output}
    if status in {"canceled", "cancelled", "expired", "failed"}:
        return {"status": "cancelled", "output": output}
    return {"status": status, "output": output}


def _extract_output(task: TaskResponse) -> str:
    if task.result and task.result.message:
        return task.result.message

    if task.result and task.result.deliverables:
        lines = ["CodeVF task completed. Deliverables:"]
        for deliverable in task.result.deliverables:
            lines.append(f"- {deliverable.file_name}: {deliverable.url}")
        return "\n".join(lines)

    if task.status.lower() == "failed":
        return "CodeVF task failed without a text response."

    if task.status.lower() in {"canceled", "cancelled", "expired"}:
        return "CodeVF task was cancelled."

    return "CodeVF task completed without a text response."


def _resolve_timeout_seconds(timeout: Optional[float], max_credits: Optional[int]) -> Optional[float]:
    if timeout is None:
        return _compute_default_timeout_seconds(max_credits)

    timeout_value = _coerce_float(timeout, field_name="timeout")
    if timeout_value == INFINITE_TIMEOUT_SENTINEL:
        return None
    if timeout_value <= 0:
        raise ValueError("timeout must be -1 for infinite wait or a positive number of seconds.")
    return timeout_value


def _compute_default_timeout_seconds(max_credits: Optional[int]) -> float:
    if max_credits is None:
        raise ValueError(
            "max_credits configuration is required to compute the default invoke timeout."
        )

    if isinstance(max_credits, bool):
        raise ValueError("max_credits must be an integer.")

    try:
        max_credits_value = int(max_credits)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_credits must be an integer.") from exc

    return float(
        (DEFAULT_TIMEOUT_PER_CREDIT_SECONDS * max_credits_value) + DEFAULT_TIMEOUT_BUFFER_SECONDS
    )


def _coerce_float(value: float, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _format_timeout_for_log(timeout: Optional[float]) -> str:
    if timeout is None:
        return "infinite"
    if timeout.is_integer():
        return f"{int(timeout)}s"
    return f"{timeout:g}s"


def _format_elapsed_time(elapsed: float) -> str:
    if elapsed.is_integer():
        return f"{int(elapsed)}s"
    return f"{elapsed:.2f}s"
