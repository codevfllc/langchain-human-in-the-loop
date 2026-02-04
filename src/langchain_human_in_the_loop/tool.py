from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ConfigDict

from codevf import CodeVFClient
from codevf.models.task import ServiceMode, TaskResponse


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

    model_config = ConfigDict(extra="forbid")


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
        timeout: float = 300.0,
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
        self.timeout = timeout
        self.tag_id = tag_id
        self.metadata = metadata

    def invoke(
        self,
        prompt: str,
        *,
        attachments: Optional[List[Union[AttachmentInput, Dict[str, Any]]]] = None,
    ) -> Dict[str, str]:
        return self._run(prompt, attachments)

    async def ainvoke(
        self,
        prompt: str,
        *,
        attachments: Optional[List[Union[AttachmentInput, Dict[str, Any]]]] = None,
    ) -> Dict[str, str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, prompt, attachments)

    def _run(
        self,
        prompt: str,
        attachments: Optional[List[Union[AttachmentInput, Dict[str, Any]]]] = None,
    ) -> Dict[str, str]:
        attachment_payload = _normalize_attachments(attachments)

        task = self.client.tasks.create(
            prompt=prompt,
            max_credits=self.max_credits,
            project_id=self.project_id,
            mode=self.mode,
            metadata=self.metadata,
            attachments=attachment_payload,
            tag_id=self.tag_id,
        )

        start = time.monotonic()
        while True:
            current = self.client.tasks.retrieve(task.id)
            status = current.status.lower()
            if status in {"completed", "failed", "canceled", "cancelled", "expired"}:
                return _format_hitl_result(current)

            if (time.monotonic() - start) > self.timeout:
                raise TimeoutError(
                    f"CodeVF task '{task.id}' did not complete within {self.timeout} seconds."
                )

            time.sleep(self.poll_interval)

    def as_langchain_tool(self) -> StructuredTool:
        def _run_tool(prompt: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
            result = self.invoke(prompt, attachments=attachments)
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
