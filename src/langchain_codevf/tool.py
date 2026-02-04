from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from langchain_core.tools import BaseTool
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


class CodeVFReviewInput(BaseModel):
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


class CodeVFReviewTool(BaseTool):
    name: str = "codevf_review"
    description: str = (
        "Send a request to CodeVF for human code review, debugging, or verification."
    )
    args_schema: type[BaseModel] = CodeVFReviewInput

    client: CodeVFClientProtocol = Field(exclude=True)
    project_id: int = Field(..., description="CodeVF project ID for task organization.")
    max_credits: int = Field(50, description="Max credits to spend per request.")
    mode: ServiceMode | str = Field(ServiceMode.STANDARD, description="Service mode.")
    poll_interval: float = Field(2.0, description="Seconds between status checks.")
    timeout: float = Field(300.0, description="Max seconds to wait for completion.")
    tag_id: Optional[int] = Field(default=None, description="Optional CodeVF tag ID.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata.")

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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
        super().__init__(
            client=client,
            project_id=project_id,
            max_credits=max_credits,
            mode=mode,
            poll_interval=poll_interval,
            timeout=timeout,
            tag_id=tag_id,
            metadata=metadata,
        )

    def _run(self, prompt: str, attachments: Optional[List[AttachmentInput]] = None) -> str:
        attachment_payload = [item.to_mapping() for item in attachments] if attachments else None

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
                return self._format_result(current)

            if (time.monotonic() - start) > self.timeout:
                raise TimeoutError(
                    f"CodeVF task '{task.id}' did not complete within {self.timeout} seconds."
                )

            time.sleep(self.poll_interval)

    async def _arun(self, prompt: str, attachments: Optional[List[AttachmentInput]] = None) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, prompt, attachments)

    @staticmethod
    def _format_result(task: TaskResponse) -> str:
        status = task.status.lower()
        if status != "completed":
            raise RuntimeError(f"CodeVF task '{task.id}' finished with status '{task.status}'.")

        if task.result and task.result.message:
            return task.result.message

        if task.result and task.result.deliverables:
            lines = ["CodeVF task completed. Deliverables:"]
            for deliverable in task.result.deliverables:
                lines.append(f"- {deliverable.file_name}: {deliverable.url}")
            return "\n".join(lines)

        return "CodeVF task completed without a text response."
