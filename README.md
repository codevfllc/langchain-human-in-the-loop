# LangChain + CodeVF

Minimal LangChain tool integration for routing review, debugging, and verification requests to CodeVF using the official Python SDK.

## Quick Start

1. Install dependencies

```bash
pip install codevf-sdk langchain-core pydantic
```

2. Set your API key

```bash
export CODEVF_API_KEY="cvf_live_..."
```

3. Run a direct review (no agent needed)

```python
from langchain_codevf import CodeVFReviewTool

codevf_tool = CodeVFReviewTool(
    project_id=123,
    max_credits=50,
    mode="fast",
    timeout=300,
)

result = codevf_tool.invoke(
    {
        "prompt": "Review this function for security issues and suggest fixes.",
        "attachments": [
            {
                "file_name": "app.py",
                "mime_type": "text/x-python",
                "content": "def login(user, pwd): ...",
            }
        ],
    }
)

print(result)
```

## Install From Source

```bash
pip install -e .
```

## Usage

### Attachment Format

Each attachment accepts either `content` (plain text) or `base64` (for binary files). The SDK enforces limits and supported file types.

### Example: Multiple Files

```python
from langchain_codevf import CodeVFReviewTool

codevf_tool = CodeVFReviewTool(project_id=123, max_credits=50)

result = codevf_tool.invoke(
    {
        "prompt": "Find concurrency bugs and propose fixes.",
        "attachments": [
            {
                "file_name": "worker.py",
                "mime_type": "text/x-python",
                "content": "def run(queue): ...",
            },
            {
                "file_name": "README.md",
                "mime_type": "text/markdown",
                "content": "# Worker\n\nHow it is supposed to work...",
            },
        ],
    }
)

print(result)
```

### Example: Binary Attachment (Base64)

```python
import base64
from langchain_codevf import CodeVFReviewTool

with open("diagram.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("ascii")

codevf_tool = CodeVFReviewTool(project_id=123, max_credits=50)
result = codevf_tool.invoke(
    {
        "prompt": "Review the architecture diagram for missing components.",
        "attachments": [
            {
                "file_name": "diagram.png",
                "mime_type": "image/png",
                "base64": encoded,
            }
        ],
    }
)

print(result)
```

## Using With A LangChain Agent

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_codevf import CodeVFReviewTool

load_dotenv()

codevf_tool = CodeVFReviewTool(project_id=123, max_credits=50)
llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_agent(
    model=llm,
    tools=[codevf_tool],
    system_prompt="You are a software engineering assistant.",
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Ask CodeVF to debug this error stack trace."}
        ]
    }
)

messages = result.get("messages", [])
last = messages[-1] if messages else None
print(getattr(last, "content", None) or (last or {}).get("content", ""))
```

See `examples/codevf_tool_agent.py` for a complete runnable snippet.  
If you want to test CodeVF without an LLM, use `examples/codevf_direct.py`.

## Testing

```bash
pip install -e .[dev]
pytest
```
