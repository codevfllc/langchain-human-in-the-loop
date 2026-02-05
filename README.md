# LangChain Human In The Loop (CodeVF)

Minimal LangChain tool integration for routing review, debugging, and verification requests to CodeVF using the official Python SDK.

## Quick Start

1. Install

```bash
pip install langchain-human-in-the-loop
```

Dependencies (`codevf-sdk`, `langchain-core`, `pydantic`) are installed automatically.

2. Set your API key

```bash
export CODEVF_API_KEY="cvf_live_..."
```

3. Run a direct review (no agent needed)

```python
from langchain_human_in_the_loop import HumanInTheLoop

hitl = HumanInTheLoop(
    api_key="CODEVF_API_KEY",
    project_id=123,
    max_credits=50,
    mode="fast",
    timeout=300,
)
result = hitl.invoke(
    "Review this function for security issues and suggest fixes.",
    attachments=[
        {
            "file_name": "app.py",
            "mime_type": "text/x-python",
            "content": "def login(user, pwd): ...",
        }
    ],
)

print(result)
```

Install page:

```text
https://pypi.org/project/langchain-human-in-the-loop/
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
from langchain_human_in_the_loop import HumanInTheLoop

hitl = HumanInTheLoop(project_id=123, max_credits=50)

result = hitl.invoke(
    "Find concurrency bugs and propose fixes.",
    attachments=[
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
)

print(result)
```

### Example: Binary Attachment (Base64)

```python
import base64
from langchain_human_in_the_loop import HumanInTheLoop

with open("diagram.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("ascii")

hitl = HumanInTheLoop(project_id=123, max_credits=50)
result = hitl.invoke(
    "Review the architecture diagram for missing components.",
    attachments=[
        {
            "file_name": "diagram.png",
            "mime_type": "image/png",
            "base64": encoded,
        }
    ],
)

print(result)
```

### LangChain Tool Helper

You can expose CodeVF as a LangChain structured tool via:

```python
from langchain_human_in_the_loop import HumanInTheLoop, HumanInTheLoopInput

hitl = HumanInTheLoop(project_id=123)
codevf_tool = hitl.as_langchain_tool()

# Optional: use the tool input schema for structured calls
schema = HumanInTheLoopInput.schema()
```

See `examples/human_in_the_loop.py` for a complete runnable snippet.  
If you want to test CodeVF with LangChain, use `examples/codevf_tool_agent.py`.

### Research / University Samples

- `examples/reproducibility_audit.py` reviews an ML experiment for reproducibility gaps.
- `examples/data_privacy_review.py` checks a student-data pipeline for privacy and policy risks.
- `examples/paper_review.py` reviews methods/results for missing baselines and reporting issues.

## Testing

```bash
pip install -e .[dev]
pytest
```
