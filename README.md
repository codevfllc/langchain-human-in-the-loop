# LangChain + CodeVF

Minimal LangChain Tool integration for routing review, debugging, and verification requests to CodeVF using the official Python SDK.

## Install

```bash
pip install codevf-sdk langchain-core pydantic
```

If you are working from source:

```bash
pip install -e .
```

Set your API key:

```bash
export CODEVF_API_KEY="cvf_live_..."
```

## Usage

```python
from langchain_codevf import CodeVFReviewTool

codevf_tool = CodeVFReviewTool(
    project_id=123,
    max_credits=50,
    mode="fast",
    timeout=300,
)

result = codevf_tool.invoke({
    "prompt": "Review this function for security issues and suggest fixes.",
    "attachments": [
        {
            "file_name": "app.py",
            "mime_type": "text/x-python",
            "content": "def login(user, pwd): ...",
        }
    ],
})

print(result)
```

### Attachment format

Each attachment accepts either `content` (plain text) or `base64` (for binary files). The SDK enforces limits and supported file types.

## Using With A LangChain Agent

```python
from langchain.agents import create_agent
from langchain_codevf import CodeVFReviewTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

codevf_tool = CodeVFReviewTool(project_id=123, max_credits=50)
llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_agent(
    model=llm,
    tools=[codevf_tool],
    system_prompt="You are a software engineering assistant.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Ask CodeVF to debug this error stack trace."}],
})

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
