import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_codevf import CodeVFReviewTool
from langchain_openai import ChatOpenAI


def _extract_text(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    return getattr(last, "content", None) or last.get("content", "")


def main() -> None:
    load_dotenv()

    # Requires CODEVF_API_KEY in env
    if "CODEVF_API_KEY" not in os.environ:
        raise RuntimeError("Set CODEVF_API_KEY before running this example.")

    codevf_tool = CodeVFReviewTool(project_id=123, max_credits=50)
    llm = ChatOpenAI(model="gpt-4o-mini")

    agent = create_agent(
        model=llm,
        tools=[codevf_tool],
        system_prompt="You are a software engineering assistant.",
    )

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Ask CodeVF to review my authentication flow for security bugs.",
            }
        ]
    })

    print(_extract_text(result))


if __name__ == "__main__":
    main()
