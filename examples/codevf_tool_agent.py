import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_human_in_the_loop import HumanInTheLoop


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

    hitl = HumanInTheLoop(project_id=123, max_credits=50)
    codevf_tool = hitl.as_langchain_tool()
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
                "content": (
                    "Use the codevf_review tool with prompt "
                    "\"Review my authentication flow for security bugs.\" and attach "
                    "a file auth.py with mime_type text/x-python and content "
                    "\"def login(user, pwd): return user == 'admin' and pwd == 'admin'\"."
                ),
            }
        ]
    })

    print(_extract_text(result))


if __name__ == "__main__":
    main()
