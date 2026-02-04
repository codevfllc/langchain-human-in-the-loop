import os

from dotenv import load_dotenv
from langchain_codevf import CodeVFReviewTool


def main() -> None:
    load_dotenv()

    if "CODEVF_API_KEY" not in os.environ:
        raise RuntimeError("Set CODEVF_API_KEY before running this example.")

    tool = CodeVFReviewTool(project_id=19, max_credits=50, timeout=600)
    result = tool.invoke({
        "prompt": "Review this authentication snippet for security issues.",
        "attachments": [
            {
                "file_name": "auth.py",
                "mime_type": "text/x-python",
                "content": "def login(user, pwd): return user == 'admin' and pwd == 'admin'",
            }
        ],
    })

    print(result)


if __name__ == "__main__":
    main()
