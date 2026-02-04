import os

from dotenv import load_dotenv

from langchain_human_in_the_loop import HumanInTheLoop


def main() -> None:
    load_dotenv()

    if "CODEVF_API_KEY" not in os.environ:
        raise RuntimeError("Set CODEVF_API_KEY before running this example.")
    if "CODEVF_PROJECT_ID" not in os.environ:
        raise RuntimeError("Set CODEVF_PROJECT_ID before running this example.")

    hitl = HumanInTheLoop(
        api_key="CODEVF_API_KEY",
        project_id=int(os.environ["CODEVF_PROJECT_ID"]),
        max_credits=50,
        timeout=300,
    )

    result = hitl.invoke("Review this output")
    print(result)


if __name__ == "__main__":
    main()
