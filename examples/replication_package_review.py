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
        project_id=int(os.environ["CODEVF_PROJECT_ID"]),
        max_credits=80,
        timeout=600,
    )

    prompt = (
        "Review this replication package for a university research release. "
        "Check for documentation gaps, missing dependencies, unclear licensing, "
        "data anonymization risks, and propose a release checklist."
    )

    attachments = [
        {
            "file_name": "release_checklist.md",
            "mime_type": "text/plain",
            "content": (
                "# Release Checklist (Draft)\n\n"
                "- Code archived in repo\n"
                "- Data uploaded to shared drive\n"
                "- No license selected yet\n"
            ),
        },
        {
            "file_name": "data_readme.md",
            "mime_type": "text/plain",
            "content": (
                "# Dataset\n\n"
                "Contains interview transcripts and demographics.\n"
                "Some fields may include names or locations.\n"
                "No de-identification steps documented.\n"
            ),
        },
        {
            "file_name": "environment.yml",
            "mime_type": "text/yaml",
            "content": (
                "name: research-env\n"
                "dependencies:\n"
                "  - python=3.10\n"
                "  - pandas\n"
                "  - scikit-learn\n"
                "  - matplotlib\n"
            ),
        },
    ]

    result = hitl.invoke(prompt, attachments=attachments)
    print(result)


if __name__ == "__main__":
    main()
