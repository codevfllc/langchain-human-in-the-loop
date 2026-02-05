import os
from pathlib import Path

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
        "You are an IRB reviewer for a university study. Review the protocol, consent "
        "materials, and survey instrument. Identify risks, consent gaps, data minimization "
        "issues, and required safeguards. Provide specific revision requests."
    )

    base_dir = Path(__file__).resolve().parent
    protocol_md = (base_dir / "sample_protocol.md").read_text(encoding="utf-8")

    attachments = [
        {
            "file_name": "protocol.md",
            "mime_type": "text/plain",
            "content": protocol_md,
        },
        {
            "file_name": "consent_form.md",
            "mime_type": "text/plain",
            "content": (
                "# Consent Form\n\n"
                "The study is about learning habits.\n"
                "We will collect your responses and schedule screenshot.\n"
                "There are no risks.\n"
                "Contact the PI with questions.\n"
            ),
        },
        {
            "file_name": "survey_questions.csv",
            "mime_type": "text/csv",
            "content": (
                "id,question\n"
                "q1,How many hours do you study per week?\n"
                "q2,Provide your student ID.\n"
                "q3,Do you have any diagnosed mental health conditions?\n"
            ),
        },
    ]

    result = hitl.invoke(prompt, attachments=attachments)
    print(result)


if __name__ == "__main__":
    main()
