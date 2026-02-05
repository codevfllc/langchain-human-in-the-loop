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
        max_credits=75,
        timeout=600,
    )

    prompt = (
        "Review this university research data pipeline for privacy risks. "
        "Call out PII handling, retention, access controls, and propose a "
        "de-identification and audit logging plan suitable for student data."
    )

    attachments = [
        {
            "file_name": "pipeline.py",
            "mime_type": "text/x-python",
            "content": (
                "import pandas as pd\n\n"
                "def load_students(path):\n"
                "    df = pd.read_csv(path)\n"
                "    df['email_hash'] = df['email']\n"
                "    df['advisor_note'] = df['advisor_note'].fillna('')\n"
                "    return df\n\n"
                "def export_for_model(df):\n"
                "    keep = ['student_id', 'email_hash', 'gpa', 'major', 'advisor_note']\n"
                "    return df[keep]\n\n"
                "if __name__ == '__main__':\n"
                "    data = load_students('students.csv')\n"
                "    export_for_model(data).to_csv('training.csv', index=False)\n"
            ),
        },
        {
            "file_name": "students_schema.csv",
            "mime_type": "text/csv",
            "content": (
                "student_id,email,phone,major,gpa,advisor_note\n"
                "12345,student@university.edu,555-0100,CS,3.8,Needs tutoring\n"
            ),
        },
        {
            "file_name": "data_policy.md",
            "mime_type": "text/plain",
            "content": (
                "# Data Policy Draft\n\n"
                "We will store raw student data on a shared drive for the project.\n"
                "No data deletion timeline is defined yet.\n"
            ),
        },
    ]

    result = hitl.invoke(prompt, attachments=attachments)
    print(result)


if __name__ == "__main__":
    main()
