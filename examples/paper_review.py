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
        max_credits=60,
        timeout=600,
    )

    prompt = (
        "Act as a research reviewer. Assess clarity of the methods, "
        "missing baselines, statistical reporting, and threats to validity. "
        "Provide concrete revisions for the authors."
    )

    attachments = [
        {
            "file_name": "methods.md",
            "mime_type": "text/plain",
            "content": (
                "# Methods\n\n"
                "We train a transformer on 5M sequences from a public corpus. "
                "The baseline is an LSTM. We report accuracy on a held-out test set.\n\n"
                "Hyperparameters were tuned by hand. We did not report variance across runs.\n"
            ),
        },
        {
            "file_name": "results.csv",
            "mime_type": "text/csv",
            "content": (
                "model,accuracy\n"
                "lstm,0.81\n"
                "transformer,0.85\n"
            ),
        },
        {
            "file_name": "related_work.md",
            "mime_type": "text/plain",
            "content": (
                "# Related Work\n\n"
                "Prior work reports comparable accuracy on similar datasets.\n"
            ),
        },
    ]

    result = hitl.invoke(prompt, attachments=attachments)
    print(result)


if __name__ == "__main__":
    main()
