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
        "Audit this research dataset and model training summary for bias and fairness risks. "
        "Look for sampling issues, proxy variables, evaluation gaps across groups, and "
        "recommend mitigation steps and reporting requirements."
    )

    attachments = [
        {
            "file_name": "dataset_stats.csv",
            "mime_type": "text/csv",
            "content": (
                "group,count,positive_outcome_rate\n"
                "female,220,0.62\n"
                "male,780,0.74\n"
                "nonbinary,12,0.50\n"
                "group_unknown,145,0.79\n"
            ),
        },
        {
            "file_name": "features.csv",
            "mime_type": "text/csv",
            "content": (
                "feature,description\n"
                "sat_score,standardized test score\n"
                "zip_code,home ZIP\n"
                "legacy_flag,parent alumni status\n"
                "first_gen,first generation college student\n"
            ),
        },
        {
            "file_name": "training_summary.md",
            "mime_type": "text/plain",
            "content": (
                "# Training Summary\n\n"
                "We trained a logistic regression model to predict admissions outcomes.\n"
                "Overall accuracy is 0.83. We did not report performance by subgroup.\n"
                "The dataset was collected from the 2018-2020 applicant pool.\n"
            ),
        },
    ]

    result = hitl.invoke(prompt, attachments=attachments)
    print(result)


if __name__ == "__main__":
    main()
