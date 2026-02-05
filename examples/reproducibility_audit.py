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
        "You are reviewing an academic ML experiment for reproducibility. "
        "Identify missing controls, logging, data/version tracking, random seeds, "
        "and propose a minimal reproducibility checklist for the authors."
    )

    attachments = [
        {
            "file_name": "train.py",
            "mime_type": "text/x-python",
            "content": (
                "import random\n"
                "import numpy as np\n"
                "import torch\n\n"
                "def train(model, loader, epochs=5):\n"
                "    optimizer = torch.optim.Adam(model.parameters())\n"
                "    for _ in range(epochs):\n"
                "        for batch in loader:\n"
                "            loss = model(batch).mean()\n"
                "            loss.backward()\n"
                "            optimizer.step()\n"
                "            optimizer.zero_grad()\n\n"
                "if __name__ == '__main__':\n"
                "    # TODO: dataset path set by user\n"
                "    print('Training...')\n"
            ),
        },
        {
            "file_name": "config.yaml",
            "mime_type": "text/yaml",
            "content": (
                "dataset: /data/imagenet\n"
                "batch_size: 64\n"
                "epochs: 5\n"
                "seed: null\n"
                "augmentation: standard\n"
                "model: resnet18\n"
            ),
        },
        {
            "file_name": "requirements.txt",
            "mime_type": "text/plain",
            "content": "torch\nnumpy\nscikit-learn\n",
        },
        {
            "file_name": "README.md",
            "mime_type": "text/markdown",
            "content": (
                "# Experiment\n\n"
                "Run training with `python train.py`.\n\n"
                "Results are reported in `results.csv`.\n"
            ),
        },
    ]

    result = hitl.invoke(prompt, attachments=attachments)
    print(result)


if __name__ == "__main__":
    main()
