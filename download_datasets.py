#!/usr/bin/env python3
"""Download one Hugging Face dataset repository directly into .data/."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


# Update these variables, or set the matching values in .env.
HF_USERNAME = os.getenv("HF_USERNAME", "your-huggingface-username")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "your-dataset-repo")
HF_DATASET_REVISION = os.getenv("HF_DATASET_REVISION", "main")
HF_TOKEN = os.getenv("HF_TOKEN") or None

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / ".data"


def load_local_env() -> None:
    """Load simple KEY=VALUE entries from .env without overriding shell values."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def main() -> int:
    load_local_env()

    username = os.getenv("HF_USERNAME", HF_USERNAME)
    dataset_repo = os.getenv("HF_DATASET_REPO", HF_DATASET_REPO)
    revision = os.getenv("HF_DATASET_REVISION", HF_DATASET_REVISION)
    token = os.getenv("HF_TOKEN") or HF_TOKEN
    if token and token.startswith("hf_placeholder_"):
        token = None

    if username == "your-huggingface-username" or dataset_repo == "your-dataset-repo":
        raise SystemExit(
            "Set HF_USERNAME and HF_DATASET_REPO in .env or at the top of this script."
        )

    dataset_id = f"{username}/{dataset_repo}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset: {dataset_id}")
    print(f"Destination: {DATA_DIR}")

    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(DATA_DIR),
        token=token,
    )
    print("Dataset downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
