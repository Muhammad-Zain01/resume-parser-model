#!/usr/bin/env python3
"""Download the public resume-parser dataset into the project's .data/ folder."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / ".data"


def main() -> int:
    """Restore or update .data from the configured Hugging Face dataset repo."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    repo_id = os.getenv("HF_DATASET_REPO", "").strip()
    if not repo_id:
        username = os.getenv("HF_USERNAME", "").strip()
        if username:
            repo_id = f"{username}/resume-parser-dataset"
        else:
            raise SystemExit(
                "Set HF_DATASET_REPO=owner/repository-name in the project .env file."
            )
    elif "/" not in repo_id:
        username = os.getenv("HF_USERNAME", "").strip()
        if not username:
            raise SystemExit(
                "HF_DATASET_REPO must be owner/repository-name, or set HF_USERNAME too."
            )
        repo_id = f"{username}/{repo_id}"

    revision = os.getenv("HF_DATASET_REVISION", "main").strip() or "main"
    token = os.getenv("HF_TOKEN", "").strip() or None
    if token and (token.startswith("hf_your_") or token.startswith("hf_placeholder_")):
        token = None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {repo_id} ({revision})")
    print(f"Destination: {DATA_DIR}")
    print("Downloading/updating remote files; local-only files will be left untouched.")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(DATA_DIR),
        token=token,
        tqdm_class=tqdm,
    )
    print("Dataset download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
