#!/usr/bin/env python3
"""Upload the complete .data pipeline to the configured Hugging Face dataset repo."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / ".data"


def _repo_id_from_env() -> str:
    """Resolve HF_DATASET_REPO, with optional legacy HF_USERNAME support."""
    repo_id = os.getenv("HF_DATASET_REPO", "").strip()
    if not repo_id:
        username = os.getenv("HF_USERNAME", "").strip()
        if username:
            repo_id = f"{username}/resume-parser-dataset"
        else:
            raise SystemExit("Set HF_DATASET_REPO=owner/repository-name in .env.")
    elif "/" not in repo_id:
        username = os.getenv("HF_USERNAME", "").strip()
        if not username:
            raise SystemExit("HF_DATASET_REPO must be owner/repository-name.")
        repo_id = f"{username}/{repo_id}"

    if repo_id.startswith("/") or repo_id.endswith("/") or repo_id.count("/") != 1:
        raise SystemExit("HF_DATASET_REPO must use the format owner/repository-name.")
    return repo_id


def _ensure_dataset_card() -> None:
    """Create the Hub root card from curated split statistics if it is missing."""
    card_path = DATA_DIR / "README.md"
    if card_path.exists():
        return

    stats_path = DATA_DIR / "curated" / "stats.json"
    if not stats_path.is_file():
        raise SystemExit(
            f"Dataset card and split statistics are missing. Expected {stats_path}."
        )

    curation = importlib.import_module("src.steps.03_data_curation.curation")
    summary = json.loads(stats_path.read_text(encoding="utf-8"))
    curation.write_dataset_card(summary, output_dir=DATA_DIR)
    print(f"Created Dataset Card: {card_path}")


def _local_inventory() -> tuple[int, int]:
    """Return the upload file count and byte size, excluding local metadata."""
    files = (
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.name != ".DS_Store"
        and ".cache/huggingface" not in path.as_posix()
    )
    count = 0
    total_bytes = 0
    for path in files:
        count += 1
        total_bytes += path.stat().st_size
    return count, total_bytes


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    repo_id = _repo_id_from_env()
    token = os.getenv("HF_TOKEN", "").strip()
    if not token or token.startswith(("hf_your_", "hf_placeholder_")):
        raise SystemExit("Set a valid HF_TOKEN in .env; upload access is required.")
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {DATA_DIR}")

    expected_stages = ("raw", "extracted", "annotated", "curated")
    missing_stages = [name for name in expected_stages if not (DATA_DIR / name).is_dir()]
    if missing_stages:
        raise SystemExit(f"Missing .data stages: {', '.join(missing_stages)}")

    _ensure_dataset_card()
    file_count, total_bytes = _local_inventory()
    size_gib = total_bytes / (1024**3)
    revision = os.getenv("HF_DATASET_REVISION", "main").strip() or "main"

    print(f"Target dataset: {repo_id} ({revision})")
    print(f"Source folder: {DATA_DIR}")
    print(f"Files to upload: {file_count:,} ({size_gib:.2f} GiB)")
    print("Creating/updating the dataset as public, then uploading all four stages...")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )
    api.update_repo_settings(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        token=token,
    )
    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message="Upload complete resume dataset pipeline",
        ignore_patterns=[
            ".cache/huggingface/**",
            ".DS_Store",
            "**/.DS_Store",
        ],
        token=token,
    )

    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
