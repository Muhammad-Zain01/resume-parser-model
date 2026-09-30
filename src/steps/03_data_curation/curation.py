"""Curate annotated resumes into train/validation/test datasets.

Inputs come from the earlier pipeline steps: extracted resume text (01) and
schema-validated annotations (02). The test split uses the same sampling call as
the evaluation notebook, so with matching ``test_size`` and ``seed`` the test
split is exactly the evaluation sample and training never sees it.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema.resume_output import ResumeOutput


DATA_DIR = PROJECT_ROOT / ".data"
EXTRACTED_DIR = DATA_DIR / "extracted"
ANNOTATED_DIR = DATA_DIR / "annotated"
CURATED_DIR = DATA_DIR / "curated"

INSTRUCTION = (
    "You extract structured information from a resume. "
    "Use only facts explicitly supported by the resume. Do not guess, infer, or invent. "
    "Use null for unavailable scalar values and [] for unavailable list values. "
    "Return exactly one JSON object."
)


@dataclass(slots=True)
class CuratedItem:
    """One resume's text paired with its schema-validated target JSON."""

    resume_id: str
    category: str
    resume_text: str
    target: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        """Render one chat-format training record."""
        return {
            "resume_id": self.resume_id,
            "category": self.category,
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": self.resume_text},
                {
                    "role": "assistant",
                    "content": json.dumps(self.target, ensure_ascii=False, separators=(",", ":")),
                },
            ],
        }


def pair_paths(
    extracted_dir: Path = EXTRACTED_DIR,
    annotated_dir: Path = ANNOTATED_DIR,
) -> list[str]:
    """Sorted relative paths that have both an extracted text and an annotation."""
    texts = {p.relative_to(extracted_dir).with_suffix("").as_posix() for p in extracted_dir.rglob("*.txt")}
    labels = {p.relative_to(annotated_dir).with_suffix("").as_posix() for p in annotated_dir.rglob("*.json")}
    return sorted(texts & labels)


def load_pair(
    pair: str,
    extracted_dir: Path = EXTRACTED_DIR,
    annotated_dir: Path = ANNOTATED_DIR,
) -> CuratedItem:
    """Load one pair and canonicalize its target through the schema."""
    raw = json.loads((annotated_dir / f"{pair}.json").read_text(encoding="utf-8"))
    target = ResumeOutput.model_validate(raw).model_dump(mode="json")
    return CuratedItem(
        resume_id=Path(pair).stem,
        category=Path(pair).parent.as_posix(),
        resume_text=(extracted_dir / f"{pair}.txt").read_text(encoding="utf-8"),
        target=target,
    )


def load_items(
    pairs: Iterable[str],
    extracted_dir: Path = EXTRACTED_DIR,
    annotated_dir: Path = ANNOTATED_DIR,
) -> tuple[list[CuratedItem], list[dict[str, str]]]:
    """Load and schema-validate each pair, reporting invalid targets."""
    items: list[CuratedItem] = []
    invalid: list[dict[str, str]] = []
    for pair in pairs:
        try:
            items.append(load_pair(pair, extracted_dir, annotated_dir))
        except (ValidationError, json.JSONDecodeError) as error:
            invalid.append({"pair": pair, "error": f"{type(error).__name__}: {error}"[:300]})
    return items, invalid


def split_pairs(
    pairs: list[str],
    *,
    test_size: int = 15,
    validation_size: int = 500,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministic pair split with the evaluation sample reserved for test."""
    if test_size < 1 or validation_size < 0:
        raise ValueError("test_size must be positive and validation_size non-negative.")
    if test_size + validation_size > len(pairs):
        raise ValueError("test_size + validation_size exceeds the number of pairs.")
    test = random.Random(seed).sample(pairs, test_size)
    remaining = [pair for pair in pairs if pair not in set(test)]
    validation = random.Random(seed + 1).sample(remaining, validation_size) if validation_size else []
    reserved = set(test) | set(validation)
    train = [pair for pair in pairs if pair not in reserved]
    return {"train": train, "validation": validation, "test": test}


def build_dataset(
    *,
    test_size: int = 15,
    validation_size: int = 500,
    seed: int = 42,
    extracted_dir: Path = EXTRACTED_DIR,
    annotated_dir: Path = ANNOTATED_DIR,
) -> tuple[dict[str, list[CuratedItem]], list[dict[str, str]]]:
    """Pair, split, then load and validate every target."""
    pair_splits = split_pairs(
        pair_paths(extracted_dir, annotated_dir),
        test_size=test_size,
        validation_size=validation_size,
        seed=seed,
    )
    splits: dict[str, list[CuratedItem]] = {}
    invalid: list[dict[str, str]] = []
    for name, split in pair_splits.items():
        items, split_invalid = load_items(split, extracted_dir, annotated_dir)
        splits[name] = items
        invalid.extend({"split": name, **entry} for entry in split_invalid)
    return splits, invalid


def summarize(splits: dict[str, list[CuratedItem]]) -> dict[str, Any]:
    """Row counts per split and category spread."""
    summary: dict[str, Any] = {
        name: {
            "count": len(items),
            "categories": dict(sorted(Counter(item.category for item in items).items())),
        }
        for name, items in splits.items()
    }
    summary["total"] = sum(len(items) for items in splits.values())
    return summary


def write_jsonl(items: Iterable[CuratedItem], path: Path) -> Path:
    """Write records as JSON Lines, one chat record per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_record(), ensure_ascii=False) + "\n")
    return path


def write_dataset_card(summary: dict[str, Any], output_dir: Path = CURATED_DIR) -> Path:
    """Write a Hugging Face dataset card with the split counts."""
    lines = [
        "---",
        "task_categories:",
        "- text-generation",
        "language:",
        "- en",
        "---",
        "",
        "# Resume extraction dataset",
        "",
        "Chat-format pairs for fine-tuning models that extract structured JSON from resumes.",
        "",
        "| split | rows |",
        "| --- | ---: |",
    ]
    for name in ("train", "validation", "test"):
        if name in summary:
            lines.append(f"| {name} | {summary[name]['count']:,} |")
    lines.extend((
        "",
        "Each record has `resume_id`, `category`, and `messages` ",
        "(system instruction, resume text, target JSON).",
        "",
        "The test split matches the project's evaluation sample, so reported ",
        "evaluation scores stay comparable with models trained on this dataset.",
        "",
    ))
    path = Path(output_dir) / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_splits(
    splits: dict[str, list[CuratedItem]],
    output_dir: Path = CURATED_DIR,
) -> dict[str, Path]:
    """Write one JSONL file per split plus ``stats.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: write_jsonl(items, output_dir / f"{name}.jsonl")
        for name, items in splits.items()
    }
    (output_dir / "stats.json").write_text(
        json.dumps(summarize(splits), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def push_to_hub(
    folder: Path = CURATED_DIR,
    repo_id: str | None = None,
    *,
    private: bool = True,
    token: str | None = None,
) -> str:
    """Upload the curated folder to a Hugging Face dataset repository."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to push datasets. Run: pip install -r requirements.txt"
        ) from error
    repo_id = repo_id or f"{os.getenv('HF_USERNAME', '')}/resume-curation"
    if not repo_id or repo_id.startswith("/"):
        raise RuntimeError("Set HF_USERNAME in .env or pass repo_id explicitly.")
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from the environment or project .env.")
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    HfApi().upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


__all__ = [
    "ANNOTATED_DIR",
    "CURATED_DIR",
    "CuratedItem",
    "EXTRACTED_DIR",
    "INSTRUCTION",
    "build_dataset",
    "load_items",
    "load_pair",
    "pair_paths",
    "push_to_hub",
    "split_pairs",
    "summarize",
    "write_dataset_card",
    "write_jsonl",
    "write_splits",
]
