"""Curate annotated resumes into train/validation/test datasets.

Inputs come from the earlier pipeline steps: extracted resume text (01) and
schema-validated annotations (02). The dataset is split by category into
train/validation/test partitions using configurable fractions.
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
DEFAULT_SPLIT_RATIOS = {"train": 0.85, "validation": 0.05, "test": 0.10}

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
        """Render one model-independent input/target record."""
        return {
            "resume_id": self.resume_id,
            "category": self.category,
            "resume_text": self.resume_text,
            "target_json": self.target,
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
    train_fraction: float = DEFAULT_SPLIT_RATIOS["train"],
    validation_fraction: float = DEFAULT_SPLIT_RATIOS["validation"],
    test_fraction: float = DEFAULT_SPLIT_RATIOS["test"],
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create deterministic, approximately stratified splits by resume category."""
    ratios = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    if any(ratio < 0 or ratio > 1 for ratio in ratios.values()):
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction <= 0 or abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1 and train_fraction must be positive.")
    if len(set(pairs)) != len(pairs):
        raise ValueError("pairs must not contain duplicate paths.")

    by_category: dict[str, list[str]] = {}
    for pair in pairs:
        category = Path(pair).parent.as_posix()
        by_category.setdefault(category, []).append(pair)

    category_names = sorted(by_category)
    total = len(pairs)
    raw_totals = {name: total * ratio for name, ratio in ratios.items()}
    target_totals = {name: int(count) for name, count in raw_totals.items()}
    remaining = total - sum(target_totals.values())
    split_order = sorted(
        ratios,
        key=lambda name: (raw_totals[name] - target_totals[name], name),
        reverse=True,
    )
    for name in split_order[:remaining]:
        target_totals[name] += 1

    # Largest-remainder allocation keeps each category represented in roughly
    # the same proportions while preserving exact overall split sizes.
    raw_test = {name: len(by_category[name]) * test_fraction for name in category_names}
    test_counts = {name: int(count) for name, count in raw_test.items()}
    for name in sorted(
        category_names,
        key=lambda category: (raw_test[category] - test_counts[category], category),
        reverse=True,
    )[: target_totals["test"] - sum(test_counts.values())]:
        test_counts[name] += 1

    remaining_sizes = {
        name: len(by_category[name]) - test_counts[name]
        for name in category_names
    }
    remaining_fraction = validation_fraction / (train_fraction + validation_fraction)
    raw_validation = {
        name: remaining_sizes[name] * remaining_fraction for name in category_names
    }
    validation_counts = {name: int(count) for name, count in raw_validation.items()}
    for name in sorted(
        category_names,
        key=lambda category: (
            raw_validation[category] - validation_counts[category], category
        ),
        reverse=True,
    )[: target_totals["validation"] - sum(validation_counts.values())]:
        validation_counts[name] += 1

    splits = {name: [] for name in ratios}
    rng = random.Random(seed)
    for category in category_names:
        category_pairs = by_category[category]
        rng.shuffle(category_pairs)
        test_end = test_counts[category]
        validation_end = test_end + validation_counts[category]
        splits["test"].extend(category_pairs[:test_end])
        splits["validation"].extend(category_pairs[test_end:validation_end])
        splits["train"].extend(category_pairs[validation_end:])

    return splits


def build_dataset(
    *,
    train_fraction: float = DEFAULT_SPLIT_RATIOS["train"],
    validation_fraction: float = DEFAULT_SPLIT_RATIOS["validation"],
    test_fraction: float = DEFAULT_SPLIT_RATIOS["test"],
    seed: int = 42,
    extracted_dir: Path = EXTRACTED_DIR,
    annotated_dir: Path = ANNOTATED_DIR,
) -> tuple[dict[str, list[CuratedItem]], list[dict[str, str]]]:
    """Pair, split, then load and validate every target."""
    pair_splits = split_pairs(
        pair_paths(extracted_dir, annotated_dir),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
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
    total = sum(len(items) for items in splits.values())
    summary: dict[str, Any] = {
        name: {
            "count": len(items),
            "percentage": round(100 * len(items) / total, 3) if total else 0.0,
            "categories": dict(sorted(Counter(item.category for item in items).items())),
        }
        for name, items in splits.items()
    }
    summary["total"] = total
    return summary


def write_jsonl(items: Iterable[CuratedItem], path: Path) -> Path:
    """Write records as JSON Lines, one chat record per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_record(), ensure_ascii=False) + "\n")
    return path


def write_dataset_card(summary: dict[str, Any], output_dir: Path = DATA_DIR) -> Path:
    """Write the Hugging Face dataset card and pipeline documentation."""
    lines = [
        "---",
        "pretty_name: Resume Parser Dataset",
        "task_categories:",
        "- text-generation",
        "language:",
        "- en",
        "tags:",
        "- resume-parsing",
        "- information-extraction",
        "- structured-output",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: curated/train.jsonl",
        "  - split: validation",
        "    path: curated/validation.jsonl",
        "  - split: test",
        "    path: curated/test.jsonl",
        "---",
        "",
        "# Resume Parser Dataset and Processing Pipeline",
        "",
        "This repository documents and packages the data pipeline for the [resume-parser-model project](https://github.com/Muhammad-Zain01/resume-parser-model). The project explores fine-tuning compact language models to turn resume text into structured JSON, using only facts supported by each resume.",
        "",
        "## What is in this repository?",
        "",
        "The folders preserve the stages used to build the dataset:",
        "",
        "| folder | contents |",
        "| --- | --- |",
        "| `raw/` | Original source resumes, organized by occupation/category. |",
        "| `extracted/` | Text extracted from each source file; image-only or failed extractions may be represented by a status marker. |",
        "| `annotated/` | Resume-level target JSON, validated against the project's Pydantic schema. |",
        "| `curated/` | JSONL records with separate resume text and target JSON fields, plus split statistics. |",
        "",
        "Resume files and their derived text/annotations use matching category folders and resume IDs so the processing trail can be followed across stages.",
        "",
        "## Training example format",
        "",
        "Each JSONL record contains four top-level fields:",
        "",
        "- `resume_id`: the stable resume identifier shared across processing stages.",
        "- `category`: the resume's occupation/category label.",
        "- `resume_text`: the extracted text used as model input (X).",
        "- `target_json`: the schema-validated JSON object used as the expected output (Y).",
        "",
        "The target follows the project's [`ResumeOutput` Pydantic schema](https://github.com/Muhammad-Zain01/resume-parser-model/blob/main/src/schema/resume_output.py). It covers contact details, professional summary and objectives, roles, work history, education, skills, certifications, projects, awards, publications, volunteering, memberships, references, and additional sections. Missing scalar values use `null`; missing collections use empty lists. Convert these fields to a model-specific prompt or chat format during training preprocessing as needed.",
        "",
        "## Dataset splits",
        "",
        "| split | rows | percentage |",
        "| --- | ---: | ---: |",
    ]
    for name in ("train", "validation", "test"):
        if name in summary:
            lines.append(
                f"| {name} | {summary[name]['count']:,} | {summary[name]['percentage']:.3f}% |"
            )
    if summary.get("total") is not None:
        lines.append(f"| **total** | **{summary['total']:,}** | **100%** |")
    lines.extend((
        "",
        "Splits are approximately stratified by resume category and created with seed 42. The test split is held out from model training and validation for final evaluation.",
        "",
        "## Source datasets",
        "",
        "The source materials were collected from:",
        "",
        "- [Mehyaar – Annotated NER PDF Resumes](https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes)",
        "- [opensporks – Resumes](https://huggingface.co/datasets/opensporks/resumes)",
        "- [hadikp – Resume Dataset PDF](https://www.kaggle.com/datasets/hadikp/resume-data-pdf/data)",
        "",
        "## Intended use and limitations",
        "",
        "This corpus is intended for research and experimentation in resume text extraction and structured-output fine-tuning. It is not a resume-ranking dataset and should not be used to make hiring or other high-impact decisions. Resume layouts, extraction quality, annotation coverage, and category balance vary; results should be evaluated on a separate, representative test set.",
        "",
        "Resumes may contain personal information. The source datasets have their own terms and licensing; review those terms and applicable privacy requirements before reusing or redistributing the files. No single blanket license is asserted for all source materials in this combined corpus.",
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
    folder: Path = DATA_DIR,
    repo_id: str | None = None,
    *,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Upload the complete local data pipeline to a Hugging Face dataset repository."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to push datasets. Run: pip install -r requirements.txt"
        ) from error
    repo_id = repo_id or os.getenv("HF_DATASET_REPO", "").strip()
    if not repo_id:
        username = os.getenv("HF_USERNAME", "").strip()
        if not username:
            raise RuntimeError(
                "Set HF_DATASET_REPO (owner/repo-name) in .env or pass repo_id explicitly."
            )
        repo_id = f"{username}/resume-parser-dataset"
    elif "/" not in repo_id:
        username = os.getenv("HF_USERNAME", "").strip()
        if not username:
            raise RuntimeError(
                "Set HF_DATASET_REPO as owner/repo-name, or set HF_USERNAME too."
            )
        repo_id = f"{username}/{repo_id}"
    if repo_id.startswith("/") or repo_id.endswith("/"):
        raise RuntimeError("HF_DATASET_REPO must be owner/repo-name.")
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from the environment or project .env.")
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    HfApi().upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        ignore_patterns=[".cache/huggingface/**"],
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
