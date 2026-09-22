#!/usr/bin/env python3
"""Create one deduplicated, consistently named PDF directory."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / ".data" / "pdf"

CATEGORY_ALIASES = {
    "agricultural": "agriculture",
    "architects": "architect",
    "businessanalyst": "business analyst",
    "civilengineer": "civil engineer",
    "datascience": "data science",
    "devopsengineer": "devops engineer",
    "dotnetdeveloper": "dot net developer",
}

SOURCES = (
    ("mehyaar", Path("/Volumes/Code/Codex/resumes/ResumesPDF")),
    ("opensporks", Path("/Volumes/Code/Codex/opensporks-resumes/data/data")),
    ("hadikp", PROJECT_ROOT / ".data" / "Resumes PDF"),
)


def normalized_category(source: str, pdf_path: Path, source_root: Path) -> str:
    """Return a stable category folder name for a source PDF."""
    if source == "mehyaar":
        return "Uncategorized"

    relative_parts = pdf_path.relative_to(source_root).parts
    category = relative_parts[0] if len(relative_parts) > 1 else "Uncategorized"
    category = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", category)
    category = re.sub(r"\s+resumes?$", "", category, flags=re.IGNORECASE)
    category = re.sub(r"[^A-Za-z0-9]+", " ", category).strip()
    category_key = category.replace(" ", "").casefold()
    category_key = CATEGORY_ALIASES.get(category_key, category.casefold())
    if not category_key:
        return "Uncategorized"

    words = category_key.split()
    special_cases = {
        "bpo": "BPO",
        "devops": "DevOps",
        "dot": "DOT",
        "hr": "HR",
        "it": "IT",
        "qa": "QA",
        "ui": "UI",
        "ux": "UX",
    }
    return " ".join(special_cases.get(word, word.capitalize()) for word in words)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_files(source_root: Path) -> list[Path]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source directory: {source_root}")
    return sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
    )


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    hash_prefixes: dict[str, str] = {}
    copied = 0
    duplicates = 0

    for source, source_root in SOURCES:
        files = pdf_files(source_root)
        print(f"{source}: {len(files)} PDFs found")

        for pdf_path in files:
            full_hash = sha256(pdf_path)
            short_hash = full_hash[:16]

            previous_hash = hash_prefixes.get(short_hash)
            if previous_hash is not None and previous_hash != full_hash:
                raise RuntimeError(
                    f"16-character hash collision between {pdf_path} and "
                    f"another PDF: {short_hash}"
                )
            hash_prefixes[short_hash] = full_hash

            if full_hash in seen_hashes:
                duplicates += 1
                continue

            category = normalized_category(source, pdf_path, source_root)
            destination_dir = OUTPUT_ROOT / category
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"resume_{short_hash}.pdf"
            if destination.exists():
                raise FileExistsError(f"Output already exists: {destination}")

            shutil.copy2(pdf_path, destination)
            seen_hashes.add(full_hash)
            copied += 1

    print(f"Unique PDFs copied: {copied}")
    print(f"Duplicate PDFs skipped: {duplicates}")
    print(f"Output directory: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
