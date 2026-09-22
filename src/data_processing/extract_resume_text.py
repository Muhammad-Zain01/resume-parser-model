#!/usr/bin/env python3
"""Extract text from the normalized resume PDFs.

The extractor mirrors the category folders from ``.data/raw`` into
``.data/extracted``. A PDF that has no selectable text receives a text file
containing ``[IMAGE_BASED]`` so it can be found and OCR-processed later.

This script does not perform OCR and does not create annotations.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".data"
IMAGE_BASED_MARKER = "[IMAGE_BASED]"
EXTRACTION_FAILED_MARKER = "[EXTRACTION_FAILED]"


@dataclass(frozen=True)
class ExtractionRecord:
    """One manifest row for one resume."""

    resume_id: str
    category: str
    raw_path: str
    extracted_path: str
    extraction_status: str
    page_count: int
    character_count: int
    extraction_error: str = ""


class ResumeTextExtractor:
    """Extract PDF text and maintain a simple, repeatable manifest."""

    def __init__(
        self,
        raw_dir: Path | None = None,
        extracted_dir: Path | None = None,
        manifest_path: Path | None = None,
        overwrite: bool = False,
    ) -> None:
        self.raw_dir = raw_dir or self._default_raw_dir()
        self.extracted_dir = extracted_dir or DATA_DIR / "extracted"
        self.manifest_path = manifest_path or DATA_DIR / "manifests" / "resume_manifest.csv"
        self.overwrite = overwrite

    @staticmethod
    def _default_raw_dir() -> Path:
        """Prefer the new ``raw`` name while supporting the current ``pdf`` name."""
        raw_dir = DATA_DIR / "raw"
        legacy_pdf_dir = DATA_DIR / "pdf"
        if raw_dir.is_dir():
            return raw_dir
        return legacy_pdf_dir

    def find_pdfs(self) -> list[Path]:
        """Return all PDFs below the raw directory in stable order."""
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"Raw resume directory does not exist: {self.raw_dir}")
        return sorted(
            path
            for path in self.raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )

    def _relative_category(self, pdf_path: Path) -> str:
        """Return the category folder, or ``Uncategorized`` for root PDFs."""
        relative_parent = pdf_path.relative_to(self.raw_dir).parent
        return str(relative_parent) if str(relative_parent) != "." else "Uncategorized"

    def _text_path(self, pdf_path: Path) -> Path:
        relative_path = pdf_path.relative_to(self.raw_dir).with_suffix(".txt")
        return self.extracted_dir / relative_path

    @staticmethod
    def _has_selectable_text(text: str) -> bool:
        """Treat any non-whitespace extracted content as selectable text."""
        return bool(re.search(r"\S", text))

    def _read_pdf(self, pdf_path: Path) -> tuple[str, int, str]:
        """Read all pages and return text, page count, and status."""
        reader = PdfReader(str(pdf_path))
        page_text = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(page for page in page_text if page)
        status = "TEXT_EXTRACTED" if self._has_selectable_text(text) else "IMAGE_BASED"
        return text, len(reader.pages), status

    def _write_text_file(self, text_path: Path, content: str) -> None:
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(content, encoding="utf-8")

    def _process_pdf(self, pdf_path: Path) -> ExtractionRecord:
        text_path = self._text_path(pdf_path)
        category = self._relative_category(pdf_path)
        resume_id = pdf_path.stem

        try:
            extracted_text, page_count, extraction_status = self._read_pdf(pdf_path)

            # Protect a later OCR result from being overwritten by a rerun.
            if text_path.exists() and not self.overwrite:
                current_text = text_path.read_text(encoding="utf-8", errors="replace")
                character_count = len(current_text)
            else:
                content = extracted_text if extracted_text else IMAGE_BASED_MARKER
                self._write_text_file(text_path, content)
                character_count = len(content)

            return ExtractionRecord(
                resume_id=resume_id,
                category=category,
                raw_path=self._relative_project_path(pdf_path),
                extracted_path=self._relative_project_path(text_path),
                extraction_status=extraction_status,
                page_count=page_count,
                character_count=character_count,
            )
        except Exception as error:  # Keep one broken PDF from stopping the batch.
            self._write_text_file(text_path, EXTRACTION_FAILED_MARKER)
            return ExtractionRecord(
                resume_id=resume_id,
                category=category,
                raw_path=self._relative_project_path(pdf_path),
                extracted_path=self._relative_project_path(text_path),
                extraction_status="EXTRACTION_FAILED",
                page_count=0,
                character_count=len(EXTRACTION_FAILED_MARKER),
                extraction_error=str(error),
            )

    @staticmethod
    def _relative_project_path(path: Path) -> str:
        """Store portable project-relative paths in the CSV."""
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()

    def _write_manifest(self, records: list[ExtractionRecord]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(asdict(records[0]).keys()) if records else list(ExtractionRecord.__annotations__)
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)

    def run(self) -> list[ExtractionRecord]:
        """Extract every PDF and write the non-annotation manifest."""
        records = [self._process_pdf(pdf_path) for pdf_path in self.find_pdfs()]
        self._write_manifest(records)
        return records

    def find_image_based_texts(self) -> list[Path]:
        """Find extracted text files that still need OCR."""
        if not self.extracted_dir.is_dir():
            return []
        return sorted(
            path
            for path in self.extracted_dir.rglob("*.txt")
            if IMAGE_BASED_MARKER in path.read_text(encoding="utf-8", errors="replace")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--extracted-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing text files. Do not use after manual OCR without a backup.",
    )
    parser.add_argument(
        "--list-image-based",
        action="store_true",
        help="List .txt files containing the image-based marker without extracting PDFs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extractor = ResumeTextExtractor(
        raw_dir=args.raw_dir,
        extracted_dir=args.extracted_dir,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
    )

    if args.list_image_based:
        image_files = extractor.find_image_based_texts()
        for path in image_files:
            print(path)
        print(f"Image-based text files: {len(image_files)}")
        return 0

    records = extractor.run()
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.extraction_status] = status_counts.get(record.extraction_status, 0) + 1

    print(f"Processed PDFs: {len(records)}")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")
    print(f"Manifest: {extractor.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

