#!/usr/bin/env python3
"""Extract selectable PDF text and mark image-based resumes for later OCR."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / ".data"
IMAGE_BASED_MARKER = "[IMAGE_BASED]"
EXTRACTION_FAILED_MARKER = "[EXTRACTION_FAILED]"


@dataclass(frozen=True)
class ExtractionRecord:
    """Result metadata for one processed resume."""

    resume_id: str
    category: str
    raw_path: str
    extracted_path: str
    extraction_status: str
    page_count: int
    character_count: int
    extraction_error: str = ""


class ResumeTextExtractor:
    """Extract text while preserving the raw category directory structure."""

    def __init__(
        self,
        raw_dir: Path | None = None,
        extracted_dir: Path | None = None,
        overwrite: bool = False,
        workers: int = 8,
    ) -> None:
        self.raw_dir = raw_dir or self._default_raw_dir()
        self.extracted_dir = extracted_dir or DATA_DIR / "extracted"
        self.overwrite = overwrite
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = workers

    @staticmethod
    def _default_raw_dir() -> Path:
        """Use ``raw`` when available and support the current ``pdf`` name."""
        raw_dir = DATA_DIR / "raw"
        return raw_dir if raw_dir.is_dir() else DATA_DIR / "pdf"

    def find_pdfs(self) -> list[Path]:
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"Raw resume directory does not exist: {self.raw_dir}")
        return sorted(
            path
            for path in self.raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )

    def _category_for(self, pdf_path: Path) -> str:
        parent = pdf_path.relative_to(self.raw_dir).parent
        return str(parent) if str(parent) != "." else "Uncategorized"

    def _text_path_for(self, pdf_path: Path) -> Path:
        relative_path = pdf_path.relative_to(self.raw_dir).with_suffix(".txt")
        return self.extracted_dir / relative_path

    @staticmethod
    def _extract(pdf_path: Path) -> tuple[str, int, str]:
        reader = PdfReader(str(pdf_path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
        status = "TEXT_EXTRACTED" if re.search(r"\S", text) else "IMAGE_BASED"
        return text, len(reader.pages), status

    @staticmethod
    def _relative_path(path: Path) -> str:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _process_pdf(self, pdf_path: Path) -> ExtractionRecord:
        text_path = self._text_path_for(pdf_path)
        category = self._category_for(pdf_path)

        try:
            text, page_count, status = self._extract(pdf_path)

            # Existing files are preserved so a future OCR result is not lost.
            if not text_path.exists() or self.overwrite:
                self._write_text(text_path, text or IMAGE_BASED_MARKER)

            current_text = text_path.read_text(encoding="utf-8", errors="replace")
            return ExtractionRecord(
                resume_id=pdf_path.stem,
                category=category,
                raw_path=self._relative_path(pdf_path),
                extracted_path=self._relative_path(text_path),
                extraction_status=status,
                page_count=page_count,
                character_count=len(current_text),
            )
        except Exception as error:  # Continue processing if one PDF is damaged.
            self._write_text(text_path, EXTRACTION_FAILED_MARKER)
            return ExtractionRecord(
                resume_id=pdf_path.stem,
                category=category,
                raw_path=self._relative_path(pdf_path),
                extracted_path=self._relative_path(text_path),
                extraction_status="EXTRACTION_FAILED",
                page_count=0,
                character_count=len(EXTRACTION_FAILED_MARKER),
                extraction_error=str(error),
            )

    def run(self) -> list[ExtractionRecord]:
        """Process all PDFs and return extraction results."""
        pdf_paths = self.find_pdfs()
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = executor.map(self._process_pdf, pdf_paths)
            return list(
                tqdm(
                    results,
                    total=len(pdf_paths),
                    desc=f"Extracting resumes ({self.workers} workers)",
                    unit="pdf",
                )
            )

    def find_image_based_texts(self) -> list[Path]:
        """Return text files that still contain the image-based marker."""
        if not self.extracted_dir.is_dir():
            return []
        return sorted(
            path
            for path in self.extracted_dir.rglob("*.txt")
            if IMAGE_BASED_MARKER in path.read_text(encoding="utf-8", errors="replace")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--extracted-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing text files; do not use after OCR without a backup.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel PDF workers (default: 8).",
    )
    parser.add_argument(
        "--list-image-based",
        action="store_true",
        help="List text files containing [IMAGE_BASED] without extracting PDFs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extractor = ResumeTextExtractor(
        raw_dir=args.raw_dir,
        extracted_dir=args.extracted_dir,
        overwrite=args.overwrite,
        workers=args.workers,
    )

    if args.list_image_based:
        image_files = extractor.find_image_based_texts()
        for path in image_files:
            print(path)
        print(f"Image-based text files: {len(image_files)}")
        return 0

    records = extractor.run()
    counts: dict[str, int] = {}
    for record in records:
        counts[record.extraction_status] = counts.get(record.extraction_status, 0) + 1
    print(f"Processed PDFs: {len(records)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
