#!/usr/bin/env python3
"""Show progress for the batch-based image-resume vision extraction."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / ".data"
DEFAULT_RAW_DIR = DATA_DIR / "raw"
DEFAULT_EXTRACTED_DIR = DATA_DIR / "extracted"
IMAGE_BASED_MARKER = "[IMAGE_BASED]"
VISION_FAILED_MARKER = "[VISION_EXTRACTION_FAILED]"
EXTRACTION_FAILED_MARKER = "[EXTRACTION_FAILED]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size used to estimate remaining runs (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if not args.extracted_dir.is_dir():
        raise SystemExit(f"Extracted directory does not exist: {args.extracted_dir}")

    counts = {
        "total_text_files": 0,
        "remaining_image_based": 0,
        "remaining_with_raw_pdf": 0,
        "text_available": 0,
        "vision_failed": 0,
        "source_extraction_failed": 0,
        "missing_raw_pdf": 0,
    }

    for text_path in args.extracted_dir.rglob("*.txt"):
        counts["total_text_files"] += 1
        content = text_path.read_text(encoding="utf-8", errors="replace").strip()
        raw_path = (args.raw_dir / text_path.relative_to(args.extracted_dir)).with_suffix(".pdf")

        if not raw_path.is_file():
            counts["missing_raw_pdf"] += 1
        if content == IMAGE_BASED_MARKER:
            counts["remaining_image_based"] += 1
            if raw_path.is_file():
                counts["remaining_with_raw_pdf"] += 1
        elif content == VISION_FAILED_MARKER:
            counts["vision_failed"] += 1
        elif content == EXTRACTION_FAILED_MARKER:
            counts["source_extraction_failed"] += 1
        else:
            counts["text_available"] += 1

    remaining = counts["remaining_with_raw_pdf"]
    batches = (remaining + args.batch_size - 1) // args.batch_size

    print(f"Total extracted text files: {counts['total_text_files']}")
    print(f"Still marked image-based: {counts['remaining_image_based']}")
    print(f"Still needing vision extraction with a raw PDF: {remaining}")
    print(f"Text already available: {counts['text_available']}")
    print(f"Vision extraction failed markers: {counts['vision_failed']}")
    print(f"Original extraction failures: {counts['source_extraction_failed']}")
    print(f"Missing raw PDFs: {counts['missing_raw_pdf']}")
    print(f"Estimated {args.batch_size}-file batches remaining: {batches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
