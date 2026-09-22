#!/usr/bin/env python3
"""Replace [IMAGE_BASED] markers with text extracted by Codex vision input."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / ".data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
DEFAULT_LOG = DATA_DIR / "logs" / "vision_extraction.jsonl"
IMAGE_BASED_MARKER = "[IMAGE_BASED]"
VISION_FAILED_MARKER = "[VISION_EXTRACTION_FAILED]"

EXTRACTION_PROMPT = """You are a faithful resume document transcription system.

The attached images are pages from one resume. Extract only the visible text.

Rules:
- Return only the extracted plain text; no explanation, JSON, Markdown, or code fences.
- Preserve reading order, headings, dates, bullets, contact details, URLs, and line breaks.
- Do not summarize, interpret, correct, complete, or invent anything.
- Keep uncertain text as visible rather than guessing.
- If a small region is unreadable, write [UNREADABLE] in that location.
"""


class VisionExtractionError(RuntimeError):
    """Raised when rendering or Codex extraction fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--extracted-dir", type=Path, default=EXTRACTED_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many randomly selected image-based PDFs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF rendering resolution passed to pdftoppm (default: 200).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files that already contain extracted text.",
    )
    return parser.parse_args()


def _matching_pdf(txt_path: Path, raw_dir: Path, extracted_dir: Path) -> Path:
    relative = txt_path.relative_to(extracted_dir).with_suffix(".pdf")
    pdf_path = raw_dir / relative
    if not pdf_path.is_file():
        raise VisionExtractionError(f"Matching PDF not found: {pdf_path}")
    return pdf_path


def _render_pdf(pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    if shutil.which("pdftoppm") is None:
        raise VisionExtractionError(
            "pdftoppm was not found. Install Poppler before running vision extraction."
        )

    prefix = output_dir / "page"
    command = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(prefix),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise VisionExtractionError(f"PDF rendering failed: {detail[:500]}")

    pages = sorted(output_dir.glob("page-*.png"))
    if not pages:
        raise VisionExtractionError("PDF rendering produced no page images.")
    return pages


def _run_codex(images: list[Path], prompt: str, model: str, reasoning_effort: str) -> str:
    with tempfile.NamedTemporaryFile(prefix="codex-output-", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        command = [
            "codex",
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--output-last-message",
            str(output_path),
        ]
        for image in images:
            command.extend(["--image", str(image)])
        # Use stdin for the prompt because --image accepts repeated values and
        # can otherwise consume a following positional prompt on some CLI builds.
        command.append("-")

        result = subprocess.run(command, input=prompt, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise VisionExtractionError(f"Codex failed: {detail[-1000:]}")

        text = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        if not text or text in {IMAGE_BASED_MARKER, VISION_FAILED_MARKER}:
            raise VisionExtractionError("Codex returned no usable resume text.")
        return text
    finally:
        output_path.unlink(missing_ok=True)


def _process_one(
    txt_path: Path,
    raw_dir: Path,
    extracted_dir: Path,
    model: str,
    reasoning_effort: str,
    dpi: int,
    overwrite: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    relative = txt_path.relative_to(extracted_dir)
    record: dict[str, object] = {
        "resume_id": txt_path.stem,
        "text_path": str(txt_path.relative_to(PROJECT_ROOT)),
        "status": "FAILED",
        "model": model,
        "reasoning_effort": reasoning_effort,
    }

    try:
        current = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if current != IMAGE_BASED_MARKER and not overwrite:
            record["status"] = "SKIPPED_ALREADY_EXTRACTED"
            return record

        pdf_path = _matching_pdf(txt_path, raw_dir, extracted_dir)
        record["raw_path"] = str(pdf_path.relative_to(PROJECT_ROOT))
        with tempfile.TemporaryDirectory(prefix=f"vision-{txt_path.stem}-") as temp_dir:
            images = _render_pdf(pdf_path, Path(temp_dir), dpi)
            record["page_count"] = len(images)
            text = _run_codex(images, EXTRACTION_PROMPT, model, reasoning_effort)

        # Replace the marker only after a successful, non-empty extraction.
        txt_path.write_text(text + "\n", encoding="utf-8")
        record["status"] = "VISION_EXTRACTED"
        record["character_count"] = len(text)
    except Exception as error:  # Keep the batch running when one file fails.
        record["status"] = "VISION_EXTRACTION_FAILED"
        record["error"] = str(error)
        if txt_path.exists() and txt_path.read_text(encoding="utf-8", errors="replace").strip() != IMAGE_BASED_MARKER:
            txt_path.write_text(VISION_FAILED_MARKER + "\n", encoding="utf-8")
    finally:
        record["duration_seconds"] = round(time.perf_counter() - started, 3)
        record["relative_path"] = str(relative)
    return record


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    if not args.raw_dir.is_dir():
        raise SystemExit(f"Raw directory does not exist: {args.raw_dir}")
    if not args.extracted_dir.is_dir():
        raise SystemExit(f"Extracted directory does not exist: {args.extracted_dir}")

    candidates = [
        path
        for path in args.extracted_dir.rglob("*.txt")
        if path.read_text(encoding="utf-8", errors="replace").strip() == IMAGE_BASED_MARKER
        or args.overwrite
    ]
    candidates = [
        path
        for path in candidates
        if (args.raw_dir / path.relative_to(args.extracted_dir)).with_suffix(".pdf").is_file()
    ]
    random.Random(args.seed).shuffle(candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Vision extraction candidates: {len(candidates)}")
    if not candidates:
        return 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [
            executor.submit(
                _process_one,
                path,
                args.raw_dir,
                args.extracted_dir,
                args.model,
                args.reasoning_effort,
                args.dpi,
                args.overwrite,
            )
            for path in candidates
        ]
        records = []
        for job in tqdm(jobs, desc=f"Codex vision ({args.workers} workers)", unit="resume"):
            record = job.result()
            records.append(record)
            with args.log_file.open("a", encoding="utf-8") as log_handle:
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"Completed: {len(records)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print(f"Log: {args.log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
