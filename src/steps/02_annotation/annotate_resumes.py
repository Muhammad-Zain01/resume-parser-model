#!/usr/bin/env python3
"""Create Pydantic-validated JSON annotations from extracted resume text."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

from pydantic import ValidationError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema.resume_output import ResumeOutput


DATA_DIR = PROJECT_ROOT / ".data"
DEFAULT_INPUT_DIR = DATA_DIR / "extracted"
DEFAULT_OUTPUT_DIR = DATA_DIR / "annotated"
DEFAULT_LOG_FILE = DATA_DIR / "logs" / "resume_annotation.jsonl"
PROMPT_PATH = Path(__file__).with_name("annotation_prompt.md")
NON_TEXT_MARKERS = {
    "[IMAGE_BASED]",
    "[NO_TEXT]",
    "[EXTRACTION_FAILED]",
    "[VISION_EXTRACTION_FAILED]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--limit",
        type=int,
        help="Annotate at most this many randomly selected resumes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing JSON annotations.",
    )
    return parser.parse_args()


def _output_path(text_path: Path, input_dir: Path, output_dir: Path) -> Path:
    return output_dir / text_path.relative_to(input_dir).with_suffix(".json")


def _make_prompt(resume_text: str, schema: dict[str, object]) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace(
        "{{JSON_SCHEMA}}", json.dumps(schema, ensure_ascii=False, indent=2)
    ).replace(
        "{{RESUME_TEXT_JSON}}", json.dumps(resume_text, ensure_ascii=False)
    )


def _make_strict_schema(schema: dict[str, object]) -> dict[str, object]:
    """Make Pydantic's schema compatible with Codex strict structured output."""
    def normalize(value: object) -> object:
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if not isinstance(value, dict):
            return value

        normalized = {key: normalize(item) for key, item in value.items() if key != "default"}
        properties = normalized.get("properties")
        if normalized.get("type") == "object" and isinstance(properties, dict):
            normalized["required"] = list(properties)
            normalized["additionalProperties"] = False
        return normalized

    normalized_schema = normalize(schema)
    assert isinstance(normalized_schema, dict)
    return normalized_schema


def _run_codex(
    prompt: str,
    schema: dict[str, object],
    model: str,
    reasoning_effort: str,
    timeout: int,
) -> str:
    with tempfile.TemporaryDirectory(prefix="resume-annotation-") as temp_dir:
        temp_path = Path(temp_dir)
        schema_path = temp_path / "resume-output.schema.json"
        response_path = temp_path / "response.json"
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False), encoding="utf-8"
        )
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
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            "-",
        ]
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Codex failed: {detail[-1200:]}")
        if not response_path.is_file():
            raise RuntimeError("Codex did not write a final response.")
        return response_path.read_text(encoding="utf-8").strip()


def _process_one(
    text_path: Path,
    input_dir: Path,
    output_dir: Path,
    schema: dict[str, object],
    model: str,
    reasoning_effort: str,
    timeout: int,
    overwrite: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    output_path = _output_path(text_path, input_dir, output_dir)
    record: dict[str, object] = {
        "resume_id": text_path.stem,
        "input_path": text_path.relative_to(input_dir).as_posix(),
        "output_path": output_path.relative_to(output_dir).as_posix(),
        "model": model,
        "reasoning_effort": reasoning_effort,
    }

    try:
        if output_path.exists() and not overwrite:
            record["status"] = "SKIPPED_EXISTS"
            return record

        resume_text = text_path.read_text(encoding="utf-8", errors="replace").strip()
        if not resume_text or resume_text in NON_TEXT_MARKERS:
            record["status"] = "SKIPPED_NO_EXTRACTED_TEXT"
            return record

        raw_response = _run_codex(
            _make_prompt(resume_text, schema),
            schema,
            model,
            reasoning_effort,
            timeout,
        )
        # Parse and validate before creating or replacing an annotation file.
        parsed = json.loads(raw_response)
        validated = ResumeOutput.model_validate(parsed)
        serialized = json.dumps(
            validated.model_dump(mode="json"), ensure_ascii=False, indent=2
        ) + "\n"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output_path.with_suffix(".json.tmp")
        temp_output.write_text(serialized, encoding="utf-8")
        temp_output.replace(output_path)
        record["status"] = "ANNOTATED"
        record["character_count"] = len(resume_text)
    except json.JSONDecodeError as error:
        record["status"] = "INVALID_JSON"
        record["error"] = str(error)
    except ValidationError as error:
        record["status"] = "SCHEMA_VALIDATION_FAILED"
        record["error"] = error.errors(include_url=False)
    except Exception as error:  # Keep other resumes moving after a single failure.
        record["status"] = "FAILED"
        record["error"] = str(error)
    finally:
        record["duration_seconds"] = round(time.perf_counter() - started, 3)
    return record


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if shutil.which("codex") is None:
        raise SystemExit("Codex CLI was not found on PATH.")

    schema = _make_strict_schema(ResumeOutput.model_json_schema())
    candidates = [
        path
        for path in args.input_dir.rglob("*.txt")
        if args.overwrite
        or not _output_path(path, args.input_dir, args.output_dir).exists()
    ]
    random.Random(args.seed).shuffle(candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Annotation candidates: {len(candidates)}")
    if not candidates:
        return 0

    totals: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [
            executor.submit(
                _process_one,
                path,
                args.input_dir,
                args.output_dir,
                schema,
                args.model,
                args.reasoning_effort,
                args.timeout,
                args.overwrite,
            )
            for path in candidates
        ]
        with args.log_file.open("a", encoding="utf-8") as log_handle:
            for job in tqdm(
                as_completed(jobs),
                total=len(jobs),
                desc=f"Annotating ({args.workers} workers)",
                unit="resume",
            ):
                record = job.result()
                status = str(record["status"])
                totals[status] = totals.get(status, 0) + 1
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_handle.flush()

    print(f"Processed: {len(candidates)}")
    for status, count in sorted(totals.items()):
        print(f"{status}: {count}")
    print(f"Annotations: {args.output_dir}")
    print(f"Log: {args.log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
