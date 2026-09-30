#!/usr/bin/env python3
"""Hybrid evaluation for structured resume extraction."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import unicodedata
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema.resume_output import ResumeOutput


DATA_DIR = PROJECT_ROOT / ".data"
DEFAULT_PROMPT_PATH = Path(__file__).with_name("evaluation_prompt.md")

# These fields need meaning comparison when normalized exact comparison differs.
SEMANTIC_FIELDS = {
    "professional_title",
    "professional_summary",
    "career_objective",
    "target_roles",
    "job_title",
    "degree",
    "field_of_study",
    "description",
    "responsibilities",
    "achievements",
    "role",
    "proficiency",
    "relationship",
    "content",
    "technical",
    "tools_and_software",
    "domain",
    "soft",
}

# Stable keys align records without relying on their array order.
RECORD_KEYS: dict[str, tuple[str, ...]] = {
    "skills.spoken_languages": ("language",),
    "work_experience": ("company", "start_date", "end_date"),
    "education": ("institution", "graduation_date"),
    "certifications": ("name", "issuer"),
    "projects": ("name",),
    "awards_and_honors": ("name", "issuer"),
    "publications": ("title", "publisher"),
    "volunteer_experience": ("organization", "start_date"),
    "professional_memberships": ("organization",),
    "references": ("name", "company"),
    "additional_sections": ("section_name",),
}

IDENTITY_PATHS = (
    "personal_information.full_name",
    "personal_information.email",
    "personal_information.phone_numbers",
)


@dataclass(slots=True)
class EvaluationItem:
    """One resume's text and already-prepared expected JSON."""

    resume_id: str
    resume_text: str
    target_json: Any
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelCallbackOutput:
    """Optional wrapper for a callback prediction and inference metadata."""

    raw_output: str | None = None
    prediction: Any = None
    prompt: str | None = None
    model_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    """One resume's validation and hybrid evaluation scores."""

    item: EvaluationItem
    status: str
    prompt: str | None = None
    raw_output: str | None = None
    target: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    model_name: str | None = None
    json_valid: bool = False
    schema_valid: bool = False
    programmatic_score: float | None = None
    semantic_score: float | None = None
    overall_score: float | None = None
    identity_score: float | None = None
    score_breakdown: list[dict[str, Any]] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_seconds: float = 0.0

    @property
    def judge_score(self) -> float | None:
        """Compatibility alias used by earlier notebook cells."""
        return self.overall_score


@dataclass(slots=True)
class EvaluationReport:
    """Aggregate results from one evaluation run."""

    metrics: dict[str, Any]
    results: list[EvaluationResult]
    output_dir: Path
    manifest: list[dict[str, Any]]


ModelCallback = Callable[[EvaluationItem], Any]
ItemFilter = Callable[[EvaluationItem], bool]
ResultCallback = Callable[[EvaluationResult], None]


def render_json_prompt(
    resume_text: str,
    *,
    schema: type[BaseModel] = ResumeOutput,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
) -> str:
    """Render the JSON-only prompt used by model callbacks."""
    template = prompt_path.read_text(encoding="utf-8")
    return template.replace(
        "{{JSON_SCHEMA}}",
        json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2),
    ).replace("{{RESUME_TEXT_JSON}}", json.dumps(resume_text, ensure_ascii=False))


def _normalise_date(value: str) -> str:
    text = value.strip().casefold()
    if text in {"present", "current", "now", "ongoing"}:
        return "current"
    formats = (
        "%m/%Y", "%m-%Y", "%m.%Y", "%m/%y", "%m-%y",
        "%Y-%m", "%Y/%m", "%B %Y", "%b %Y",
        "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%B %d, %Y", "%b %d, %Y", "%Y",
    )
    for date_format in formats:
        try:
            parsed = datetime.strptime(text, date_format)
        except ValueError:
            continue
        if "%d" in date_format:
            return parsed.strftime("%Y-%m-%d")
        if "%m" in date_format or "%B" in date_format or "%b" in date_format:
            return parsed.strftime("%Y-%m")
        return parsed.strftime("%Y")
    return re.sub(r"\s+", " ", text)


def _normalise(value: Any, path: str) -> Any:
    """Normalize harmless presentation differences before exact comparison."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value

    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        return None
    leaf = path.rsplit(".", maxsplit=1)[-1].replace("[*]", "")
    if leaf == "email":
        return text.casefold()
    if "phone" in leaf:
        digits = re.sub(r"\D", "", text)
        return digits or text.casefold()
    if leaf in {"url", "linkedin", "portfolio", "other_links"} or leaf.endswith("_link"):
        candidate = text if "://" in text else f"https://{text}"
        parts = urlsplit(candidate)
        host = parts.netloc.casefold()
        url_path = parts.path.rstrip("/")
        return urlunsplit((parts.scheme.casefold(), host, url_path, parts.query, ""))
    if leaf.endswith("date") or leaf == "date":
        return _normalise_date(text)
    return re.sub(r"\s+", " ", text).casefold()


def _stable_json(value: Any, path: str) -> str:
    """Canonical JSON string for comparing unordered list values and record keys."""
    if isinstance(value, Mapping):
        value = {
            str(key): _normalise(child, f"{path}.{key}")
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    else:
        value = _normalise(value, path)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _set_metrics(expected: list[Any], generated: list[Any], path: str) -> dict[str, float | int]:
    """Order-independent multiset precision/recall/F1, including empty lists."""
    expected_counts = Counter(_stable_json(value, path) for value in expected)
    generated_counts = Counter(_stable_json(value, path) for value in generated)
    true_positive = sum((expected_counts & generated_counts).values())
    false_positive = sum((generated_counts - expected_counts).values())
    false_negative = sum((expected_counts - generated_counts).values())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


class Evaluator:
    """Validate resume JSON, compare structured data deterministically, and
    ask Jev only about non-identical semantic text fields.

    ``judge_score`` remains as a compatibility alias for ``overall_score``.
    """

    def __init__(self, items: Iterable[EvaluationItem | Mapping[str, Any]]) -> None:
        self.items = [self._coerce_item(item) for item in items]
        self.schema = ResumeOutput

    @staticmethod
    def _coerce_item(item: EvaluationItem | Mapping[str, Any]) -> EvaluationItem:
        if isinstance(item, EvaluationItem):
            return item
        if not isinstance(item, Mapping):
            raise TypeError("Each item must be an EvaluationItem or mapping.")
        resume_text = item.get("resume_text", item.get("resume"))
        target_json = item.get("target_json", item.get("target", item.get("output")))
        if not isinstance(resume_text, str):
            raise ValueError("Each item needs string 'resume_text' (or 'resume').")
        if target_json is None:
            raise ValueError("Each item needs 'target_json' (or 'target'/'output').")
        return EvaluationItem(
            resume_id=str(item.get("resume_id", item.get("id", ""))),
            resume_text=resume_text,
            target_json=target_json,
            category=item.get("category"),
            metadata=dict(item.get("metadata", {})),
        )

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("Expected JSON text, a mapping, or a Pydantic model.")

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): Evaluator._json_safe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [Evaluator._json_safe(child) for child in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    @staticmethod
    def _is_semantic(path: str) -> bool:
        leaf = path.rsplit(".", maxsplit=1)[-1].replace("[*]", "")
        return leaf in SEMANTIC_FIELDS

    @staticmethod
    def _record_key(collection: str, record: Mapping[str, Any]) -> str:
        keys = RECORD_KEYS.get(collection, ())
        parts = [_stable_json(record.get(key), f"{collection}[*].{key}") for key in keys]
        if not keys or all(part in {"null", '""'} for part in parts):
            return _stable_json(record, f"{collection}[*]")
        return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _is_empty(value: Any) -> bool:
        return value is None or value == "" or value == []

    def _compare(
        self,
        path: str,
        expected: Any,
        generated: Any,
        programmatic: list[dict[str, Any]],
        semantic_cases: dict[str, list[tuple[Any, Any]]],
        semantic_direct: dict[str, list[float]],
    ) -> None:
        if isinstance(expected, Mapping) and isinstance(generated, Mapping):
            for key in expected.keys() | generated.keys():
                child_path = f"{path}.{key}" if path else str(key)
                self._compare(
                    child_path,
                    expected.get(key),
                    generated.get(key),
                    programmatic,
                    semantic_cases,
                    semantic_direct,
                )
            return

        if isinstance(expected, list) and isinstance(generated, list):
            if path in RECORD_KEYS or any(
                isinstance(value, Mapping) for value in expected + generated
            ):
                self._compare_record_lists(
                    path, expected, generated, programmatic, semantic_cases, semantic_direct
                )
                return
            metrics = _set_metrics(expected, generated, path)
            if self._is_semantic(path):
                programmatic.append({
                    "path": path,
                    "group": "programmatic",
                    "metric": "unordered_list_precision_recall_f1",
                    "value": metrics["f1"],
                    **metrics,
                })
                if metrics["f1"] == 1.0:
                    semantic_direct[path].append(1.0)
                else:
                    semantic_cases[path].append((expected, generated))
            else:
                programmatic.append({
                    "path": path,
                    "group": "programmatic",
                    "metric": "unordered_list_precision_recall_f1",
                    "value": metrics["f1"],
                    **metrics,
                })
            return

        if self._is_semantic(path):
            if _normalise(expected, path) == _normalise(generated, path):
                semantic_direct[path].append(1.0)
            elif self._is_empty(expected) or self._is_empty(generated):
                # Presence/absence is factual; Jev should not override it.
                semantic_direct[path].append(0.0)
            else:
                semantic_cases[path].append((expected, generated))
            return

        matches = _normalise(expected, path) == _normalise(generated, path)
        programmatic.append({
            "path": path,
            "group": "programmatic",
            "metric": "normalized_exact_match",
            "value": 1.0 if matches else 0.0,
            "match": matches,
        })

    def _compare_record_lists(
        self,
        path: str,
        expected: list[Mapping[str, Any]],
        generated: list[Mapping[str, Any]],
        programmatic: list[dict[str, Any]],
        semantic_cases: dict[str, list[tuple[Any, Any]]],
        semantic_direct: dict[str, list[float]],
    ) -> None:
        expected_keys = [self._record_key(path, record) for record in expected]
        generated_keys = [self._record_key(path, record) for record in generated]
        metrics = _set_metrics(expected_keys, generated_keys, path)
        programmatic.append({
            "path": path,
            "group": "programmatic",
            "metric": "unordered_record_precision_recall_f1",
            "value": metrics["f1"],
            **metrics,
        })

        expected_by_key: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
        generated_by_key: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
        for key, record in zip(expected_keys, expected):
            expected_by_key[key].append(record)
        for key, record in zip(generated_keys, generated):
            generated_by_key[key].append(record)

        for key in expected_by_key.keys() & generated_by_key.keys():
            while expected_by_key[key] and generated_by_key[key]:
                expected_record = expected_by_key[key].popleft()
                generated_record = generated_by_key[key].popleft()
                for field_name in expected_record.keys() | generated_record.keys():
                    self._compare(
                        f"{path}[*].{field_name}",
                        expected_record.get(field_name),
                        generated_record.get(field_name),
                        programmatic,
                        semantic_cases,
                        semantic_direct,
                    )

    @staticmethod
    def _load_jev() -> tuple[Any, Any]:
        try:
            from typesafe_sdk import Noul, TypeSafeClient
        except ImportError as error:
            raise RuntimeError(
                "Install requirements.txt to use Jev for semantic comparisons."
            ) from error
        if not os.getenv("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is missing from the environment or project .env.")
        return TypeSafeClient, Noul

    def _judge_semantics(
        self,
        item: EvaluationItem,
        cases: dict[str, list[tuple[Any, Any]]],
    ) -> dict[str, float]:
        """Ask Jev only about non-identical, present semantic values."""
        if not cases:
            return {}
        TypeSafeClient, Noul = self._load_jev()
        questions: dict[str, Any] = {}
        state_cases: dict[str, Any] = {}
        for index, (path, pairs) in enumerate(sorted(cases.items())):
            name = f"semantic_field_{index}"
            questions[name] = Noul(
                instructions=(
                    f"Compare each expected and generated value for the resume field {path}. "
                    "Judge whether the generated value preserves the same factual meaning. "
                    "Accept faithful paraphrases and clear synonyms. Penalize omissions, "
                    "contradictions, and unsupported claims. If values are arrays or represent "
                    "records, ignore ordering and compare corresponding items. The resume is evidence."
                )
            )
            state_cases[name] = [
                {"expected": expected, "generated": generated}
                for expected, generated in pairs
            ]
        state = {
            "resume_text": item.resume_text,
            "semantic_field_comparisons": json.dumps(state_cases, ensure_ascii=False),
        }
        with TypeSafeClient() as client:
            response = client.system_one(state=state, questions=questions)
        scores = {}
        for name, path in zip(questions, sorted(cases)):
            scores[path] = float(response.nouls[name].noul)
        return scores

    def _score_prediction(
        self,
        item: EvaluationItem,
        target: dict[str, Any],
        prediction: dict[str, Any],
    ) -> tuple[float, float | None, float, float, list[dict[str, Any]], str | None]:
        programmatic: list[dict[str, Any]] = []
        semantic_cases: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        semantic_direct: dict[str, list[float]] = defaultdict(list)
        self._compare("", target, prediction, programmatic, semantic_cases, semantic_direct)

        semantic_error = None
        try:
            semantic_jev = self._judge_semantics(item, semantic_cases)
        except Exception as error:
            semantic_jev = {}
            semantic_error = f"{type(error).__name__}: {error}"
        semantic_breakdown: list[dict[str, Any]] = []
        semantic_paths = semantic_direct.keys() | semantic_cases.keys()
        for path in sorted(semantic_paths):
            direct_scores = semantic_direct.get(path, [])
            jev_score = semantic_jev.get(path)
            case_count = len(semantic_cases.get(path, []))
            scores = direct_scores + ([jev_score] * case_count if jev_score is not None else [])
            value = (
                sum(scores) / len(scores)
                if scores and (not case_count or jev_score is not None)
                else None
            )
            semantic_breakdown.append({
                "path": path,
                "group": "semantic",
                "metric": "exact_or_jev_semantic_match",
                "value": value,
                "exact_cases": len(direct_scores),
                "jev_cases": case_count,
                "jev_score": jev_score,
                "error": semantic_error if case_count and jev_score is None else None,
            })

        programmatic_score = (
            sum(float(entry["value"]) for entry in programmatic) / len(programmatic)
            if programmatic
            else 1.0
        )
        semantic_values = [
            float(entry["value"])
            for entry in semantic_breakdown
            if entry["value"] is not None
        ]
        semantic_score = (
            sum(semantic_values) / len(semantic_values)
            if semantic_values and semantic_error is None
            else None
        )
        overall_score = (
            (programmatic_score + semantic_score) / 2
            if semantic_score is not None
            else None if semantic_error else programmatic_score
        )

        fields_by_path = {entry["path"]: float(entry["value"]) for entry in programmatic}
        fields_by_path.update({
            entry["path"]: float(entry["value"])
            for entry in semantic_breakdown
            if entry["value"] is not None
        })
        identity_values = [fields_by_path[path] for path in IDENTITY_PATHS if path in fields_by_path]
        identity_score = sum(identity_values) / len(identity_values) if identity_values else 1.0
        breakdown = programmatic + semantic_breakdown
        return programmatic_score, semantic_score, overall_score, identity_score, breakdown, semantic_error

    def _evaluate_one(self, item: EvaluationItem, callback: ModelCallback) -> EvaluationResult:
        started = time.perf_counter()
        result = EvaluationResult(item=item, status="PENDING")
        try:
            try:
                target_value = self._json_value(item.target_json)
                target = self.schema.model_validate(target_value).model_dump(mode="json")
                result.target = target
            except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
                result.status = "INVALID_TARGET"
                result.error = str(error)
                return result

            try:
                callback_output = callback(item)
                if isinstance(callback_output, ModelCallbackOutput):
                    result.prompt = callback_output.prompt
                    result.raw_output = callback_output.raw_output
                    result.model_name = callback_output.model_name
                    result.model_metadata = callback_output.metadata
                    prediction_value = (
                        callback_output.prediction
                        if callback_output.prediction is not None
                        else callback_output.raw_output
                    )
                else:
                    prediction_value = callback_output
                result.raw_output = (
                    prediction_value
                    if isinstance(prediction_value, str)
                    else json.dumps(self._json_safe(prediction_value), ensure_ascii=False)
                )
                parsed = self._json_value(prediction_value)
                result.json_valid = True
            except json.JSONDecodeError as error:
                result.status = "INVALID_JSON"
                result.error = str(error)
                return result
            except Exception as error:
                result.status = "CALLBACK_FAILED"
                result.error = f"{type(error).__name__}: {error}"
                return result

            try:
                prediction = self.schema.model_validate(parsed).model_dump(mode="json")
                result.prediction = prediction
                result.schema_valid = True
            except ValidationError as error:
                result.status = "INVALID_PREDICTION_SCHEMA"
                result.error = error.json(include_url=False)
                return result

            try:
                (
                    result.programmatic_score,
                    result.semantic_score,
                    result.overall_score,
                    result.identity_score,
                    result.score_breakdown,
                    semantic_error,
                ) = self._score_prediction(item, target, prediction)
                result.error = semantic_error
                result.status = "SCORED_WITH_SEMANTIC_ERROR" if semantic_error else "SCORED"
            except Exception as error:
                result.status = "SEMANTIC_JUDGE_FAILED"
                result.error = f"{type(error).__name__}: {error}"
            return result
        finally:
            result.duration_seconds = round(time.perf_counter() - started, 4)

    @staticmethod
    def _mean(results: list[EvaluationResult], field_name: str) -> float | None:
        values = [getattr(result, field_name) for result in results]
        values = [float(value) for value in values if value is not None]
        return round(sum(values) / len(values), 6) if values else None

    def _summarize(self, results: list[EvaluationResult]) -> dict[str, Any]:
        valid_targets = [result for result in results if result.target is not None]
        scored = [result for result in valid_targets if result.overall_score is not None]
        denominator = len(valid_targets)
        return {
            "items_evaluated": len(results),
            "valid_targets": denominator,
            "json_valid_count": sum(result.json_valid for result in results),
            "schema_valid_count": sum(result.schema_valid for result in results),
            "scored_count": len(scored),
            "mean_programmatic_score": self._mean(valid_targets, "programmatic_score"),
            "mean_semantic_score": self._mean(valid_targets, "semantic_score"),
            "mean_overall_score": self._mean(valid_targets, "overall_score"),
            "mean_identity_score": self._mean(valid_targets, "identity_score"),
            "overall_score_including_invalid_predictions": (
                round(sum(result.overall_score or 0.0 for result in valid_targets) / denominator, 6)
                if denominator
                else None
            ),
            "status_counts": dict(Counter(result.status for result in results)),
            "score_range": [0.0, 1.0],
        }

    def evaluate(
        self,
        callback: ModelCallback,
        *,
        workers: int = 2,
        test_size: int | float | None = None,
        seed: int = 42,
        item_filter: ItemFilter | None = None,
        on_result: ResultCallback | None = None,
        output_dir: Path | str | None = None,
        show_progress: bool = True,
    ) -> EvaluationReport:
        """Run inference, deterministic scoring, and semantic fallback where needed."""
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if test_size is not None:
            if isinstance(test_size, bool) or test_size <= 0:
                raise ValueError("test_size must be a positive count or fraction")
            if isinstance(test_size, float) and test_size > 1:
                raise ValueError("fractional test_size must be <= 1")

        selected = [item for item in self.items if item_filter is None or item_filter(item)]
        selected.sort(key=lambda item: item.resume_id)
        if test_size is not None:
            count = max(1, round(len(selected) * test_size)) if isinstance(test_size, float) else test_size
            if count > len(selected):
                raise ValueError(f"test_size ({count}) exceeds item count ({len(selected)}).")
            selected = random.Random(seed).sample(selected, count)
            selected.sort(key=lambda item: item.resume_id)

        run_dir = Path(output_dir) if output_dir else (
            DATA_DIR / "evaluation" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = [
            {"resume_id": item.resume_id, "category": item.category, "metadata": self._json_safe(item.metadata)}
            for item in selected
        ]
        (run_dir / "test_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        results: list[EvaluationResult] = []
        result_path = run_dir / "results.jsonl"
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._evaluate_one, item, callback) for item in selected]
            with result_path.open("w", encoding="utf-8") as handle:
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Evaluating resumes ({workers} workers)",
                    unit="resume",
                    disable=not show_progress,
                ):
                    result = future.result()
                    if on_result is not None:
                        try:
                            on_result(result)
                        except Exception as error:
                            result.error = f"on_result failed: {type(error).__name__}: {error}"
                    results.append(result)
                    handle.write(json.dumps(self._result_dict(result), ensure_ascii=False) + "\n")
                    handle.flush()

        metrics = self._summarize(results)
        metrics.update({
            "requested_test_size": test_size,
            "seed": seed,
            "workers": workers,
            "selected_resume_ids": [item.resume_id for item in selected],
        })
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return EvaluationReport(metrics=metrics, results=results, output_dir=run_dir, manifest=manifest)

    def plot_results(
        self,
        report: EvaluationReport,
        *,
        max_resumes: int = 30,
        max_fields: int = 15,
        save_path: Path | str | None = None,
        show: bool = True,
    ) -> Any:
        """Plot run-level metrics, score distribution, resume scores, and field scores.

        Returns the Matplotlib figure so callers can customize or save it. Install
        Matplotlib with ``pip install -r requirements.txt`` to use this method.
        """
        if not isinstance(report, EvaluationReport):
            raise TypeError("report must be the EvaluationReport returned by evaluate().")
        if max_resumes < 1 or max_fields < 1:
            raise ValueError("max_resumes and max_fields must be positive.")

        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "Matplotlib is required for plotting. Run: pip install -r requirements.txt"
            ) from error

        scored = [result for result in report.results if result.overall_score is not None]
        figure, axes = plt.subplots(2, 2, figsize=(15, 10))
        figure.suptitle("Resume parser evaluation", fontsize=16, fontweight="bold")

        # Aggregate score and output-validity metrics.
        metric_names = {
            "mean_overall_score": "Overall",
            "mean_programmatic_score": "Programmatic",
            "mean_semantic_score": "Semantic",
            "mean_identity_score": "Identity",
        }
        metric_values = [
            (label, report.metrics.get(key))
            for key, label in metric_names.items()
            if report.metrics.get(key) is not None
        ]
        metric_values.extend((
            ("JSON valid", report.metrics.get("json_valid_count", 0) / max(len(report.results), 1)),
            ("Schema valid", report.metrics.get("schema_valid_count", 0) / max(len(report.results), 1)),
        ))
        metric_axis = axes[0, 0]
        if metric_values:
            labels, values = zip(*metric_values)
            bars = metric_axis.bar(labels, [value * 100 for value in values], color="#3977b8")
            metric_axis.bar_label(bars, fmt="%.1f%%", padding=3)
            metric_axis.set_ylim(0, 110)
            metric_axis.tick_params(axis="x", rotation=20)
            metric_axis.set_ylabel("Score / valid outputs (%)")
        else:
            metric_axis.text(0.5, 0.5, "No aggregate metrics", ha="center", va="center")
        metric_axis.set_title("Run summary")
        metric_axis.grid(axis="y", alpha=0.2)

        # Distribution shows whether a high mean hides weak individual resumes.
        distribution_axis = axes[0, 1]
        if scored:
            distribution_axis.hist(
                [result.overall_score * 100 for result in scored],
                bins=10,
                range=(0, 100),
                color="#4b9b79",
                edgecolor="white",
            )
            distribution_axis.set_xlim(0, 100)
            distribution_axis.set_xlabel("Overall score (%)")
            distribution_axis.set_ylabel("Number of resumes")
        else:
            distribution_axis.text(0.5, 0.5, "No scored resumes", ha="center", va="center")
        distribution_axis.set_title("Overall-score distribution")
        distribution_axis.grid(axis="y", alpha=0.2)

        # Show the lowest-scoring resumes first to make inspection actionable.
        resume_axis = axes[1, 0]
        lowest_scores = sorted(scored, key=lambda result: result.overall_score)[:max_resumes]
        if lowest_scores:
            labels = [result.item.resume_id or "(no id)" for result in lowest_scores]
            values = [result.overall_score * 100 for result in lowest_scores]
            resume_axis.barh(labels, values, color="#d48b45")
            resume_axis.set_xlim(0, 100)
            resume_axis.set_xlabel("Overall score (%)")
            resume_axis.invert_yaxis()
        else:
            resume_axis.text(0.5, 0.5, "No scored resumes", ha="center", va="center")
        resume_axis.set_title(f"Lowest-scoring resumes (up to {max_resumes})")
        resume_axis.grid(axis="x", alpha=0.2)

        # Mean field-level values reveal which parts of the schema need work.
        field_scores: dict[str, list[float]] = defaultdict(list)
        for result in scored:
            for entry in result.score_breakdown:
                value = entry.get("value")
                if value is not None:
                    label = f"{entry.get('group', 'field')}: {entry['path']}"
                    field_scores[label].append(float(value))
        field_means = sorted(
            ((path, sum(values) / len(values)) for path, values in field_scores.items()),
            key=lambda item: item[1],
        )[:max_fields]
        field_axis = axes[1, 1]
        if field_means:
            labels, values = zip(*field_means)
            field_axis.barh(labels, [value * 100 for value in values], color="#8a70b2")
            field_axis.set_xlim(0, 100)
            field_axis.set_xlabel("Mean field score (%)")
            field_axis.invert_yaxis()
        else:
            field_axis.text(0.5, 0.5, "No field scores available", ha="center", va="center")
        field_axis.set_title(f"Lowest-scoring fields (up to {max_fields})")
        field_axis.grid(axis="x", alpha=0.2)

        figure.tight_layout(rect=(0, 0, 1, 0.96))
        if save_path is not None:
            figure.savefig(save_path, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        return figure

    def _result_dict(self, result: EvaluationResult) -> dict[str, Any]:
        return {
            "resume_id": result.item.resume_id,
            "category": result.item.category,
            "metadata": self._json_safe(result.item.metadata),
            "resume_text": result.item.resume_text,
            "target_json": result.target if result.target is not None else self._json_safe(result.item.target_json),
            "prompt": result.prompt,
            "raw_output": result.raw_output,
            "prediction_json": result.prediction,
            "status": result.status,
            "json_valid": result.json_valid,
            "schema_valid": result.schema_valid,
            "programmatic_score": result.programmatic_score,
            "semantic_score": result.semantic_score,
            "overall_score": result.overall_score,
            "identity_score": result.identity_score,
            "score_breakdown": result.score_breakdown,
            "model_name": result.model_name,
            "model_metadata": self._json_safe(result.model_metadata),
            "error": result.error,
            "duration_seconds": result.duration_seconds,
        }


__all__ = [
    "EvaluationItem",
    "EvaluationReport",
    "EvaluationResult",
    "Evaluator",
    "ModelCallbackOutput",
    "render_json_prompt",
]
