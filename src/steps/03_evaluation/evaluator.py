#!/usr/bin/env python3
"""Callback-driven resume JSON evaluation using Jev as the semantic judge."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

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


def _build_schema_questions() -> tuple[dict[str, Any], ...]:
    """Create one Jev check per ResumeOutput leaf, including nested records."""
    scalar_fields = [
        "personal_information.full_name",
        "personal_information.professional_title",
        "personal_information.email",
        "personal_information.address",
        "personal_information.city",
        "personal_information.state_or_region",
        "personal_information.postal_code",
        "personal_information.country",
        "personal_information.linkedin",
        "personal_information.portfolio",
        "professional_summary",
        "career_objective",
        "work_experience[*].job_title",
        "work_experience[*].company",
        "work_experience[*].location",
        "work_experience[*].employment_type",
        "work_experience[*].start_date",
        "work_experience[*].end_date",
        "work_experience[*].is_current",
        "work_experience[*].description",
        "education[*].institution",
        "education[*].degree",
        "education[*].field_of_study",
        "education[*].location",
        "education[*].start_date",
        "education[*].end_date",
        "education[*].graduation_date",
        "education[*].gpa",
        "certifications[*].name",
        "certifications[*].type",
        "certifications[*].issuer",
        "certifications[*].issue_date",
        "certifications[*].expiry_date",
        "certifications[*].credential_id",
        "certifications[*].description",
        "projects[*].name",
        "projects[*].role",
        "projects[*].description",
        "projects[*].start_date",
        "projects[*].end_date",
        "projects[*].url",
        "awards_and_honors[*].name",
        "awards_and_honors[*].issuer",
        "awards_and_honors[*].date",
        "awards_and_honors[*].description",
        "publications[*].title",
        "publications[*].publisher",
        "publications[*].date",
        "publications[*].url",
        "volunteer_experience[*].organization",
        "volunteer_experience[*].role",
        "volunteer_experience[*].start_date",
        "volunteer_experience[*].end_date",
        "volunteer_experience[*].description",
        "professional_memberships[*].organization",
        "professional_memberships[*].role",
        "professional_memberships[*].date",
        "references[*].name",
        "references[*].relationship",
        "references[*].company",
        "references[*].contact",
        "additional_sections[*].section_name",
        "additional_sections[*].content",
    ]
    unordered_lists = [
        "personal_information.phone_numbers",
        "personal_information.other_links",
        "target_roles",
        "skills.technical",
        "skills.tools_and_software",
        "skills.domain",
        "skills.soft",
        "work_experience[*].responsibilities",
        "work_experience[*].achievements",
        "work_experience[*].skills_used",
        "education[*].honors",
        "projects[*].technologies",
    ]
    record_keys = {
        "skills.spoken_languages": "language",
        "work_experience": "job_title, company, and dates",
        "education": "institution, degree, and dates",
        "certifications": "name and issuer",
        "projects": "name and role",
        "awards_and_honors": "name and issuer",
        "publications": "title and publisher",
        "volunteer_experience": "organization and role",
        "professional_memberships": "organization and role",
        "references": "name and company",
        "additional_sections": "section_name",
    }
    scalar_fields.extend((
        "skills.spoken_languages[*].language",
        "skills.spoken_languages[*].proficiency",
    ))

    def make_question(path: str, *, is_list: bool = False) -> dict[str, Any]:
        collection = path.split("[*]", maxsplit=1)[0] if "[*]" in path else None
        is_semantic = any(
            term in path
            for term in (
                "summary", "objective", "description", "responsibilities",
                "achievements", "additional_sections[*].content",
            )
        ) or path.startswith(("target_roles", "skills."))
        if collection:
            anchors = record_keys.get(collection, "the record's identifying fields")
            comparison = (
                f"Treat {collection} as an unordered array. Match records by {anchors}; "
                f"compare only {path} within every matched record. Ignore record order, "
                "but count missing or extra records and missing, incorrect, or unsupported "
                "values for this field as mismatches."
            )
            if is_list:
                comparison += " If this field is itself an array, ignore item order and check both coverage and unsupported extras."
        elif is_list and path in record_keys:
            comparison = (
                f"Treat {path} as an unordered array of records. Match records by "
                f"{record_keys[path]}, compare every field in matched records, and "
                "penalize missing or extra records. Ignore record order."
            )
        elif is_list:
            comparison = (
                f"Compare {path} as an unordered list: ignore ordering, require all expected "
                "items to be present, and penalize missing or unsupported extra items."
            )
        else:
            comparison = (
                f"Compare {path} in actual_output with the same field in expected_output "
                "and verify it against the resume text. Treat both absent values as a match; "
                "penalize missing expected values, contradictions, and unsupported additions."
            )
        if is_semantic:
            comparison += " Accept meaning-preserving paraphrases and clear synonyms; do not accept changed or omitted material facts."
        else:
            comparison += " Allow harmless capitalization, whitespace, or formatting differences only; the underlying fact must match."

        question_name = path.replace("[*]", "_item").replace(".", "_") + "_match"
        return {
            "type": "noul",
            "name": question_name,
            "path": path,
            "question": comparison,
            "weight": 1.0,
            "zero_threshold": 0.10 if is_semantic else 0.20,
            "full_threshold": (
                0.70
                if is_list or "[*]" in path or path.endswith(("_date", ".date"))
                else 0.80
            ),
        }

    questions = [make_question(path) for path in scalar_fields]
    questions.extend(make_question(path, is_list=True) for path in unordered_lists)
    questions.extend((
        make_question("skills.spoken_languages", is_list=True),
        make_question("work_experience", is_list=True),
        make_question("education", is_list=True),
        make_question("certifications", is_list=True),
        make_question("projects", is_list=True),
        make_question("awards_and_honors", is_list=True),
        make_question("publications", is_list=True),
        make_question("volunteer_experience", is_list=True),
        make_question("professional_memberships", is_list=True),
        make_question("references", is_list=True),
        make_question("additional_sections", is_list=True),
    ))
    return tuple(questions)


@dataclass(slots=True)
class EvaluationItem:
    """One already-prepared case: raw resume text and expected JSON."""

    resume_id: str
    resume_text: str
    target_json: Any
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelCallbackOutput:
    """Optional wrapper for a callback's prediction and inference details."""

    raw_output: str | None = None
    prediction: Any = None
    prompt: str | None = None
    model_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    """Inference result and Jev's score for one resume."""

    item: EvaluationItem
    status: str
    prompt: str | None = None
    raw_output: str | None = None
    target: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    model_name: str | None = None
    json_valid: bool = False
    schema_valid: bool = False
    judge_score: float | None = None
    judge_reason: str | None = None
    score_breakdown: list[dict[str, Any]] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass(slots=True)
class EvaluationReport:
    """Aggregate results from an evaluation run."""

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
    """Render the default JSON-only prompt for a model callback."""
    template = prompt_path.read_text(encoding="utf-8")
    return template.replace(
        "{{JSON_SCHEMA}}",
        json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2),
    ).replace("{{RESUME_TEXT_JSON}}", json.dumps(resume_text, ensure_ascii=False))


class Evaluator:
    """Run model inference, validate JSON, and score it with TypeSafe Jev.

    Prepare ``EvaluationItem`` objects outside this class. The callback receives
    one item and returns the model's JSON; Jev then compares the prediction with
    the target while using the original resume text as evidence.

    Custom questions use simple dictionaries with ``type`` (``noul``, ``score``
    or ``choice``), ``name``, ``question``, and ``weight``. ``levels`` is used
    by score questions; ``options`` maps choice labels to credits from 0 to 1,
    or ``None`` when that choice means the question does not apply.
    """

    DEFAULT_QUESTIONS: tuple[dict[str, Any], ...] = (
        {
            "type": "noul",
            "name": "name_match",
            "question": (
                "The full name in actual_output.personal_information matches "
                "expected_output.personal_information, ignoring harmless "
                "capitalization and spacing differences. If expected_output "
                "has no name and the resume gives no name, treat that as correct."
            ),
            "weight": 1.5,
            "zero_threshold": 0.20,
            "full_threshold": 0.90,
        },
        {
            "type": "noul",
            "name": "professional_title_match",
            "question": (
                "The professional title in actual_output.personal_information "
                "matches the expected title and is supported by the resume. "
                "Accept harmless wording or capitalization differences, but "
                "do not treat a different role as a match."
            ),
            "weight": 1.0,
            "zero_threshold": 0.20,
            "full_threshold": 0.90,
        },
        {
            "type": "noul",
            "name": "email_match",
            "question": (
                "The email in actual_output.personal_information matches "
                "expected_output and the resume. Do not accept a different or invented address."
            ),
            "weight": 1.25,
            "zero_threshold": 0.20,
            "full_threshold": 0.90,
        },
        {
            "type": "noul",
            "name": "phone_numbers_match",
            "question": (
                "Phone numbers in actual_output.personal_information match "
                "the expected numbers regardless of list order or harmless "
                "formatting such as spaces, parentheses, or hyphens."
            ),
            "weight": 1.0,
            "zero_threshold": 0.20,
            "full_threshold": 0.90,
        },
        {
            "type": "noul",
            "name": "location_and_links_match",
            "question": (
                "Address, city, region, postal code, country, LinkedIn, "
                "portfolio, and other links in actual_output.personal_information "
                "match expected_output and are supported by the resume."
            ),
            "weight": 1.0,
            "zero_threshold": 0.20,
            "full_threshold": 0.90,
        },
        {
            "type": "noul",
            "name": "target_roles_match",
            "question": (
                "The target roles in actual_output.target_roles cover the "
                "expected roles regardless of order, without changing their meaning."
            ),
            "weight": 1.0,
            "zero_threshold": 0.10,
            "full_threshold": 0.80,
        },
        {
            "type": "noul",
            "name": "skill_recall",
            "question": (
                "Every skill listed in expected_output.skills is represented "
                "in actual_output.skills, allowing clear synonyms and ignoring "
                "list order. Judge coverage, not wording."
            ),
            "weight": 2.0,
            "zero_threshold": 0.10,
            "full_threshold": 0.80,
        },
        {
            "type": "noul",
            "name": "skill_grounding",
            "question": (
                "Skills listed in actual_output.skills are supported by the "
                "resume text. Do not penalize a clearly supported skill only "
                "because it is absent from expected_output.skills."
            ),
            "weight": 1.5,
            "zero_threshold": 0.10,
            "full_threshold": 0.90,
        },
        {
            "type": "score",
            "name": "work_experience_core_facts",
            "question": (
                "Across corresponding work experience records, are job titles, "
                "employers, locations, employment types, and start/end dates "
                "correct against expected_output and the resume? Ignore record order "
                "and harmless date formatting differences."
            ),
            "levels": [
                "Mostly wrong or unsupported",
                "Several key role facts are wrong or missing",
                "Mostly correct with minor omissions or formatting differences",
                "All expected core role facts are correctly represented and grounded",
            ],
            "weight": 2.5,
            "zero_threshold": 0.20,
            "full_threshold": 0.90,
        },
        {
            "type": "score",
            "name": "work_experience_quality",
            "question": (
                "How accurately does actual_output.work_experience capture "
                "expected responsibilities, achievements, and skills_used? "
                "Ignore ordering differences and harmless paraphrases; penalize "
                "material omissions, contradictions, and unsupported claims."
            ),
            "levels": [
                "Mostly wrong, missing, or unsupported",
                "Major facts are wrong or missing",
                "Mostly accurate, with some omissions or minor errors",
                "All material expected facts are accurately represented and grounded",
            ],
            "weight": 3.0,
            "zero_threshold": 0.10,
            "full_threshold": 0.80,
        },
        {
            "type": "choice",
            "name": "long_text_meaning",
            "question": (
                "Compare professional_summary, career_objective, work/project/" 
                "certification/volunteer descriptions, and additional-section content "
                "in actual_output with expected_output and the resume. Choose the "
                "best description of factual meaning; paraphrases are acceptable, "
                "but omissions and unsupported claims matter."
            ),
            "options": {
                "exact_or_equivalent_meaning": 1.0,
                "same_meaning_with_minor_omission": 0.75,
                "partly_correct_with_material_omissions": 0.4,
                "contradictory_or_unsupported": 0.0,
                "no_comparable_long_text_present": None,
            },
            "weight": 2.5,
            "zero_threshold": 0.10,
            "full_threshold": 0.80,
        },
        {
            "type": "score",
            "name": "education_quality",
            "question": (
                "How accurately does actual_output.education match the expected "
                "education facts and the resume, ignoring ordering and harmless "
                "format differences in dates or degree wording?"
            ),
            "levels": [
                "Mostly wrong or unsupported",
                "Major education facts are missing or incorrect",
                "Mostly correct with minor omissions or differences",
                "Expected education facts are correctly represented and grounded",
            ],
            "weight": 1.5,
            "zero_threshold": 0.20,
            "full_threshold": 0.90,
        },
        {
            "type": "score",
            "name": "certifications_quality",
            "question": (
                "How accurately do certification names, types, issuers, dates, "
                "credential IDs, and descriptions in actual_output match the "
                "expected records and the resume? Ignore order, not factual differences."
            ),
            "levels": [
                "Mostly wrong or unsupported",
                "Major certification facts are missing or incorrect",
                "Mostly correct with minor omissions or differences",
                "All expected certification facts are accurately represented and grounded",
            ],
            "weight": 1.25,
            "zero_threshold": 0.20,
            "full_threshold": 0.90,
        },
        {
            "type": "score",
            "name": "projects_quality",
            "question": (
                "How accurately do project names, roles, technologies, dates, URLs, "
                "and descriptions in actual_output match expected_output and the resume? "
                "Ignore project ordering and accept meaning-preserving paraphrases."
            ),
            "levels": [
                "Mostly wrong or unsupported",
                "Major project facts are missing or incorrect",
                "Mostly correct with minor omissions or differences",
                "All expected project facts are accurately represented and grounded",
            ],
            "weight": 1.25,
            "zero_threshold": 0.10,
            "full_threshold": 0.80,
        },
        {
            "type": "score",
            "name": "other_sections_quality",
            "question": (
                "How accurately do certifications, projects, awards, publications, "
                "volunteer experience, memberships, references, and additional "
                "sections in actual_output match expected_output and the resume? "
                "Ignore ordering; penalize missing expected facts and unsupported additions."
            ),
            "levels": [
                "Mostly wrong or unsupported",
                "Several major omissions or errors",
                "Mostly accurate with some minor omissions or errors",
                "All applicable expected facts are accurate and grounded",
            ],
            "weight": 2.0,
            "zero_threshold": 0.10,
            "full_threshold": 0.80,
        },
        {
            "type": "noul",
            "name": "no_unsupported_facts",
            "question": (
                "Every factual claim in actual_output is supported by the resume; "
                "the output does not invent names, dates, qualifications, employers, "
                "skills, or achievements."
            ),
            "weight": 2.5,
            "zero_threshold": 0.10,
            "full_threshold": 0.90,
        },
    )

    # Replace the former broad section-level checks with one check per schema leaf.
    DEFAULT_QUESTIONS = _build_schema_questions()

    def __init__(
        self,
        items: Iterable[EvaluationItem | Mapping[str, Any]],
        *,
        questions: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.items = [self._coerce_item(item) for item in items]
        self.schema = ResumeOutput
        self.questions = [dict(question) for question in (questions or self.DEFAULT_QUESTIONS)]
        self._validate_questions()

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

    def _validate_questions(self) -> None:
        if not self.questions:
            raise ValueError("Provide at least one Jev question.")
        for question in self.questions:
            kind = question.get("type")
            if kind not in {"noul", "score", "choice"}:
                raise ValueError(f"Unsupported Jev question type: {kind!r}.")
            if not question.get("name") or not question.get("question"):
                raise ValueError("Each Jev question needs a name and question.")
            if float(question.get("weight", 1)) <= 0:
                raise ValueError("Jev question weights must be positive.")
            zero_threshold = float(question.get("zero_threshold", 0.0))
            full_threshold = float(question.get("full_threshold", 1.0))
            if (
                not math.isfinite(zero_threshold)
                or not math.isfinite(full_threshold)
                or not 0.0 <= zero_threshold < full_threshold <= 1.0
            ):
                raise ValueError(
                    f"Question {question['name']!r} must have thresholds satisfying "
                    "0 <= zero_threshold < full_threshold <= 1."
                )
            if kind == "score" and len(question.get("levels", [])) < 2:
                raise ValueError("Score questions need at least two ordered levels.")
            if kind == "choice" and not question.get("options"):
                raise ValueError("Choice questions need an options mapping.")

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
    def _calibrate_score(value: float, *, zero_threshold: float, full_threshold: float) -> float:
        """Snap weak/strong Jev scores to 0/1; preserve scores between cutoffs."""
        if value <= zero_threshold:
            return 0.0
        if value >= full_threshold:
            return 1.0
        return value

    @staticmethod
    def _load_jev_dependencies() -> tuple[Any, Any, Any, Any]:
        """Load TypeSafe SDK lazily so importing the evaluator stays light."""
        try:
            from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
        except ImportError as error:
            raise RuntimeError(
                "Jev dependencies are missing. Install requirements.txt to use the evaluator."
            ) from error

        if not os.getenv("TYPESAFE_API_KEY"):
            raise RuntimeError(
                "TYPESAFE_API_KEY is missing. Add it to the project .env file "
                "or export it in your shell."
            )
        return TypeSafeClient, Noul, Score, Choice

    def _make_jev_questions(self) -> tuple[dict[str, Any], dict[str, dict[str, float | None]]]:
        TypeSafeClient, Noul, Score, Choice = self._load_jev_dependencies()
        questions: dict[str, Any] = {}
        choice_credits: dict[str, dict[str, float | None]] = {}
        for spec in self.questions:
            name = spec["name"]
            question_type = spec["type"]
            if question_type == "noul":
                questions[name] = Noul(instructions=spec["question"])
            elif question_type == "score":
                questions[name] = Score(
                    instructions=spec["question"], criteria=list(spec["levels"])
                )
            else:
                criteria: dict[str, str | None] = {}
                choice_credits[name] = {}
                for option, value in spec["options"].items():
                    if isinstance(value, Mapping):
                        credit = value.get("credit")
                        description = value.get("description", option.replace("_", " "))
                    else:
                        credit = value
                        description = f"{option.replace('_', ' ')} (quality credit {credit})"
                    choice_credits[name][option] = None if credit is None else float(credit)
                    criteria[option] = None if credit is None else str(description)
                questions[name] = Choice(instructions=spec["question"], criteria=criteria)
        return questions, choice_credits

    def _judge(self, item: EvaluationItem, target: dict[str, Any], prediction: dict[str, Any]):
        TypeSafeClient, Noul, Score, Choice = self._load_jev_dependencies()
        questions, choice_credits = self._make_jev_questions()
        state = {
            "resume_text": item.resume_text,
            "expected_json": json.dumps(target, ensure_ascii=False),
            "actual_json": json.dumps(prediction, ensure_ascii=False),
        }

        with TypeSafeClient() as client:
            response = client.system_one(state=state, questions=questions)
        breakdown = []
        weighted_total = 0.0
        total_weight = 0.0
        for spec in self.questions:
            name = spec["name"]
            kind = spec["type"]
            weight = float(spec.get("weight", 1.0))
            if kind == "noul":
                answer = response.nouls[name]
                value = float(answer.noul)
                probabilities = {"yes": value, "no": 1.0 - value}
            elif kind == "score":
                answer = response.scores[name]
                highest = len(spec["levels"]) - 1
                value = float(answer.score) / highest
                probabilities = self._json_safe(getattr(answer, "probabilities", {}))
            else:
                answer = response.choices[name]
                probabilities = self._json_safe(getattr(answer, "probabilities", {}))
                credits = choice_credits[name]
                applicable_mass = sum(
                    float(probabilities.get(option, 0.0))
                    for option, credit in credits.items()
                    if credit is not None
                )
                if applicable_mass >= 0.5:
                    value = sum(
                        float(probabilities.get(option, 0.0)) * float(credit)
                        for option, credit in credits.items()
                        if credit is not None
                    ) / applicable_mass
                else:
                    value = None

            applies = value is not None
            raw_value = value
            zero_threshold = float(spec.get("zero_threshold", 0.0))
            full_threshold = float(spec.get("full_threshold", 1.0))
            if applies:
                value = self._calibrate_score(
                    float(raw_value),
                    zero_threshold=zero_threshold,
                    full_threshold=full_threshold,
                )
            breakdown.append({
                "name": name,
                "path": spec.get("path"),
                "type": kind,
                "weight": weight,
                "value": round(value, 6) if applies else None,
                "raw_value": round(raw_value, 6) if applies else None,
                "thresholds": {
                    "zero_at_or_below": zero_threshold,
                    "full_at_or_above": full_threshold,
                },
                "calibrated": bool(applies and value != raw_value),
                "applicable": applies,
                "answer": self._json_safe(getattr(answer, "choice", getattr(answer, "score", getattr(answer, "noul", None)))),
                "confidence": self._json_safe(getattr(answer, "confidence", None)),
                "probabilities": probabilities,
            })
            if applies:
                weighted_total += weight * float(value)
                total_weight += weight

        final_score = weighted_total / total_weight if total_weight else 1.0
        reason = "; ".join(
            f"{entry['name']}: {entry['answer']} (score={entry['value']})"
            for entry in breakdown
            if entry["applicable"]
        )
        return round(final_score, 6), reason, breakdown

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

                if isinstance(prediction_value, str):
                    result.raw_output = result.raw_output or prediction_value
                else:
                    result.raw_output = json.dumps(
                        self._json_safe(prediction_value), ensure_ascii=False
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
                result.judge_score, result.judge_reason, result.score_breakdown = self._judge(
                    item, target, prediction
                )
                result.status = "SCORED"
            except Exception as error:
                result.status = "JUDGE_FAILED"
                result.error = f"{type(error).__name__}: {error}"
            return result
        finally:
            result.duration_seconds = round(time.perf_counter() - started, 4)

    def _summarize(self, results: list[EvaluationResult]) -> dict[str, Any]:
        valid_targets = [result for result in results if result.target is not None]
        judged = [result for result in valid_targets if result.judge_score is not None]
        denominator = len(valid_targets)
        status_counts = dict(Counter(result.status for result in results))
        return {
            "items_evaluated": len(results),
            "valid_targets": denominator,
            "json_valid_count": sum(result.json_valid for result in results),
            "schema_valid_count": sum(result.schema_valid for result in results),
            "judged_count": len(judged),
            "mean_jev_score_on_judged": (
                round(sum(result.judge_score for result in judged) / len(judged), 6)
                if judged
                else None
            ),
            # Invalid predictions count as zero; invalid targets are excluded.
            "overall_score_including_invalid_predictions": (
                round(sum(result.judge_score or 0.0 for result in valid_targets) / denominator, 6)
                if denominator
                else None
            ),
            "status_counts": status_counts,
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
        """Run the callback and Jev judge on all or a deterministic test sample."""
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
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._evaluate_one, item, callback): item for item in selected}
            with result_path.open("w", encoding="utf-8") as handle:
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Evaluating with Jev ({workers} workers)",
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
            "question_weights": {q["name"]: q.get("weight", 1.0) for q in self.questions},
        })
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return EvaluationReport(metrics=metrics, results=results, output_dir=run_dir, manifest=manifest)

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
            "jev_score": result.judge_score,
            "jev_reason": result.judge_reason,
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
