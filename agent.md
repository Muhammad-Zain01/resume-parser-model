# Agent Instructions

## Project purpose

This repository is for building and fine-tuning a small language model that parses resumes efficiently and quickly while maintaining excellent accuracy across diverse resume formats, industries, and languages.

## Core requirements

- Prioritize accurate extraction of structured resume information.
- Optimize for low latency and efficient resource usage.
- Support varied layouts, sections, writing styles, and document quality.
- Preserve important details such as names, contact information, skills, work experience, education, certifications, projects, and dates.
- Measure quality with reproducible evaluation datasets and clearly reported metrics.
- Keep training, evaluation, preprocessing, and inference workflows reproducible.
- Protect personal and sensitive information contained in resumes.
- Never commit secrets, tokens, private resume data, model credentials, or generated datasets.

## Engineering expectations

- Prefer simple, maintainable implementations with clear documentation.
- Validate changes with focused tests or data-quality checks before considering them complete.
- Record assumptions and limitations when introducing a dataset, model, parser, or metric.
- Avoid claiming global accuracy or production readiness without representative benchmark evidence.
- Keep generated data under `.data/`, which is intentionally ignored by Git.

## Environment configuration

Local environment variables belong in `.env`, which is ignored by Git. Use `.env.example` as the configuration template. The Hugging Face credential is provided through `HF_TOKEN`.
