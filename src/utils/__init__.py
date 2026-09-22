"""Shared utilities for data processing, training, and evaluation."""

from .paths import (
    DATA_DIR,
    EVALUATION_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    ensure_data_directories,
)

__all__ = [
    "DATA_DIR",
    "EVALUATION_DIR",
    "MODELS_DIR",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "ensure_data_directories",
]
