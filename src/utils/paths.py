"""Shared project paths used by notebooks and Python modules."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
EVALUATION_DIR = DATA_DIR / "evaluation"


def ensure_data_directories() -> None:
    """Create the directories used by later pipeline stages."""
    for directory in (DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, EVALUATION_DIR):
        directory.mkdir(parents=True, exist_ok=True)
