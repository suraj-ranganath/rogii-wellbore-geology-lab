from __future__ import annotations

from pathlib import Path

COMPETITION_SLUG = "rogii-wellbore-geology-prediction"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def raw_competition_dir() -> Path:
    return PROJECT_ROOT / "data" / "raw" / COMPETITION_SLUG


def processed_dir() -> Path:
    return PROJECT_ROOT / "data" / "processed"


def model_dir() -> Path:
    return PROJECT_ROOT / "models"


def output_dir() -> Path:
    return PROJECT_ROOT / "outputs"


def submission_dir() -> Path:
    return PROJECT_ROOT / "submissions"


def logs_dir() -> Path:
    return PROJECT_ROOT / "logs"
