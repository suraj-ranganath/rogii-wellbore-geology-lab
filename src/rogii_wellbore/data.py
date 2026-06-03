from __future__ import annotations

import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rogii_wellbore.paths import COMPETITION_SLUG

HORIZONTAL_RE = re.compile(r"(?P<well>Well\d+)__horizontal_well\.csv$", re.IGNORECASE)
TYPEWELL_RE = re.compile(
    r"(?P<well>Well\d+)__typewell__(?P<typewell>Typewell\d+)\.csv$", re.IGNORECASE
)


@dataclass(frozen=True)
class WellPair:
    well_id: str
    typewell_id: str | None
    horizontal_path: Path
    typewell_path: Path | None
    split: str


def infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if {"train", "training"} & parts:
        return "train"
    if {"test", "testing", "validation", "val"} & parts:
        return "test"
    return "unknown"


def parse_horizontal_filename(path: Path) -> str | None:
    match = HORIZONTAL_RE.search(path.name)
    return match.group("well") if match else None


def parse_typewell_filename(path: Path) -> tuple[str, str] | None:
    match = TYPEWELL_RE.search(path.name)
    if not match:
        return None
    return match.group("well"), match.group("typewell")


def scan_wells(root: Path) -> list[WellPair]:
    """Find horizontal/typewell file pairs under a competition data directory."""
    root = Path(root)
    horizontal_paths: dict[str, Path] = {}
    typewell_paths: dict[str, tuple[str, Path]] = {}

    for path in sorted(root.rglob("*.csv")):
        well_id = parse_horizontal_filename(path)
        if well_id:
            horizontal_paths[well_id] = path
            continue

        parsed_typewell = parse_typewell_filename(path)
        if parsed_typewell:
            well_id, typewell_id = parsed_typewell
            typewell_paths[well_id] = (typewell_id, path)

    pairs: list[WellPair] = []
    for well_id, horizontal_path in sorted(horizontal_paths.items()):
        typewell_id, typewell_path = typewell_paths.get(well_id, (None, None))
        pairs.append(
            WellPair(
                well_id=well_id,
                typewell_id=typewell_id,
                horizontal_path=horizontal_path,
                typewell_path=typewell_path,
                split=infer_split(horizontal_path),
            )
        )
    return pairs


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def find_sample_submission(root: Path) -> Path | None:
    candidates = sorted(Path(root).rglob("*sample*submission*.csv"))
    return candidates[0] if candidates else None


def summarize_columns(paths: Iterable[Path], nrows: int = 200) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        frame = read_csv(path, nrows=nrows)
        for column in frame.columns:
            rows.append(
                {
                    "file": str(path),
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "missing_frac_sample": float(frame[column].isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def download_competition(root: Path, force: bool = False) -> Path:
    """Download and unzip competition files using the Kaggle API.

    This requires either `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and
    `KAGGLE_KEY` in the environment.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - covered by environment setup
        msg = "Install dependencies first with `uv sync --extra dev`."
        raise RuntimeError(msg) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local credentials
        msg = (
            "Kaggle authentication failed. Run `uv run kaggle auth login`, "
            "or export KAGGLE_API_TOKEN from Kaggle settings."
        )
        raise RuntimeError(msg) from exc

    api.competition_download_files(COMPETITION_SLUG, path=str(root), force=force, quiet=False)

    zip_path = root / f"{COMPETITION_SLUG}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
    return root
