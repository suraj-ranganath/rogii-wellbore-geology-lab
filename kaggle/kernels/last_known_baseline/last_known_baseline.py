from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION_SLUG = "rogii-wellbore-geology-prediction"


def candidate_data_dirs() -> list[Path]:
    return [
        Path("/kaggle/input") / COMPETITION_SLUG,
        Path("/kaggle/input/competitions") / COMPETITION_SLUG,
        Path("data/raw") / COMPETITION_SLUG,
        Path("../data/raw") / COMPETITION_SLUG,
        Path("../../data/raw") / COMPETITION_SLUG,
        Path("../../../data/raw") / COMPETITION_SLUG,
    ]


def find_data_dir() -> Path:
    for path in candidate_data_dirs():
        if (path / "sample_submission.csv").exists() and (path / "test").exists():
            return path
    searched = "\n".join(str(path) for path in candidate_data_dirs())
    raise FileNotFoundError(f"Could not find competition data. Searched:\n{searched}")


def parse_submission_id(value: str) -> tuple[str, int]:
    well_id, row_idx = value.rsplit("_", 1)
    return well_id, int(row_idx)


def last_known_tvt(horizontal: pd.DataFrame, row_idx: int | None = None) -> float:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    if row_idx is not None and row_idx >= 0:
        prefix = tvt_input.iloc[: min(row_idx, len(tvt_input) - 1) + 1]
        if prefix.notna().any():
            return float(prefix.dropna().iloc[-1])
    if tvt_input.notna().any():
        return float(tvt_input.dropna().iloc[-1])
    return float("nan")


def build_submission(data_dir: Path, output_path: Path) -> Path:
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    test_dir = data_dir / "test"

    horizontal_cache: dict[str, pd.DataFrame] = {}
    well_defaults: dict[str, float] = {}
    global_known: list[float] = []
    predictions: list[float] = []

    for submission_id in sample["id"].astype(str):
        well_id, row_idx = parse_submission_id(submission_id)
        if well_id not in horizontal_cache:
            path = test_dir / f"{well_id}__horizontal_well.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing horizontal well file for {well_id}: {path}")
            horizontal = pd.read_csv(path)
            if "TVT_input" not in horizontal.columns:
                raise KeyError(f"{path} does not contain TVT_input")
            horizontal_cache[well_id] = horizontal
            well_defaults[well_id] = last_known_tvt(horizontal)
            if np.isfinite(well_defaults[well_id]):
                global_known.append(well_defaults[well_id])

        pred = last_known_tvt(horizontal_cache[well_id], row_idx=row_idx)
        if not np.isfinite(pred):
            pred = well_defaults[well_id]
        predictions.append(float(pred))

    fallback = float(np.nanmedian(global_known)) if global_known else 0.0
    sample["tvt"] = pd.Series(predictions, dtype="float64").fillna(fallback)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    print(f"Wrote {len(sample)} rows to {output_path}")
    print(f"Wells: {len(horizontal_cache)}")
    print(f"Prediction range: {sample['tvt'].min():.3f} .. {sample['tvt'].max():.3f}")
    return output_path


if __name__ == "__main__":
    build_submission(find_data_dir(), Path("submission.csv"))
