from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION_SLUG = "rogii-wellbore-geology-prediction"
CANDIDATE_NAME = "public_sunny_last_extrapolate"
EXPECTED_PUBLIC_RMSE = "7.861"
NOTE = "Public-score calibrated Sunny+v10 -> last-known extrapolation."


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
            print(f"INPUT_DIR={path}")
            return path
    searched = "\n".join(str(path) for path in candidate_data_dirs())
    raise FileNotFoundError(f"Could not find competition data. Searched:\n{searched}")


def main() -> None:
    data_dir = find_data_dir()
    sample = pd.read_csv(data_dir / "sample_submission.csv")[["id"]]
    embedded_path = Path(__file__).with_name("candidate_submission.csv.gz")
    with gzip.open(embedded_path, "rt") as handle:
        candidate = pd.read_csv(handle)

    if list(candidate.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected embedded columns: {list(candidate.columns)}")
    if candidate["id"].duplicated().any():
        raise ValueError("Embedded candidate has duplicate ids")

    sample_ids = set(sample["id"].astype(str))
    candidate_ids = set(candidate["id"].astype(str))
    missing = sample_ids.difference(candidate_ids)
    extra = candidate_ids.difference(sample_ids)
    if missing or extra:
        raise ValueError(
            f"Static public candidate only matches the current public sample: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    submission = sample.merge(candidate, on="id", how="left")
    values = submission["tvt"].to_numpy(dtype=float)
    if submission["tvt"].isna().any() or not np.isfinite(values).all():
        raise ValueError("Non-finite predictions after sample alignment")

    submission.to_csv("submission.csv", index=False)
    print(f"candidate={CANDIDATE_NAME}")
    print(f"expected_public_rmse={EXPECTED_PUBLIC_RMSE}")
    print(f"note={NOTE}")
    print(f"rows={len(submission)}")
    print(f"prediction_range={submission['tvt'].min():.3f}..{submission['tvt'].max():.3f}")


if __name__ == "__main__":
    main()
