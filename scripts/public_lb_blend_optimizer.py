from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SUBMISSIONS = {
    "last_known": (Path("outputs/kaggle_last_known_v1/submission.csv"), 15.883),
    "pf_selector_spread3": (Path("outputs/kaggle_pf_selector_spread3_v1/submission.csv"), 8.781),
    "physical_noise_pf": (Path("outputs/kaggle_physical_noise_pf_v1/submission.csv"), 8.777),
    "sunny_v10_artifact_blend": (
        Path("outputs/kaggle_sunny_v10_artifact_blend_v1/submission.csv"),
        None,
    ),
    "target_free_alignment_gated": (
        Path("outputs/kaggle_target_free_alignment_gated_v1/submission.csv"),
        None,
    ),
}


@dataclass(frozen=True)
class Submission:
    name: str
    path: Path
    score: float
    predictions: np.ndarray


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_name_value(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=value, got {value!r}")
        name, raw = value.split("=", 1)
        parsed[name.strip()] = raw.strip()
    return parsed


def load_submission(name: str, path: Path, score: float, sample: pd.DataFrame) -> Submission:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "tvt"]:
        raise ValueError(f"{path}: expected columns ['id', 'tvt'], got {list(frame.columns)}")
    if not frame["id"].equals(sample["id"]):
        raise ValueError(f"{path}: ids do not match sample_submission order")
    pred = frame["tvt"].to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError(f"{path}: non-finite predictions")
    return Submission(name=name, path=path, score=float(score), predictions=pred)


def public_optimal_blend(base: Submission, other: Submission) -> dict[str, float]:
    """Solve the exact 1D optimum implied by two public RMSEs and two prediction vectors."""
    delta = other.predictions - base.predictions
    delta_mse = float(np.mean(delta * delta))
    if delta_mse <= 1e-12:
        raise ValueError(f"{base.name} and {other.name} have nearly identical predictions")

    base_mse = base.score**2
    other_mse = other.score**2
    cross = 0.5 * (other_mse - base_mse - delta_mse)
    alpha = -cross / delta_mse
    best_mse = base_mse - (cross * cross / delta_mse)
    return {
        "base": base.name,
        "other": other.name,
        "base_score": base.score,
        "other_score": other.score,
        "delta_rmse": math.sqrt(delta_mse),
        "cross": cross,
        "alpha": alpha,
        "predicted_public_rmse": math.sqrt(max(best_mse, 0.0)),
    }


def write_candidate(
    sample: pd.DataFrame,
    base: Submission,
    other: Submission,
    alpha: float,
    predicted_score: float,
    out_dir: Path,
) -> Path:
    pred = base.predictions + alpha * (other.predictions - base.predictions)
    out = sample.copy()
    out["tvt"] = pred
    filename = (
        f"{safe_name(base.name)}__to__{safe_name(other.name)}"
        f"__alpha_{alpha:.6f}__pred_{predicted_score:.4f}.csv"
    )
    path = out_dir / filename
    out.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate score-calibrated public-LB blend/extrapolation submissions. "
            "This is for public-leaderboard experimentation, not private-safe validation."
        )
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/raw/rogii-wellbore-geology-prediction/sample_submission.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/public_lb_blend_candidates"),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional submission CSV to consider.",
    )
    parser.add_argument(
        "--score",
        action="append",
        default=[],
        metavar="NAME=PUBLIC_RMSE",
        help="Override or provide a public score, e.g. sunny_v10_artifact_blend=8.293.",
    )
    parser.add_argument(
        "--write-top",
        type=int,
        default=5,
        help="Write the top N candidate CSVs by predicted public RMSE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = pd.read_csv(args.sample)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"{args.sample}: expected columns ['id', 'tvt']")

    score_overrides = {name: float(raw) for name, raw in parse_name_value(args.score).items()}
    path_overrides = {name: Path(raw) for name, raw in parse_name_value(args.candidate).items()}

    specs: dict[str, tuple[Path, float | None]] = dict(DEFAULT_SUBMISSIONS)
    for name, path in path_overrides.items():
        specs[name] = (path, score_overrides.get(name))

    submissions: list[Submission] = []
    for name, (path, default_score) in specs.items():
        score = score_overrides.get(name, default_score)
        if score is None or not path.exists():
            continue
        submissions.append(load_submission(name, path, float(score), sample))

    if len(submissions) < 2:
        raise RuntimeError("Need at least two submissions with known scores")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for base in submissions:
        for other in submissions:
            if base.name == other.name:
                continue
            try:
                rows.append(public_optimal_blend(base, other))
            except ValueError as exc:
                print(f"skip: {exc}")

    report = pd.DataFrame(rows).sort_values("predicted_public_rmse")
    report_path = args.out_dir / "blend_report.csv"
    report.to_csv(report_path, index=False)
    print(report.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"wrote report: {report_path}")

    for row in report.head(max(args.write_top, 0)).itertuples(index=False):
        base = next(sub for sub in submissions if sub.name == row.base)
        other = next(sub for sub in submissions if sub.name == row.other)
        path = write_candidate(
            sample=sample,
            base=base,
            other=other,
            alpha=float(row.alpha),
            predicted_score=float(row.predicted_public_rmse),
            out_dir=args.out_dir,
        )
        print(f"wrote candidate: {path}")


if __name__ == "__main__":
    main()
