from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction"
DEFAULT_OUTPUT = ROOT / "outputs/visible_overlap_scores.json"


@dataclass(frozen=True)
class SubmissionSpec:
    name: str
    path: Path
    kaggle_public: float | None = None


DEFAULT_SUBMISSIONS = [
    SubmissionSpec("last_known", ROOT / "outputs/kaggle_last_known_v1/submission.csv", 15.883),
    SubmissionSpec(
        "pf_selector_spread3",
        ROOT / "outputs/kaggle_pf_selector_spread3_v1/submission.csv",
        8.781,
    ),
    SubmissionSpec(
        "physical_noise_pf",
        ROOT / "outputs/kaggle_physical_noise_pf_v1/submission.csv",
        8.777,
    ),
    SubmissionSpec(
        "sunny_v10_artifact",
        ROOT / "outputs/kaggle_sunny_v10_artifact_blend_v1/submission.csv",
        8.421,
    ),
    SubmissionSpec(
        "target_free_align",
        ROOT / "outputs/kaggle_target_free_alignment_gated_v1/submission.csv",
        10.626,
    ),
    SubmissionSpec(
        "super_solution_top3",
        ROOT / "outputs/next_window/super_solution_top3_fixed/submission.csv",
        10.150,
    ),
    SubmissionSpec(
        "ridge_w020",
        ROOT / "outputs/queue_20260606_hidden_safe/ravaghi_ridge_w020/submission.csv",
        8.233,
    ),
    SubmissionSpec(
        "ridge_w025",
        ROOT / "outputs/queue_20260606_hidden_safe/ravaghi_ridge_w025/submission.csv",
        8.439,
    ),
    SubmissionSpec(
        "ridge_w030",
        ROOT / "outputs/queue_20260606_hidden_safe/ravaghi_ridge_w030/submission.csv",
        8.187,
    ),
    SubmissionSpec(
        "ridge_w035",
        ROOT / "outputs/queue_20260606_hidden_safe/ravaghi_ridge_w035/submission.csv",
        8.108,
    ),
    SubmissionSpec(
        "ridge_w040_best",
        ROOT / "outputs/queue_20260606_hidden_safe/ravaghi_ridge_w040/submission.csv",
        7.906,
    ),
    SubmissionSpec(
        "ridge_w040_pf40",
        ROOT / "outputs/queue_20260607/ridge_w040_pf40/submission.csv",
    ),
    SubmissionSpec(
        "ridge_w040_selector070",
        ROOT / "outputs/queue_20260607/ridge_w040_selector070/submission.csv",
    ),
    SubmissionSpec(
        "ridge_w040_selector080",
        ROOT / "outputs/queue_20260607/ridge_w040_selector080/submission.csv",
    ),
    SubmissionSpec(
        "ridge_w040_prefix_gate",
        ROOT / "outputs/queue_20260607/ridge_w040_prefix_gate/submission.csv",
    ),
    SubmissionSpec(
        "ridge_w040_formprefix_gate",
        ROOT / "outputs/queue_20260607/ridge_w040_formprefix_gate/submission.csv",
    ),
]


def load_visible_truth(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    truth_by_well: dict[str, np.ndarray] = {}
    for well_id in sorted(sample["id"].str[:8].unique()):
        path = data_dir / "train" / f"{well_id}__horizontal_well.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Visible-overlap well is not in train: {path}")
        truth_by_well[well_id] = pd.read_csv(path, usecols=["TVT"])["TVT"].to_numpy(float)

    truth = []
    for row_id in sample["id"]:
        well_id, row_idx_text = row_id.split("_", maxsplit=1)
        truth.append(float(truth_by_well[well_id][int(row_idx_text)]))
    return sample, np.asarray(truth, dtype=float)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        raise ValueError("No finite rows to score.")
    diff = y_true[finite] - y_pred[finite]
    return float(np.sqrt(np.mean(diff * diff)))


def score_submission(spec: SubmissionSpec, sample: pd.DataFrame, truth: np.ndarray) -> dict[str, Any]:
    sub = pd.read_csv(spec.path)
    if list(sub.columns) != ["id", "tvt"]:
        raise ValueError(f"{spec.path}: expected columns ['id', 'tvt'], got {list(sub.columns)}")
    if not sub["id"].equals(sample["id"]):
        if set(sub["id"]) != set(sample["id"]):
            raise ValueError(f"{spec.path}: ids do not match sample_submission")
        sub = sample[["id"]].merge(sub, on="id", how="left")
    prediction = sub["tvt"].to_numpy(float)
    local_rmse = rmse(truth, prediction)
    result = {
        "name": spec.name,
        "path": str(spec.path),
        "visible_overlap_rmse": local_rmse,
        "rows": int(len(sub)),
    }
    if spec.kaggle_public is not None:
        result["kaggle_public"] = spec.kaggle_public
        result["visible_minus_kaggle"] = local_rmse - spec.kaggle_public
    return result


def parse_submission_args(values: list[str]) -> list[SubmissionSpec]:
    specs = []
    for value in values:
        if "=" in value:
            name, path_text = value.split("=", maxsplit=1)
        else:
            path = Path(value)
            name = path.parent.name or path.stem
            path_text = value
        specs.append(SubmissionSpec(name=name, path=Path(path_text).resolve()))
    return specs


def correlation_summary(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    paired = [
        (row["visible_overlap_rmse"], row["kaggle_public"])
        for row in rows
        if "kaggle_public" in row
    ]
    if len(paired) < 2:
        return None
    visible = pd.Series([item[0] for item in paired])
    kaggle = pd.Series([item[1] for item in paired])
    return {
        "pearson": float(visible.corr(kaggle, method="pearson")),
        "spearman": float(visible.corr(kaggle, method="spearman")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score submissions on the downloadable visible-overlap wells. "
            "This is a plumbing/diagnostic score, not a Kaggle public-LB emulator."
        )
    )
    parser.add_argument("submissions", nargs="*", help="Optional name=path or path entries.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sample, truth = load_visible_truth(args.data_dir)
    specs = parse_submission_args(args.submissions) if args.submissions else DEFAULT_SUBMISSIONS
    rows = [
        score_submission(spec, sample, truth)
        for spec in specs
        if spec.path.is_file()
    ]
    rows = sorted(rows, key=lambda row: row["visible_overlap_rmse"])
    summary = {
        "rows": rows,
        "correlation_with_known_kaggle_public": correlation_summary(rows),
        "warning": (
            "The downloadable test wells overlap train and are not representative of the "
            "hidden/public rerun distribution. Use this score only for diagnostics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{'name':28s} {'visible_rmse':>12s} {'kaggle_public':>13s} {'delta':>10s}")
    for row in rows:
        kaggle = row.get("kaggle_public")
        kaggle_text = "" if kaggle is None else f"{kaggle:.3f}"
        delta = row.get("visible_minus_kaggle")
        delta_text = "" if delta is None else f"{delta:.3f}"
        print(
            f"{row['name']:28s} {row['visible_overlap_rmse']:12.4f} "
            f"{kaggle_text:>13s} {delta_text:>10s}"
        )
    if summary["correlation_with_known_kaggle_public"]:
        corr = summary["correlation_with_known_kaggle_public"]
        print(f"pearson={corr['pearson']:.4f} spearman={corr['spearman']:.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
