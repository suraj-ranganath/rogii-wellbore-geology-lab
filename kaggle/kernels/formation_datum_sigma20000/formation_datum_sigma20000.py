#!/usr/bin/env python3
"""Dynamic formation-datum invariant candidate for ROGII.

This adapts the public formation-datum idea into a hidden-test compatible
script. It predicts a train-derived ANCC surface from X/Y coordinates, anchors
the per-well invariant from the visible TVT_input prefix, and writes predictions
in sample_submission order for whichever wells Kaggle mounts at inference time.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

FORMATION_COL = "ANCC"
BLEND_SIGMA = 20000.0
MAX_POINTS_PER_WELL = 200
RANDOM_STATE = 42


def find_data_dir() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("data/raw/rogii-wellbore-geology-prediction"),
    ]
    override = os.environ.get("ROGII_INPUT_DIR")
    if override:
        candidates.insert(0, Path(override))
    for path in candidates:
        if (path / "train").is_dir() and (path / "test").is_dir() and (
            path / "sample_submission.csv"
        ).is_file():
            return path
    raise FileNotFoundError("Could not locate competition data directory")


def well_id_from_filename(path: Path) -> str:
    return path.name.replace("__horizontal_well.csv", "")


def load_horizontal_wells(split_dir: Path) -> dict[str, pd.DataFrame]:
    wells: dict[str, pd.DataFrame] = {}
    for path in sorted(split_dir.glob("*__horizontal_well.csv")):
        wells[well_id_from_filename(path)] = pd.read_csv(path)
    return wells


def sample_training_surface(
    train_data: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    xy_parts: list[np.ndarray] = []
    ancc_parts: list[np.ndarray] = []
    for df in train_data.values():
        if FORMATION_COL not in df.columns:
            continue
        valid = (
            df["X"].notna().to_numpy()
            & df["Y"].notna().to_numpy()
            & df[FORMATION_COL].notna().to_numpy()
        )
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) < 10:
            continue
        take = np.linspace(0, len(valid_idx) - 1, min(MAX_POINTS_PER_WELL, len(valid_idx)))
        idx = valid_idx[np.unique(take.astype(int))]
        xy_parts.append(df.loc[idx, ["X", "Y"]].to_numpy(dtype=np.float64))
        ancc_parts.append(df.loc[idx, FORMATION_COL].to_numpy(dtype=np.float64))
    if not xy_parts:
        raise ValueError(f"No usable {FORMATION_COL} surface points found")
    return np.vstack(xy_parts), np.concatenate(ancc_parts)


def fit_surface_model(train_data: dict[str, pd.DataFrame]):
    xy, ancc = sample_training_surface(train_data)
    poly = PolynomialFeatures(degree=3, include_bias=True)
    x_poly = poly.fit_transform(xy)
    trend = Ridge(alpha=100.0, random_state=RANDOM_STATE)
    trend.fit(x_poly, ancc)
    residuals = ancc - trend.predict(x_poly)
    knn = KNeighborsRegressor(n_neighbors=20, weights="distance")
    knn.fit(xy, residuals)
    r2 = 1.0 - float(np.var(residuals) / np.var(ancc))
    print(f"surface_points={len(xy)} trend_r2={r2:.5f}", flush=True)

    def predict(xy_new: np.ndarray) -> np.ndarray:
        xy_new = np.asarray(xy_new, dtype=np.float64)
        return trend.predict(poly.transform(xy_new)) + knn.predict(xy_new)

    return predict


def last_known_fallback(df: pd.DataFrame) -> np.ndarray:
    values = df["TVT_input"].to_numpy(dtype=float)
    known = np.flatnonzero(np.isfinite(values))
    if len(known) == 0:
        return np.zeros(len(df), dtype=float)
    return np.full(len(df), float(values[known[-1]]), dtype=float)


def predict_well(df: pd.DataFrame, predict_surface) -> np.ndarray:
    required = {"X", "Y", "Z", "TVT_input"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    tvt_input = df["TVT_input"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    xy = df[["X", "Y"]].to_numpy(dtype=float)
    has_input = np.isfinite(tvt_input)
    if has_input.sum() == 0 or not np.isfinite(xy).all() or not np.isfinite(z).all():
        return last_known_fallback(df)

    ancc_pred = predict_surface(xy)
    invariant = tvt_input[has_input] + z[has_input] - ancc_pred[has_input]
    invariant = invariant[np.isfinite(invariant)]
    if len(invariant) == 0:
        return last_known_fallback(df)

    form_pred = float(np.median(invariant)) - z + ancc_pred
    form_pred[has_input] = tvt_input[has_input]

    flat_pred = last_known_fallback(df)
    anchor_pos = np.flatnonzero(has_input)
    dists, _ = cKDTree(anchor_pos.reshape(-1, 1)).query(
        np.arange(len(df)).reshape(-1, 1)
    )
    weights = np.exp(-0.5 * (dists.ravel() / BLEND_SIGMA) ** 2)
    pred = weights * form_pred + (1.0 - weights) * flat_pred
    pred[has_input] = tvt_input[has_input]
    pred = np.where(np.isfinite(pred), pred, flat_pred)
    return pred.astype(float)


def main() -> None:
    data_dir = find_data_dir()
    train_data = load_horizontal_wells(data_dir / "train")
    test_data = load_horizontal_wells(data_dir / "test")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    sample["well"] = sample["id"].str.extract(r"^(.+)_\d+$", expand=False)
    sample["row_idx"] = sample["id"].str.extract(r"_(\d+)$", expand=False).astype(int)
    print(
        f"data_dir={data_dir} train_wells={len(train_data)} "
        f"test_wells={len(test_data)} sample_rows={len(sample)}",
        flush=True,
    )

    predict_surface = fit_surface_model(train_data)
    rows: list[dict[str, float | str]] = []
    for well_id, df in sorted(test_data.items()):
        full_pred = predict_well(df, predict_surface)
        well_sample = sample[sample["well"] == well_id]
        for row in well_sample.itertuples(index=False):
            row_idx = int(row.row_idx)
            if row_idx < 0 or row_idx >= len(full_pred):
                raise IndexError(f"{well_id}: row_idx {row_idx} outside 0..{len(full_pred)-1}")
            rows.append({"id": row.id, "tvt": float(full_pred[row_idx])})
        print(f"{well_id}: wrote {len(well_sample)} rows", flush=True)

    pred_df = pd.DataFrame(rows)
    submission = sample[["id"]].merge(pred_df, on="id", how="left")
    if submission["tvt"].isna().any():
        missing = int(submission["tvt"].isna().sum())
        print(f"missing rows after merge={missing}; filling with per-well last known", flush=True)
        fallback_rows: list[dict[str, float | str]] = []
        for well_id, df in test_data.items():
            fallback = last_known_fallback(df)
            well_sample = sample[sample["well"] == well_id]
            for row in well_sample.itertuples(index=False):
                fallback_rows.append({"id": row.id, "fallback": float(fallback[int(row.row_idx)])})
        fallback_df = pd.DataFrame(fallback_rows)
        submission = submission.merge(fallback_df, on="id", how="left")
        submission["tvt"] = submission["tvt"].fillna(submission["fallback"])
        submission = submission[["id", "tvt"]]

    if len(submission) != len(sample):
        raise ValueError(f"row count mismatch: {len(submission)} != {len(sample)}")
    if not submission["id"].equals(sample["id"]):
        raise ValueError("submission id order does not match sample_submission")
    if submission["tvt"].isna().any() or not np.isfinite(submission["tvt"]).all():
        raise ValueError("submission contains non-finite predictions")

    submission.to_csv("submission.csv", index=False)
    print(
        "saved submission.csv "
        f"rows={len(submission)} min={submission['tvt'].min():.3f} "
        f"max={submission['tvt'].max():.3f} mean={submission['tvt'].mean():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
