from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from rogii_wellbore.data import WellPair, read_csv, scan_wells
from rogii_wellbore.features import canonicalize, prediction_mask
from rogii_wellbore.metrics import rmse

FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


@dataclass
class FormationSurfaceKNN:
    well_ids: list[str]
    xy: np.ndarray
    formations: np.ndarray
    scale: np.ndarray
    tree: cKDTree

    @classmethod
    def from_pairs(cls, pairs: list[WellPair]) -> FormationSurfaceKNN:
        rows: list[dict[str, float | str]] = []
        for pair in pairs:
            frame = read_csv(pair.horizontal_path)
            required = {"X", "Y", *FORMATION_COLUMNS}
            if not required.issubset(frame.columns):
                continue
            clean = frame[list(required)].dropna()
            if clean.empty:
                continue
            row: dict[str, float | str] = {
                "well_id": pair.well_id,
                "x": float(clean["X"].median()),
                "y": float(clean["Y"].median()),
            }
            for column in FORMATION_COLUMNS:
                row[column] = float(clean[column].median())
            rows.append(row)

        if not rows:
            raise ValueError("No training wells with formation columns were found.")

        table = pd.DataFrame(rows)
        xy = table[["x", "y"]].to_numpy(dtype=float)
        scale = xy.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        formations = table[list(FORMATION_COLUMNS)].to_numpy(dtype=float)
        return cls(
            well_ids=table["well_id"].astype(str).tolist(),
            xy=xy,
            formations=formations,
            scale=scale,
            tree=cKDTree(xy / scale),
        )

    def predict(
        self,
        xy_query: np.ndarray,
        exclude_well_id: str | None = None,
        k: int = 10,
        plane_fit: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        xy_query = np.asarray(xy_query, dtype=float)
        if xy_query.ndim != 2 or xy_query.shape[1] != 2:
            raise ValueError("xy_query must have shape (n_rows, 2).")

        n_fetch = min(len(self.well_ids), max(k + 1, k))
        distances, indices = self.tree.query(xy_query / self.scale, k=n_fetch)
        if n_fetch == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        if exclude_well_id is not None and exclude_well_id in self.well_ids:
            exclude_idx = self.well_ids.index(exclude_well_id)
            distances = np.where(indices == exclude_idx, np.inf, distances)

        order = np.argsort(distances, axis=1)[:, : min(k, distances.shape[1])]
        knn_dist = np.take_along_axis(distances, order, axis=1)
        knn_idx = np.take_along_axis(indices, order, axis=1)
        valid = np.isfinite(knn_dist)
        nearest_dist = np.where(valid.any(axis=1), np.nanmin(knn_dist, axis=1), np.nan)

        if not plane_fit:
            weights = np.where(valid, 1.0 / (knn_dist + 1e-3), 0.0)
            weight_sum = weights.sum(axis=1)
            neighbor_formations = self.formations[knn_idx]
            pred = (neighbor_formations * weights[:, :, None]).sum(axis=1)
            global_mean = np.nanmean(self.formations, axis=0)
            good = weight_sum > 0
            pred[good] = pred[good] / weight_sum[good, None]
            pred[~good] = global_mean
            return pred, nearest_dist

        pred = np.empty((len(xy_query), len(FORMATION_COLUMNS)), dtype=float)
        global_mean = np.nanmean(self.formations, axis=0)

        for row_idx in range(len(xy_query)):
            row_valid = valid[row_idx]
            if not row_valid.any():
                pred[row_idx] = global_mean
                continue
            idx = knn_idx[row_idx, row_valid]
            dist = knn_dist[row_idx, row_valid]
            neighbor_xy = self.xy[idx]
            neighbor_form = self.formations[idx]
            weights = 1.0 / (dist + 1e-3)

            if plane_fit and len(idx) >= 3:
                pred[row_idx] = _weighted_plane_predict(
                    query_xy=xy_query[row_idx],
                    neighbor_xy=neighbor_xy,
                    neighbor_values=neighbor_form,
                    weights=weights,
                )
            else:
                pred[row_idx] = np.average(neighbor_form, axis=0, weights=weights)

        return pred, nearest_dist


@dataclass
class DenseFormationKNN:
    well_ids: np.ndarray
    xy: np.ndarray
    formations: np.ndarray
    scale: np.ndarray
    tree: cKDTree

    @classmethod
    def from_pairs(cls, pairs: list[WellPair], samples_per_well: int = 80) -> DenseFormationKNN:
        xy_parts: list[np.ndarray] = []
        formation_parts: list[np.ndarray] = []
        well_parts: list[np.ndarray] = []
        for pair in pairs:
            frame = read_csv(pair.horizontal_path)
            required = ["X", "Y", *FORMATION_COLUMNS]
            if not set(required).issubset(frame.columns):
                continue
            clean = frame[required].dropna()
            if clean.empty:
                continue
            n = min(samples_per_well, len(clean))
            sample_idx = np.linspace(0, len(clean) - 1, n, dtype=int)
            sample = clean.iloc[sample_idx]
            xy_parts.append(sample[["X", "Y"]].to_numpy(dtype=float))
            formation_parts.append(sample[list(FORMATION_COLUMNS)].to_numpy(dtype=float))
            well_parts.append(np.full(n, pair.well_id, dtype=object))

        if not xy_parts:
            raise ValueError("No dense formation samples were found.")

        xy = np.vstack(xy_parts)
        formations = np.vstack(formation_parts)
        well_ids = np.concatenate(well_parts)
        scale = xy.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(
            well_ids=well_ids,
            xy=xy,
            formations=formations,
            scale=scale,
            tree=cKDTree(xy / scale),
        )

    def predict(
        self,
        xy_query: np.ndarray,
        exclude_well_id: str | None = None,
        k: int = 30,
        n_fetch: int = 80,
    ) -> tuple[np.ndarray, np.ndarray]:
        xy_query = np.asarray(xy_query, dtype=float)
        n_fetch = min(len(self.xy), max(k + 10, n_fetch))
        distances, indices = self.tree.query(xy_query / self.scale, k=n_fetch)
        if n_fetch == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        valid = np.isfinite(distances)
        if exclude_well_id is not None:
            valid &= self.well_ids[indices] != exclude_well_id

        masked_distances = np.where(valid, distances, np.inf)
        order = np.argsort(masked_distances, axis=1)[:, : min(k, masked_distances.shape[1])]
        knn_dist = np.take_along_axis(masked_distances, order, axis=1)
        knn_idx = np.take_along_axis(indices, order, axis=1)
        valid_knn = np.isfinite(knn_dist)
        weights = np.where(valid_knn, 1.0 / (knn_dist + 1e-3), 0.0)
        weight_sum = weights.sum(axis=1)
        neighbor_formations = self.formations[knn_idx]
        pred = (neighbor_formations * weights[:, :, None]).sum(axis=1)
        global_mean = np.nanmean(self.formations, axis=0)
        good = weight_sum > 0
        pred[good] = pred[good] / weight_sum[good, None]
        pred[~good] = global_mean
        nearest_dist = np.where(valid_knn.any(axis=1), np.nanmin(knn_dist, axis=1), np.nan)
        return pred, nearest_dist


def _weighted_plane_predict(
    query_xy: np.ndarray,
    neighbor_xy: np.ndarray,
    neighbor_values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        [
            neighbor_xy[:, 0],
            neighbor_xy[:, 1],
            np.ones(len(neighbor_xy)),
        ]
    )
    sqrt_w = np.sqrt(weights)
    weighted_design = design * sqrt_w[:, None]
    weighted_values = neighbor_values * sqrt_w[:, None]
    try:
        coef, *_ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
        return query_xy[0] * coef[0] + query_xy[1] * coef[1] + coef[2]
    except np.linalg.LinAlgError:
        return np.average(neighbor_values, axis=0, weights=weights)


def formation_prior_for_well(
    pair: WellPair,
    imputer: FormationSurfaceKNN,
    k: int = 10,
    formation: str = "ANCC",
    plane_fit: bool = False,
) -> pd.DataFrame:
    if formation not in FORMATION_COLUMNS:
        raise ValueError(f"Unknown formation {formation!r}. Expected one of {FORMATION_COLUMNS}.")

    raw = read_csv(pair.horizontal_path)
    horizontal = canonicalize(raw, required=("x", "y", "z", "tvt_input"))
    mask = prediction_mask(horizontal, require_target="tvt" in horizontal)
    if not mask.any():
        return pd.DataFrame()

    xy = horizontal[["x", "y"]].to_numpy(dtype=float)
    predicted_formations, nearest_dist = imputer.predict(
        xy,
        exclude_well_id=pair.well_id if pair.split == "train" else None,
        k=k,
        plane_fit=plane_fit,
    )
    form_idx = FORMATION_COLUMNS.index(formation)
    form_pred = predicted_formations[:, form_idx]

    known = horizontal["tvt_input"].notna()
    bias_values = (
        pd.to_numeric(horizontal.loc[known, "tvt_input"], errors="coerce").to_numpy(dtype=float)
        + pd.to_numeric(horizontal.loc[known, "z"], errors="coerce").to_numpy(dtype=float)
        - form_pred[known.to_numpy()]
    )
    bias = float(np.nanmedian(bias_values)) if np.isfinite(bias_values).any() else 0.0
    tvt_pred = -pd.to_numeric(horizontal["z"], errors="coerce").to_numpy(dtype=float) + form_pred + bias

    result = pd.DataFrame(
        {
            "well_id": pair.well_id,
            "row_idx": raw.index.to_numpy(),
            "formation": formation,
            "formation_prior_tvt": tvt_pred,
            "formation_bias": bias,
            "formation_knn_dist": nearest_dist,
        },
        index=raw.index,
    ).loc[mask]
    if "tvt" in horizontal:
        result["target"] = pd.to_numeric(horizontal.loc[mask, "tvt"], errors="coerce").to_numpy()
    return result.reset_index(drop=True)


def evaluate_formation_priors(data_dir: Path, k_values: list[int] | None = None) -> dict[str, Any]:
    data_dir = Path(data_dir)
    k_values = k_values or [5, 10, 15]
    train_pairs = [pair for pair in scan_wells(data_dir) if pair.split == "train"]
    imputer = FormationSurfaceKNN.from_pairs(train_pairs)

    stats: dict[tuple[int, str], dict[str, Any]] = {
        (k, formation): {"sse": 0.0, "n": 0, "well_scores": []}
        for k in k_values
        for formation in FORMATION_COLUMNS
    }

    for pair in train_pairs:
        raw = read_csv(pair.horizontal_path)
        horizontal = canonicalize(raw, required=("x", "y", "z", "tvt_input", "tvt"))
        mask = prediction_mask(horizontal, require_target=True).to_numpy()
        if not mask.any():
            continue

        xy = horizontal[["x", "y"]].to_numpy(dtype=float)
        z = pd.to_numeric(horizontal["z"], errors="coerce").to_numpy(dtype=float)
        tvt_input = pd.to_numeric(horizontal["tvt_input"], errors="coerce").to_numpy(dtype=float)
        y_true_all = pd.to_numeric(horizontal["tvt"], errors="coerce").to_numpy(dtype=float)
        known = np.isfinite(tvt_input)

        for k in k_values:
            predicted_formations, _ = imputer.predict(
                xy,
                exclude_well_id=pair.well_id,
                k=k,
                plane_fit=False,
            )
            for form_idx, formation in enumerate(FORMATION_COLUMNS):
                bias_values = tvt_input[known] + z[known] - predicted_formations[known, form_idx]
                if not np.isfinite(bias_values).any():
                    continue
                bias = float(np.nanmedian(bias_values))
                y_pred_all = -z + predicted_formations[:, form_idx] + bias
                y_true = y_true_all[mask]
                y_pred = y_pred_all[mask]
                finite = np.isfinite(y_true) & np.isfinite(y_pred)
                if not finite.any():
                    continue

                err = y_true[finite] - y_pred[finite]
                score = rmse(y_true[finite], y_pred[finite])
                bucket = stats[(k, formation)]
                bucket["sse"] += float(np.sum(err**2))
                bucket["n"] += int(finite.sum())
                bucket["well_scores"].append(
                    {"well_id": pair.well_id, "rmse": score, "rows": int(finite.sum())}
                )

    rows: list[dict[str, Any]] = []
    for (k, formation), values in stats.items():
        well_scores = values["well_scores"]
        rmse_values = [item["rmse"] for item in well_scores]
        n = values["n"]
        rmse_value = float(np.sqrt(values["sse"] / n)) if n else np.nan
        rows.append(
            {
                "k": k,
                "formation": formation,
                "rmse": rmse_value,
                "rows": n,
                "wells": len(well_scores),
                "well_rmse_mean": float(np.mean(rmse_values)) if rmse_values else np.nan,
                "well_rmse_median": float(np.median(rmse_values)) if rmse_values else np.nan,
                "worst_wells": sorted(well_scores, key=lambda item: item["rmse"], reverse=True)[:10],
            }
        )

    rows = sorted(rows, key=lambda item: item["rmse"])
    return {
        "n_train_wells": len(train_pairs),
        "formation_columns": list(FORMATION_COLUMNS),
        "results": rows,
        "best": rows[0] if rows else None,
    }


def evaluate_dense_formation_priors(
    data_dir: Path,
    k_values: list[int] | None = None,
    formation_names: list[str] | None = None,
    samples_per_well: int = 80,
    max_wells: int | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    k_values = k_values or [40]
    formation_names = formation_names or ["ANCC"]
    unknown_formations = set(formation_names) - set(FORMATION_COLUMNS)
    if unknown_formations:
        raise ValueError(f"Unknown formation columns: {sorted(unknown_formations)}")
    train_pairs = [pair for pair in scan_wells(data_dir) if pair.split == "train"]
    imputer = DenseFormationKNN.from_pairs(train_pairs, samples_per_well=samples_per_well)
    eval_pairs = train_pairs[:max_wells] if max_wells else train_pairs
    stats: dict[tuple[int, str], dict[str, Any]] = {
        (k, formation): {"sse": 0.0, "n": 0, "well_scores": []}
        for k in k_values
        for formation in formation_names
    }

    for pair in eval_pairs:
        raw = read_csv(pair.horizontal_path)
        horizontal = canonicalize(raw, required=("x", "y", "z", "tvt_input", "tvt"))
        mask = prediction_mask(horizontal, require_target=True).to_numpy()
        if not mask.any():
            continue

        xy = horizontal[["x", "y"]].to_numpy(dtype=float)
        z = pd.to_numeric(horizontal["z"], errors="coerce").to_numpy(dtype=float)
        tvt_input = pd.to_numeric(horizontal["tvt_input"], errors="coerce").to_numpy(dtype=float)
        y_true_all = pd.to_numeric(horizontal["tvt"], errors="coerce").to_numpy(dtype=float)
        known = np.isfinite(tvt_input)

        for k in k_values:
            predicted_formations, _ = imputer.predict(
                xy,
                exclude_well_id=pair.well_id,
                k=k,
            )
            for formation in formation_names:
                form_idx = FORMATION_COLUMNS.index(formation)
                bias_values = tvt_input[known] + z[known] - predicted_formations[known, form_idx]
                if not np.isfinite(bias_values).any():
                    continue
                bias = float(np.nanmedian(bias_values))
                y_pred_all = -z + predicted_formations[:, form_idx] + bias
                y_true = y_true_all[mask]
                y_pred = y_pred_all[mask]
                finite = np.isfinite(y_true) & np.isfinite(y_pred)
                if not finite.any():
                    continue

                err = y_true[finite] - y_pred[finite]
                score = rmse(y_true[finite], y_pred[finite])
                bucket = stats[(k, formation)]
                bucket["sse"] += float(np.sum(err**2))
                bucket["n"] += int(finite.sum())
                bucket["well_scores"].append(
                    {"well_id": pair.well_id, "rmse": score, "rows": int(finite.sum())}
                )

    rows: list[dict[str, Any]] = []
    for (k, formation), values in stats.items():
        well_scores = values["well_scores"]
        rmse_values = [item["rmse"] for item in well_scores]
        n = values["n"]
        rows.append(
            {
                "k": k,
                "formation": formation,
                "rmse": float(np.sqrt(values["sse"] / n)) if n else np.nan,
                "rows": n,
                "wells": len(well_scores),
                "well_rmse_mean": float(np.mean(rmse_values)) if rmse_values else np.nan,
                "well_rmse_median": float(np.median(rmse_values)) if rmse_values else np.nan,
                "worst_wells": sorted(well_scores, key=lambda item: item["rmse"], reverse=True)[:10],
            }
        )

    rows = sorted(rows, key=lambda item: item["rmse"])
    return {
        "n_train_wells": len(train_pairs),
        "n_eval_wells": len(eval_pairs),
        "samples_per_well": samples_per_well,
        "max_wells": max_wells,
        "formation_columns": list(FORMATION_COLUMNS),
        "results": rows,
        "best": rows[0] if rows else None,
    }
