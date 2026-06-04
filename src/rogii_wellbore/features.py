from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from rogii_wellbore.alignment import prefix_ncc_features, typewell_beam_features
from rogii_wellbore.data import WellPair, read_csv

CANONICAL_COLUMNS = {
    "md": ("md", "measured_depth", "measured depth"),
    "x": ("x", "x_loc", "xloc", "easting"),
    "y": ("y", "y_loc", "yloc", "northing"),
    "z": ("z", "tvd", "depth"),
    "gr": ("gr", "gamma", "gamma_ray", "gamma ray"),
    "tvt": ("tvt", "target"),
    "tvt_input": ("tvt_input", "tvt input", "tvtinput"),
    "geology": ("geology", "formation", "layer"),
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace("/", "_")


def find_column(frame: pd.DataFrame, canonical: str) -> str | None:
    candidates = {_normalize_name(value) for value in CANONICAL_COLUMNS[canonical]}
    for column in frame.columns:
        normalized = _normalize_name(column)
        if normalized in candidates:
            return column
    return None


def canonicalize(frame: pd.DataFrame, required: Iterable[str] = ()) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    missing: list[str] = []
    for canonical in CANONICAL_COLUMNS:
        column = find_column(frame, canonical)
        if column is None:
            if canonical in required:
                missing.append(canonical)
            continue
        result[canonical] = frame[column]

    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(frame.columns)}")
    return result


def _safe_gradient(values: pd.Series, spacing: pd.Series) -> np.ndarray:
    values_arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    spacing_arr = pd.to_numeric(spacing, errors="coerce").to_numpy(dtype=float)
    if len(values_arr) < 2 or not np.isfinite(values_arr).any() or not np.isfinite(spacing_arr).any():
        return np.zeros(len(values_arr), dtype=float)

    values_filled = pd.Series(values_arr).interpolate(limit_direction="both").to_numpy()
    spacing_filled = pd.Series(spacing_arr).interpolate(limit_direction="both").to_numpy()
    if np.nanmax(spacing_filled) == np.nanmin(spacing_filled):
        return np.zeros(len(values_arr), dtype=float)
    return np.gradient(values_filled, spacing_filled)


def _last_known_slope(md: pd.Series, tvt_input: pd.Series, window: int = 50) -> float:
    known = pd.DataFrame({"md": md, "tvt": tvt_input}).dropna().tail(window)
    if len(known) < 2:
        return 0.0
    x = known["md"].to_numpy(dtype=float)
    y = known["tvt"].to_numpy(dtype=float)
    if np.nanmax(x) == np.nanmin(x):
        return 0.0
    slope, _ = np.polyfit(x, y, deg=1)
    if not math.isfinite(float(slope)):
        return 0.0
    return float(slope)


def prediction_mask(horizontal: pd.DataFrame, require_target: bool) -> pd.Series:
    tvt_input = horizontal.get("tvt_input")
    mask = pd.Series(True, index=horizontal.index) if tvt_input is None else tvt_input.isna()

    if require_target:
        tvt = horizontal.get("tvt")
        if tvt is None:
            return pd.Series(False, index=horizontal.index)
        mask &= tvt.notna()
    return mask


def build_horizontal_features(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame | None = None,
    include_extra_numeric: bool = False,
    include_prefix_ncc: bool = False,
    include_typewell_beam: bool = False,
) -> pd.DataFrame:
    raw_horizontal = horizontal.copy()
    horizontal = canonicalize(horizontal, required=("md",))
    features = pd.DataFrame(index=horizontal.index)
    features["row_idx"] = np.arange(len(horizontal), dtype=float)
    features["row_frac"] = features["row_idx"] / max(len(horizontal) - 1, 1)

    md = pd.to_numeric(horizontal["md"], errors="coerce")
    features["md"] = md
    features["md_from_start"] = md - md.min()

    if "gr" in horizontal:
        gr = pd.to_numeric(horizontal["gr"], errors="coerce")
        features["gr"] = gr
        features["gr_missing"] = gr.isna().astype(float)
        gr_filled = gr.interpolate(limit_direction="both")
        for window in (5, 25, 101):
            features[f"gr_roll_mean_{window}"] = gr_filled.rolling(
                window, center=True, min_periods=1
            ).mean()
            features[f"gr_roll_std_{window}"] = (
                gr_filled.rolling(window, center=True, min_periods=2).std().fillna(0.0)
            )
        features["gr_grad_md"] = _safe_gradient(gr_filled, md)

    for coord in ("x", "y", "z"):
        if coord not in horizontal:
            continue
        values = pd.to_numeric(horizontal[coord], errors="coerce")
        features[coord] = values
        features[f"{coord}_from_start"] = values - values.iloc[0]
        features[f"d{coord}_dmd"] = _safe_gradient(values, md)

    if {"x", "y"}.issubset(horizontal.columns):
        x = pd.to_numeric(horizontal["x"], errors="coerce")
        y = pd.to_numeric(horizontal["y"], errors="coerce")
        features["xy_dist_from_start"] = np.sqrt((x - x.iloc[0]) ** 2 + (y - y.iloc[0]) ** 2)
        features["azimuth_sin"] = np.sin(np.arctan2(y - y.iloc[0], x - x.iloc[0]))
        features["azimuth_cos"] = np.cos(np.arctan2(y - y.iloc[0], x - x.iloc[0]))

    if "tvt_input" in horizontal:
        tvt_input = pd.to_numeric(horizontal["tvt_input"], errors="coerce")
        known = tvt_input.notna()
        features["tvt_input_known"] = known.astype(float)
        features["tvt_input_ffill"] = tvt_input.ffill().bfill()

        if known.any():
            ps_idx = int(np.flatnonzero(known.to_numpy())[-1])
            ps_md = float(md.iloc[ps_idx])
            last_tvt = float(tvt_input.iloc[ps_idx])
        else:
            ps_idx = 0
            ps_md = float(md.iloc[0])
            last_tvt = 0.0

        slope = _last_known_slope(md, tvt_input)
        features["ps_row_idx"] = float(ps_idx)
        features["row_from_ps"] = features["row_idx"] - float(ps_idx)
        features["md_from_ps"] = md - ps_md
        features["last_known_tvt"] = last_tvt
        features["last_known_tvt_slope"] = slope
        features["linear_tvt_prior"] = last_tvt + slope * (md - ps_md)

        if include_prefix_ncc and "gr" in horizontal:
            ncc_features = prefix_ncc_features(
                gr=pd.to_numeric(horizontal["gr"], errors="coerce"),
                tvt_input=tvt_input,
            )
            features = pd.concat([features, ncc_features], axis=1)

    if typewell is not None:
        typewell_features = _typewell_projection_features(typewell, features.get("linear_tvt_prior"))
        features = pd.concat([features, typewell_features], axis=1)
        if include_typewell_beam and {"gr", "tvt_input"}.issubset(horizontal.columns):
            typewell_canonical = canonicalize(typewell, required=("tvt", "gr"))
            beam_features = typewell_beam_features(
                gr=pd.to_numeric(horizontal["gr"], errors="coerce"),
                tvt_input=pd.to_numeric(horizontal["tvt_input"], errors="coerce"),
                typewell_tvt=pd.to_numeric(typewell_canonical["tvt"], errors="coerce"),
                typewell_gr=pd.to_numeric(typewell_canonical["gr"], errors="coerce"),
            )
            features = pd.concat([features, beam_features], axis=1)

    if include_extra_numeric:
        extra_numeric = _extra_numeric_features(raw_horizontal, features.columns)
        if not extra_numeric.empty:
            features = pd.concat([features, extra_numeric], axis=1)

    return features.replace([np.inf, -np.inf], np.nan)


def _extra_numeric_features(horizontal: pd.DataFrame, existing_columns: Iterable[str]) -> pd.DataFrame:
    existing = set(existing_columns)
    skip = set(CANONICAL_COLUMNS)
    extras = pd.DataFrame(index=horizontal.index)
    for column in horizontal.columns:
        normalized = _normalize_name(column)
        if normalized in skip or normalized in existing:
            continue
        values = pd.to_numeric(horizontal[column], errors="coerce")
        if values.notna().any():
            extras[f"raw_{normalized}"] = values
    return extras


def _typewell_projection_features(typewell: pd.DataFrame, tvt_prior: pd.Series | None) -> pd.DataFrame:
    typewell = canonicalize(typewell, required=("tvt",))
    result = pd.DataFrame(index=tvt_prior.index if tvt_prior is not None else None)
    if tvt_prior is None or "gr" not in typewell:
        return result

    tw = typewell[["tvt", "gr"]].dropna().sort_values("tvt")
    if len(tw) < 2:
        return result

    target_tvt = pd.to_numeric(tvt_prior, errors="coerce").to_numpy(dtype=float)
    tw_tvt = pd.to_numeric(tw["tvt"], errors="coerce").to_numpy(dtype=float)
    tw_gr = pd.to_numeric(tw["gr"], errors="coerce").to_numpy(dtype=float)
    result["typewell_gr_at_linear_tvt"] = np.interp(target_tvt, tw_tvt, tw_gr)
    result["typewell_tvt_min"] = float(np.nanmin(tw_tvt))
    result["typewell_tvt_max"] = float(np.nanmax(tw_tvt))
    result["linear_tvt_prior_clipped"] = np.clip(target_tvt, np.nanmin(tw_tvt), np.nanmax(tw_tvt))
    return result


def load_training_frame(
    pairs: Iterable[WellPair],
    include_extra_numeric: bool = False,
    include_prefix_ncc: bool = False,
    include_typewell_beam: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rows: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    groups: list[pd.Series] = []

    for pair in pairs:
        if pair.typewell_path is None:
            continue
        horizontal_raw = read_csv(pair.horizontal_path)
        horizontal = canonicalize(horizontal_raw)
        if "tvt" not in horizontal:
            continue
        typewell_raw = read_csv(pair.typewell_path)
        feature_frame = build_horizontal_features(
            horizontal_raw,
            typewell_raw,
            include_extra_numeric=include_extra_numeric,
            include_prefix_ncc=include_prefix_ncc,
            include_typewell_beam=include_typewell_beam,
        )
        mask = prediction_mask(horizontal, require_target=True)
        if not mask.any():
            continue
        rows.append(feature_frame.loc[mask])
        targets.append(pd.to_numeric(horizontal.loc[mask, "tvt"], errors="coerce"))
        groups.append(pd.Series(pair.well_id, index=horizontal.index[mask]))

    if not rows:
        raise ValueError("No training rows found. Check the data directory and column names.")
    return pd.concat(rows), pd.concat(targets), pd.concat(groups)
