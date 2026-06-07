from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rogii_wellbore.data import WellPair, read_csv, scan_wells
from rogii_wellbore.features import build_horizontal_features, canonicalize, prediction_mask
from rogii_wellbore.metrics import rmse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction"
DEFAULT_OUTPUT = ROOT / "outputs/local_tail_cv.json"


@dataclass
class LocalFrame:
    x: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    ids: pd.Series
    priors: dict[str, pd.Series]


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def last_known_stats(horizontal: pd.DataFrame) -> dict[str, float]:
    tvt_input = pd.to_numeric(horizontal["tvt_input"], errors="coerce")
    md = pd.to_numeric(horizontal["md"], errors="coerce")
    z = pd.to_numeric(horizontal["z"], errors="coerce") if "z" in horizontal else None
    known = tvt_input.notna()
    if not known.any():
        return {
            "last_known_tvt": 0.0,
            "last_known_md": float(md.iloc[0]) if len(md) else 0.0,
            "last_known_z": 0.0,
            "known_len": 0.0,
            "linear_md_slope": 0.0,
            "z_anchor_offset": 0.0,
        }

    known_idx = np.flatnonzero(known.to_numpy())
    last_idx = int(known_idx[-1])
    last_known_tvt = float(tvt_input.iloc[last_idx])
    last_known_md = float(md.iloc[last_idx])
    last_known_z = float(z.iloc[last_idx]) if z is not None else 0.0

    tail = pd.DataFrame({"md": md[known], "tvt": tvt_input[known]}).dropna().tail(120)
    if len(tail) >= 2 and tail["md"].max() > tail["md"].min():
        linear_md_slope = float(np.polyfit(tail["md"].to_numpy(), tail["tvt"].to_numpy(), deg=1)[0])
    else:
        linear_md_slope = 0.0

    if z is not None:
        z_bias = tvt_input[known] + z[known]
        z_anchor_offset = float(np.nanmedian(z_bias.to_numpy(dtype=float)))
    else:
        z_anchor_offset = last_known_tvt

    return {
        "last_known_tvt": last_known_tvt,
        "last_known_md": last_known_md,
        "last_known_z": last_known_z,
        "known_len": float(len(known_idx)),
        "linear_md_slope": linear_md_slope,
        "z_anchor_offset": z_anchor_offset,
    }


def build_well_frame(
    pair: WellPair,
    include_extra_numeric: bool,
    include_prefix_ncc: bool,
    include_typewell_beam: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, pd.Series]] | None:
    horizontal_raw = read_csv(pair.horizontal_path)
    horizontal = canonicalize(horizontal_raw, required=("md", "z", "tvt_input", "tvt"))
    mask = prediction_mask(horizontal, require_target=True)
    if not mask.any():
        return None

    typewell_raw = read_csv(pair.typewell_path) if pair.typewell_path is not None else None
    features = build_horizontal_features(
        horizontal_raw,
        typewell_raw,
        include_extra_numeric=include_extra_numeric,
        include_prefix_ncc=include_prefix_ncc,
        include_typewell_beam=include_typewell_beam,
    )

    stats = last_known_stats(horizontal)
    md = pd.to_numeric(horizontal["md"], errors="coerce")
    z = pd.to_numeric(horizontal["z"], errors="coerce")
    row_idx = pd.Series(np.arange(len(horizontal), dtype=float), index=horizontal.index)

    priors = {
        "last_known": pd.Series(stats["last_known_tvt"], index=horizontal.index, dtype=float),
        "linear_md": pd.Series(
            stats["last_known_tvt"] + stats["linear_md_slope"] * (md - stats["last_known_md"]),
            index=horizontal.index,
            dtype=float,
        ),
        "z_anchor": pd.Series(stats["z_anchor_offset"] - z, index=horizontal.index, dtype=float),
    }

    x = features.copy()
    x["well_row_idx"] = row_idx
    x["well_rows"] = float(len(horizontal))
    x["eval_rows"] = float(mask.sum())
    x["known_len"] = stats["known_len"]
    x["z_from_last_known"] = z - stats["last_known_z"]
    x["z_anchor_tvt"] = priors["z_anchor"]
    x["linear_md_tvt"] = priors["linear_md"]
    x["last_known_tvt_prior"] = priors["last_known"]

    y = pd.to_numeric(horizontal.loc[mask, "tvt"], errors="coerce")
    ids = pd.Series([f"{pair.well_id}_{idx}" for idx in horizontal.index[mask]], index=horizontal.index[mask])
    return x.loc[mask], y, ids, {name: values.loc[mask] for name, values in priors.items()}


def select_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)


def load_local_frame(args: argparse.Namespace) -> LocalFrame:
    pairs = [pair for pair in scan_wells(args.data_dir) if pair.split == "train"]
    if args.max_wells:
        pairs = pairs[: args.max_wells]
    if args.well_stride > 1:
        pairs = pairs[:: args.well_stride]
    if not pairs:
        raise ValueError(f"No train wells found under {args.data_dir}")

    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    id_parts: list[pd.Series] = []
    group_parts: list[pd.Series] = []
    prior_parts: dict[str, list[pd.Series]] = {
        "last_known": [],
        "linear_md": [],
        "z_anchor": [],
    }

    for index, pair in enumerate(pairs, start=1):
        built = build_well_frame(
            pair,
            include_extra_numeric=args.include_extra_numeric,
            include_prefix_ncc=args.include_prefix_ncc,
            include_typewell_beam=args.include_typewell_beam,
        )
        if built is None:
            continue
        x, y, ids, priors = built
        x_parts.append(x.reset_index(drop=True))
        y_parts.append(y.reset_index(drop=True))
        id_parts.append(ids.reset_index(drop=True))
        group_parts.append(pd.Series(pair.well_id, index=range(len(y))))
        for name, values in priors.items():
            prior_parts[name].append(values.reset_index(drop=True))
        if args.progress and index % args.progress == 0:
            print(f"loaded {index}/{len(pairs)} wells", flush=True)

    x_all = select_numeric(pd.concat(x_parts, ignore_index=True))
    y_all = pd.concat(y_parts, ignore_index=True)
    ids_all = pd.concat(id_parts, ignore_index=True)
    groups_all = pd.concat(group_parts, ignore_index=True)
    priors_all = {name: pd.concat(parts, ignore_index=True) for name, parts in prior_parts.items()}
    return LocalFrame(x=x_all, y=y_all, groups=groups_all, ids=ids_all, priors=priors_all)


def make_splits(groups: pd.Series, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = groups.nunique()
    n_splits = min(n_splits, int(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two wells for local CV.")
    if unique_groups == n_splits:
        splitter = GroupKFold(n_splits=n_splits)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        wells = pd.Series(sorted(groups.unique()))
        fold_pairs = []
        for train_well_idx, valid_well_idx in splitter.split(wells):
            train_wells = set(wells.iloc[train_well_idx])
            valid_wells = set(wells.iloc[valid_well_idx])
            train_idx = np.flatnonzero(groups.isin(train_wells).to_numpy())
            valid_idx = np.flatnonzero(groups.isin(valid_wells).to_numpy())
            fold_pairs.append((train_idx, valid_idx))
        return fold_pairs
    return list(splitter.split(np.zeros(len(groups)), groups=groups))


def fit_ridge_model(x: pd.DataFrame, target: pd.Series, alpha: float) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=0)),
        ]
    )
    model.set_output(transform="pandas")
    model.fit(x, target)
    return model


def fit_lgbm_model(x: pd.DataFrame, target: pd.Series, args: argparse.Namespace):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("LightGBM is not installed in this environment") from exc

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "lgbm",
                LGBMRegressor(
                    n_estimators=args.lgbm_estimators,
                    learning_rate=args.lgbm_learning_rate,
                    num_leaves=args.lgbm_num_leaves,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.85,
                    reg_lambda=5.0,
                    min_child_samples=30,
                    objective="regression",
                    random_state=args.seed,
                    n_jobs=args.n_jobs,
                    verbosity=-1,
                    force_row_wise=True,
                ),
            ),
        ]
    )
    model.set_output(transform="pandas")
    model.fit(x, target)
    return model


def fit_catboost_model(x: pd.DataFrame, target: pd.Series, args: argparse.Namespace):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("CatBoost is not installed in this environment") from exc

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "catboost",
                CatBoostRegressor(
                    iterations=args.catboost_iterations,
                    learning_rate=args.catboost_learning_rate,
                    depth=args.catboost_depth,
                    loss_function="RMSE",
                    bootstrap_type="Bernoulli",
                    subsample=0.8,
                    random_seed=args.seed,
                    task_type=args.catboost_task_type,
                    devices=args.catboost_devices,
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    )
    model.set_output(transform="pandas")
    model.fit(x, target)
    return model


def score_prediction(y: pd.Series, pred: np.ndarray, groups: pd.Series) -> dict[str, Any]:
    finite = np.isfinite(y.to_numpy(float)) & np.isfinite(pred)
    y_finite = y.to_numpy(float)[finite]
    pred_finite = pred[finite]
    groups_finite = groups.iloc[np.flatnonzero(finite)]
    well_scores = []
    for well_id, group_index in groups_finite.groupby(groups_finite).groups.items():
        idx = np.asarray(list(group_index), dtype=int)
        local_y = y.to_numpy(float)[idx]
        local_pred = pred[idx]
        local_finite = np.isfinite(local_y) & np.isfinite(local_pred)
        if local_finite.any():
            well_scores.append(
                {
                    "well": str(well_id),
                    "rmse": rmse(local_y[local_finite], local_pred[local_finite]),
                    "rows": int(local_finite.sum()),
                }
            )
    well_rmse_values = [row["rmse"] for row in well_scores]
    return {
        "rmse": rmse(y_finite, pred_finite),
        "rows": int(finite.sum()),
        "wells": int(len(well_scores)),
        "well_rmse_mean": float(np.mean(well_rmse_values)) if well_rmse_values else None,
        "well_rmse_median": float(np.median(well_rmse_values)) if well_rmse_values else None,
        "worst_wells": sorted(well_scores, key=lambda row: row["rmse"], reverse=True)[:10],
    }


def run_cv(frame: LocalFrame, args: argparse.Namespace) -> dict[str, Any]:
    splits = make_splits(frame.groups, args.folds, args.seed)
    method_names = ["last_known", "linear_md", "z_anchor"]
    if args.include_ridge:
        method_names.extend(["ridge_last_residual", "ridge_z_residual"])
    if args.include_lgbm:
        method_names.extend(["lgbm_last_residual", "lgbm_z_residual"])
    if args.include_catboost:
        method_names.extend(["catboost_last_residual", "catboost_z_residual"])

    oof = {name: np.full(len(frame.y), np.nan, dtype=float) for name in method_names}
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        x_train = frame.x.iloc[train_idx]
        x_valid = frame.x.iloc[valid_idx]
        y_train = frame.y.iloc[train_idx]
        y_valid = frame.y.iloc[valid_idx]

        for name in ("last_known", "linear_md", "z_anchor"):
            oof[name][valid_idx] = frame.priors[name].iloc[valid_idx].to_numpy(float)

        if args.include_ridge:
            last_train = frame.priors["last_known"].iloc[train_idx]
            last_valid = frame.priors["last_known"].iloc[valid_idx].to_numpy(float)
            z_train = frame.priors["z_anchor"].iloc[train_idx]
            z_valid = frame.priors["z_anchor"].iloc[valid_idx].to_numpy(float)

            ridge_last = fit_ridge_model(x_train, y_train - last_train, alpha=args.ridge_alpha)
            oof["ridge_last_residual"][valid_idx] = last_valid + ridge_last.predict(x_valid)

            ridge_z = fit_ridge_model(x_train, y_train - z_train, alpha=args.ridge_alpha)
            oof["ridge_z_residual"][valid_idx] = z_valid + ridge_z.predict(x_valid)

        if args.include_lgbm:
            last_train = frame.priors["last_known"].iloc[train_idx]
            last_valid = frame.priors["last_known"].iloc[valid_idx].to_numpy(float)
            z_train = frame.priors["z_anchor"].iloc[train_idx]
            z_valid = frame.priors["z_anchor"].iloc[valid_idx].to_numpy(float)

            lgbm_last = fit_lgbm_model(x_train, y_train - last_train, args)
            oof["lgbm_last_residual"][valid_idx] = last_valid + lgbm_last.predict(x_valid)

            lgbm_z = fit_lgbm_model(x_train, y_train - z_train, args)
            oof["lgbm_z_residual"][valid_idx] = z_valid + lgbm_z.predict(x_valid)

        if args.include_catboost:
            last_train = frame.priors["last_known"].iloc[train_idx]
            last_valid = frame.priors["last_known"].iloc[valid_idx].to_numpy(float)
            z_train = frame.priors["z_anchor"].iloc[train_idx]
            z_valid = frame.priors["z_anchor"].iloc[valid_idx].to_numpy(float)

            cat_last = fit_catboost_model(x_train, y_train - last_train, args)
            oof["catboost_last_residual"][valid_idx] = last_valid + cat_last.predict(x_valid)

            cat_z = fit_catboost_model(x_train, y_train - z_train, args)
            oof["catboost_z_residual"][valid_idx] = z_valid + cat_z.predict(x_valid)

        fold_result = {"fold": fold, "train_rows": int(len(train_idx)), "valid_rows": int(len(valid_idx))}
        for name in method_names:
            fold_result[name] = safe_float(rmse(y_valid.to_numpy(float), oof[name][valid_idx]))
        fold_rows.append(fold_result)
        print(json.dumps(fold_result), flush=True)

    results = {
        name: score_prediction(frame.y, pred, frame.groups)
        for name, pred in sorted(oof.items(), key=lambda item: rmse(frame.y.to_numpy(float), item[1]))
    }
    return {
        "rows": int(len(frame.y)),
        "wells": int(frame.groups.nunique()),
        "features": int(frame.x.shape[1]),
        "folds": fold_rows,
        "results": results,
        "best": next(iter(results.items())) if results else None,
        "args": {
            "folds": args.folds,
            "max_wells": args.max_wells,
            "well_stride": args.well_stride,
            "include_extra_numeric": args.include_extra_numeric,
            "include_prefix_ncc": args.include_prefix_ncc,
            "include_typewell_beam": args.include_typewell_beam,
            "include_ridge": args.include_ridge,
            "include_lgbm": args.include_lgbm,
            "include_catboost": args.include_catboost,
            "catboost_task_type": args.catboost_task_type,
            "catboost_devices": args.catboost_devices,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local train-tail CV scorer for ROGII candidate ideas."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-wells", type=int, default=80)
    parser.add_argument("--well-stride", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=204)
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument("--include-extra-numeric", action="store_true")
    parser.add_argument("--include-prefix-ncc", action="store_true")
    parser.add_argument("--include-typewell-beam", action="store_true")
    parser.add_argument("--include-ridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-lgbm", action="store_true")
    parser.add_argument("--include-catboost", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgbm-num-leaves", type=int, default=63)
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.03)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--catboost-task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--catboost-devices", default="0")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    frame = load_local_frame(args)
    result = run_cv(frame, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)
    print(json.dumps(result["best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
