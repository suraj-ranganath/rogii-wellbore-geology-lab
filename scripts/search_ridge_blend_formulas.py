from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rogii_wellbore.metrics import rmse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENTS = ROOT / "outputs/local_ridge_final_blend_components_80_s8p128.parquet"
DEFAULT_OUTPUT = ROOT / "outputs/ridge_blend_formula_search_80_s8p128.json"


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 2:
            raise ValueError(f"Bin count must be at least 2: {value}")
        values.append(value)
    if not values:
        raise ValueError("Expected at least one bin count")
    return values


def score(y: np.ndarray, pred: np.ndarray) -> float:
    finite = np.isfinite(y) & np.isfinite(pred)
    return rmse(y[finite], pred[finite])


def global_blend(y: np.ndarray, ridge: np.ndarray, selector: np.ndarray, clip: bool = True) -> float:
    diff = ridge - selector
    denom = float(np.dot(diff, diff))
    if denom <= 1e-12:
        return 0.5
    weight = float(np.dot(y - selector, diff) / denom)
    if clip:
        weight = float(np.clip(weight, 0.0, 1.0))
    return weight


def blend(weight: np.ndarray | float, ridge: np.ndarray, selector: np.ndarray) -> np.ndarray:
    return np.asarray(weight, dtype=float) * ridge + (1.0 - np.asarray(weight, dtype=float)) * selector


def split_rows(frame: pd.DataFrame, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["well"].astype(str).to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n_splits).split(frame, groups=groups))


def cv_global_weight(frame: pd.DataFrame, folds: int) -> dict[str, Any]:
    y = frame["truth"].to_numpy(float)
    ridge = frame["ridge_pp"].to_numpy(float)
    selector = frame["selector"].to_numpy(float)
    pred = np.zeros(len(frame), dtype=float)
    weights = []
    for train_idx, valid_idx in split_rows(frame, folds):
        weight = global_blend(y[train_idx], ridge[train_idx], selector[train_idx])
        weights.append(weight)
        pred[valid_idx] = blend(weight, ridge[valid_idx], selector[valid_idx])
    return {
        "rmse": score(y, pred),
        "weights": weights,
        "mean_weight": float(np.mean(weights)),
        "std_weight": float(np.std(weights)),
    }


def well_summary(frame: pd.DataFrame, feature: str) -> pd.Series:
    if feature == "abs_disagreement":
        values = (frame["ridge_pp"] - frame["selector"]).abs()
        return values.groupby(frame["well"].astype(str)).median()
    if feature == "signed_disagreement":
        values = frame["ridge_pp"] - frame["selector"]
        return values.groupby(frame["well"].astype(str)).median()
    if feature == "selector_score":
        score_cols = [col for col in ("sc8_sc", "sc15_sc", "sc25_sc") if col in frame.columns]
        values = frame[score_cols].max(axis=1) if score_cols else pd.Series(0.0, index=frame.index)
        return values.groupby(frame["well"].astype(str)).median()
    if feature not in frame.columns:
        raise KeyError(feature)
    return frame.groupby(frame["well"].astype(str))[feature].median(numeric_only=True)


def assign_bins(train_values: pd.Series, all_values: pd.Series, n_bins: int) -> pd.Series:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    cuts = np.unique(np.nanquantile(train_values.to_numpy(float), quantiles))
    if len(cuts) == 0:
        return pd.Series(0, index=all_values.index)
    return pd.Series(np.searchsorted(cuts, all_values.to_numpy(float), side="right"), index=all_values.index)


def cv_binned_weight(frame: pd.DataFrame, feature: str, n_bins: int, folds: int) -> dict[str, Any]:
    y = frame["truth"].to_numpy(float)
    ridge = frame["ridge_pp"].to_numpy(float)
    selector = frame["selector"].to_numpy(float)
    wells = frame["well"].astype(str)
    pred = np.zeros(len(frame), dtype=float)
    fold_meta = []
    feature_by_well = well_summary(frame, feature)
    for train_idx, valid_idx in split_rows(frame, folds):
        train_wells = set(wells.iloc[train_idx])
        train_values = feature_by_well.loc[list(train_wells)]
        bins_by_well = assign_bins(train_values, feature_by_well, n_bins)
        train_bins = wells.iloc[train_idx].map(bins_by_well).to_numpy(int)
        valid_bins = wells.iloc[valid_idx].map(bins_by_well).to_numpy(int)
        fallback = global_blend(y[train_idx], ridge[train_idx], selector[train_idx])
        bin_weights = {}
        for bin_id in range(n_bins):
            local_train = train_idx[train_bins == bin_id]
            if len(local_train) < 100:
                bin_weights[bin_id] = fallback
            else:
                bin_weights[bin_id] = global_blend(
                    y[local_train],
                    ridge[local_train],
                    selector[local_train],
                )
        valid_weights = np.array([bin_weights.get(int(bin_id), fallback) for bin_id in valid_bins])
        pred[valid_idx] = blend(valid_weights, ridge[valid_idx], selector[valid_idx])
        fold_meta.append(
            {
                "fallback": fallback,
                "bin_weights": {str(key): float(value) for key, value in bin_weights.items()},
            }
        )
    return {
        "rmse": score(y, pred),
        "feature": feature,
        "bins": n_bins,
        "folds": fold_meta,
    }


def row_feature(frame: pd.DataFrame, feature: str) -> pd.Series:
    if feature == "abs_disagreement":
        return (frame["ridge_pp"] - frame["selector"]).abs()
    if feature == "signed_disagreement":
        return frame["ridge_pp"] - frame["selector"]
    if feature == "selector_score":
        score_cols = [col for col in ("sc8_sc", "sc15_sc", "sc25_sc") if col in frame.columns]
        return frame[score_cols].max(axis=1) if score_cols else pd.Series(0.0, index=frame.index)
    if feature not in frame.columns:
        raise KeyError(feature)
    return frame[feature].astype(float)


def cv_row_binned_weight(frame: pd.DataFrame, feature: str, n_bins: int, folds: int) -> dict[str, Any]:
    y = frame["truth"].to_numpy(float)
    ridge = frame["ridge_pp"].to_numpy(float)
    selector = frame["selector"].to_numpy(float)
    pred = np.zeros(len(frame), dtype=float)
    values = row_feature(frame, feature).to_numpy(float)
    fold_meta = []
    for train_idx, valid_idx in split_rows(frame, folds):
        train_values = values[train_idx]
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        cuts = np.unique(np.nanquantile(train_values[np.isfinite(train_values)], quantiles))
        train_bins = np.searchsorted(cuts, values[train_idx], side="right")
        valid_bins = np.searchsorted(cuts, values[valid_idx], side="right")
        fallback = global_blend(y[train_idx], ridge[train_idx], selector[train_idx])
        bin_weights = {}
        for bin_id in range(n_bins):
            local_train = train_idx[train_bins == bin_id]
            if len(local_train) < 500:
                bin_weights[bin_id] = fallback
            else:
                bin_weights[bin_id] = global_blend(
                    y[local_train],
                    ridge[local_train],
                    selector[local_train],
                )
        valid_weights = np.array([bin_weights.get(int(bin_id), fallback) for bin_id in valid_bins])
        pred[valid_idx] = blend(valid_weights, ridge[valid_idx], selector[valid_idx])
        fold_meta.append(
            {
                "cuts": [float(value) for value in cuts],
                "fallback": fallback,
                "bin_weights": {str(key): float(value) for key, value in bin_weights.items()},
            }
        )
    return {
        "rmse": score(y, pred),
        "feature": feature,
        "bins": n_bins,
        "folds": fold_meta,
    }


def model_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["ridge_pp"] = frame["ridge_pp"].astype(float)
    out["selector"] = frame["selector"].astype(float)
    out["last_known"] = frame["last_known"].astype(float)
    out["ridge_minus_selector"] = frame["ridge_pp"].astype(float) - frame["selector"].astype(float)
    out["abs_ridge_minus_selector"] = out["ridge_minus_selector"].abs()
    for column in [
        "eval_len",
        "known_len",
        "md_since",
        "frac",
        "z",
        "sig_std",
        "beam_std_d",
        "pf_ancc_std",
        "sc8_sc",
        "sc15_sc",
        "sc25_sc",
        "sc_ens_d",
        "dense_std",
        "dense_dist",
        "form_std_d",
        "pfx_rmse",
        "ktvt_range",
        "ktvt_std",
    ]:
        if column in frame.columns:
            out[column] = frame[column].astype(float)
    if {"sc8_sc", "sc15_sc", "sc25_sc"}.issubset(out.columns):
        out["selector_score"] = out[["sc8_sc", "sc15_sc", "sc25_sc"]].max(axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def cv_residual_model(
    frame: pd.DataFrame,
    folds: int,
    base_weight: float,
    model_name: str,
) -> dict[str, Any]:
    y = frame["truth"].to_numpy(float)
    ridge = frame["ridge_pp"].to_numpy(float)
    selector = frame["selector"].to_numpy(float)
    base = blend(base_weight, ridge, selector)
    x = model_feature_frame(frame)
    pred = np.zeros(len(frame), dtype=float)
    fold_scores = []
    for train_idx, valid_idx in split_rows(frame, folds):
        if model_name == "ridge_residual":
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=80.0, random_state=0)),
                ]
            )
        elif model_name == "hgb_residual":
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "hgb",
                        HistGradientBoostingRegressor(
                            max_iter=120,
                            learning_rate=0.04,
                            max_leaf_nodes=15,
                            l2_regularization=20.0,
                            min_samples_leaf=800,
                            random_state=0,
                        ),
                    ),
                ]
            )
        elif model_name == "ridge_direct":
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=40.0, random_state=0)),
                ]
            )
        else:
            raise ValueError(model_name)

        target = y[train_idx] if model_name == "ridge_direct" else y[train_idx] - base[train_idx]
        model.fit(x.iloc[train_idx], target)
        local_pred = model.predict(x.iloc[valid_idx])
        pred[valid_idx] = local_pred if model_name == "ridge_direct" else base[valid_idx] + local_pred
        fold_scores.append(score(y[valid_idx], pred[valid_idx]))
    return {
        "rmse": score(y, pred),
        "fold_scores": fold_scores,
        "base_weight": base_weight,
        "model": model_name,
    }


def fixed_weight_scores(frame: pd.DataFrame, weights: list[float]) -> dict[str, Any]:
    y = frame["truth"].to_numpy(float)
    ridge = frame["ridge_pp"].to_numpy(float)
    selector = frame["selector"].to_numpy(float)
    return {
        f"fixed_w{int(round(weight * 1000)):03d}": {"rmse": score(y, blend(weight, ridge, selector))}
        for weight in weights
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search deployable Ridge/selector blend formulas.")
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--well-bin-counts", default="2,3,4")
    parser.add_argument("--row-bin-counts", default="2,3,4,5,6,8,10,11,12,13,14,15,16,17,18,19,20")
    args = parser.parse_args()

    frame = pd.read_parquet(args.components)
    y = frame["truth"].to_numpy(float)
    ridge = frame["ridge_pp"].to_numpy(float)
    selector = frame["selector"].to_numpy(float)
    global_weight = global_blend(y, ridge, selector)

    results: dict[str, Any] = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "fixed": fixed_weight_scores(
            frame,
            [0.40, 0.42, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, global_weight],
        ),
        "global_in_sample": {
            "weight": global_weight,
            "rmse": score(y, blend(global_weight, ridge, selector)),
        },
        "global_cv": cv_global_weight(frame, args.folds),
        "binned_cv": {},
        "row_binned_cv": {},
        "model_cv": {},
    }

    well_bin_counts = parse_int_list(args.well_bin_counts)
    row_bin_counts = parse_int_list(args.row_bin_counts)

    for feature in [
        "abs_disagreement",
        "signed_disagreement",
        "selector_score",
        "eval_len",
        "known_len",
        "sig_std",
        "beam_std_d",
        "pf_ancc_std",
        "dense_std",
        "pfx_rmse",
        "ktvt_range",
    ]:
        for n_bins in well_bin_counts:
            key = f"{feature}_q{n_bins}"
            try:
                results["binned_cv"][key] = cv_binned_weight(frame, feature, n_bins, args.folds)
            except Exception as exc:
                results["binned_cv"][key] = {"error": repr(exc)}

    for feature in [
        "frac",
        "md_since",
        "abs_disagreement",
        "signed_disagreement",
        "selector_score",
        "sig_std",
        "beam_std_d",
        "pf_ancc_std",
        "dense_std",
        "pfx_rmse",
        "z",
    ]:
        for n_bins in row_bin_counts:
            key = f"{feature}_rowq{n_bins}"
            try:
                results["row_binned_cv"][key] = cv_row_binned_weight(
                    frame,
                    feature,
                    n_bins,
                    args.folds,
                )
            except Exception as exc:
                results["row_binned_cv"][key] = {"error": repr(exc)}

    base_weight = results["global_cv"]["mean_weight"]
    for model_name in ("ridge_residual", "ridge_direct", "hgb_residual"):
        results["model_cv"][model_name] = cv_residual_model(
            frame,
            folds=args.folds,
            base_weight=base_weight,
            model_name=model_name,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    print(f"rows={results['rows']} wells={results['wells']}")
    print(
        f"global in-sample weight={results['global_in_sample']['weight']:.4f} "
        f"rmse={results['global_in_sample']['rmse']:.4f}"
    )
    print(
        f"global CV mean_weight={results['global_cv']['mean_weight']:.4f} "
        f"rmse={results['global_cv']['rmse']:.4f}"
    )
    rows = []
    for section in ("fixed", "binned_cv", "row_binned_cv", "model_cv"):
        for name, result in results[section].items():
            if "rmse" in result:
                rows.append((result["rmse"], section, name))
    for value, section, name in sorted(rows)[:20]:
        print(f"{section:10s} {name:32s} rmse={value:.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
