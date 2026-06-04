from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from rogii_wellbore.data import scan_wells
from rogii_wellbore.features import load_training_frame
from rogii_wellbore.metrics import rmse


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_regressor(config: dict[str, Any]):
    model_name = config.get("baseline", {}).get("model", "lightgbm")
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(**config.get("baseline", {}).get("lgbm", {}))
        except ImportError:
            pass

    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(random_state=config["validation"]["random_state"])


def make_pipeline(config: dict[str, Any]) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", make_regressor(config)),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def target_mode(config: dict[str, Any]) -> str:
    return str(config.get("baseline", {}).get("target", "absolute"))


def select_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.select_dtypes(include=[np.number]).copy()


def select_train_pairs(config: dict[str, Any]):
    raw_dir = Path(config["paths"]["raw_dir"])
    pairs = [pair for pair in scan_wells(raw_dir) if pair.split in {"train", "unknown"}]
    max_wells = config.get("data", {}).get("max_wells")
    if max_wells:
        pairs = pairs[: int(max_wells)]
    return pairs


def cross_validate_baseline(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    pairs = select_train_pairs(config)
    x_raw, y, groups = load_training_frame(
        pairs,
        include_extra_numeric=bool(config.get("features", {}).get("include_extra_numeric", False)),
        include_prefix_ncc=bool(config.get("features", {}).get("include_prefix_ncc", False)),
        include_typewell_beam=bool(config.get("features", {}).get("include_typewell_beam", False)),
    )
    x = select_numeric_features(x_raw)

    n_splits = min(int(config["validation"]["n_splits"]), groups.nunique())
    if n_splits < 2:
        raise ValueError("Need at least two wells with target rows for GroupKFold.")

    fold_scores: list[dict[str, float | int]] = []
    oof = pd.Series(np.nan, index=y.index, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = make_pipeline(config)
        y_train = training_target(y.iloc[train_idx], x.iloc[train_idx], config)
        model.fit(x.iloc[train_idx], y_train)
        pred = restore_prediction(model.predict(x.iloc[valid_idx]), x.iloc[valid_idx], config)
        score = rmse(y.iloc[valid_idx].to_numpy(), pred)
        oof.iloc[valid_idx] = pred
        fold_scores.append(
            {
                "fold": fold,
                "rmse": score,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
            }
        )

    cv_score = rmse(y.to_numpy(), oof.to_numpy())
    return {
        "cv_rmse": cv_score,
        "folds": fold_scores,
        "n_rows": int(len(x)),
        "n_features": int(x.shape[1]),
        "n_wells": int(groups.nunique()),
        "features": list(x.columns),
    }


def train_full_baseline(config_path: Path) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    model_dir = Path(config["paths"]["model_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = select_train_pairs(config)
    x_raw, y, _ = load_training_frame(
        pairs,
        include_extra_numeric=bool(config.get("features", {}).get("include_extra_numeric", False)),
        include_prefix_ncc=bool(config.get("features", {}).get("include_prefix_ncc", False)),
        include_typewell_beam=bool(config.get("features", {}).get("include_typewell_beam", False)),
    )
    x = select_numeric_features(x_raw)
    model = make_pipeline(config)
    model.fit(x, training_target(y, x, config))

    artifact = {
        "model": model,
        "feature_names": list(x.columns),
        "target_mode": target_mode(config),
        "config": config,
    }
    model_path = model_dir / "baseline.joblib"
    joblib.dump(artifact, model_path)

    summary = {
        "model_path": str(model_path),
        "n_rows": int(len(x)),
        "n_features": int(x.shape[1]),
        "features": list(x.columns),
    }
    with (output_dir / "baseline_train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return model_path, summary


def training_target(y: pd.Series, x: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    if target_mode(config) == "residual_last_known":
        return y - x["last_known_tvt"]
    return y


def restore_prediction(prediction: np.ndarray, x: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    if target_mode(config) == "residual_last_known":
        return np.asarray(prediction, dtype=float) + x["last_known_tvt"].to_numpy(dtype=float)
    return np.asarray(prediction, dtype=float)
