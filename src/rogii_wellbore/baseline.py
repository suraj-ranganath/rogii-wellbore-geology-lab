from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from rogii_wellbore.data import scan_wells
from rogii_wellbore.features import load_training_frame


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
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", make_regressor(config)),
        ]
    )


def select_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.select_dtypes(include=[np.number]).copy()


def cross_validate_baseline(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_dir"])
    pairs = [pair for pair in scan_wells(raw_dir) if pair.split in {"train", "unknown"}]
    x_raw, y, groups = load_training_frame(pairs)
    x = select_numeric_features(x_raw)

    n_splits = min(int(config["validation"]["n_splits"]), groups.nunique())
    if n_splits < 2:
        raise ValueError("Need at least two wells with target rows for GroupKFold.")

    fold_scores: list[dict[str, float | int]] = []
    oof = pd.Series(np.nan, index=y.index, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = make_pipeline(config)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[valid_idx])
        score = float(mean_squared_error(y.iloc[valid_idx], pred, squared=False))
        oof.iloc[valid_idx] = pred
        fold_scores.append(
            {
                "fold": fold,
                "rmse": score,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
            }
        )

    cv_score = float(mean_squared_error(y, oof, squared=False))
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
    raw_dir = Path(config["paths"]["raw_dir"])
    model_dir = Path(config["paths"]["model_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [pair for pair in scan_wells(raw_dir) if pair.split in {"train", "unknown"}]
    x_raw, y, _ = load_training_frame(pairs)
    x = select_numeric_features(x_raw)
    model = make_pipeline(config)
    model.fit(x, y)

    artifact = {
        "model": model,
        "feature_names": list(x.columns),
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
