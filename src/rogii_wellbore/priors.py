from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii_wellbore.data import read_csv, scan_wells
from rogii_wellbore.features import build_horizontal_features, canonicalize, prediction_mask
from rogii_wellbore.metrics import rmse

PRIOR_COLUMNS = ("last_known_tvt", "linear_tvt_prior")


def evaluate_priors(data_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    pairs = [pair for pair in scan_wells(data_dir) if pair.split == "train"]
    totals = {
        name: {"sse": 0.0, "n": 0, "well_scores": []}
        for name in PRIOR_COLUMNS
    }

    for pair in pairs:
        horizontal_raw = read_csv(pair.horizontal_path)
        horizontal = canonicalize(horizontal_raw)
        if "tvt" not in horizontal:
            continue
        mask = prediction_mask(horizontal, require_target=True)
        if not mask.any():
            continue

        features = build_horizontal_features(horizontal_raw)
        y_true = pd.to_numeric(horizontal.loc[mask, "tvt"], errors="coerce").to_numpy(dtype=float)
        for name in PRIOR_COLUMNS:
            y_pred = pd.to_numeric(features.loc[mask, name], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(y_true) & np.isfinite(y_pred)
            if not finite.any():
                continue
            errors = y_true[finite] - y_pred[finite]
            score = rmse(y_true[finite], y_pred[finite])
            totals[name]["sse"] += float(np.sum(errors**2))
            totals[name]["n"] += int(finite.sum())
            totals[name]["well_scores"].append(
                {
                    "well_id": pair.well_id,
                    "rmse": score,
                    "rows": int(finite.sum()),
                }
            )

    metrics: dict[str, Any] = {"n_wells": len(pairs), "priors": {}}
    for name, values in totals.items():
        well_scores = values["well_scores"]
        rmse_values = [item["rmse"] for item in well_scores]
        metrics["priors"][name] = {
            "rmse": float(np.sqrt(values["sse"] / values["n"])) if values["n"] else None,
            "rows": values["n"],
            "wells": len(well_scores),
            "well_rmse_mean": float(np.mean(rmse_values)) if rmse_values else None,
            "well_rmse_median": float(np.median(rmse_values)) if rmse_values else None,
            "worst_wells": sorted(well_scores, key=lambda item: item["rmse"], reverse=True)[:10],
        }
    return metrics
