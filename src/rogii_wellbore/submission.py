from __future__ import annotations

from pathlib import Path

import joblib

from rogii_wellbore.data import find_sample_submission, read_csv, scan_wells
from rogii_wellbore.features import build_horizontal_features


def parse_submission_id(submission_id: str) -> tuple[str, int]:
    well_id, row_idx = submission_id.rsplit("_", 1)
    return well_id, int(row_idx)


def make_model_submission(data_dir: Path, model_path: Path, output_path: Path) -> Path:
    data_dir = Path(data_dir)
    model_path = Path(model_path)
    output_path = Path(output_path)

    sample_submission_path = find_sample_submission(data_dir)
    if sample_submission_path is None:
        raise FileNotFoundError(f"No sample submission found under {data_dir}.")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    target_mode = artifact.get("target_mode", "absolute")

    predictions: dict[str, float] = {}
    pairs = [pair for pair in scan_wells(data_dir) if pair.split == "test"]
    for pair in pairs:
        horizontal = read_csv(pair.horizontal_path)
        typewell = read_csv(pair.typewell_path) if pair.typewell_path is not None else None
        features = build_horizontal_features(horizontal, typewell)
        numeric = features.select_dtypes(include=["number"]).reindex(columns=feature_names)
        pred = model.predict(numeric)
        if target_mode == "residual_last_known":
            pred = pred + numeric["last_known_tvt"].to_numpy(dtype=float)
        for row_idx, value in enumerate(pred):
            predictions[f"{pair.well_id}_{row_idx}"] = float(value)

    submission = read_csv(sample_submission_path)
    missing = sorted(set(submission["id"]) - set(predictions))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing predictions for {len(missing)} submission ids: {preview}")

    submission["tvt"] = submission["id"].map(predictions).astype(float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path
