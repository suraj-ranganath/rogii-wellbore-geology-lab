from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

from rogii_wellbore.metrics import rmse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction"
DEFAULT_ARTIFACT_DIR = ROOT / "data/external/ravaghi/wellbore-geology-prediction-artifacts"
DEFAULT_KERNEL = ROOT / "kaggle/kernels/ravaghi_ridge_w040/ravaghi_ridge_w040.py"
DEFAULT_OUTPUT = ROOT / "outputs/local_ridge_final_blend_cv.json"

ARTIFACT_DATASET = "ravaghi/wellbore-geology-prediction-artifacts"
MODEL_FILES = {
    "lightgbm-1": "lgbmregressor_trainer_20260526182612.pkl",
    "lightgbm-2": "lgbmregressor_trainer_20260526190415.pkl",
    "lightgbm-3": "lgbmregressor_trainer_20260526192806.pkl",
    "catboost-1": "catboostregressor_trainer_20260526193740.pkl",
    "catboost-2": "catboostregressor_trainer_20260526194838.pkl",
}
REMOTE_MODEL_FILES = {
    "lightgbm-1": "models/lightgbm-1/lgbmregressor_trainer_20260526182612.pkl",
    "lightgbm-2": "models/lightgbm-2/lgbmregressor_trainer_20260526190415.pkl",
    "lightgbm-3": "models/lightgbm-3/lgbmregressor_trainer_20260526192806.pkl",
    "catboost-1": "models/catboost-1/catboostregressor_trainer_20260526193740.pkl",
    "catboost-2": "models/catboost-2/catboostregressor_trainer_20260526194838.pkl",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    ridge_weight: float
    selector_weight: float
    postprocess_mode: str = "none"
    prefix_max_weight: float = 0.0
    formation_max_weight: float = 0.0


CANDIDATES = [
    Candidate("ridge_w020", 0.20, 0.80),
    Candidate("ridge_w025", 0.25, 0.75),
    Candidate("ridge_w030", 0.30, 0.70),
    Candidate("ridge_w035", 0.35, 0.65),
    Candidate("ridge_w040", 0.40, 0.60),
    Candidate("ridge_w040_pf40", 0.40, 0.60),
    Candidate("ridge_w040_selector070", 0.30, 0.70),
    Candidate("ridge_w040_selector080", 0.20, 0.80),
    Candidate("ridge_w040_prefix_gate", 0.32, 0.68, "prefix", prefix_max_weight=0.12),
    Candidate(
        "ridge_w040_formprefix_gate",
        0.32,
        0.68,
        "formprefix",
        prefix_max_weight=0.08,
        formation_max_weight=0.08,
    ),
]


def missing_model_files(artifact_dir: Path) -> list[str]:
    return [name for name, filename in MODEL_FILES.items() if not (artifact_dir / filename).is_file()]


def download_model_files(artifact_dir: Path, names: list[str]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        remote_path = REMOTE_MODEL_FILES[name]
        cmd = [
            "uv",
            "run",
            "kaggle",
            "datasets",
            "download",
            "-d",
            ARTIFACT_DATASET,
            "-f",
            remote_path,
            "-p",
            str(artifact_dir),
            "--unzip",
        ]
        print("$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def load_ridge_namespace(kernel_path: Path, data_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    source = kernel_path.read_text()
    start = source.index("# %% cell 2")
    stop = source.index("\n# %% cell 7")
    prefix = source[start:stop]
    prefix = prefix.replace("@njit(cache=True)", "@njit(cache=False)")
    prefix = prefix.replace(
        'Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")',
        f"Path({str(data_dir)!r})",
    )
    prefix = prefix.replace(
        'Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")',
        f"Path({str(artifact_dir)!r})",
    )
    namespace: dict[str, Any] = {"__file__": str(kernel_path), "_sys": sys}
    exec(compile(prefix, str(kernel_path), "exec"), namespace)
    params_start = source.index("# %% cell 9")
    params_stop = source.index("\n# %% cell 10")
    params_block = source[params_start:params_stop]
    exec(compile(params_block, str(kernel_path), "exec"), namespace)
    pp_start = source.index("# %% cell 19")
    pp_stop = source.index("\n# %% cell 20")
    pp_block = source[pp_start:pp_stop]
    exec(compile(pp_block, str(kernel_path), "exec"), namespace)
    namespace["CFG"].dataset_path = data_dir
    namespace["CFG"].artifacts_path = artifact_dir
    return namespace


def reconstruct_train_targets(data_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    train_paths = sorted((data_dir / "train").glob("*__horizontal_well.csv"))
    for path in train_paths:
        well = path.stem.replace("__horizontal_well", "")
        typewell_path = path.parent / f"{well}__typewell.csv"
        if not typewell_path.is_file():
            continue
        try:
            horizontal = pd.read_csv(path, usecols=["TVT_input", "TVT", "MD", "Z"])
            typewell = pd.read_csv(typewell_path, usecols=["TVT"])
        except Exception:
            continue
        known = horizontal["TVT_input"].notna()
        eval_mask = horizontal["TVT_input"].isna()
        if not eval_mask.any() or known.sum() < 10 or typewell["TVT"].nunique(dropna=True) < 3:
            continue
        if horizontal["TVT"].isna().all():
            continue
        last_known_tvt = float(horizontal.loc[known, "TVT_input"].iloc[-1])
        eval_frame = horizontal.loc[eval_mask, ["TVT", "MD", "Z"]].copy()
        eval_frame["well"] = well
        eval_frame["row_idx"] = eval_frame.index.astype(int)
        eval_frame["id"] = [f"{well}_{idx}" for idx in eval_frame.index]
        eval_frame["last_known_tvt"] = last_known_tvt
        eval_frame["target"] = eval_frame["TVT"].astype(float) - last_known_tvt
        rows.append(eval_frame.reset_index(drop=True))
    if not rows:
        raise ValueError(f"No usable training target rows under {data_dir}")
    return pd.concat(rows, ignore_index=True)


def load_base_oof_predictions(artifact_dir: Path) -> pd.DataFrame:
    columns = {}
    for name, filename in MODEL_FILES.items():
        trainer = joblib.load(artifact_dir / filename)
        columns[name] = np.asarray(trainer.oof_preds, dtype=float)
    lengths = {name: len(values) for name, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Base trainer OOF lengths disagree: {lengths}")
    return pd.DataFrame(columns)


def fit_ridge_oof(base_oof: pd.DataFrame, target_frame: pd.DataFrame, ns: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    if len(base_oof) != len(target_frame):
        raise ValueError(
            f"Base OOF length {len(base_oof)} does not match reconstructed targets {len(target_frame)}"
        )
    y = target_frame["target"].astype(float).reset_index(drop=True)
    groups = target_frame["well"].astype(str).reset_index(drop=True)
    params = dict(ns["ridge_params"])
    preds = np.zeros(len(y), dtype=float)
    fold_scores = []
    y_min = float(y.min())
    y_max = float(y.max())
    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(base_oof, y, groups=groups),
        start=1,
    ):
        model = clone(Ridge(**params))
        model.fit(base_oof.iloc[train_idx], y.iloc[train_idx])
        fold_pred = model.predict(base_oof.iloc[valid_idx]).clip(y_min, y_max)
        preds[valid_idx] = fold_pred
        fold_score = root_mean_squared_error(y.iloc[valid_idx], fold_pred)
        fold_scores.append(float(fold_score))
        print(f"ridge meta fold {fold}: {fold_score:.4f}", flush=True)
    overall = root_mean_squared_error(y, preds)
    info = {
        "rmse": float(overall),
        "fold_scores": fold_scores,
        "mean_fold": float(np.mean(fold_scores)),
        "std_fold": float(np.std(fold_scores)),
    }
    return pd.Series(preds, index=target_frame["id"].astype(str), name="ridge_residual"), info


def selected_train_paths(data_dir: Path, max_wells: int, well_stride: int) -> list[Path]:
    paths = sorted((data_dir / "train").glob("*__horizontal_well.csv"))
    if well_stride > 1:
        paths = paths[::well_stride]
    if max_wells:
        paths = paths[:max_wells]
    return paths


def build_selected_feature_frame(ns: dict[str, Any], paths: list[Path], n_jobs: int) -> pd.DataFrame:
    ns["NCPU"] = max(1, int(n_jobs))
    frame = ns["build_dataset"](paths, is_train=True, label="local-ridge-final")
    if frame.empty:
        raise ValueError("Selected Ridge feature frame is empty")
    return frame.reset_index(drop=True)


def apply_ridge_pp(ns: dict[str, Any], frame: pd.DataFrame, ridge_residual_by_id: pd.Series) -> pd.Series:
    ids = frame["id"].astype(str)
    missing = ids[~ids.isin(ridge_residual_by_id.index)]
    if len(missing):
        raise ValueError(f"Missing Ridge OOF residuals for {len(missing)} selected rows")
    ridge_residual = ridge_residual_by_id.reindex(ids).to_numpy(float)
    base = frame["last_known_tvt"].to_numpy(float)
    pf_delta = frame["pf_ancc"].to_numpy(float) - base
    delta = ns["apply_pp"](frame, ridge_residual, pf_delta, **ns["pp_params"])
    smooth_frame = frame[["id", "well"]].copy()
    smooth_frame["pred"] = base + delta
    smooth_frame = ns["sg_smooth"](smooth_frame, "pred")
    return pd.Series(smooth_frame["pred"].to_numpy(float), index=ids, name="ridge_pp")


def selector_predictions_for_paths(
    ns: dict[str, Any],
    paths: list[Path],
    n_particles: int,
    n_seeds: int,
    progress: int,
) -> pd.Series:
    rows: list[pd.Series] = []
    for idx, path in enumerate(paths, start=1):
        well = path.stem.replace("__horizontal_well", "")
        typewell_path = path.parent / f"{well}__typewell.csv"
        if not typewell_path.is_file():
            continue
        horizontal = pd.read_csv(path)
        typewell = pd.read_csv(typewell_path)
        eval_mask = horizontal["TVT_input"].isna().to_numpy()
        if not eval_mask.any():
            continue
        pf_by_scale = ns["run_pf_lik_ensemble_scales"](
            horizontal,
            typewell,
            n_particles=n_particles,
            n_seeds=n_seeds,
        )
        try:
            beam = ns["run_beam_ensemble"](horizontal, typewell)
        except Exception:
            beam = pf_by_scale["pf_scale_8"].copy()
        _, variant, _, _ = ns["selector_well_code"](horizontal)
        known = horizontal["TVT_input"].dropna()
        last_known_tvt = float(known.iloc[-1]) if len(known) else float(np.nanmean(beam))
        selector = ns["apply_selector_variant"](variant, pf_by_scale, beam, last_known_tvt)
        ids = [f"{well}_{row_idx}" for row_idx in np.flatnonzero(eval_mask)]
        rows.append(pd.Series(selector[eval_mask].astype(float), index=ids))
        if progress and idx % progress == 0:
            print(f"selector replayed {idx}/{len(paths)} wells", flush=True)
    if not rows:
        raise ValueError("No selector predictions were generated")
    return pd.concat(rows).rename("selector")


def score_prediction(y_true: pd.Series, pred: pd.Series, groups: pd.Series) -> dict[str, Any]:
    aligned = pred.reindex(y_true.index).to_numpy(float)
    truth = y_true.to_numpy(float)
    finite = np.isfinite(truth) & np.isfinite(aligned)
    if not finite.any():
        raise ValueError(f"No finite rows for {pred.name}")
    well_rows = []
    groups_aligned = groups.reindex(y_true.index)
    for well, group_ids in groups_aligned.groupby(groups_aligned).groups.items():
        local_index = list(group_ids)
        local_truth = y_true.reindex(local_index).to_numpy(float)
        local_pred = pred.reindex(local_index).to_numpy(float)
        local_finite = np.isfinite(local_truth) & np.isfinite(local_pred)
        if local_finite.any():
            well_rows.append(
                {
                    "well": str(well),
                    "rmse": rmse(local_truth[local_finite], local_pred[local_finite]),
                    "rows": int(local_finite.sum()),
                }
            )
    well_values = [row["rmse"] for row in well_rows]
    return {
        "rmse": rmse(truth[finite], aligned[finite]),
        "rows": int(finite.sum()),
        "wells": int(len(well_rows)),
        "well_rmse_mean": float(np.mean(well_values)) if well_values else None,
        "well_rmse_median": float(np.median(well_values)) if well_values else None,
        "worst_wells": sorted(well_rows, key=lambda row: row["rmse"], reverse=True)[:10],
    }


def apply_candidate_postprocess(candidate: Candidate, frame: pd.DataFrame, base_pred: pd.Series) -> pd.Series:
    ids = frame["id"].astype(str)
    values = base_pred.reindex(ids).to_numpy(float)

    if candidate.postprocess_mode in {"prefix", "formprefix"} and candidate.prefix_max_weight > 0:
        required = [
            "last_known_tvt",
            "sc_ens_d",
            "sc8_sc",
            "sc15_sc",
            "sc25_sc",
            "known_len",
            "sig_std",
        ]
        if set(required).issubset(frame.columns):
            prefix_tvt = frame["last_known_tvt"].to_numpy(float) + frame["sc_ens_d"].to_numpy(float)
            prefix_score = frame[["sc8_sc", "sc15_sc", "sc25_sc"]].max(axis=1).to_numpy(float)
            conf = np.clip((prefix_score - 0.58) / 0.22, 0.0, 1.0)
            conf *= np.clip(frame["known_len"].to_numpy(float) / 260.0, 0.0, 1.0)
            conf *= np.clip(18.0 / (np.abs(frame["sig_std"].to_numpy(float)) + 18.0), 0.0, 1.0)
            weights = candidate.prefix_max_weight * conf
            values = (1.0 - weights) * values + weights * prefix_tvt

    if candidate.postprocess_mode == "formprefix" and candidate.formation_max_weight > 0:
        required = [
            "last_known_tvt",
            "tvt_dense_d",
            "dense_std",
            "form_std_d",
            "dense_dist",
        ]
        if set(required).issubset(frame.columns):
            form_tvt = frame["last_known_tvt"].to_numpy(float) + frame["tvt_dense_d"].to_numpy(float)
            conf = np.clip(1.0 - np.abs(frame["dense_std"].to_numpy(float)) / 90.0, 0.0, 1.0)
            conf *= np.clip(1.0 - np.abs(frame["form_std_d"].to_numpy(float)) / 140.0, 0.0, 1.0)
            conf *= np.clip(1.0 / (1.0 + np.abs(frame["dense_dist"].to_numpy(float)) / 6.0), 0.0, 1.0)
            weights = candidate.formation_max_weight * conf
            values = (1.0 - weights) * values + weights * form_tvt

    return pd.Series(values, index=ids, name=candidate.name)


def candidate_scores(
    frame: pd.DataFrame,
    ridge_pp: pd.Series,
    selector: pd.Series,
    ridge_weight_grid: list[float],
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.Series]]:
    ids = frame["id"].astype(str)
    y_true = pd.Series(
        frame["last_known_tvt"].to_numpy(float) + frame["target"].to_numpy(float),
        index=ids,
        name="truth",
    )
    groups = pd.Series(frame["well"].astype(str).to_numpy(), index=ids)
    predictions: dict[str, pd.Series] = {
        "last_known": pd.Series(frame["last_known_tvt"].to_numpy(float), index=ids, name="last_known"),
        "ridge_pp": ridge_pp.reindex(ids).rename("ridge_pp"),
        "selector": selector.reindex(ids).rename("selector"),
    }
    for candidate in CANDIDATES:
        raw = (
            candidate.ridge_weight * predictions["ridge_pp"].to_numpy(float)
            + candidate.selector_weight * predictions["selector"].to_numpy(float)
        )
        raw_series = pd.Series(raw, index=ids, name=f"{candidate.name}_raw")
        predictions[candidate.name] = apply_candidate_postprocess(candidate, frame, raw_series)
    for weight in ridge_weight_grid:
        selector_weight = 1.0 - weight
        name = f"grid_ridge_w{int(round(weight * 1000)):03d}"
        predictions[name] = pd.Series(
            weight * predictions["ridge_pp"].to_numpy(float)
            + selector_weight * predictions["selector"].to_numpy(float),
            index=ids,
            name=name,
        )
    scores = {name: score_prediction(y_true, pred, groups) for name, pred in predictions.items()}
    return dict(sorted(scores.items(), key=lambda item: item[1]["rmse"])), predictions


def parse_grid(raw: str) -> list[float]:
    if not raw:
        return []
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Ridge weight outside [0, 1]: {value}")
        values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score final Ridge/PF selector blend variants on train hidden tails."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--kernel-code", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-wells", type=int, default=20)
    parser.add_argument("--well-stride", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--selector-particles", type=int, default=128)
    parser.add_argument("--selector-seeds", type=int, default=8)
    parser.add_argument("--ridge-weight-grid", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--progress", type=int, default=5)
    args = parser.parse_args()

    missing = missing_model_files(args.artifact_dir)
    if missing:
        if not args.download_models:
            raise FileNotFoundError(
                f"Missing artifact trainers {missing} under {args.artifact_dir}. "
                "Rerun with --download-models to fetch the public Kaggle artifact models."
            )
        download_model_files(args.artifact_dir, missing)

    start = time.time()
    ns = load_ridge_namespace(args.kernel_code, args.data_dir, args.artifact_dir)
    print("loaded Ridge namespace", flush=True)

    target_frame = reconstruct_train_targets(args.data_dir)
    base_oof = load_base_oof_predictions(args.artifact_dir)
    ridge_residual, ridge_meta = fit_ridge_oof(base_oof, target_frame, ns)
    print(f"ridge meta overall={ridge_meta['rmse']:.4f}", flush=True)

    paths = selected_train_paths(args.data_dir, args.max_wells, args.well_stride)
    feature_frame = build_selected_feature_frame(ns, paths, args.n_jobs)
    ridge_pp = apply_ridge_pp(ns, feature_frame, ridge_residual)
    selector = selector_predictions_for_paths(
        ns,
        paths,
        n_particles=args.selector_particles,
        n_seeds=args.selector_seeds,
        progress=args.progress,
    )
    scores, _ = candidate_scores(
        feature_frame,
        ridge_pp,
        selector,
        ridge_weight_grid=parse_grid(args.ridge_weight_grid),
    )

    result = {
        "args": {
            "max_wells": args.max_wells,
            "well_stride": args.well_stride,
            "selector_particles": args.selector_particles,
            "selector_seeds": args.selector_seeds,
            "ridge_weight_grid": parse_grid(args.ridge_weight_grid),
        },
        "elapsed_seconds": float(time.time() - start),
        "ridge_meta": ridge_meta,
        "feature_rows": int(len(feature_frame)),
        "feature_wells": int(feature_frame["well"].nunique()),
        "results": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\nFinal Ridge/PF blend proxy ({result['feature_wells']} wells):")
    for name, row in list(scores.items())[:20]:
        print(
            f"{name:28s} rmse={row['rmse']:.4f} "
            f"median_well={row['well_rmse_median']:.4f} wells={row['wells']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
