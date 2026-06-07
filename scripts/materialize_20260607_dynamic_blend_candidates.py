from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from rogii_wellbore.metrics import rmse

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "kaggle" / "kernels" / "ravaghi_ridge_w040"
BASE_CODE = BASE_DIR / "ravaghi_ridge_w040.py"
OUT_ROOT = ROOT / "kaggle" / "kernels"
COMPONENTS = ROOT / "outputs" / "local_ridge_final_blend_components_80_s32p256.parquet"


@dataclass(frozen=True)
class DynamicCandidate:
    dirname: str
    kernel_id: str
    title: str
    code_file: str
    profile: str
    bins: int
    shrink: float = 0.0
    final_seeds: int = 32


CANDIDATES = [
    DynamicCandidate(
        dirname="ridge_w040_zq08",
        kernel_id="surajranganath17/rogii-ridge-w040-zq08",
        title="ROGII Ridge W040 ZQ08",
        code_file="ridge_w040_zq08.py",
        profile="w040_zq08",
        bins=8,
    ),
    DynamicCandidate(
        dirname="ridge_w040_zq11",
        kernel_id="surajranganath17/rogii-ridge-w040-zq11",
        title="ROGII Ridge W040 ZQ11",
        code_file="ridge_w040_zq11.py",
        profile="w040_zq11",
        bins=11,
    ),
    DynamicCandidate(
        dirname="ridge_w040_zq12",
        kernel_id="surajranganath17/rogii-ridge-w040-zq12",
        title="ROGII Ridge W040 ZQ12",
        code_file="ridge_w040_zq12.py",
        profile="w040_zq12",
        bins=12,
    ),
    DynamicCandidate(
        dirname="ridge_w040_zq12_shr1000",
        kernel_id="surajranganath17/rogii-ridge-w040-zq12-shr1000",
        title="ROGII Ridge W040 ZQ12 Shr1000",
        code_file="ridge_w040_zq12_shr1000.py",
        profile="w040_zq12_shr1000",
        bins=12,
        shrink=1000.0,
    ),
    DynamicCandidate(
        dirname="ridge_w040_zq13",
        kernel_id="surajranganath17/rogii-ridge-w040-zq13",
        title="ROGII Ridge W040 ZQ13",
        code_file="ridge_w040_zq13.py",
        profile="w040_zq13",
        bins=13,
    ),
]


ORIGINAL_FINAL_BLOCK = """sub = (
    sub_1.merge(sub_2, on='id', suffixes=('_1', '_2'))
       .assign(tvt=lambda x: 0.40 * x['tvt_1'] + 0.60 * x['tvt_2'])
       [['id', 'tvt']]
)
sub.to_csv("submission.csv", index=False)
sub
"""


def score(y: np.ndarray, pred: np.ndarray) -> float:
    finite = np.isfinite(y) & np.isfinite(pred)
    return rmse(y[finite], pred[finite])


def blend(weight: np.ndarray | float, ridge: np.ndarray, selector: np.ndarray) -> np.ndarray:
    weight = np.asarray(weight, dtype=float)
    return weight * ridge + (1.0 - weight) * selector


def global_blend(y: np.ndarray, ridge: np.ndarray, selector: np.ndarray, idx: np.ndarray) -> float:
    diff = ridge[idx] - selector[idx]
    denom = float(np.dot(diff, diff))
    if denom <= 1e-12:
        return 0.5
    weight = float(np.dot(y[idx] - selector[idx], diff) / denom)
    return float(np.clip(weight, 0.0, 1.0))


def fit_z_bins(
    frame: pd.DataFrame,
    bins: int,
    shrink: float,
    min_rows: int = 500,
) -> dict[str, Any]:
    y = frame["truth"].to_numpy(float)
    ridge = frame["ridge_pp"].to_numpy(float)
    selector = frame["selector"].to_numpy(float)
    z = frame["z"].to_numpy(float)
    all_idx = np.arange(len(frame))
    fallback = global_blend(y, ridge, selector, all_idx)
    cuts = np.unique(np.nanquantile(z[np.isfinite(z)], np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    assigned = np.searchsorted(cuts, z, side="right")
    weights = []
    counts = []
    for bin_id in range(bins):
        local_idx = all_idx[assigned == bin_id]
        counts.append(int(len(local_idx)))
        if len(local_idx) < min_rows:
            weight = fallback
        else:
            weight = global_blend(y, ridge, selector, local_idx)
            if shrink > 0:
                weight = (len(local_idx) * weight + shrink * fallback) / (len(local_idx) + shrink)
            weight = float(np.clip(weight, 0.0, 1.0))
        weights.append(weight)
    pred = blend(np.asarray(weights)[assigned], ridge, selector)
    return {
        "cuts": [float(value) for value in cuts],
        "weights": [float(value) for value in weights],
        "counts": counts,
        "fallback_weight": float(fallback),
        "fallback_z": float(np.nanmedian(z)),
        "in_sample_rmse": float(score(y, pred)),
    }


def cv_z_bins(
    frame: pd.DataFrame,
    bins: int,
    shrink: float,
    folds: int = 5,
    min_rows: int = 500,
) -> dict[str, Any]:
    y = frame["truth"].to_numpy(float)
    ridge = frame["ridge_pp"].to_numpy(float)
    selector = frame["selector"].to_numpy(float)
    z = frame["z"].to_numpy(float)
    groups = frame["well"].astype(str).to_numpy()
    pred = np.zeros(len(frame), dtype=float)
    fold_scores = []
    split_rows = GroupKFold(n_splits=folds).split(frame, groups=groups)
    for train_idx, valid_idx in split_rows:
        train_z = z[train_idx]
        cuts = np.unique(
            np.nanquantile(train_z[np.isfinite(train_z)], np.linspace(0.0, 1.0, bins + 1)[1:-1])
        )
        train_bins = np.searchsorted(cuts, z[train_idx], side="right")
        valid_bins = np.searchsorted(cuts, z[valid_idx], side="right")
        fallback = global_blend(y, ridge, selector, train_idx)
        weights = []
        for bin_id in range(bins):
            local_idx = train_idx[train_bins == bin_id]
            if len(local_idx) < min_rows:
                weight = fallback
            else:
                weight = global_blend(y, ridge, selector, local_idx)
                if shrink > 0:
                    weight = (len(local_idx) * weight + shrink * fallback) / (len(local_idx) + shrink)
                weight = float(np.clip(weight, 0.0, 1.0))
            weights.append(weight)
        valid_weights = np.asarray(weights)[np.clip(valid_bins, 0, len(weights) - 1)]
        pred[valid_idx] = blend(valid_weights, ridge[valid_idx], selector[valid_idx])
        fold_scores.append(float(score(y[valid_idx], pred[valid_idx])))
    return {
        "cv_rmse": float(score(y, pred)),
        "fold_scores": fold_scores,
    }


def array_literal(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.12g}" for value in values) + "]"


def final_block(candidate: DynamicCandidate, spec: dict[str, Any]) -> str:
    cuts = array_literal(spec["cuts"])
    weights = array_literal(spec["weights"])
    fallback_weight = spec["fallback_weight"]
    fallback_z = spec["fallback_z"]
    return f"""CANDIDATE_NAME = {candidate.profile!r}
Z_BLEND_BINS = {candidate.bins}
Z_BLEND_SHRINK = {candidate.shrink:.8f}
Z_BLEND_CUTS = np.array({cuts}, dtype=float)
Z_BLEND_WEIGHTS = np.array({weights}, dtype=float)
Z_BLEND_FALLBACK_WEIGHT = {fallback_weight:.12g}
Z_BLEND_FALLBACK_Z = {fallback_z:.12g}

_blend = sub_1.merge(sub_2, on='id', suffixes=('_1', '_2'))
_z = test_df[['id', 'z']].copy()
_blend = _blend.merge(_z, on='id', how='left')
_z_values = _blend['z'].to_numpy(float)
_z_for_bins = np.where(np.isfinite(_z_values), _z_values, Z_BLEND_FALLBACK_Z)
_bins = np.searchsorted(Z_BLEND_CUTS, _z_for_bins, side='right')
_bins = np.clip(_bins, 0, len(Z_BLEND_WEIGHTS) - 1)
_weights = Z_BLEND_WEIGHTS[_bins]
_weights = np.where(np.isfinite(_z_values), _weights, Z_BLEND_FALLBACK_WEIGHT)
_blend['tvt'] = (
    _weights * _blend['tvt_1'].astype(float).to_numpy()
    + (1.0 - _weights) * _blend['tvt_2'].astype(float).to_numpy()
)

sub = _blend[['id', 'tvt']].copy()
if len(sub) != len(sample):
    raise RuntimeError(f"{{CANDIDATE_NAME}} final row mismatch: {{len(sub)}} != {{len(sample)}}")
if not sub['id'].equals(sample['id']):
    raise RuntimeError(f"{{CANDIDATE_NAME}} final ids do not match sample order")
if sub['tvt'].isna().any() or not np.isfinite(sub['tvt'].astype(float)).all():
    raise RuntimeError(f"{{CANDIDATE_NAME}} final submission contains non-finite tvt values")
print(
    f"{{CANDIDATE_NAME}} dynamic_z bins={{Z_BLEND_BINS}} shrink={{Z_BLEND_SHRINK:.1f}} "
    f"weight_range={{float(np.min(_weights)):.3f}}..{{float(np.max(_weights)):.3f}} "
    f"mean_w={{float(np.mean(_weights)):.3f}} "
    f"tvt_range={{float(sub['tvt'].min()):.3f}}..{{float(sub['tvt'].max()):.3f}}",
    flush=True,
)
sub.to_csv("submission.csv", index=False)
sub
"""


def materialize_candidate(candidate: DynamicCandidate, frame: pd.DataFrame) -> None:
    spec = fit_z_bins(frame, bins=candidate.bins, shrink=candidate.shrink)
    spec.update(cv_z_bins(frame, bins=candidate.bins, shrink=candidate.shrink))
    spec.update(
        {
            "candidate": candidate.dirname,
            "kernel_id": candidate.kernel_id,
            "source_components": str(COMPONENTS.relative_to(ROOT)),
            "bins": candidate.bins,
            "shrink": candidate.shrink,
        }
    )

    target_dir = OUT_ROOT / candidate.dirname
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    for wheel in BASE_DIR.glob("*.whl"):
        shutil.copy2(wheel, target_dir / wheel.name)

    code = BASE_CODE.read_text()
    code = code.replace(
        'ROGII_HIDDEN_SAFE_PROFILE = "w040"',
        f'ROGII_HIDDEN_SAFE_PROFILE = "{candidate.profile}"',
    )
    code = code.replace("FINAL_SELECTOR_PF_SEEDS=32", f"FINAL_SELECTOR_PF_SEEDS={candidate.final_seeds}")
    code = code.replace(
        'warnings.filterwarnings("ignore")\n',
        'warnings.filterwarnings("ignore")\nnp.random.seed(20260607)\n',
        1,
    )
    if ORIGINAL_FINAL_BLOCK not in code:
        raise RuntimeError("Base final block not found; aborting materialization")
    code = code.replace(ORIGINAL_FINAL_BLOCK, final_block(candidate, spec))
    (target_dir / candidate.code_file).write_text(code)

    metadata = {
        "id": candidate.kernel_id,
        "title": candidate.title,
        "code_file": candidate.code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["ridge", "hidden-safe", "dynamic-z"],
        "dataset_sources": ["ravaghi/wellbore-geology-prediction-artifacts"],
        "kernel_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "model_sources": [],
    }
    (target_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (target_dir / "dynamic-blend-spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if not COMPONENTS.is_file():
        raise FileNotFoundError(f"Missing component cache: {COMPONENTS}")
    frame = pd.read_parquet(COMPONENTS)
    for candidate in CANDIDATES:
        materialize_candidate(candidate, frame)
        print(
            f"materialized {candidate.dirname}: {candidate.kernel_id} "
            f"bins={candidate.bins} shrink={candidate.shrink:g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
