"""Score tail-replay component/blend submissions against held-out truth.

Given a synthetic competition dir (with ``truth.csv``) and a kernel run's output
dir containing ``sp45_projection_submission.csv`` and
``fleongg_pretrained_submission.csv``, this computes:

- Global row RMSE of the pure SP45 and pure fleongg components.
- The blend RMSE over a grid of ``w_sp45`` weights (the key open question).
- The optimal ``w_sp45`` on this held-out split.
- Per-well RMSE distribution (mean/median/worst-decile) for each component.

When run over multiple replicate output dirs (different seeds, same split), it
also reports replicate-to-replicate RMSE variance per component, quantifying the
bagging gain available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_GRID = [0.55, 0.60, 0.65, 0.72, 0.80, 0.90, 1.00]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_component(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"id", "tvt"}.issubset(df.columns):
        raise ValueError(f"{path} must contain id,tvt columns; got {list(df.columns)}")
    return df[["id", "tvt"]].copy()


def per_well_rmse(merged: pd.DataFrame, pred_col: str) -> dict[str, float]:
    merged = merged.copy()
    merged["well"] = merged["id"].str[:8]
    scores = []
    for _, grp in merged.groupby("well"):
        scores.append(rmse(grp["tvt_truth"].to_numpy(), grp[pred_col].to_numpy()))
    scores = np.array(scores, dtype=float)
    return {
        "well_rmse_mean": float(np.mean(scores)),
        "well_rmse_median": float(np.median(scores)),
        "well_rmse_worst_decile": float(np.mean(np.sort(scores)[::-1][: max(1, len(scores) // 10)])),
        "well_rmse_max": float(np.max(scores)),
        "n_wells": int(len(scores)),
    }


def score_one_run(truth: pd.DataFrame, run_dir: Path, grid: list[float]) -> dict:
    sp45 = load_component(run_dir / "sp45_projection_submission.csv").rename(
        columns={"tvt": "tvt_sp45"}
    )
    fle = load_component(run_dir / "fleongg_pretrained_submission.csv").rename(
        columns={"tvt": "tvt_fleongg"}
    )
    merged = truth.merge(sp45, on="id", how="inner").merge(fle, on="id", how="inner")

    n_truth = len(truth)
    n_merged = len(merged)
    coverage = n_merged / n_truth if n_truth else 0.0

    for col in ("tvt_sp45", "tvt_fleongg"):
        if not np.isfinite(merged[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {col} for {run_dir}")

    y = merged["tvt_truth"].to_numpy(dtype=float)
    sp45_pred = merged["tvt_sp45"].to_numpy(dtype=float)
    fle_pred = merged["tvt_fleongg"].to_numpy(dtype=float)

    blend = {}
    for w in grid:
        pred = w * sp45_pred + (1.0 - w) * fle_pred
        blend[f"{w:.2f}"] = rmse(y, pred)
    best_w = min(blend, key=lambda k: blend[k])

    return {
        "run_dir": str(run_dir),
        "n_truth_rows": n_truth,
        "n_scored_rows": n_merged,
        "coverage": coverage,
        "rmse_sp45": rmse(y, sp45_pred),
        "rmse_fleongg": rmse(y, fle_pred),
        "blend_grid": blend,
        "best_w_sp45": float(best_w),
        "best_blend_rmse": blend[best_w],
        "per_well_sp45": per_well_rmse(merged, "tvt_sp45"),
        "per_well_fleongg": per_well_rmse(merged, "tvt_fleongg"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True, help="Path to truth.csv")
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Kernel output dir with component CSVs (repeatable for replicates).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--grid",
        type=float,
        nargs="*",
        default=DEFAULT_GRID,
    )
    parser.add_argument("--runtime-notes", type=str, default="")
    args = parser.parse_args()

    truth = pd.read_csv(args.truth).rename(columns={"tvt": "tvt_truth"})

    runs = [score_one_run(truth, rd, args.grid) for rd in args.run_dir]

    # Replicate variance per component / blend point.
    def col(key: str) -> list[float]:
        return [r[key] for r in runs]

    replicate = {
        "n_replicates": len(runs),
        "rmse_sp45_mean": float(np.mean(col("rmse_sp45"))),
        "rmse_sp45_std": float(np.std(col("rmse_sp45"), ddof=0)),
        "rmse_fleongg_mean": float(np.mean(col("rmse_fleongg"))),
        "rmse_fleongg_std": float(np.std(col("rmse_fleongg"), ddof=0)),
    }
    # Bagging gain estimate: RMSE of the per-row mean of replicate component preds.
    if len(args.run_dir) >= 2:
        y_idx = truth.set_index("id")["tvt_truth"]
        for comp, fname in (
            ("sp45", "sp45_projection_submission.csv"),
            ("fleongg", "fleongg_pretrained_submission.csv"),
        ):
            preds = [
                load_component(Path(rd) / fname).set_index("id")["tvt"]
                for rd in args.run_dir
            ]
            wide = pd.concat(preds, axis=1, join="inner")
            bagged = wide.mean(axis=1)
            y = y_idx.reindex(bagged.index)
            replicate[f"rmse_{comp}_bagged"] = rmse(y.to_numpy(), bagged.to_numpy())

    # Grid averaged across replicates.
    grid_keys = [f"{w:.2f}" for w in args.grid]
    avg_grid = {k: float(np.mean([r["blend_grid"][k] for r in runs])) for k in grid_keys}
    best_w_avg = min(avg_grid, key=lambda k: avg_grid[k])

    result = {
        "grid": args.grid,
        "runs": runs,
        "replicate": replicate,
        "avg_blend_grid": avg_grid,
        "best_w_sp45_avg": float(best_w_avg),
        "best_blend_rmse_avg": avg_grid[best_w_avg],
        "runtime_notes": args.runtime_notes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "rmse_sp45_mean": replicate["rmse_sp45_mean"],
        "rmse_fleongg_mean": replicate["rmse_fleongg_mean"],
        "avg_blend_grid": avg_grid,
        "best_w_sp45_avg": best_w_avg,
    }, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
