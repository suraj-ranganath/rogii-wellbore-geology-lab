from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii_wellbore.data import read_csv, scan_wells
from rogii_wellbore.metrics import rmse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction"
DEFAULT_KERNEL = ROOT / "kaggle/kernels/pf_selector_spread3/pf_selector_spread3.py"
DEFAULT_OUTPUT = ROOT / "outputs/local_pf_selector_cv.json"


def load_pf_namespace(kernel_path: Path) -> dict[str, Any]:
    source = kernel_path.read_text()
    marker = "\nsample = pd.read_csv"
    if marker not in source:
        raise ValueError(f"Could not find notebook main marker in {kernel_path}")
    prefix = source.split(marker, maxsplit=1)[0]
    namespace: dict[str, Any] = {"__file__": str(kernel_path)}
    exec(compile(prefix, str(kernel_path), "exec"), namespace)
    return namespace


def last_known_prediction(hw: pd.DataFrame) -> np.ndarray:
    values = hw["TVT_input"].to_numpy(float).copy()
    known = pd.Series(values).dropna()
    last_value = float(known.iloc[-1]) if len(known) else 0.0
    values[np.isnan(values)] = last_value
    return values


def evaluate_well(
    pair,
    ns: dict[str, Any],
    n_particles: int,
    n_seeds: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    hw = read_csv(pair.horizontal_path)
    tw = read_csv(pair.typewell_path)
    eval_mask = hw["TVT_input"].isna().to_numpy()
    if not eval_mask.any():
        return {}, {}
    target = hw.loc[eval_mask, "TVT"].to_numpy(float)

    predictions: dict[str, np.ndarray] = {"last_known": last_known_prediction(hw)}
    pf_by_scale = ns["run_pf_lik_ensemble_scales"](
        hw,
        tw,
        n_particles=n_particles,
        n_seeds=n_seeds,
    )
    for name, values in pf_by_scale.items():
        predictions[name] = values

    try:
        predictions["beam"] = ns["run_beam_ensemble"](hw, tw)
    except Exception:
        predictions["beam"] = pf_by_scale["pf_scale_8"].copy()

    selector_code, selector_variant, n_eval, z_span = ns["selector_well_code"](hw)
    known = hw["TVT_input"].dropna()
    last_tvt = float(known.iloc[-1]) if len(known) else float(np.nanmean(pf_by_scale["pf_scale_8"]))
    predictions["selector"] = ns["apply_selector_variant"](
        selector_variant,
        pf_by_scale,
        predictions["beam"],
        last_tvt,
    )

    scores = {
        name: rmse(target, values[eval_mask])
        for name, values in predictions.items()
    }
    meta = {
        "selector_code": int(selector_code),
        "selector_variant": selector_variant,
        "n_eval": int(n_eval),
        "z_span": float(z_span),
        "rows": int(eval_mask.sum()),
    }
    return scores, meta


def summarize(method_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {}
    for method, rows in method_rows.items():
        if not rows:
            continue
        sse = sum(float(row["sse"]) for row in rows)
        n = sum(int(row["rows"]) for row in rows)
        well_rmse = [float(row["rmse"]) for row in rows]
        summary[method] = {
            "rmse": float(np.sqrt(sse / n)) if n else None,
            "rows": int(n),
            "wells": int(len(rows)),
            "well_rmse_mean": float(np.mean(well_rmse)),
            "well_rmse_median": float(np.median(well_rmse)),
            "worst_wells": sorted(
                [
                    {"well": row["well"], "rmse": float(row["rmse"]), "rows": int(row["rows"])}
                    for row in rows
                ],
                key=lambda row: row["rmse"],
                reverse=True,
            )[:10],
        }
    return dict(sorted(summary.items(), key=lambda item: item[1]["rmse"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay PF/beam/selector methods on train hidden tails."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--kernel-code", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-wells", type=int, default=40)
    parser.add_argument("--well-stride", type=int, default=1)
    parser.add_argument("--n-particles", type=int, default=256)
    parser.add_argument("--n-seeds", type=int, default=16)
    parser.add_argument("--progress", type=int, default=5)
    args = parser.parse_args()

    ns = load_pf_namespace(args.kernel_code)
    pairs = [pair for pair in scan_wells(args.data_dir) if pair.split == "train" and pair.typewell_path]
    if args.well_stride > 1:
        pairs = pairs[:: args.well_stride]
    if args.max_wells:
        pairs = pairs[: args.max_wells]

    method_rows: dict[str, list[dict[str, Any]]] = {}
    well_rows = []
    start = time.time()
    for idx, pair in enumerate(pairs, start=1):
        well_start = time.time()
        try:
            scores, meta = evaluate_well(pair, ns, args.n_particles, args.n_seeds)
        except Exception as exc:
            well_rows.append({"well": pair.well_id, "error": repr(exc)})
            print(f"{pair.well_id}: failed {exc}", flush=True)
            continue
        if not scores:
            continue
        target_rows = int(meta["rows"])
        well_record = {"well": pair.well_id, **meta, "scores": scores}
        well_rows.append(well_record)
        for method, score in scores.items():
            method_rows.setdefault(method, []).append(
                {
                    "well": pair.well_id,
                    "rmse": float(score),
                    "rows": target_rows,
                    "sse": float(score * score * target_rows),
                }
            )
        if args.progress and idx % args.progress == 0:
            best = min(scores.items(), key=lambda item: item[1])
            print(
                f"{idx}/{len(pairs)} {pair.well_id} rows={target_rows} "
                f"best={best[0]}:{best[1]:.3f} elapsed={time.time() - well_start:.1f}s",
                flush=True,
            )

    result = {
        "args": {
            "kernel_code": str(args.kernel_code),
            "max_wells": args.max_wells,
            "well_stride": args.well_stride,
            "n_particles": args.n_particles,
            "n_seeds": args.n_seeds,
        },
        "elapsed_seconds": float(time.time() - start),
        "wells": well_rows,
        "results": summarize(method_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\nPF selector replay summary ({len(well_rows)} wells):")
    for method, row in result["results"].items():
        print(
            f"{method:16s} rmse={row['rmse']:.4f} "
            f"median_well={row['well_rmse_median']:.4f} wells={row['wells']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
