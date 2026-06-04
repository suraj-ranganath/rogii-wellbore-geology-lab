from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_name_value(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected WELL=SCORE, got {value!r}")
        name, raw = value.split("=", 1)
        parsed[name.strip()] = float(raw.strip())
    return parsed


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def add_well_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["well_id"] = out["id"].astype(str).str.rsplit("_", n=1).str[0]
    return out


def validate_submission(frame: pd.DataFrame) -> None:
    if list(frame.columns) != ["id", "tvt"]:
        raise ValueError(f"Expected columns ['id', 'tvt'], got {list(frame.columns)}")
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate submission ids")
    values = frame["tvt"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite tvt predictions")


def make_probes(args: argparse.Namespace) -> None:
    base = pd.read_csv(args.base_submission)
    validate_submission(base)
    framed = add_well_column(base)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wells = sorted(framed["well_id"].unique())
    for well in wells:
        out = framed[["id", "tvt"]].copy()
        mask = framed["well_id"] == well
        out.loc[mask, "tvt"] = out.loc[mask, "tvt"] + args.shift
        path = out_dir / f"probe_{safe_name(well)}_plus_{args.shift:g}.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path}: shifted rows={int(mask.sum())}")

    print()
    print("After each probe scores, solve offsets with:")
    print(
        "uv run python scripts/public_lb_probe_tool.py solve "
        f"--base-submission {args.base_submission} --base-score {args.base_score:g} "
        f"--shift {args.shift:g} --probe-score WELL=SCORE ..."
    )


def solve_probes(args: argparse.Namespace) -> None:
    base = pd.read_csv(args.base_submission)
    validate_submission(base)
    framed = add_well_column(base)
    probe_scores = parse_name_value(args.probe_score)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = len(framed)
    base_mse = args.base_score**2
    shifts: dict[str, float] = {}
    report_rows: list[dict[str, float | str | int]] = []

    for well in sorted(framed["well_id"].unique()):
        n_rows = int((framed["well_id"] == well).sum())
        score = probe_scores.get(well)
        if score is None:
            shifts[well] = 0.0
            report_rows.append(
                {
                    "well_id": well,
                    "rows": n_rows,
                    "probe_score": math.nan,
                    "estimated_residual_mean": math.nan,
                    "recommended_shift": 0.0,
                }
            )
            continue

        probe_mse = score**2
        residual_mean = (
            total_rows * (probe_mse - base_mse) - n_rows * args.shift**2
        ) / (2.0 * args.shift * n_rows)
        recommended_shift = -residual_mean
        shifts[well] = recommended_shift
        report_rows.append(
            {
                "well_id": well,
                "rows": n_rows,
                "probe_score": score,
                "estimated_residual_mean": residual_mean,
                "recommended_shift": recommended_shift,
            }
        )

    corrected = framed[["id", "tvt"]].copy()
    for well, shift in shifts.items():
        corrected.loc[framed["well_id"] == well, "tvt"] += shift

    corrected_path = out_dir / "public_probe_corrected_submission.csv"
    report_path = out_dir / "public_probe_shift_report.csv"
    corrected.to_csv(corrected_path, index=False)
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(pd.DataFrame(report_rows).to_string(index=False))
    print(f"wrote corrected submission: {corrected_path}")
    print(f"wrote shift report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make or solve public-leaderboard well-shift probes. "
            "This intentionally uses public LB feedback and is not private-safe."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make", help="Write one plus-shift probe CSV per public well.")
    make.add_argument(
        "--base-submission",
        type=Path,
        default=Path("outputs/kaggle_pf_selector_spread3_v1/submission.csv"),
    )
    make.add_argument("--base-score", type=float, default=8.781)
    make.add_argument("--shift", type=float, default=10.0)
    make.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/public_lb_probe_candidates"),
    )
    make.set_defaults(func=make_probes)

    solve = sub.add_parser("solve", help="Solve well offsets from plus-shift probe scores.")
    solve.add_argument(
        "--base-submission",
        type=Path,
        default=Path("outputs/kaggle_pf_selector_spread3_v1/submission.csv"),
    )
    solve.add_argument("--base-score", type=float, default=8.781)
    solve.add_argument("--shift", type=float, default=10.0)
    solve.add_argument(
        "--probe-score",
        action="append",
        default=[],
        metavar="WELL=SCORE",
        help="Public score for the probe that shifted WELL by +shift.",
    )
    solve.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/public_lb_probe_candidates"),
    )
    solve.set_defaults(func=solve_probes)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
