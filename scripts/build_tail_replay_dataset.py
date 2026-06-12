"""Build a synthetic ROGII competition directory for a local hidden-tail replay.

We hold out N train wells, move them into a synthetic ``test/`` directory, and
mask the tail portion of each held-out well's ``TVT_input`` to ``NaN`` exactly
the way the real Kaggle test wells are masked (contiguous suffix, id =
``{well8}_{rowpos}`` of every NaN row). The held-out wells are dropped from the
synthetic ``train/`` directory so the champion pipeline can be retrained without
leakage and replayed on real hidden-tail rows whose ground-truth ``TVT`` we keep
in ``truth.csv``.

Mirrors conventions from ``scripts/local_tail_cv.py``:
- Only wells that have a real ``TVT`` target on the masked rows are eligible
  (``prediction_mask(require_target=True)`` equivalent).
- Deterministic well selection by seed.

The synthetic ``test/`` and ``train/`` horizontal CSVs are written with only the
columns present in the *real* test wells (``MD, X, Y, Z, GR, TVT_input``) so the
pipeline never sees the train-only formation/TVT columns it would not have at
inference time. Typewell CSVs are copied with only ``TVT, GR`` (the real test
typewell schema).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/raw/rogii-wellbore-geology-prediction"

# Columns that exist in the real Kaggle test horizontal CSVs (train-only feature
# columns such as ANCC/ASTNU/.../BUDA and the TVT target are NOT present there).
TEST_HORIZONTAL_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
TEST_TYPEWELL_COLUMNS = ["TVT", "GR"]


def list_train_wells(data_dir: Path) -> list[str]:
    wells = sorted(
        p.name.replace("__horizontal_well.csv", "")
        for p in (data_dir / "train").glob("*__horizontal_well.csv")
    )
    return wells


def measured_tail_fraction(data_dir: Path, n: int = 60) -> float:
    """Estimate the tail (hidden) fraction from real test wells if present."""
    fracs: list[float] = []
    for p in sorted((data_dir / "test").glob("*__horizontal_well.csv"))[:n]:
        df = pd.read_csv(p, usecols=["TVT_input"])
        if len(df):
            fracs.append(float(df["TVT_input"].isna().mean()))
    return float(np.median(fracs)) if fracs else 0.72


def select_eval_wells(
    data_dir: Path, n_wells: int, seed: int, min_eval_rows: int, tail_frac: float
) -> list[str]:
    """Pick eligible wells deterministically.

    Eligible = horizontal has a usable real TVT target on its tail rows and a
    sizeable eval zone, mirroring ``local_tail_cv`` well filtering.
    """
    wells = list_train_wells(data_dir)
    rng = np.random.default_rng(seed)
    rng.shuffle(wells)
    eligible: list[str] = []
    for wid in wells:
        hw_path = data_dir / "train" / f"{wid}__horizontal_well.csv"
        tw_path = data_dir / "train" / f"{wid}__typewell.csv"
        if not tw_path.exists():
            continue
        try:
            df = pd.read_csv(hw_path, usecols=["TVT", "TVT_input"])
        except (ValueError, KeyError):
            continue
        if "TVT" not in df.columns or df["TVT"].isna().all():
            continue
        n_tail = int(round(len(df) * tail_frac))
        if n_tail < min_eval_rows:
            continue
        # Tail TVT must be fully finite to score against.
        tail_tvt = df["TVT"].iloc[len(df) - n_tail :]
        if not np.isfinite(tail_tvt.to_numpy(dtype=float)).all():
            continue
        eligible.append(wid)
        if len(eligible) >= n_wells:
            break
    if len(eligible) < n_wells:
        raise ValueError(
            f"Only found {len(eligible)} eligible eval wells; requested {n_wells}."
        )
    return sorted(eligible)


def write_synthetic_test_well(
    data_dir: Path,
    out_dir: Path,
    wid: str,
    tail_frac: float,
) -> pd.DataFrame:
    """Write a masked synthetic test well; return its truth rows (id, tvt)."""
    hw = pd.read_csv(data_dir / "train" / f"{wid}__horizontal_well.csv")
    n = len(hw)
    n_tail = int(round(n * tail_frac))
    n_tail = max(1, min(n_tail, n))
    tail_start = n - n_tail

    # Build the masked test horizontal: keep only real-test columns.
    test_hw = hw.reindex(columns=TEST_HORIZONTAL_COLUMNS).copy()
    # Ground-truth TVT for the tail rows comes from the train TVT column.
    truth_tvt = pd.to_numeric(hw["TVT"], errors="coerce").to_numpy(dtype=float)
    # Mask the contiguous tail of TVT_input (positions tail_start..n-1).
    test_hw.loc[test_hw.index[tail_start:], "TVT_input"] = np.nan

    out_dir.mkdir(parents=True, exist_ok=True)
    test_hw.to_csv(out_dir / f"{wid}__horizontal_well.csv", index=False)

    # Typewell: copy with the real-test schema (TVT, GR only).
    tw = pd.read_csv(data_dir / "train" / f"{wid}__typewell.csv")
    tw.reindex(columns=TEST_TYPEWELL_COLUMNS).to_csv(
        out_dir / f"{wid}__typewell.csv", index=False
    )

    rows = [
        {"id": f"{wid}_{pos}", "tvt": truth_tvt[pos]}
        for pos in range(tail_start, n)
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-eval-wells", type=int, default=40)
    parser.add_argument("--seed", type=int, default=204)
    parser.add_argument("--min-eval-rows", type=int, default=200)
    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.0,
        help="Hidden-tail fraction per well; 0 => median of real test wells.",
    )
    parser.add_argument(
        "--max-train-wells",
        type=int,
        default=0,
        help="Cap synthetic train wells (0 = use all remaining). Speeds smoke runs.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    tail_frac = args.tail_frac if args.tail_frac > 0 else measured_tail_fraction(data_dir)

    eval_wells = select_eval_wells(
        data_dir, args.n_eval_wells, args.seed, args.min_eval_rows, tail_frac
    )
    eval_set = set(eval_wells)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    # Synthetic train: all remaining wells (optionally capped).
    train_wells = [w for w in list_train_wells(data_dir) if w not in eval_set]
    if args.max_train_wells:
        train_wells = sorted(train_wells)[: args.max_train_wells]
    for wid in train_wells:
        for suffix in ("__horizontal_well.csv", "__typewell.csv"):
            src = data_dir / "train" / f"{wid}{suffix}"
            if src.exists():
                shutil.copy2(src, out_dir / "train" / f"{wid}{suffix}")

    # Synthetic test: held-out eval wells, tail-masked.
    truth_parts: list[pd.DataFrame] = []
    for wid in eval_wells:
        truth_parts.append(
            write_synthetic_test_well(data_dir, out_dir / "test", wid, tail_frac)
        )
    truth = pd.concat(truth_parts, ignore_index=True)
    truth.to_csv(out_dir / "truth.csv", index=False)

    # sample_submission.csv with the exact id format and ordering used by Kaggle.
    sample = truth[["id"]].copy()
    sample["tvt"] = 0.0
    sample.to_csv(out_dir / "sample_submission.csv", index=False)

    if not all(len(wid) == 8 for wid in eval_wells):
        raise ValueError("Well ids are not 8 chars; id format would not match Kaggle.")

    meta = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "tail_frac": tail_frac,
        "n_eval_wells": len(eval_wells),
        "n_train_wells": len(train_wells),
        "n_truth_rows": int(len(truth)),
        "min_eval_rows": args.min_eval_rows,
        "max_train_wells": args.max_train_wells,
        "eval_wells": eval_wells,
    }
    (out_dir / "replay_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "eval_wells"}, indent=2))
    print(f"wrote synthetic competition dir -> {out_dir}")


if __name__ == "__main__":
    main()
