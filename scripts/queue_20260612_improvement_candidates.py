"""Queue the 2026-06-12 improvement candidates after the daily cap resets.

The user explicitly approved spending the next five submission slots. The
candidate set is restricted to completed, downloaded, validated Kaggle commits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import queue_20260607_heavy_ridge_candidates as queue

queue.STATE_PATH = Path("outputs/queue_20260612_improvement_state.json")

OUT_BASE = Path("outputs/queue_20260612_improvement_submit")

QUEUE_SPEC = [
    (
        "fle3n_v5_exact_r2",
        "surajranganath17/rogii-fle3n-v5-exact-r2",
        1,
        "improve queue fle3n v5 exact r2",
        "kaggle/kernels/fle3n_v5_exact_r2",
    ),
    (
        "fle3n_v5f_exact_r2",
        "surajranganath17/rogii-fle3n-v5f-exact-r2",
        1,
        "improve queue fle3n v5f exact r2",
        "kaggle/kernels/fle3n_v5f_exact_r2",
    ),
    (
        "jaemin_seed7_mtoshi_beicicc",
        "surajranganath17/rogii-jaemin-seed7-mtoshi-beicicc",
        1,
        "improve queue jaemin seed7 mtoshi beicicc",
        "kaggle/kernels/jaemin_seed7_mtoshi_beicicc",
    ),
    (
        "jaemin_affine_seed7_mtoshi",
        "surajranganath17/rogii-jaemin-affine-seed7-mtoshi",
        1,
        "improve queue jaemin affine seed7 mtoshi",
        "kaggle/kernels/jaemin_affine_seed7_mtoshi",
    ),
    (
        "jaemin_sp45_fleongg_w065s",
        "surajranganath17/rogii-sp45-fleongg-w065s-runtime-safe",
        1,
        "improve queue sp45 fleongg w065 override",
        "kaggle/kernels/jaemin_sp45_fleongg_w065s",
    ),
]

queue.CANDIDATES = [
    queue.Candidate(
        name=name,
        kernel=kernel,
        version=version,
        message=message,
        kernel_dir=Path(kernel_dir),
        output_dir=OUT_BASE / name,
    )
    for name, kernel, version, message, kernel_dir in QUEUE_SPEC
]


def seed_pushed_state() -> None:
    state = queue.load_state()
    for candidate in queue.CANDIDATES:
        if candidate.name not in state["pushed"]:
            state["pushed"][candidate.name] = {
                "kernel": candidate.kernel,
                "version": candidate.version,
                "pushed_at": dt.datetime.now(dt.UTC).isoformat(),
                "seeded": True,
            }
    queue.save_state(state)
    print("seeded pushed-state:")
    print(json.dumps(state["pushed"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-minutes", type=int, default=1440)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    seed_pushed_state()
    queue.monitor(args.timeout_minutes, args.poll_seconds)


if __name__ == "__main__":
    main()
