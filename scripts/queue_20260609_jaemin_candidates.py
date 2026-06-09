from __future__ import annotations

import argparse
from pathlib import Path

import queue_20260607_heavy_ridge_candidates as queue

queue.STATE_PATH = Path("outputs/queue_20260609_jaemin_state.json")

queue.CANDIDATES = [
    queue.Candidate(
        name="jaemin_sp45_fleongg_exact",
        kernel="surajranganath17/rogii-sp45-fleongg-blend-exact",
        version=1,
        message="jaemin sp45 fleongg blend exact",
        kernel_dir=Path("kaggle/kernels/jaemin_sp45_fleongg_exact"),
        output_dir=Path("outputs/queue_20260609_jaemin/jaemin_sp45_fleongg_exact"),
    ),
    queue.Candidate(
        name="jaemin_sp45_fleongg_w060",
        kernel="surajranganath17/rogii-sp45-fleongg-w060",
        version=1,
        message="jaemin sp45 fleongg blend w060",
        kernel_dir=Path("kaggle/kernels/jaemin_sp45_fleongg_w060"),
        output_dir=Path("outputs/queue_20260609_jaemin/jaemin_sp45_fleongg_w060"),
    ),
    queue.Candidate(
        name="yaroslav_sel15_forced_selector",
        kernel="surajranganath17/rogii-sel15-forced-selector",
        version=1,
        message="yaroslav sel15 forced selector reference",
        kernel_dir=Path("kaggle/kernels/yaroslav_sel15_forced_selector"),
        output_dir=Path("outputs/queue_20260609_jaemin/yaroslav_sel15_forced_selector"),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-minutes", type=int, default=2400)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    queue.monitor(args.timeout_minutes, args.poll_seconds)


if __name__ == "__main__":
    main()
