"""Queue the 2026-06-11 final five recovery/upside candidates.

This queue is meant to run after the UTC submission cap resets. It reuses
already-pushed, already-validated version-1 Kaggle commits:

1. Runtime-safe SP45/fleongg w0.60 + guarded overlap override.
2. Runtime-safe pure-SP45 w1.00 + guarded overlap override.
3. Public fle3n v5 exact hedge, reported LB 7.528.
4. fle3n v5 SP45-heavy / hedge-weight optimum probe.
5. Public fle3n v5f exact hedge probe.

Do not run this unless the user has approved spending submissions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import queue_20260607_heavy_ridge_candidates as queue

queue.STATE_PATH = Path("outputs/queue_20260611_final_state.json")

OUT_BASE = Path("outputs/queue_20260611_final_submit")

QUEUE_SPEC = [
    (
        "jaemin_sp45_fleongg_w060s",
        "surajranganath17/rogii-sp45-fleongg-w060s",
        1,
        "final queue sp45 fleongg w060 override",
        "kaggle/kernels/jaemin_sp45_fleongg_w060s",
    ),
    (
        "jaemin_sp45_fleongg_w100s",
        "surajranganath17/rogii-sp45-fleongg-w100s-runtime-safe",
        1,
        "final queue sp45 fleongg w100 override",
        "kaggle/kernels/jaemin_sp45_fleongg_w100s",
    ),
    (
        "fle3n_v5_exact_h050",
        "surajranganath17/rogii-fle3n-v5-exact-h050",
        1,
        "final queue fle3n v5 exact h050",
        "kaggle/kernels/fle3n_v5_exact_h050",
    ),
    (
        "fle3n_v5_w060_h0455",
        "surajranganath17/rogii-fle3n-v5-w060-h0455",
        1,
        "final queue fle3n v5 w060 h0455",
        "kaggle/kernels/fle3n_v5_w060_h0455",
    ),
    (
        "fle3n_v5f_exact_h050",
        "surajranganath17/rogii-fle3n-v5f-exact-h050",
        1,
        "final queue fle3n v5f exact h050",
        "kaggle/kernels/fle3n_v5f_exact_h050",
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
