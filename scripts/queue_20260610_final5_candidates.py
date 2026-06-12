"""Queue the final-five 2026-06-10 candidates.

Submits at most five validated candidates after the 2026-06-11 00:00 UTC reset.
This queue intentionally avoids five public-duplicate overlap outputs:

1. SP45/drift QP mix with guarded override (already completed).
2. SP45-heavy w0.72 bagged probe with guarded override (already completed).
3. Pure bagged SP45 probe with guarded override (already completed).
4. Exact medali CNN-MTP inference with guarded override.
5. Bagged pure-SP45 / medali CNN-MTP 80/20 blend with guarded override.

The first three kernels were pushed and completed by the SP45-heavy batch. The
two CNN kernels are pushed by this queue if needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import queue_20260607_heavy_ridge_candidates as queue

queue.STATE_PATH = Path("outputs/queue_20260610_final5_state.json")

OUT_BASE = Path("outputs/queue_20260610_final5")

QUEUE_SPEC = [
    (
        "sp45h_drift_mix",
        "surajranganath17/rogii-sp45h-drift-mix",
        1,
        "final5 sp45h drift mix 083 017 override",
        True,
    ),
    (
        "sp45h_bag3_w072",
        "surajranganath17/rogii-sp45h-bag3-w072",
        1,
        "final5 sp45h bag3 w072 override",
        True,
    ),
    (
        "sp45h_bag3_w100",
        "surajranganath17/rogii-sp45h-bag3-w100",
        1,
        "final5 sp45h bag3 w100 pure override",
        True,
    ),
    (
        "medali_cnn_mtp_exact_override",
        "surajranganath17/rogii-medali-cnn-mtp-exact-override",
        3,
        "final5 medali cnn mtp exact override",
        False,
    ),
    (
        "sp45h_cnn_mtp_blend020",
        "surajranganath17/rogii-sp45h-cnn-mtp-blend020",
        3,
        "final5 sp45h cnn mtp blend020 override",
        False,
    ),
]

queue.CANDIDATES = [
    queue.Candidate(
        name=name,
        kernel=kernel,
        version=version,
        message=message,
        kernel_dir=Path(f"kaggle/kernels/{name}"),
        output_dir=OUT_BASE / name,
    )
    for name, kernel, version, message, _already_pushed in QUEUE_SPEC
]


def seed_existing_pushed_state() -> None:
    state = queue.load_state()
    for name, kernel, version, _message, already_pushed in QUEUE_SPEC:
        if already_pushed and name not in state["pushed"]:
            state["pushed"][name] = {
                "kernel": kernel,
                "version": version,
                "pushed_at": dt.datetime.now(dt.UTC).isoformat(),
                "seeded": True,
            }
    queue.save_state(state)
    print("seeded existing pushed-state:")
    print(json.dumps(state["pushed"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-minutes", type=int, default=1440)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    seed_existing_pushed_state()
    queue.monitor(args.timeout_minutes, args.poll_seconds)


if __name__ == "__main__":
    main()
