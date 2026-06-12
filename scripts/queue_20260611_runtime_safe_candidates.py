"""Queue runtime-safe recovery candidates after the 2026-06-11 timeout batch.

The 2026-06-11 final-five submission batch showed that bag3 SP45 kernels can
finish as public commits but still exceed the hidden competition replay runtime.
This queue therefore uses only single-run JAEMIN SP45/fleongg variants with the
guarded overlap override.

Do not run this for submissions without explicit user approval.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import queue_20260607_heavy_ridge_candidates as queue

queue.STATE_PATH = Path("outputs/queue_20260611_runtime_safe_state.json")

OUT_BASE = Path("outputs/queue_20260611_runtime_safe")

QUEUE_SPEC = [
    ("jaemin_sp45_fleongg_w060s", "rogii-sp45-fleongg-w060s", "runtime safe sp45 fleongg w060 override"),
    (
        "jaemin_sp45_fleongg_w065s",
        "rogii-sp45-fleongg-w065s-runtime-safe",
        "runtime safe sp45 fleongg w065 override",
    ),
    (
        "jaemin_sp45_fleongg_w072s",
        "rogii-sp45-fleongg-w072s-runtime-safe",
        "runtime safe sp45 fleongg w072 override",
    ),
    (
        "jaemin_sp45_fleongg_w080s",
        "rogii-sp45-fleongg-w080s-runtime-safe",
        "runtime safe sp45 fleongg w080 override",
    ),
    (
        "jaemin_sp45_fleongg_w100s",
        "rogii-sp45-fleongg-w100s-runtime-safe",
        "runtime safe sp45 fleongg w100 override",
    ),
]

queue.CANDIDATES = [
    queue.Candidate(
        name=name,
        kernel=f"surajranganath17/{slug}",
        version=1,
        message=message,
        kernel_dir=Path(f"kaggle/kernels/{name}"),
        output_dir=OUT_BASE / name,
    )
    for name, slug, message in QUEUE_SPEC
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
