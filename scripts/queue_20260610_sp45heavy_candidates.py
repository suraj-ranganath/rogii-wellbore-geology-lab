"""Queue the 2026-06-10 SP45-heavy + overlap-override candidates.

Priority order: the five bagged orchestrator kernels first (weight ladder
w0.72 / drift mix / w0.80 / w1.00 / w0.65, all with the pixiux guarded
overlap override), then the single-run plan-B backups. The monitor submits the
first five candidates that are COMPLETE with a validated output after the
2026-06-11 00:00 UTC reset; failed kernels are skipped automatically.

All nine kernels were pushed manually before this queue starts, so the state
file is pre-seeded by scripts/seed_20260610_queue_state.py (run automatically
when this module is executed) to prevent the monitor from re-pushing and
restarting the runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import queue_20260607_heavy_ridge_candidates as queue

queue.STATE_PATH = Path("outputs/queue_20260610_sp45heavy_state.json")

OUT_BASE = Path("outputs/queue_20260610_sp45heavy")

QUEUE_SPEC = [
    # (name, slug, message)
    ("sp45h_bag3_w072", "rogii-sp45h-bag3-w072", "sp45h bag3 w072 overlap override"),
    ("sp45h_drift_mix", "rogii-sp45h-drift-mix", "sp45h drift mix 083 017 overlap override"),
    ("sp45h_bag3_w080", "rogii-sp45h-bag3-w080", "sp45h bag3 w080 overlap override"),
    ("sp45h_bag3_w100", "rogii-sp45h-bag3-w100", "sp45h bag3 w100 pure sp45 overlap override"),
    ("sp45h_bag3_w065", "rogii-sp45h-bag3-w065", "sp45h bag3 w065 overlap override"),
    ("jaemin_sp45_fleongg_w072s", "rogii-sp45-fleongg-w072s", "sp45 fleongg w072 single overlap override"),
    ("jaemin_sp45_fleongg_w080s", "rogii-sp45-fleongg-w080s", "sp45 fleongg w080 single overlap override"),
    ("jaemin_sp45_fleongg_w065s", "rogii-sp45-fleongg-w065s", "sp45 fleongg w065 single overlap override"),
    ("jaemin_sp45_fleongg_w100s", "rogii-sp45-fleongg-w100s", "sp45 fleongg w100 single overlap override"),
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
    print(f"seeded pushed-state for {len(queue.CANDIDATES)} candidates")
    print(json.dumps(sorted(state["pushed"]), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-minutes", type=int, default=2400)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    seed_pushed_state()
    queue.monitor(args.timeout_minutes, args.poll_seconds)


if __name__ == "__main__":
    main()
