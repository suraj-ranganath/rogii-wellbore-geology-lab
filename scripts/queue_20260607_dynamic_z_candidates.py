from __future__ import annotations

import argparse
from pathlib import Path

import queue_20260607_heavy_ridge_candidates as queue

queue.STATE_PATH = Path("outputs/queue_20260607_dynamic_z_state.json")

queue.CANDIDATES = [
    queue.Candidate(
        name="ridge_w040_zq12",
        kernel="surajranganath17/rogii-ridge-w040-zq12",
        version=1,
        message="ridge w040 dynamic zq12 exact80",
        kernel_dir=Path("kaggle/kernels/ridge_w040_zq12"),
        output_dir=Path("outputs/queue_20260607_dynamic_z/ridge_w040_zq12"),
    ),
    queue.Candidate(
        name="ridge_w040_zq11",
        kernel="surajranganath17/rogii-ridge-w040-zq11",
        version=1,
        message="ridge w040 dynamic zq11 exact80",
        kernel_dir=Path("kaggle/kernels/ridge_w040_zq11"),
        output_dir=Path("outputs/queue_20260607_dynamic_z/ridge_w040_zq11"),
    ),
    queue.Candidate(
        name="ridge_w040_zq12_shr1000",
        kernel="surajranganath17/rogii-ridge-w040-zq12-shr1000",
        version=1,
        message="ridge w040 dynamic zq12 shrink1000 exact80",
        kernel_dir=Path("kaggle/kernels/ridge_w040_zq12_shr1000"),
        output_dir=Path("outputs/queue_20260607_dynamic_z/ridge_w040_zq12_shr1000"),
    ),
    queue.Candidate(
        name="ridge_w040_zq13",
        kernel="surajranganath17/rogii-ridge-w040-zq13",
        version=1,
        message="ridge w040 dynamic zq13 exact80",
        kernel_dir=Path("kaggle/kernels/ridge_w040_zq13"),
        output_dir=Path("outputs/queue_20260607_dynamic_z/ridge_w040_zq13"),
    ),
    queue.Candidate(
        name="ridge_w040_zq08",
        kernel="surajranganath17/rogii-ridge-w040-zq08",
        version=1,
        message="ridge w040 dynamic zq08 exact80",
        kernel_dir=Path("kaggle/kernels/ridge_w040_zq08"),
        output_dir=Path("outputs/queue_20260607_dynamic_z/ridge_w040_zq08"),
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
