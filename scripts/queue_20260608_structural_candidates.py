from __future__ import annotations

import argparse
from pathlib import Path

import queue_20260607_heavy_ridge_candidates as queue

queue.STATE_PATH = Path("outputs/queue_20260608_structural_state.json")

queue.CANDIDATES = [
    queue.Candidate(
        name="ridge_sp7776_exact",
        kernel="surajranganath17/rogii-ridge-sp7776-exact",
        version=1,
        message="ridge sp7776 exact public structural",
        kernel_dir=Path("kaggle/kernels/ridge_sp7776_exact"),
        output_dir=Path("outputs/queue_20260608_structural/ridge_sp7776_exact"),
    ),
    queue.Candidate(
        name="ridge_sp45_proj",
        kernel="surajranganath17/rogii-ridge-sp45-projection",
        version=1,
        message="ridge sp45 robust u projection",
        kernel_dir=Path("kaggle/kernels/ridge_sp45_proj"),
        output_dir=Path("outputs/queue_20260608_structural/ridge_sp45_proj"),
    ),
    queue.Candidate(
        name="ridge_sp7776_proj",
        kernel="surajranganath17/rogii-ridge-sp7776-projection",
        version=1,
        message="ridge sp7776 tuned robust u projection",
        kernel_dir=Path("kaggle/kernels/ridge_sp7776_proj"),
        output_dir=Path("outputs/queue_20260608_structural/ridge_sp7776_proj"),
    ),
    queue.Candidate(
        name="ridge_sp7776_projdeg3",
        kernel="surajranganath17/rogii-ridge-sp7776-projection-deg3",
        version=1,
        message="ridge sp7776 tuned u projection deg3",
        kernel_dir=Path("kaggle/kernels/ridge_sp7776_projdeg3"),
        output_dir=Path("outputs/queue_20260608_structural/ridge_sp7776_projdeg3"),
    ),
    queue.Candidate(
        name="drift_geosteering_infer",
        kernel="surajranganath17/rogii-drift-geosteering-infer",
        version=1,
        message="drift geosteering pretrained structural",
        kernel_dir=Path("kaggle/kernels/drift_geosteering_infer"),
        output_dir=Path("outputs/queue_20260608_structural/drift_geosteering_infer"),
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
