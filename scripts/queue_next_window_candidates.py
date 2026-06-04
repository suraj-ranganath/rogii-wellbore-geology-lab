from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

COMPETITION = "rogii-wellbore-geology-prediction"
DAILY_SUBMISSION_CAP = 5
SAMPLE_PATH = Path("data/raw/rogii-wellbore-geology-prediction/sample_submission.csv")
STATE_PATH = Path("outputs/next_window_queue_state.json")


@dataclass(frozen=True)
class Candidate:
    name: str
    kernel: str
    version: int
    message: str
    kernel_dir: Path
    output_dir: Path
    push_args: list[str] = field(default_factory=list)


CANDIDATES = [
    Candidate(
        name="super_solution_top3_fixed",
        kernel="surajranganath17/rogii-super-solution-top3",
        version=2,
        message="super solution top3 fixed catboost bootstrap",
        kernel_dir=Path("kaggle/kernels/super_solution_top3"),
        output_dir=Path("outputs/next_window/super_solution_top3_fixed"),
        push_args=["-t", "32400", "--accelerator", "gpu"],
    ),
    Candidate(
        name="public_anti_target_free_extrapolate",
        kernel="surajranganath17/rogii-public-anti-target-free-extrapolate",
        version=2,
        message="public anti target-free extrapolate moonshot",
        kernel_dir=Path("kaggle/kernels/public_anti_target_free_extrapolate"),
        output_dir=Path("outputs/next_window/public_anti_target_free_extrapolate"),
    ),
    Candidate(
        name="public_pf_sunny_extrapolate",
        kernel="surajranganath17/rogii-public-pf-sunny-extrapolate",
        version=2,
        message="public pf sunny extrapolate alpha 5.05551",
        kernel_dir=Path("kaggle/kernels/public_pf_sunny_extrapolate"),
        output_dir=Path("outputs/next_window/public_pf_sunny_extrapolate"),
    ),
    Candidate(
        name="public_sunny_last_extrapolate",
        kernel="surajranganath17/rogii-public-sunny-last-extrapolate",
        version=2,
        message="public sunny last-known extrapolate alpha -0.279964",
        kernel_dir=Path("kaggle/kernels/public_sunny_last_extrapolate"),
        output_dir=Path("outputs/next_window/public_sunny_last_extrapolate"),
    ),
    Candidate(
        name="public_train_tvt_extrapolate",
        kernel="surajranganath17/rogii-public-train-tvt-extrapolate",
        version=1,
        message="public train tvt extrapolate beta 0.157936",
        kernel_dir=Path("kaggle/kernels/public_train_tvt_extrapolate"),
        output_dir=Path("outputs/next_window/public_train_tvt_extrapolate"),
    ),
]


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(args), flush=True)
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}")
    return result


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"pushed": {}, "submitted": {}, "failed": {}, "push_failures": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def submissions_text() -> str:
    result = run_cmd(
        ["uv", "run", "kaggle", "competitions", "submissions", "-c", COMPETITION]
    )
    return result.stdout


def utc_submission_count_today(text: str) -> int:
    today = dt.datetime.now(dt.UTC).date().isoformat()
    count = 0
    for line in text.splitlines():
        match = re.match(r"\s*\d+\s+\S+\s+(\d{4}-\d{2}-\d{2})\s+", line)
        if match and match.group(1) == today:
            count += 1
    return count


def already_submitted(text: str, message: str) -> bool:
    return message in text


def kernel_status(kernel: str) -> str:
    result = run_cmd(["uv", "run", "kaggle", "kernels", "status", kernel], check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = detail[-1] if detail else f"exit={result.returncode}"
        return f"UNAVAILABLE: {suffix}"
    match = re.search(r'has status "([^"]+)"', result.stdout)
    if not match:
        return "UNKNOWN"
    return match.group(1)


def push_candidate(candidate: Candidate, state: dict) -> None:
    if candidate.name in state["pushed"]:
        return
    if not candidate.kernel_dir.is_dir():
        raise FileNotFoundError(candidate.kernel_dir)
    args = [
        "uv",
        "run",
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(candidate.kernel_dir),
        *candidate.push_args,
    ]
    result = run_cmd(args, check=False)
    if result.returncode == 0:
        state["pushed"][candidate.name] = {
            "kernel": candidate.kernel,
            "version": candidate.version,
            "pushed_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        state["push_failures"].pop(candidate.name, None)
    else:
        failures = state["push_failures"].setdefault(candidate.name, [])
        failures.append(
            {
                "failed_at": dt.datetime.now(dt.UTC).isoformat(),
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1000:],
                "stderr_tail": result.stderr[-1000:],
            }
        )
    save_state(state)


def validate_submission(path: Path) -> None:
    sample = pd.read_csv(SAMPLE_PATH)
    sub = pd.read_csv(path)
    if list(sub.columns) != ["id", "tvt"]:
        raise ValueError(f"{path}: expected columns ['id', 'tvt'], got {list(sub.columns)}")
    if len(sub) != len(sample):
        raise ValueError(f"{path}: expected {len(sample)} rows, got {len(sub)}")
    if not sub["id"].equals(sample["id"]):
        raise ValueError(f"{path}: ids do not match sample_submission order")
    if sub["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ids")
    values = sub["tvt"].astype(float)
    if sub["tvt"].isna().any() or not values.map(math.isfinite).all():
        raise ValueError(f"{path}: non-finite tvt predictions")
    print(
        f"validated {path}: rows={len(sub)} range=({values.min():.4f}, {values.max():.4f})",
        flush=True,
    )


def download_output(candidate: Candidate) -> Path:
    if candidate.output_dir.exists():
        shutil.rmtree(candidate.output_dir)
    candidate.output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "uv",
            "run",
            "kaggle",
            "kernels",
            "output",
            candidate.kernel,
            "-p",
            str(candidate.output_dir),
        ]
    )
    output_path = candidate.output_dir / "submission.csv"
    if not output_path.is_file():
        raise FileNotFoundError(f"{candidate.name}: missing {output_path}")
    return output_path


def submit_candidate(candidate: Candidate) -> None:
    run_cmd(
        [
            "uv",
            "run",
            "kaggle",
            "competitions",
            "submit",
            COMPETITION,
            "-k",
            candidate.kernel,
            "-v",
            str(candidate.version),
            "-f",
            "submission.csv",
            "-m",
            candidate.message,
        ]
    )


def monitor(timeout_minutes: int, poll_seconds: int) -> None:
    start = time.time()
    state = load_state()
    save_state(state)
    while time.time() - start < timeout_minutes * 60:
        for candidate in CANDIDATES:
            if candidate.name in state["submitted"] or candidate.name in state["failed"]:
                continue
            push_candidate(candidate, state)

        submission_table = submissions_text()
        today_count = utc_submission_count_today(submission_table)
        print(f"UTC submissions today: {today_count}/{DAILY_SUBMISSION_CAP}", flush=True)

        for candidate in CANDIDATES:
            if candidate.name in state["submitted"] or candidate.name in state["failed"]:
                continue
            if already_submitted(submission_table, candidate.message):
                state["submitted"][candidate.name] = {
                    "message": candidate.message,
                    "detected_at": dt.datetime.now(dt.UTC).isoformat(),
                }
                save_state(state)
                continue
            if candidate.name not in state["pushed"]:
                continue
            if today_count >= DAILY_SUBMISSION_CAP:
                print("daily submission cap reached; waiting for UTC reset", flush=True)
                break

            status = kernel_status(candidate.kernel)
            print(f"{candidate.name}: {status}", flush=True)
            if "COMPLETE" in status:
                try:
                    output_path = download_output(candidate)
                    validate_submission(output_path)
                    submit_candidate(candidate)
                except Exception as exc:
                    print(f"{candidate.name}: submit path failed, will retry: {exc}", flush=True)
                    continue
                today_count += 1
                state["submitted"][candidate.name] = {
                    "message": candidate.message,
                    "submitted_at": dt.datetime.now(dt.UTC).isoformat(),
                    "output_path": str(output_path),
                    "kernel": candidate.kernel,
                    "version": candidate.version,
                }
                save_state(state)
            elif any(marker in status for marker in ["ERROR", "FAILED", "CANCELED"]):
                state["failed"][candidate.name] = {
                    "status": status,
                    "detected_at": dt.datetime.now(dt.UTC).isoformat(),
                    "kernel": candidate.kernel,
                    "version": candidate.version,
                }
                save_state(state)

        if len(state["submitted"]) + len(state["failed"]) >= len(CANDIDATES):
            print("all candidates have reached terminal state", flush=True)
            return
        print(f"sleeping {poll_seconds}s", flush=True)
        time.sleep(poll_seconds)
    print("monitor timeout reached", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-minutes", type=int, default=720)
    parser.add_argument("--poll-seconds", type=int, default=180)
    args = parser.parse_args()
    monitor(args.timeout_minutes, args.poll_seconds)


if __name__ == "__main__":
    main()
