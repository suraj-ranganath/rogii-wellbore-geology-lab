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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COMPETITION = "rogii-wellbore-geology-prediction"
DAILY_SUBMISSION_CAP = 5
SAMPLE_PATH = Path("data/raw/rogii-wellbore-geology-prediction/sample_submission.csv")
STATE_PATH = Path("outputs/kaggle_monitor_state.json")


@dataclass(frozen=True)
class Candidate:
    name: str
    kernel: str
    version: int
    message: str
    output_dir: Path
    kernel_state_path: Path | None = None


CANDIDATES = [
    Candidate(
        name="sunny_v10_artifact_blend",
        kernel="surajranganath17/rogii-sunny-v10-artifact-blend",
        version=1,
        message="sunny v10 artifact blend",
        output_dir=Path("outputs/kaggle_sunny_v10_artifact_blend_v1"),
    ),
    Candidate(
        name="super_solution_top3",
        kernel="surajranganath17/rogii-super-solution-top3",
        version=1,
        message="super solution top3 physics tree stack",
        output_dir=Path("outputs/kaggle_super_solution_top3_v1"),
    ),
    Candidate(
        name="target_free_alignment_gated",
        kernel="surajranganath17/rogii-strat-align-sidecar-v2",
        version=1,
        message="target free alignment gated",
        output_dir=Path("outputs/kaggle_target_free_alignment_gated_v1"),
        kernel_state_path=Path("outputs/target_free_push_state.json"),
    ),
]


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(args), flush=True)
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
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
    return {"submitted": {}, "failed": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


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


def candidate_kernel(candidate: Candidate) -> str:
    if candidate.kernel_state_path and candidate.kernel_state_path.is_file():
        state = json.loads(candidate.kernel_state_path.read_text())
        accepted_kernel = state.get("accepted_kernel")
        if accepted_kernel:
            return str(accepted_kernel)
    return candidate.kernel


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
            candidate_kernel(candidate),
            "-p",
            str(candidate.output_dir),
        ]
    )
    output_path = candidate.output_dir / "submission.csv"
    if not output_path.is_file():
        raise FileNotFoundError(f"{candidate.name}: missing {output_path}")
    return output_path


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
    if sub["tvt"].isna().any():
        raise ValueError(f"{path}: null tvt predictions")
    values = sub["tvt"].astype(float)
    if not values.map(math.isfinite).all():
        raise ValueError(f"{path}: non-finite tvt predictions")
    print(
        f"validated {path}: rows={len(sub)} "
        f"range=({values.min():.4f}, {values.max():.4f})",
        flush=True,
    )


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
            candidate_kernel(candidate),
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
        submission_table = submissions_text()
        today_count = utc_submission_count_today(submission_table)
        print(f"UTC submissions today: {today_count}/{DAILY_SUBMISSION_CAP}", flush=True)

        for candidate in CANDIDATES:
            if candidate.name in state["submitted"]:
                continue
            if candidate.name in state["failed"]:
                continue
            if already_submitted(submission_table, candidate.message):
                state["submitted"][candidate.name] = {
                    "message": candidate.message,
                    "detected_at": dt.datetime.now(dt.UTC).isoformat(),
                }
                save_state(state)
                continue
            if today_count >= DAILY_SUBMISSION_CAP:
                print(
                    "daily submission cap reached; will retry on a later poll",
                    flush=True,
                )
                break

            status = kernel_status(candidate_kernel(candidate))
            print(f"{candidate.name}: {status}", flush=True)
            if "COMPLETE" in status:
                output_path = download_output(candidate)
                validate_submission(output_path)
                try:
                    submit_candidate(candidate)
                except RuntimeError as exc:
                    print(f"{candidate.name}: submit failed, will retry: {exc}", flush=True)
                    continue
                today_count += 1
                state["submitted"][candidate.name] = {
                    "message": candidate.message,
                    "submitted_at": dt.datetime.now(dt.UTC).isoformat(),
                    "output_path": str(output_path),
                }
                save_state(state)
            elif any(marker in status for marker in ["ERROR", "FAILED", "CANCELED"]):
                state["failed"][candidate.name] = {
                    "status": status,
                    "detected_at": dt.datetime.now(dt.UTC).isoformat(),
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
    parser.add_argument("--timeout-minutes", type=int, default=480)
    parser.add_argument("--poll-seconds", type=int, default=180)
    args = parser.parse_args()
    monitor(args.timeout_minutes, args.poll_seconds)


if __name__ == "__main__":
    main()
