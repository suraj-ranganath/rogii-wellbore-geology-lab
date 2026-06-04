from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SOURCE_DIR = Path("kaggle/kernels/target_free_alignment_gated")
WORK_ROOT = Path("outputs/target_free_push_work")
STATE_PATH = Path("outputs/target_free_push_state.json")
BASE_KERNEL_ID = "surajranganath17/rogii-strat-align-sidecar-v"
BASE_TITLE = "ROGII Strat Align Sidecar V"
GPU_LIMIT_TEXT = "Maximum batch GPU session count"
RETRYABLE_TEXT = (GPU_LIMIT_TEXT, "Notebook not found")
BLOCKING_GPU_KERNELS = [
    "surajranganath17/rogii-sunny-v10-artifact-blend",
    "surajranganath17/rogii-super-solution-top3",
]


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(args), flush=True)
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"next_attempt": 2}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def status_text(kernel: str) -> str:
    result = run_cmd(["uv", "run", "kaggle", "kernels", "status", kernel])
    return f"{result.stdout}\n{result.stderr}"


def gpu_slots_still_full() -> bool:
    active = 0
    for kernel in BLOCKING_GPU_KERNELS:
        text = status_text(kernel)
        if "KernelWorkerStatus.RUNNING" in text or "KernelWorkerStatus.QUEUED" in text:
            active += 1
    print(f"known active GPU kernels: {active}/{len(BLOCKING_GPU_KERNELS)}", flush=True)
    return active >= 2


def materialize_attempt(attempt: int) -> tuple[str, Path]:
    kernel_id = f"{BASE_KERNEL_ID}{attempt}"
    title = f"{BASE_TITLE}{attempt}"
    attempt_dir = WORK_ROOT / f"v{attempt}"
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    shutil.copytree(SOURCE_DIR, attempt_dir)
    metadata_path = attempt_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["id"] = kernel_id
    metadata["title"] = title
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return kernel_id, attempt_dir


def push_kernel(attempt: int) -> tuple[bool, str, str]:
    kernel_id, attempt_dir = materialize_attempt(attempt)
    result = run_cmd(["uv", "run", "kaggle", "kernels", "push", "-p", str(attempt_dir)])
    text = f"{result.stdout}\n{result.stderr}"
    if result.returncode == 0 and "successfully pushed" in text:
        return True, kernel_id, text
    return False, kernel_id, text


def queue_push(timeout_minutes: int, poll_seconds: int) -> None:
    start = time.time()
    state = load_state()
    save_state(state)
    if state.get("accepted_kernel"):
        print(f"target-free kernel already accepted: {state['accepted_kernel']}", flush=True)
        return

    while time.time() - start < timeout_minutes * 60:
        now = dt.datetime.now(dt.UTC).isoformat()
        print(f"[{now}] checking target-free push state", flush=True)

        if gpu_slots_still_full():
            print(f"GPU batch slots still full; sleeping {poll_seconds}s", flush=True)
            time.sleep(poll_seconds)
            continue

        attempt = int(state.get("next_attempt", 2))
        pushed, kernel_id, text = push_kernel(attempt)
        if pushed:
            state["accepted_kernel"] = kernel_id
            state["accepted_at"] = dt.datetime.now(dt.UTC).isoformat()
            state["accepted_attempt"] = attempt
            save_state(state)
            print(f"target-free kernel push accepted: {kernel_id}", flush=True)
            return

        if any(marker in text for marker in RETRYABLE_TEXT):
            state["next_attempt"] = attempt + 1
            state["last_retryable_error"] = text[-1000:]
            state["last_retryable_at"] = dt.datetime.now(dt.UTC).isoformat()
            save_state(state)
            print(
                f"retryable push failure for {kernel_id}; "
                f"next attempt will use v{attempt + 1}",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue

        raise RuntimeError("target-free kernel push failed for a non-retryable reason")
    print("target-free push queue timeout reached", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-minutes", type=int, default=2160)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    queue_push(args.timeout_minutes, args.poll_seconds)


if __name__ == "__main__":
    main()
