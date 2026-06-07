from __future__ import annotations

import argparse
import json
import math
import py_compile
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "rogii-wellbore-geology-prediction"
SAMPLE_PATH = ROOT / "data/raw/rogii-wellbore-geology-prediction/sample_submission.csv"
DEFAULT_BASELINE = (
    ROOT / "outputs/queue_20260606_hidden_safe/ravaghi_ridge_w040/submission.csv"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    kernel: str
    kernel_dir: Path
    code_file: str
    output_dir: Path

    @property
    def code_path(self) -> Path:
        return self.kernel_dir / self.code_file

    @property
    def metadata_path(self) -> Path:
        return self.kernel_dir / "kernel-metadata.json"

    @property
    def submission_path(self) -> Path:
        return self.output_dir / "submission.csv"


CANDIDATES = [
    Candidate(
        name="ridge_w040_pf40",
        kernel="surajranganath17/rogii-ridge-w040-pf40",
        kernel_dir=ROOT / "kaggle/kernels/ridge_w040_pf40",
        code_file="ridge_w040_pf40.py",
        output_dir=ROOT / "outputs/queue_20260607/ridge_w040_pf40",
    ),
    Candidate(
        name="ridge_w040_selector070",
        kernel="surajranganath17/rogii-ridge-w040-selector070",
        kernel_dir=ROOT / "kaggle/kernels/ridge_w040_selector070",
        code_file="ridge_w040_selector070.py",
        output_dir=ROOT / "outputs/queue_20260607/ridge_w040_selector070",
    ),
    Candidate(
        name="ridge_w040_selector080",
        kernel="surajranganath17/rogii-ridge-w040-selector080",
        kernel_dir=ROOT / "kaggle/kernels/ridge_w040_selector080",
        code_file="ridge_w040_selector080.py",
        output_dir=ROOT / "outputs/queue_20260607/ridge_w040_selector080",
    ),
    Candidate(
        name="ridge_w040_prefix_gate",
        kernel="surajranganath17/rogii-ridge-w040-prefix-gate",
        kernel_dir=ROOT / "kaggle/kernels/ridge_w040_prefix_gate",
        code_file="ridge_w040_prefix_gate.py",
        output_dir=ROOT / "outputs/queue_20260607/ridge_w040_prefix_gate",
    ),
    Candidate(
        name="ridge_w040_formprefix_gate",
        kernel="surajranganath17/rogii-ridge-w040-formprefix-gate",
        kernel_dir=ROOT / "kaggle/kernels/ridge_w040_formprefix_gate",
        code_file="ridge_w040_formprefix_gate.py",
        output_dir=ROOT / "outputs/queue_20260607/ridge_w040_formprefix_gate",
    ),
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def run_cmd(args: list[str]) -> None:
    print("$ " + " ".join(args), flush=True)
    result = subprocess.run(args, cwd=ROOT, check=False, text=True)
    if result.returncode != 0:
        fail(f"command failed with exit {result.returncode}: {' '.join(args)}")


def refresh_kaggle_output(candidate: Candidate) -> None:
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


def check_metadata(candidate: Candidate) -> None:
    if not candidate.metadata_path.is_file():
        fail(f"{candidate.name}: missing {candidate.metadata_path}")
    metadata = json.loads(candidate.metadata_path.read_text())
    expected = {
        "id": candidate.kernel,
        "code_file": candidate.code_file,
        "kernel_type": "script",
        "language": "python",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            fail(f"{candidate.name}: metadata {key!r} expected {value!r}, got {metadata.get(key)!r}")
    if metadata.get("enable_internet") is not False:
        fail(f"{candidate.name}: metadata enable_internet must be false")
    if metadata.get("enable_gpu") is not False:
        fail(f"{candidate.name}: metadata enable_gpu must be false")
    if COMPETITION not in metadata.get("competition_sources", []):
        fail(f"{candidate.name}: metadata missing competition source {COMPETITION}")
    if not candidate.code_path.is_file():
        fail(f"{candidate.name}: missing code file {candidate.code_path}")


def compile_candidate(candidate: Candidate) -> None:
    py_compile.compile(str(candidate.code_path), doraise=True)


def check_log(candidate: Candidate) -> str:
    logs = sorted(candidate.output_dir.glob("*.log"))
    if not logs:
        fail(f"{candidate.name}: missing Kaggle output log in {candidate.output_dir}")
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    bad_markers = ["Traceback (most recent call last)", "ModuleNotFoundError", "FileNotFoundError"]
    for marker in bad_markers:
        if marker in text:
            fail(f"{candidate.name}: log contains {marker!r}")
    final_lines = [line for line in text.splitlines() if " final blend " in line]
    if not final_lines:
        fail(f"{candidate.name}: log does not contain final blend marker")
    return final_lines[-1]


def validate_submission(candidate: Candidate, sample: pd.DataFrame) -> pd.DataFrame:
    if not candidate.submission_path.is_file():
        fail(f"{candidate.name}: missing {candidate.submission_path}")
    sub = pd.read_csv(candidate.submission_path)
    if list(sub.columns) != ["id", "tvt"]:
        fail(f"{candidate.name}: expected columns ['id', 'tvt'], got {list(sub.columns)}")
    if len(sub) != len(sample):
        fail(f"{candidate.name}: expected {len(sample)} rows, got {len(sub)}")
    if not sub["id"].equals(sample["id"]):
        fail(f"{candidate.name}: ids do not match sample_submission order")
    if sub["id"].duplicated().any():
        fail(f"{candidate.name}: duplicate ids")
    values = sub["tvt"].astype(float)
    if sub["tvt"].isna().any() or not values.map(math.isfinite).all():
        fail(f"{candidate.name}: non-finite tvt predictions")
    if values.max() - values.min() < 50:
        fail(f"{candidate.name}: suspiciously narrow tvt range")
    if values.median() < 8000 or values.median() > 14000:
        fail(f"{candidate.name}: suspicious median {values.median():.4f}")
    return sub


def load_baseline(path: Path, sample: pd.DataFrame) -> pd.Series:
    if not path.is_file():
        fail(f"baseline submission not found: {path}")
    baseline = pd.read_csv(path)
    if list(baseline.columns) != ["id", "tvt"]:
        fail(f"baseline expected columns ['id', 'tvt'], got {list(baseline.columns)}")
    if len(baseline) != len(sample) or not baseline["id"].equals(sample["id"]):
        fail("baseline ids do not match sample_submission order")
    return baseline["tvt"].astype(float)


def rmse(a: pd.Series, b: pd.Series) -> float:
    diff = a.to_numpy(float) - b.to_numpy(float)
    return float(np.sqrt(np.mean(diff * diff)))


def smoke(args: argparse.Namespace) -> None:
    if args.refresh_kaggle:
        for candidate in CANDIDATES:
            refresh_kaggle_output(candidate)

    sample = pd.read_csv(SAMPLE_PATH)
    baseline = load_baseline(args.baseline, sample)
    submissions: dict[str, pd.DataFrame] = {}
    final_lines: dict[str, str] = {}

    for candidate in CANDIDATES:
        check_metadata(candidate)
        compile_candidate(candidate)
        submissions[candidate.name] = validate_submission(candidate, sample)
        final_lines[candidate.name] = check_log(candidate)

    print("\nCandidate smoke summary:")
    for candidate in CANDIDATES:
        values = submissions[candidate.name]["tvt"].astype(float)
        distance = rmse(values, baseline)
        if distance < args.min_baseline_rmse:
            fail(
                f"{candidate.name}: too close to baseline "
                f"({distance:.4f} < {args.min_baseline_rmse:.4f})"
            )
        print(
            f"{candidate.name:28s} rows={len(values):6d} "
            f"range=({values.min():10.4f}, {values.max():10.4f}) "
            f"mean={values.mean():10.4f} rmse_vs_w040={distance:8.4f}",
            flush=True,
        )
        print(f"  log: {final_lines[candidate.name]}", flush=True)

    print("\nPairwise RMSE:")
    for left_index, left in enumerate(CANDIDATES):
        left_values = submissions[left.name]["tvt"].astype(float)
        for right in CANDIDATES[left_index + 1 :]:
            right_values = submissions[right.name]["tvt"].astype(float)
            distance = rmse(left_values, right_values)
            if distance < args.min_pairwise_rmse:
                fail(
                    f"{left.name} and {right.name}: too similar "
                    f"({distance:.4f} < {args.min_pairwise_rmse:.4f})"
                )
            print(f"{left.name:28s} vs {right.name:28s} rmse={distance:8.4f}")

    print("\nPASS: local candidate smoke checks completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local smoke checks for the 2026-06-07 ROGII Kaggle candidates."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Reference w040 submission used for diversity checks.",
    )
    parser.add_argument(
        "--min-baseline-rmse",
        type=float,
        default=0.25,
        help="Minimum RMSE distance each candidate must have from the reference baseline.",
    )
    parser.add_argument(
        "--min-pairwise-rmse",
        type=float,
        default=0.25,
        help="Minimum RMSE distance required between any two candidates.",
    )
    parser.add_argument(
        "--refresh-kaggle",
        action="store_true",
        help="Download latest Kaggle notebook outputs before validating. This does not submit.",
    )
    smoke(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
