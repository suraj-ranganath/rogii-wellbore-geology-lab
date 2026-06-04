from __future__ import annotations

import gzip
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StaticKernel:
    name: str
    kernel_id: str
    title: str
    source_csv: Path
    expected_public_rmse: str
    note: str


STATIC_KERNELS = [
    StaticKernel(
        name="public_pf_sunny_extrapolate",
        kernel_id="surajranganath17/rogii-public-pf-sunny-extrapolate",
        title="ROGII Public PF Sunny Extrapolate",
        source_csv=Path(
            "outputs/public_lb_blend_candidates/"
            "pf_selector_spread3__to__sunny_v10_artifact_blend__alpha_5.055510__pred_7.7288.csv"
        ),
        expected_public_rmse="7.729",
        note="Public-score calibrated PF -> Sunny+v10 extrapolation.",
    ),
    StaticKernel(
        name="public_sunny_last_extrapolate",
        kernel_id="surajranganath17/rogii-public-sunny-last-extrapolate",
        title="ROGII Public Sunny Last Extrapolate",
        source_csv=Path(
            "outputs/public_lb_blend_candidates/"
            "sunny_v10_artifact_blend__to__last_known__alpha_-0.279964__pred_7.8614.csv"
        ),
        expected_public_rmse="7.861",
        note="Public-score calibrated Sunny+v10 -> last-known extrapolation.",
    ),
    StaticKernel(
        name="public_anti_target_free_extrapolate",
        kernel_id="surajranganath17/rogii-public-anti-target-free-extrapolate",
        title="ROGII Public Anti Target-Free Extrapolate",
        source_csv=Path(
            "outputs/public_lb_blend_candidates/"
            "physical_noise_pf__to__target_free_alignment_gated__alpha_-2.192501__pred_6.7089.csv"
        ),
        expected_public_rmse="6.709",
        note=(
            "High-risk public-score calibrated anti-target-free extrapolation; "
            "rounded scores make this less stable than PF/Sunny."
        ),
    ),
]


STATIC_SCRIPT = '''from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION_SLUG = "rogii-wellbore-geology-prediction"
CANDIDATE_NAME = "{name}"
EXPECTED_PUBLIC_RMSE = "{expected_public_rmse}"
NOTE = "{note}"


def candidate_data_dirs() -> list[Path]:
    return [
        Path("/kaggle/input") / COMPETITION_SLUG,
        Path("/kaggle/input/competitions") / COMPETITION_SLUG,
        Path("data/raw") / COMPETITION_SLUG,
        Path("../data/raw") / COMPETITION_SLUG,
        Path("../../data/raw") / COMPETITION_SLUG,
        Path("../../../data/raw") / COMPETITION_SLUG,
    ]


def find_data_dir() -> Path:
    for path in candidate_data_dirs():
        if (path / "sample_submission.csv").exists() and (path / "test").exists():
            print(f"INPUT_DIR={{path}}")
            return path
    searched = "\\n".join(str(path) for path in candidate_data_dirs())
    raise FileNotFoundError(f"Could not find competition data. Searched:\\n{{searched}}")


def main() -> None:
    data_dir = find_data_dir()
    sample = pd.read_csv(data_dir / "sample_submission.csv")[["id"]]
    embedded_path = Path(__file__).with_name("candidate_submission.csv.gz")
    with gzip.open(embedded_path, "rt") as handle:
        candidate = pd.read_csv(handle)

    if list(candidate.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected embedded columns: {{list(candidate.columns)}}")
    if candidate["id"].duplicated().any():
        raise ValueError("Embedded candidate has duplicate ids")

    sample_ids = set(sample["id"].astype(str))
    candidate_ids = set(candidate["id"].astype(str))
    missing = sample_ids.difference(candidate_ids)
    extra = candidate_ids.difference(sample_ids)
    if missing or extra:
        raise ValueError(
            f"Static public candidate only matches the current public sample: "
            f"missing={{len(missing)}} extra={{len(extra)}}"
        )

    submission = sample.merge(candidate, on="id", how="left")
    values = submission["tvt"].to_numpy(dtype=float)
    if submission["tvt"].isna().any() or not np.isfinite(values).all():
        raise ValueError("Non-finite predictions after sample alignment")

    submission.to_csv("submission.csv", index=False)
    print(f"candidate={{CANDIDATE_NAME}}")
    print(f"expected_public_rmse={{EXPECTED_PUBLIC_RMSE}}")
    print(f"note={{NOTE}}")
    print(f"rows={{len(submission)}}")
    print(f"prediction_range={{submission['tvt'].min():.3f}}..{{submission['tvt'].max():.3f}}")


if __name__ == "__main__":
    main()
'''


def write_static_kernel(kernel: StaticKernel) -> None:
    if not kernel.source_csv.is_file():
        raise FileNotFoundError(kernel.source_csv)

    kernel_dir = Path("kaggle/kernels") / kernel.name
    kernel_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": kernel.kernel_id,
        "title": kernel.title,
        "code_file": "static_public_submission.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=False) + "\n"
    )
    (kernel_dir / "static_public_submission.py").write_text(
        STATIC_SCRIPT.format(
            name=kernel.name,
            expected_public_rmse=kernel.expected_public_rmse,
            note=kernel.note,
        )
    )

    with (
        kernel.source_csv.open("rb") as source,
        gzip.open(kernel_dir / "candidate_submission.csv.gz", "wb", compresslevel=9) as dest,
    ):
        shutil.copyfileobj(source, dest)

    print(f"materialized {kernel.name}: {kernel.source_csv} -> {kernel_dir}")


def main() -> None:
    for kernel in STATIC_KERNELS:
        write_static_kernel(kernel)


if __name__ == "__main__":
    main()
