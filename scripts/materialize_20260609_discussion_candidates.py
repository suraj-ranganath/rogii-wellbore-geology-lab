from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "kaggle" / "kernels"
RESEARCH_ROOT = Path("/tmp/rogii_discussion_candidates")


@dataclass(frozen=True)
class Candidate:
    name: str
    source_dir: Path
    source_file: str
    kernel_id: str
    title: str
    dataset_sources: list[str]
    kernel_sources: list[str] | None = None
    enable_gpu: bool = False


CANDIDATES = [
    Candidate(
        name="yaroslav_sel15_forced_selector",
        source_dir=RESEARCH_ROOT / "yaroslav_sel15",
        source_file="rogii-sel15-forced-selector.ipynb",
        kernel_id="surajranganath17/rogii-sel15-forced-selector",
        title="ROGII SEL15 Forced Selector",
        dataset_sources=[
            "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
            "pilkwang/rogii-model-package",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ],
        enable_gpu=True,
    ),
    Candidate(
        name="iaztec_ridge_artifact_param",
        source_dir=RESEARCH_ROOT / "iaztec_param",
        source_file="rogii-wellbore-geology-ridge-artifact-param.ipynb",
        kernel_id="surajranganath17/rogii-ridge-artifact-param",
        title="ROGII Ridge Artifact Param",
        dataset_sources=[
            "ravaghi/wellbore-geology-prediction-artifacts",
        ],
        kernel_sources=["packagemanager/pm-122018862-at-06-09-2026-02-34-43"],
    ),
]


def source_bytes(candidate: Candidate, target_dir: Path) -> bytes:
    source_path = candidate.source_dir / candidate.source_file
    fallback_path = target_dir / candidate.source_file
    if source_path.is_file():
        return source_path.read_bytes()
    if fallback_path.is_file():
        return fallback_path.read_bytes()
    raise FileNotFoundError(f"Missing source for {candidate.name}: {source_path}")


def materialize_candidate(candidate: Candidate) -> None:
    target_dir = OUT_ROOT / candidate.name
    payload = source_bytes(candidate, target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    (target_dir / candidate.source_file).write_bytes(payload)
    metadata = {
        "id": candidate.kernel_id,
        "title": candidate.title,
        "code_file": candidate.source_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": candidate.enable_gpu,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": candidate.dataset_sources,
        "kernel_sources": candidate.kernel_sources or [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "model_sources": [],
    }
    (target_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    for candidate in CANDIDATES:
        materialize_candidate(candidate)
        print(f"materialized {candidate.name}: {candidate.kernel_id}", flush=True)


if __name__ == "__main__":
    main()
