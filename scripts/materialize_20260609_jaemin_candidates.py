from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "kaggle" / "kernels"
RESEARCH_ROOT = Path("/tmp/rogii_750_research")


@dataclass(frozen=True)
class Candidate:
    name: str
    source_dir: Path
    source_file: str
    kernel_id: str
    title: str
    code_file: str
    dataset_sources: list[str]
    final_submission_name: str | None = None


CANDIDATES = [
    Candidate(
        name="jaemin_sp45_fleongg_exact",
        source_dir=RESEARCH_ROOT / "jaemin_blend_v2",
        source_file="rogii-sp45-fleongg-blend-v2.py",
        kernel_id="surajranganath17/rogii-sp45-fleongg-blend-exact",
        title="ROGII SP45 Fleongg Blend Exact",
        code_file="jaemin_sp45_fleongg_exact.py",
        dataset_sources=[
            "phongnguyn23021656/koolbox-offline",
            "fleongg/rogii-claude-models-pub",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ],
    ),
    Candidate(
        name="jaemin_sp45_fleongg_w060",
        source_dir=RESEARCH_ROOT / "jaemin_blend_v2",
        source_file="rogii-sp45-fleongg-blend-v2.py",
        kernel_id="surajranganath17/rogii-sp45-fleongg-w060",
        title="ROGII SP45 Fleongg W060",
        code_file="jaemin_sp45_fleongg_w060.py",
        dataset_sources=[
            "phongnguyn23021656/koolbox-offline",
            "fleongg/rogii-claude-models-pub",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ],
        final_submission_name="submission_sp45_fleongg_w0.60.csv",
    ),
]


DEFAULT_FINAL_NAME = "_final_name = 'submission_sp45_fleongg_w0.55.csv'"


def source_code(candidate: Candidate, target_dir: Path) -> str:
    source_path = candidate.source_dir / candidate.source_file
    fallback_path = target_dir / candidate.code_file
    if source_path.is_file():
        return source_path.read_text()
    if fallback_path.is_file():
        return fallback_path.read_text()
    raise FileNotFoundError(f"Missing source for {candidate.name}: {source_path}")


def materialize_candidate(candidate: Candidate) -> None:
    target_dir = OUT_ROOT / candidate.name
    code = source_code(candidate, target_dir)
    if candidate.final_submission_name is not None:
        replacement = f"_final_name = {candidate.final_submission_name!r}"
        if DEFAULT_FINAL_NAME not in code:
            raise RuntimeError(f"Could not find final-name block in {candidate.name}")
        code = code.replace(DEFAULT_FINAL_NAME, replacement)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    (target_dir / candidate.code_file).write_text(code)
    metadata = {
        "id": candidate.kernel_id,
        "title": candidate.title,
        "code_file": candidate.code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": candidate.dataset_sources,
        "kernel_sources": [],
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
