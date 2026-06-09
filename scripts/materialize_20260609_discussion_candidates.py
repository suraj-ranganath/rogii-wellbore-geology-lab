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
            "phongnguyn23021656/koolbox-offline",
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


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def patch_iaztec_koolbox_shim(notebook_path: Path) -> None:
    yaroslav_path = OUT_ROOT / "yaroslav_sel15_forced_selector" / "rogii-sel15-forced-selector.ipynb"
    if not yaroslav_path.is_file():
        raise FileNotFoundError(f"Missing Yaroslav shim source: {yaroslav_path}")

    yaroslav_nb = json.loads(yaroslav_path.read_text())
    shim_source = next(
        _cell_source(cell)
        for cell in yaroslav_nb["cells"]
        if "def _install_koolbox_trainer_shim" in _cell_source(cell)
    )
    shim_start = shim_source.index("    # Resolve koolbox exactly")
    shim_end = shim_source.index("    from pathlib import Path")
    shim_block = shim_source[shim_start:shim_end]

    old_block = (
        "    try:\n"
        "        from koolbox import Trainer\n"
        "    except ModuleNotFoundError as exc:\n"
        "        raise RuntimeError('The ridge artifact profiles require the original notebook dependency: koolbox.') from exc\n"
    )

    nb = json.loads(notebook_path.read_text())
    patched = False
    for cell in nb["cells"]:
        source = _cell_source(cell)
        if old_block in source:
            source = source.replace(old_block, shim_block)
            cell["source"] = source.splitlines(keepends=True)
            patched = True
            break
    if not patched:
        if "def _install_koolbox_trainer_shim" in notebook_path.read_text():
            return
        raise RuntimeError(f"Could not patch koolbox import block in {notebook_path}")

    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, separators=(",", ":")) + "\n")


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
    if candidate.name == "iaztec_ridge_artifact_param":
        patch_iaztec_koolbox_shim(target_dir / candidate.source_file)


def main() -> None:
    for candidate in CANDIDATES:
        materialize_candidate(candidate)
        print(f"materialized {candidate.name}: {candidate.kernel_id}", flush=True)


if __name__ == "__main__":
    main()
