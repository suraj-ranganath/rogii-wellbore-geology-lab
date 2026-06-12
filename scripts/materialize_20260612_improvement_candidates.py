"""Materialize the 2026-06-12 runtime-safe improvement candidates.

The batch combines two requested exact fle3n reruns, two new public JAEMIN
seed7/affine variants, and one conservative rerun of the current best
SP45/fleongg w0.60 + guarded override candidate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle" / "kernels"
EXTERNAL = ROOT / "external_kaggle"

COMPETITION = "rogii-wellbore-geology-prediction"

FLE3N_DATASETS = [
    "phongnguyn23021656/koolbox-offline",
    "fleongg/rogii-claude-models-pub",
    "ravaghi/wellbore-geology-prediction-artifacts",
]

JAEMIN_SEED7_DATASETS = [
    "fleongg/rogii-claude-models-pub",
    "yieldsmarter/rogii2026-dependencies",
    "ravaghi/wellbore-geology-prediction-artifacts",
]


def _copy_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copy2(src, dst / src.name)


def _write_metadata(
    kdir: Path,
    *,
    slug: str,
    title: str,
    code_file: str,
    kernel_type: str,
    datasets: list[str],
) -> None:
    metadata = {
        "id": f"surajranganath17/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": kernel_type,
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": datasets,
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _notebook_candidate(
    *,
    dir_name: str,
    slug: str,
    title: str,
    source: Path,
    datasets: list[str],
) -> None:
    kdir = KERNELS / dir_name
    _copy_clean(source, kdir)
    code_file = source.name.replace("-", "_")
    (kdir / source.name).rename(kdir / code_file)

    notebook = json.loads((kdir / code_file).read_text())
    notebook.setdefault("metadata", {})
    notebook["metadata"]["kaggle"] = {
        "accelerator": "none",
        "dataSources": [],
        "isGpuEnabled": False,
        "isInternetEnabled": False,
        "language": "python",
        "sourceType": "notebook",
    }
    (kdir / code_file).write_text(json.dumps(notebook, ensure_ascii=False))
    _write_metadata(
        kdir,
        slug=slug,
        title=title,
        code_file=code_file,
        kernel_type="notebook",
        datasets=datasets,
    )
    print(f"materialized {kdir.relative_to(ROOT)}")


def _script_candidate(
    *,
    dir_name: str,
    slug: str,
    title: str,
    source: Path,
    code_file: str,
    datasets: list[str],
) -> None:
    kdir = KERNELS / dir_name
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    shutil.copy2(source, kdir / code_file)
    _write_metadata(
        kdir,
        slug=slug,
        title=title,
        code_file=code_file,
        kernel_type="script",
        datasets=datasets,
    )
    print(f"materialized {kdir.relative_to(ROOT)}")


def main() -> None:
    _notebook_candidate(
        dir_name="fle3n_v5_exact_r2",
        slug="rogii-fle3n-v5-exact-r2",
        title="rogii-fle3n-v5-exact-r2",
        source=EXTERNAL / "fle3n_v5" / "fle3n-rogii-v5.ipynb",
        datasets=FLE3N_DATASETS,
    )
    _notebook_candidate(
        dir_name="fle3n_v5f_exact_r2",
        slug="rogii-fle3n-v5f-exact-r2",
        title="rogii-fle3n-v5f-exact-r2",
        source=EXTERNAL / "fle3n_v5f_probe" / "fle3n-rogii-v5f-probe.ipynb",
        datasets=FLE3N_DATASETS,
    )
    _script_candidate(
        dir_name="jaemin_seed7_mtoshi_beicicc",
        slug="rogii-jaemin-seed7-mtoshi-beicicc",
        title="rogii-jaemin-seed7-mtoshi-beicicc",
        source=EXTERNAL
        / "jaemin_seed7_mtoshi_beicicc"
        / "rogii-seed7-mtoshi-beicicc-w483517-s265625-d2.py",
        code_file="jaemin_seed7_mtoshi_beicicc.py",
        datasets=JAEMIN_SEED7_DATASETS,
    )
    _script_candidate(
        dir_name="jaemin_affine_seed7_mtoshi",
        slug="rogii-jaemin-affine-seed7-mtoshi",
        title="rogii-jaemin-affine-seed7-mtoshi",
        source=EXTERNAL
        / "jaemin_affine_seed7_mtoshi"
        / "rogii-affine-seed7-mtoshi-beicicc-w633823m24.py",
        code_file="jaemin_affine_seed7_mtoshi.py",
        datasets=JAEMIN_SEED7_DATASETS,
    )
    _script_candidate(
        dir_name="jaemin_sp45_fleongg_w060s_r2",
        slug="rogii-sp45-fleongg-w060s-r2",
        title="rogii-sp45-fleongg-w060s-r2",
        source=KERNELS / "jaemin_sp45_fleongg_w060s" / "jaemin_sp45_fleongg_w060s.py",
        code_file="jaemin_sp45_fleongg_w060s_r2.py",
        datasets=FLE3N_DATASETS,
    )


if __name__ == "__main__":
    main()
