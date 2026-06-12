"""Materialize 2026-06-11 final queue candidates.

Creates three runtime-safe fle3n hedge candidates to pair with the already
validated single-run SP45 recovery kernels. These are notebook forks of public
code, with dynamic hidden-rerun logic intact and no static public-test outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle" / "kernels"
EXTERNAL = ROOT / "external_kaggle"

COMPETITION = "rogii-wellbore-geology-prediction"
DATASET_SOURCES = [
    "phongnguyn23021656/koolbox-offline",
    "fleongg/rogii-claude-models-pub",
    "ravaghi/wellbore-geology-prediction-artifacts",
]

CANDIDATES = [
    {
        "dir_name": "fle3n_v5_exact_h050",
        "slug": "rogii-fle3n-v5-exact-h050",
        "title": "rogii-fle3n-v5-exact-h050",
        "source": EXTERNAL / "fle3n_v5" / "fle3n-rogii-v5.ipynb",
        "replacements": [],
    },
    {
        "dir_name": "fle3n_v5_w060_h0455",
        "slug": "rogii-fle3n-v5-w060-h0455",
        "title": "rogii-fle3n-v5-w060-h0455",
        "source": EXTERNAL / "fle3n_v5" / "fle3n-rogii-v5.ipynb",
        "replacements": [
            (
                "_final_name = 'submission_sp45_fleongg_w0.55.csv'",
                "_final_name = 'submission_sp45_fleongg_w0.60.csv'",
            ),
            (
                "XR_W_XFER = 0.5      # weight of the transferred train curve",
                "XR_W_XFER = 0.455    # weight of the transferred train curve",
            ),
        ],
    },
    {
        "dir_name": "fle3n_v5f_exact_h050",
        "slug": "rogii-fle3n-v5f-exact-h050",
        "title": "rogii-fle3n-v5f-exact-h050",
        "source": EXTERNAL / "fle3n_v5f_probe" / "fle3n-rogii-v5f-probe.ipynb",
        "replacements": [],
    },
]


def write_kernel(candidate: dict[str, object]) -> None:
    dir_name = str(candidate["dir_name"])
    slug = str(candidate["slug"])
    title = str(candidate["title"])
    source = Path(candidate["source"])
    if not source.exists():
        raise FileNotFoundError(source)

    kdir = KERNELS / dir_name
    if kdir.exists():
        for path in sorted(kdir.glob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                raise RuntimeError(f"unexpected directory in kernel dir: {path}")
    else:
        kdir.mkdir(parents=True)

    text = source.read_text()
    for old, new in candidate["replacements"]:  # type: ignore[index]
        if old not in text:
            raise RuntimeError(f"{dir_name}: replacement target missing: {old!r}")
        text = text.replace(old, new, 1)

    notebook = json.loads(text)
    notebook.setdefault("metadata", {})
    notebook["metadata"]["kaggle"] = {
        "accelerator": "none",
        "dataSources": [],
        "isGpuEnabled": False,
        "isInternetEnabled": False,
        "language": "python",
        "sourceType": "notebook",
    }
    code_file = f"{dir_name}.ipynb"
    (kdir / code_file).write_text(json.dumps(notebook, ensure_ascii=False))

    metadata = {
        "id": f"surajranganath17/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"materialized {kdir.relative_to(ROOT)}")


def main() -> None:
    for candidate in CANDIDATES:
        write_kernel(candidate)


if __name__ == "__main__":
    main()
