"""Materialize final-five 2026-06-10 CNN-based candidates.

Creates two new Kaggle kernels:

1. ``medali_cnn_mtp_exact_override``: exact public CNN-MTP inference from
   medali1992/rogii-cnn-mtp-inference, converted to a script wrapper and with
   the guarded overlap override appended.
2. ``sp45h_cnn_mtp_blend020``: runs the completed bagged pure-SP45 orchestrator
   and medali CNN-MTP inference, then writes ``0.80 * SP45 + 0.20 * CNN`` before
   the same guarded override.

The other three final-five candidates reuse already-pushed kernels:
``sp45h_drift_mix``, ``sp45h_bag3_w072``, and ``sp45h_bag3_w100``.
"""

from __future__ import annotations

import base64
import json
import py_compile
import shutil
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle" / "kernels"
PUBLIC = ROOT / "outputs" / "public_kernels" / "medali_cnn_mtp"

SP45_BAGGED_SRC = KERNELS / "sp45h_bag3_w100" / "sp45h_bag3_w100.py"
OVERRIDE_SRC_PATH = KERNELS / "shared" / "pixiux_overlap_override.py"

COMPETITION = "rogii-wellbore-geology-prediction"

MEDALI_KERNEL_SOURCE = "medali1992/rogii-cnn-mtp-train"
DATASET_SOURCES = [
    "phongnguyn23021656/koolbox-offline",
    "fleongg/rogii-claude-models-pub",
    "ravaghi/wellbore-geology-prediction-artifacts",
]

OVERRIDE_PREAMBLE = '''

# ---- guarded train-overlap override (pixiux/rogii-dual-pipeline-blend) ----
import shutil as _pre_shutil
from pathlib import Path as _PrePath

_pre_w = _PrePath("/kaggle/working") if _PrePath("/kaggle/working").exists() else _PrePath(".")
if (_pre_w / "submission.csv").exists():
    _pre_shutil.copyfile(_pre_w / "submission.csv", _pre_w / "submission_no_override.csv")
'''

EXACT_WRAPPER = '''"""Exact medali CNN-MTP inference plus guarded override."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import zlib
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
RUNTIME_PAYLOADS = {runtime_payloads}


def write_runtime() -> None:
    for rel, payload in RUNTIME_PAYLOADS.items():
        path = WORK / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(zlib.decompress(base64.b85decode(payload)).decode("utf-8"))


def main() -> None:
    write_runtime()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORK) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    script = WORK / "inference.py"
    print(f"[cnn-exact] running {script}", flush=True)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(WORK), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"medali inference failed with rc={proc.returncode}")
    out = WORK / "submission.csv"
    if not out.exists():
        raise RuntimeError("medali inference did not create submission.csv")
    print(f"[cnn-exact] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
'''

BLEND_WRAPPER = '''"""Bagged pure-SP45 / medali CNN-MTP ensemble plus guarded override."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
W_SP45 = 0.80
W_CNN = 0.20
SP45_PAYLOAD = "{sp45_payload}"
RUNTIME_PAYLOADS = {runtime_payloads}


def write_runtime() -> None:
    for rel, payload in RUNTIME_PAYLOADS.items():
        path = WORK / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(zlib.decompress(base64.b85decode(payload)).decode("utf-8"))


def run_cmd(args: list[str], label: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORK) + os.pathsep + env.get("PYTHONPATH", "")
    env["SHOW_FIGS"] = "0"
    if label == "cnn_mtp":
        env["CUDA_VISIBLE_DEVICES"] = ""
    log = WORK / f"{label}.log"
    print(f"[blend] start {label}: {' '.join(args)}", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.run(args, cwd=str(WORK), env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[blend] done {label} rc={proc.returncode}", flush=True)
    if proc.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-80:]
        print("\\n".join(tail), flush=True)
        raise RuntimeError(f"{label} failed with rc={proc.returncode}")


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if list(df.columns) != ["id", "tvt"]:
        raise RuntimeError(f"{path}: bad columns {list(df.columns)}")
    df["tvt"] = df["tvt"].astype(float)
    if not np.isfinite(df["tvt"].to_numpy()).all():
        raise RuntimeError(f"{path}: non-finite predictions")
    return df


def main() -> None:
    write_runtime()
    cnn_script = WORK / "inference.py"
    run_cmd([sys.executable, str(cnn_script)], "cnn_mtp")
    cnn_path = WORK / "cnn_mtp_submission.csv"
    (WORK / "submission.csv").rename(cnn_path)

    sp45_code = zlib.decompress(base64.b85decode(SP45_PAYLOAD)).decode("utf-8")
    sp45_script = WORK / "sp45_bagged_payload.py"
    sp45_script.write_text(sp45_code)
    run_cmd([sys.executable, str(sp45_script)], "sp45_bagged")

    sp45_path = WORK / "bagged_sp45_fleongg_w1.00.csv"
    if not sp45_path.exists():
        raise RuntimeError("bagged SP45 payload did not emit bagged_sp45_fleongg_w1.00.csv")

    sp45 = load_submission(sp45_path).set_index("id")["tvt"]
    cnn = load_submission(cnn_path).set_index("id")["tvt"]
    if not sp45.index.equals(cnn.index):
        cnn = cnn.reindex(sp45.index)
        if cnn.isna().any():
            raise RuntimeError("CNN ids do not align to SP45 ids")

    final = (W_SP45 * sp45 + W_CNN * cnn).rename("tvt").reset_index()
    if not np.isfinite(final["tvt"].to_numpy()).all():
        raise RuntimeError("blend produced non-finite predictions")
    final.to_csv(WORK / "submission.csv", index=False)
    print(
        f"[blend] wrote submission.csv rows={len(final)} "
        f"range={final['tvt'].min():.3f}..{final['tvt'].max():.3f} "
        f"weights sp45={W_SP45:.2f} cnn={W_CNN:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def encode_payload(text: str) -> str:
    return base64.b85encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")


def write_metadata(
    kdir: Path,
    slug: str,
    title: str,
    code_file: str,
    *,
    dataset_sources: list[str],
    kernel_sources: list[str],
) -> None:
    meta = {
        "id": f"surajranganath17/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": dataset_sources,
        "kernel_sources": kernel_sources,
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def medali_runtime_payloads() -> str:
    files = [
        PUBLIC / "inference.py",
        PUBLIC / "src" / "__init__.py",
        PUBLIC / "src" / "config.py",
        PUBLIC / "src" / "dataset.py",
        PUBLIC / "src" / "model_sdf.py",
    ]
    if any(not path.exists() for path in files):
        raise FileNotFoundError("Run kaggle kernels output medali1992/rogii-cnn-mtp-inference first")
    payloads = {
        str(path.relative_to(PUBLIC)): encode_payload(path.read_text())
        for path in files
    }
    return repr(payloads)


def write_kernel(dir_name: str, slug: str, title: str, code: str, dataset_sources: list[str]) -> None:
    kdir = KERNELS / dir_name
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    code_file = f"{dir_name}.py"
    full_code = code + OVERRIDE_PREAMBLE + "\n" + OVERRIDE_SRC_PATH.read_text()
    (kdir / code_file).write_text(full_code)
    py_compile.compile(str(kdir / code_file), doraise=True)
    write_metadata(
        kdir,
        slug,
        title,
        code_file,
        dataset_sources=dataset_sources,
        kernel_sources=[MEDALI_KERNEL_SOURCE],
    )
    print(f"materialized {kdir.relative_to(ROOT)}")


def main() -> None:
    override_src = OVERRIDE_SRC_PATH.read_text()
    assert "GUARDED override" in override_src

    runtime_payloads = medali_runtime_payloads()
    write_kernel(
        "medali_cnn_mtp_exact_override",
        "rogii-medali-cnn-mtp-exact-override",
        "ROGII Medali CNN MTP Exact Override",
        EXACT_WRAPPER.replace("{runtime_payloads}", runtime_payloads),
        dataset_sources=[],
    )

    sp45_code = SP45_BAGGED_SRC.read_text()
    sp45_payload = encode_payload(sp45_code)
    blend_code = BLEND_WRAPPER.replace("{sp45_payload}", sp45_payload).replace(
        "{runtime_payloads}", runtime_payloads
    )
    write_kernel(
        "sp45h_cnn_mtp_blend020",
        "rogii-sp45h-cnn-mtp-blend020",
        "ROGII SP45H CNN MTP Blend020",
        blend_code,
        dataset_sources=DATASET_SOURCES,
    )


if __name__ == "__main__":
    main()
