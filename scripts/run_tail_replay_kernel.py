"""Run the UNMODIFIED champion kernel on a synthetic tail-replay dataset.

We do NOT hand-port the pipeline. We make a copy of the kernel ``.py`` and apply
a few *surgical* string replacements that redirect its hardcoded Kaggle paths to
local/synthetic locations, then execute it. All redirection targets come from
environment variables so the same patched copy is reproducible.

Redirections (string-level only):
- koolbox dir  -> ROGII_KOOLBOX
- CFG.dataset_path (line ~66) -> ROGII_DATA (the synthetic competition dir)
- CFG.artifacts_path (line ~67) -> ROGII_ARTIFACTS

The fleongg ``_find_data()`` already honours ``ROGII_DATA`` as a fallback and the
``/kaggle/input/**`` globs resolve to nothing locally, so both the SP45 and the
fleongg stacks retrain from scratch on the synthetic train wells (no held-out
leakage). To force that explicitly we point ROGII_ARTIFACTS at a directory that
does NOT contain ``data/train.csv`` or ``models/<name>/*.pkl``.

The kernel writes its outputs into the current working directory (its
``/kaggle/working`` checks fall back to ``.``), so we run it inside the run dir.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL = (
    ROOT / "kaggle/kernels/jaemin_sp45_fleongg_w060/jaemin_sp45_fleongg_w060.py"
)

REPLACEMENTS = [
    (
        "kb_dir = '/kaggle/input/koolbox-offline'",
        "kb_dir = os.environ.get('ROGII_KOOLBOX', '/kaggle/input/koolbox-offline')",
    ),
    (
        'dataset_path = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")',
        'dataset_path = Path(os.environ.get("ROGII_DATA", "/kaggle/input/competitions/rogii-wellbore-geology-prediction"))',
    ),
    (
        'artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")',
        'artifacts_path = Path(os.environ.get("ROGII_ARTIFACTS", "/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts"))',
    ),
]

# Seed-perturbation replacements: applied only when a non-zero offset is given so
# replicate runs differ. Each (old, new) rewrites a single hardcoded booster seed
# literal to add ``_ROGII_RR`` (the offset). This keeps the change purely string-level
# and leaves the default (offset 0) run byte-identical to the unmodified kernel.
SEED_REPLACEMENTS = [
    ("        seed=123\n", "        seed=123 + _ROGII_RR\n"),
    ("        random_state=0,\n", "        random_state=0 + _ROGII_RR,\n"),
    ("        random_state=29,\n", "        random_state=29 + _ROGII_RR,\n"),
    ("        random_seed=7\n", "        random_seed=7 + _ROGII_RR\n"),
    ("        random_seed=123\n", "        random_seed=123 + _ROGII_RR\n"),
    (
        "n_estimators=n, seed=123),",
        "n_estimators=n, seed=123 + _ROGII_RR),",
    ),
    (
        "n_estimators=min(2*n, 10000), random_state=0),",
        "n_estimators=min(2*n, 10000), random_state=0 + _ROGII_RR),",
    ),
    (
        "n_estimators=min(2*n, 10000), random_state=29),",
        "n_estimators=min(2*n, 10000), random_state=29 + _ROGII_RR),",
    ),
    (
        "od_wait=300, verbose=0, learning_rate=0.02, random_seed=7),",
        "od_wait=300, verbose=0, learning_rate=0.02, random_seed=7 + _ROGII_RR),",
    ),
    (
        "od_wait=300, verbose=0, learning_rate=0.03, random_seed=123),",
        "od_wait=300, verbose=0, learning_rate=0.03, random_seed=123 + _ROGII_RR),",
    ),
]

# Injected right after the first import line so ``_ROGII_RR`` is defined module-wide.
SEED_PRELUDE_ANCHOR = "import sys, os, glob, subprocess\n"
SEED_PRELUDE = (
    "import sys, os, glob, subprocess\n"
    "_ROGII_RR = int(os.environ.get('ROGII_SEED_OFFSET', '0'))\n"
)


def patch_kernel(src: Path, dst: Path, seed_offset: int) -> None:
    text = src.read_text()
    for old, new in REPLACEMENTS:
        if old not in text:
            raise SystemExit(f"Expected anchor not found in kernel: {old!r}")
        text = text.replace(old, new)
    if seed_offset:
        if SEED_PRELUDE_ANCHOR not in text:
            raise SystemExit("Seed prelude anchor not found in kernel.")
        text = text.replace(SEED_PRELUDE_ANCHOR, SEED_PRELUDE, 1)
        for old, new in SEED_REPLACEMENTS:
            if old not in text:
                raise SystemExit(f"Seed anchor not found in kernel: {old!r}")
            text = text.replace(old, new)
    dst.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--data-dir", type=Path, required=True, help="Synthetic comp dir")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output/cwd for the run")
    parser.add_argument("--koolbox", type=Path, required=True, help="Dir with koolbox wheel/pkg")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifacts dir; default = empty dir so stacks retrain (no leakage).",
    )
    parser.add_argument("--use-gpu", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--fast", type=int, default=0)
    parser.add_argument("--n-train-wells", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Added to every booster seed for replicate variance (0 = unmodified).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = args.artifacts or (run_dir / "_empty_artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    patched = run_dir / "patched_kernel.py"
    patch_kernel(args.kernel, patched, args.seed_offset)

    env = dict(os.environ)
    env["ROGII_DATA"] = str(args.data_dir.resolve())
    env["ROGII_ARTIFACTS"] = str(artifacts.resolve())
    env["ROGII_KOOLBOX"] = str(args.koolbox.resolve())
    env["USE_GPU"] = args.use_gpu
    env["FAST"] = str(args.fast)
    env["N_TRAIN_WELLS"] = str(args.n_train_wells)
    env["SHOW_FIGS"] = "0"
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # Seed override: the kernel uses fixed seeds internally; we vary the seed by
    # exporting PYTHONHASHSEED and a ROGII_SEED env that the bagging wrapper reads
    # only if present. The dominant variance source is GPU/boosting nondeterminism
    # plus subsample seeds, which differ per process run anyway.
    env["ROGII_SEED"] = str(args.seed)
    env["ROGII_SEED_OFFSET"] = str(args.seed_offset)
    env["PYTHONHASHSEED"] = str(args.seed)

    print(f"[run] cwd={run_dir} data={env['ROGII_DATA']} gpu={args.gpu_id} "
          f"use_gpu={args.use_gpu} fast={args.fast} seed={args.seed} "
          f"seed_offset={args.seed_offset}", flush=True)

    proc = subprocess.run(
        [sys.executable, str(patched.resolve())],
        cwd=str(run_dir),
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(f"kernel run failed with code {proc.returncode}")
    print(f"[run] done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
