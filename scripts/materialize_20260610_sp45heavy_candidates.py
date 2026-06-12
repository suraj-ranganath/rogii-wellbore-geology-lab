"""Materialize the 2026-06-10 SP45-heavy candidate kernels.

Two kernel families:

1. Bagged orchestrators (plan A): embed the proven JAEMIN SP45/fleongg script
   (and for the drift mix, the drift geosteering notebook code) as compressed
   payloads, run them as subprocesses N times with shifted seeds, average the
   component predictions across replicates, then blend with the target weights.
   Rationale: observed cross-run component noise on Kaggle is RMS 0.46 (SP45)
   and 1.32 (fleongg); averaging 3 replicates cuts that variance ~sqrt(3)x.

2. Single-run patched kernels (plan B backups): the exact JAEMIN code path that
   already scored 7.551/7.609, with only the emitted weight grid extended and a
   different final weight selected. Used if a plan-A preflight fails.

Both families append the pixiux guarded train-overlap override (public kernel
`pixiux/rogii-dual-pipeline-blend`, 121 votes, LB-validated 7.572 -> 7.519 on
the same blend family). It is hidden-rerun safe: at rerun time it recomputes a
physics TVT from the train copy's contacts for any test well whose id prefix
exists in train, validates against the test copy's known prefix interpolated by
MD (override only if prefix RMSE < 1 ft, >= 50 comparable prefix rows, >= 100
valid phys rows, in-range MDs only), and otherwise keeps the blend. On a
non-overlapping test set it degrades to a no-op.

Run: uv run python scripts/materialize_20260610_sp45heavy_candidates.py
Then push each kernel with: uv run kaggle kernels push -p kaggle/kernels/<dir>
"""

from __future__ import annotations

import base64
import json
import py_compile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle" / "kernels"

JAEMIN_SRC = KERNELS / "jaemin_sp45_fleongg_w060" / "jaemin_sp45_fleongg_w060.py"
DRIFT_NB = KERNELS / "drift_geosteering_infer" / "drift_geosteering_infer.ipynb"

DATASET_SOURCES = [
    "phongnguyn23021656/koolbox-offline",
    "fleongg/rogii-claude-models-pub",
    "ravaghi/wellbore-geology-prediction-artifacts",
]
COMPETITION = "rogii-wellbore-geology-prediction"

EMIT_WEIGHTS = [0.60, 0.62, 0.65, 0.70, 0.72, 0.80, 0.90, 1.00]

# (dir_name, kernel_slug, title, final_spec)
# final_spec: ("two_way", w_sp45) or ("drift_mix", w_sp45, w_drift)
BAGGED_CANDIDATES = [
    ("sp45h_bag3_w065", "rogii-sp45h-bag3-w065", "ROGII SP45H Bag3 W065", ("two_way", 0.65)),
    ("sp45h_bag3_w072", "rogii-sp45h-bag3-w072", "ROGII SP45H Bag3 W072", ("two_way", 0.72)),
    ("sp45h_bag3_w080", "rogii-sp45h-bag3-w080", "ROGII SP45H Bag3 W080", ("two_way", 0.80)),
    ("sp45h_bag3_w100", "rogii-sp45h-bag3-w100", "ROGII SP45H Bag3 W100", ("two_way", 1.00)),
    ("sp45h_drift_mix", "rogii-sp45h-drift-mix", "ROGII SP45H Drift Mix", ("drift_mix", 0.83, 0.17)),
]

SINGLE_CANDIDATES = [
    ("jaemin_sp45_fleongg_w060s", "rogii-sp45-fleongg-w060s", "ROGII SP45 Fleongg W060S", 0.60),
    (
        "jaemin_sp45_fleongg_w065s",
        "rogii-sp45-fleongg-w065s-runtime-safe",
        "ROGII SP45 Fleongg W065S Runtime Safe",
        0.65,
    ),
    (
        "jaemin_sp45_fleongg_w072s",
        "rogii-sp45-fleongg-w072s-runtime-safe",
        "ROGII SP45 Fleongg W072S Runtime Safe",
        0.72,
    ),
    (
        "jaemin_sp45_fleongg_w080s",
        "rogii-sp45-fleongg-w080s-runtime-safe",
        "ROGII SP45 Fleongg W080S Runtime Safe",
        0.80,
    ),
    (
        "jaemin_sp45_fleongg_w100s",
        "rogii-sp45-fleongg-w100s-runtime-safe",
        "ROGII SP45 Fleongg W100S Runtime Safe",
        1.00,
    ),
]

OLD_WEIGHT_LINE = "for _w_sp45 in [0.50, 0.52, 0.55, 0.58, 0.60]:"
NEW_WEIGHT_LINE = "for _w_sp45 in [0.60, 0.62, 0.65, 0.70, 0.72, 0.80, 0.90, 1.00]:"
OLD_FINAL_LINE = "_final_name = 'submission_sp45_fleongg_w0.60.csv'"

OVERRIDE_SRC_PATH = KERNELS / "shared" / "pixiux_overlap_override.py"

OVERRIDE_PREAMBLE = '''

# ---- guarded train-overlap override (pixiux/rogii-dual-pipeline-blend) ----
import shutil as _pre_shutil
from pathlib import Path as _PrePath

_pre_w = _PrePath("/kaggle/working") if _PrePath("/kaggle/working").exists() else _PrePath(".")
if (_pre_w / "submission.csv").exists():
    _pre_shutil.copyfile(_pre_w / "submission.csv", _pre_w / "submission_no_override.csv")
'''


ORCHESTRATOR_TEMPLATE = '''"""Bagged SP45/fleongg{maybe_drift} orchestrator.

Runs the proven JAEMIN SP45+fleongg pipeline N_JAEMIN_REPS times (seed-shifted
replicates) as subprocesses, averages component predictions across replicates,
then writes the final weighted blend. Hidden-rerun safe: every replicate
recomputes predictions from the mounted competition data at run time.
"""

import base64
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

N_JAEMIN_REPS = {n_jaemin_reps}
N_DRIFT_REPS = {n_drift_reps}
EMIT_WEIGHTS = {emit_weights}
FINAL_SPEC = {final_spec}

PAYLOADS = {{
{payload_entries}
}}


def decode_payload(name):
    return zlib.decompress(base64.b85decode(PAYLOADS[name])).decode("utf-8")


def run_payload(name, rep, replacements):
    code = decode_payload(name)
    for old, new in replacements:
        if old not in code:
            raise RuntimeError(f"replacement target missing in {{name}}: {{old!r}}")
        code = code.replace(old, new)
    script = WORK / f"{{name}}_rep{{rep}}.py"
    script.write_text(code)
    log = WORK / f"{{name}}_rep{{rep}}.log"
    env = dict(os.environ)
    env["SHOW_FIGS"] = "0"
    print(f"[orchestrator] start {{name}} rep {{rep}}", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(WORK),
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(f"[orchestrator] done {{name}} rep {{rep}} rc={{proc.returncode}}", flush=True)
    if proc.returncode != 0:
        tail = log.read_text().splitlines()[-30:]
        print("\\n".join(tail), flush=True)
    return proc.returncode == 0


def seed_replacements(rep):
    if rep == 0:
        return []
    return [
        ("SEED=42", f"SEED={{42 + 1000 * rep}}"),
        ("seed = 42", f"seed = {{42 + 1000 * rep}}"),
        ("seed_base=0", f"seed_base={{1000 * rep}}"),
    ]


def collect(src_name, dest_name):
    src = WORK / src_name
    if not src.exists():
        raise RuntimeError(f"expected output missing: {{src}}")
    shutil.move(str(src), str(WORK / dest_name))


def average_reps(prefix, n_reps):
    frames = []
    for rep in range(n_reps):
        path = WORK / f"{{prefix}}_rep{{rep}}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        frames.append(df.set_index("id")["tvt"].astype(float))
    if not frames:
        raise RuntimeError(f"no successful replicates for {{prefix}}")
    base = frames[0]
    for f in frames[1:]:
        if not f.index.equals(base.index):
            raise RuntimeError(f"replicate id mismatch for {{prefix}}")
    stack = np.vstack([f.to_numpy() for f in frames])
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            d = float(np.sqrt(np.mean((stack[i] - stack[j]) ** 2)))
            print(f"[diag] {{prefix}} rep{{i}} vs rep{{j}} rms diff = {{d:.4f}}", flush=True)
    return pd.Series(stack.mean(axis=0), index=base.index, name="tvt")


def main():
    jaemin_ok = 0
    for rep in range(N_JAEMIN_REPS):
        ok = run_payload("jaemin", rep, seed_replacements(rep))
        if ok:
            try:
                collect("sp45_projection_submission.csv", f"sp45_rep{{rep}}.csv")
                collect("fleongg_pretrained_submission.csv", f"fleongg_rep{{rep}}.csv")
                jaemin_ok += 1
            except RuntimeError as exc:
                print(f"[orchestrator] rep {{rep}} outputs incomplete: {{exc}}", flush=True)
        for leftover in WORK.glob("submission_sp45_fleongg_w*.csv"):
            leftover.unlink()
        if (WORK / "submission.csv").exists():
            (WORK / "submission.csv").unlink()
    if jaemin_ok == 0:
        raise RuntimeError("all jaemin replicates failed")

    sp45 = average_reps("sp45", N_JAEMIN_REPS)
    fleongg = average_reps("fleongg", N_JAEMIN_REPS)
    if not sp45.index.equals(fleongg.index):
        raise RuntimeError("sp45/fleongg id mismatch")

    drift = None
    if N_DRIFT_REPS > 0:
        drift_ok = 0
        for rep in range(N_DRIFT_REPS):
            ok = run_payload("drift", rep, [])
            if ok:
                collect("submission.csv", f"drift_rep{{rep}}.csv")
                drift_ok += 1
        if drift_ok == 0:
            raise RuntimeError("all drift replicates failed")
        drift = average_reps("drift", N_DRIFT_REPS)
        drift = drift.reindex(sp45.index)
        if drift.isna().any():
            raise RuntimeError("drift id mismatch vs sp45")

    for w in EMIT_WEIGHTS:
        blend = w * sp45 + (1.0 - w) * fleongg
        out = blend.rename("tvt").reset_index()
        out.to_csv(WORK / f"bagged_sp45_fleongg_w{{w:.2f}}.csv", index=False)

    if FINAL_SPEC[0] == "two_way":
        w = FINAL_SPEC[1]
        final = w * sp45 + (1.0 - w) * fleongg
        desc = f"two_way w_sp45={{w}}"
    elif FINAL_SPEC[0] == "drift_mix":
        w_sp, w_dr = FINAL_SPEC[1], FINAL_SPEC[2]
        w_fle = 1.0 - w_sp - w_dr
        final = w_sp * sp45 + w_dr * drift + w_fle * fleongg
        desc = f"drift_mix w_sp45={{w_sp}} w_drift={{w_dr}} w_fleongg={{w_fle:.2f}}"
    else:
        raise RuntimeError(f"unknown final spec {{FINAL_SPEC}}")

    out = final.rename("tvt").reset_index()
    if not np.isfinite(out["tvt"]).all():
        raise RuntimeError("non-finite final predictions")
    out.to_csv(WORK / "submission.csv", index=False)
    print(
        f"[orchestrator] wrote submission.csv ({{desc}}) rows={{len(out)}} "
        f"range={{out['tvt'].min():.3f}}..{{out['tvt'].max():.3f}}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def encode_payload(text: str) -> str:
    return base64.b85encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")


def extract_drift_script() -> str:
    nb = json.loads(DRIFT_NB.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return "\n\n".join("".join(c["source"]) for c in cells)


def write_kernel(dir_name: str, slug: str, title: str, code: str, code_file: str) -> None:
    kdir = KERNELS / dir_name
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / code_file).write_text(code)
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
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    py_compile.compile(str(kdir / code_file), doraise=True)
    print(f"materialized {kdir.relative_to(ROOT)}")


def build_orchestrator(final_spec: tuple, with_drift: bool, jaemin_code: str, drift_code: str) -> str:
    payloads = [f'    "jaemin": "{encode_payload(jaemin_code)}",']
    if with_drift:
        payloads.append(f'    "drift": "{encode_payload(drift_code)}",')
    code = ORCHESTRATOR_TEMPLATE
    # un-double literal braces BEFORE inserting payloads: base85 text can
    # legitimately contain {{ or }} sequences that must stay untouched
    code = code.replace("{{", "\x00").replace("}}", "\x01")
    code = code.replace("\x00", "{").replace("\x01", "}")
    code = code.replace("{maybe_drift}", "/drift" if with_drift else "")
    code = code.replace("{n_jaemin_reps}", "3")
    code = code.replace("{n_drift_reps}", "2" if with_drift else "0")
    code = code.replace("{emit_weights}", repr(EMIT_WEIGHTS))
    code = code.replace("{final_spec}", repr(final_spec))
    code = code.replace("{payload_entries}", "\n".join(payloads))
    return code


def main() -> None:
    jaemin_code = JAEMIN_SRC.read_text()
    for target in (OLD_WEIGHT_LINE, OLD_FINAL_LINE, "SEED=42", "seed = 42", "seed_base=0"):
        assert target in jaemin_code, f"missing patch target: {target!r}"
    drift_code = extract_drift_script()
    override_code = OVERRIDE_PREAMBLE + "\n" + OVERRIDE_SRC_PATH.read_text()

    for dir_name, slug, title, final_spec in BAGGED_CANDIDATES:
        with_drift = final_spec[0] == "drift_mix"
        code = build_orchestrator(final_spec, with_drift, jaemin_code, drift_code)
        code = code + override_code
        write_kernel(dir_name, slug, title, code, f"{dir_name}.py")

    for dir_name, slug, title, w in SINGLE_CANDIDATES:
        code = jaemin_code.replace(OLD_WEIGHT_LINE, NEW_WEIGHT_LINE)
        new_final = f"_final_name = 'submission_sp45_fleongg_w{w:.2f}.csv'"
        code = code.replace(OLD_FINAL_LINE, new_final)
        assert new_final in code
        code = code + override_code
        write_kernel(dir_name, slug, title, code, f"{dir_name}.py")


if __name__ == "__main__":
    main()
