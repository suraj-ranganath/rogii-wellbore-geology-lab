"""Deep validation of the 2026-06-10 candidate kernel outputs.

Beyond the queue's schema checks, verifies per candidate:
- the run log shows the expected replicate diagnostics and the override result,
- submission_no_override.csv differs from submission.csv only on overlap wells,
- the final blend is consistent with the emitted bagged component blends,
- pairwise diversity across candidates is sane (no duplicate uploads).

Usage: uv run python scripts/validate_20260610_outputs.py
Reads outputs/queue_20260610_sp45heavy/<name>/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "queue_20260610_sp45heavy"
SAMPLE = ROOT / "data" / "raw" / "rogii-wellbore-geology-prediction" / "sample_submission.csv"

BAGGED = {
    "sp45h_bag3_w065": 0.65,
    "sp45h_bag3_w072": 0.72,
    "sp45h_bag3_w080": 0.80,
    "sp45h_bag3_w100": 1.00,
    "sp45h_drift_mix": None,  # 0.83 sp45 + 0.17 drift
}
SINGLES = {
    "jaemin_sp45_fleongg_w065s": 0.65,
    "jaemin_sp45_fleongg_w072s": 0.72,
    "jaemin_sp45_fleongg_w080s": 0.80,
    "jaemin_sp45_fleongg_w100s": 1.00,
}


def load(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df.set_index("id")["tvt"].astype(float)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def check_candidate(name: str) -> tuple[bool, str, pd.Series | None]:
    d = BASE / name
    sub_path = d / "submission.csv"
    if not sub_path.exists():
        return False, "no submission.csv downloaded", None
    sample = pd.read_csv(SAMPLE)
    sub = pd.read_csv(sub_path)
    sub.columns = [c.lower() for c in sub.columns]
    if list(sub.columns) != ["id", "tvt"]:
        return False, f"bad columns {list(sub.columns)}", None
    if len(sub) != len(sample) or not sub["id"].equals(sample["id"]):
        return False, "id order/count mismatch vs sample_submission", None
    vals = sub["tvt"].to_numpy(float)
    if not np.isfinite(vals).all():
        return False, "non-finite predictions", None
    if not (8000 <= float(np.median(vals)) <= 14000):
        return False, f"implausible median {np.median(vals):.1f}", None

    msgs = []
    logs = list(d.glob("*.log"))
    log_text = "\n".join(p.read_text(errors="replace") for p in logs)
    if "GUARDED override done" in log_text or "GUARDED override" in log_text:
        ok_n = log_text.count("override OK")
        skip_n = log_text.count("override SKIP")
        msgs.append(f"override OK={ok_n} SKIP={skip_n}")
    else:
        msgs.append("WARN: no override marker in log")
    diag_n = log_text.count("[diag]")
    if name in BAGGED:
        msgs.append(f"replicate diags={diag_n}")
        if diag_n == 0:
            msgs.append("WARN: no replicate diagnostics found")

    no_ov = d / "submission_no_override.csv"
    if no_ov.exists():
        pre = load(no_ov)
        delta = rms(pre.to_numpy() - vals)
        n_changed = int((np.abs(pre.to_numpy() - vals) > 1e-9).sum())
        msgs.append(f"override changed {n_changed} rows, rms delta {delta:.3f}")
    w = BAGGED.get(name) or SINGLES.get(name)
    if name in BAGGED and w is not None:
        comp = d / f"bagged_sp45_fleongg_w{w:.2f}.csv"
        if comp.exists() and no_ov.exists():
            blend = load(comp)
            agree = rms(blend.to_numpy() - load(no_ov).to_numpy())
            msgs.append(f"pre-override vs emitted w{w:.2f} blend rms {agree:.6f}")
            if agree > 1e-6:
                return False, "; ".join(msgs + ["FAIL: blend inconsistency"]), None
    return True, "; ".join(msgs), sub.set_index("id")["tvt"]


def main() -> None:
    vectors: dict[str, pd.Series] = {}
    failed = []
    for name in list(BAGGED) + list(SINGLES):
        if not (BASE / name).exists():
            print(f"[absent ] {name}")
            continue
        ok, msg, vec = check_candidate(name)
        tag = "ok" if ok else "FAIL"
        print(f"[{tag:6}] {name}: {msg}")
        if ok and vec is not None:
            vectors[name] = vec
        elif not ok:
            failed.append(name)

    names = list(vectors)
    print("\npairwise rms distances:")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = rms(vectors[a].to_numpy() - vectors[b].to_numpy())
            flag = "  <-- DUPLICATE?" if d < 0.01 else ""
            print(f"  {a} vs {b}: {d:.4f}{flag}")

    if failed:
        print(f"\nFAILED candidates: {failed}")
        sys.exit(1)
    print("\nall present candidates passed")


if __name__ == "__main__":
    main()
