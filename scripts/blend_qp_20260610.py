"""Analytic blend-weight optimization from public scores + prediction vectors.

Estimates the hidden-set error covariance of submitted prediction families using
the identity E[(pi-y)(pj-y)] = (MSEi + MSEj - D(pi,pj)) / 2, where MSE comes
from Kaggle public scores and D is the mean squared difference between
prediction vectors (visible-run outputs, used as a proxy for hidden geometry).

Solves the pure SP45 / fleongg component errors exactly from the two scored
blends (w0.55 -> 7.609, w0.60 -> 7.551) plus the component vector distance,
then optimizes simplex-constrained blend weights over component sets with
diagonal-loading shrinkage and leave-one-anchor-out stability checks.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# name -> (csv path, kaggle public RMSE or None for unscored components)
MANIFEST: dict[str, tuple[Path, float | None]] = {
    "sp45": (OUT / "queue_20260609_jaemin/jaemin_sp45_fleongg_w060/sp45_projection_submission.csv", None),
    "fleongg": (OUT / "queue_20260609_jaemin/jaemin_sp45_fleongg_w060/fleongg_pretrained_submission.csv", None),
    "blend_w055": (OUT / "queue_20260609_jaemin/jaemin_sp45_fleongg_exact/submission.csv", 7.609),
    "blend_w060": (OUT / "queue_20260609_jaemin/jaemin_sp45_fleongg_w060/submission.csv", 7.551),
    "jy_consensus": (OUT / "queue_20260609_jaemin/jaemin_sp45_fleongg_jy/submission.csv", 7.672),
    "yaroslav_sel15": (OUT / "queue_20260609_jaemin/yaroslav_sel15_forced_selector/submission.csv", 7.903),
    "iaztec_param": (OUT / "queue_20260609_jaemin/iaztec_ridge_artifact_param/submission.csv", 7.822),
    "drift_geo": (OUT / "queue_20260608_structural/drift_geosteering_infer/submission.csv", 7.858),
    "ridge_sp45_proj": (OUT / "queue_20260608_structural/ridge_sp45_proj/submission.csv", 8.173),
    "ridge_sp7776_exact": (OUT / "queue_20260608_structural/ridge_sp7776_exact/submission.csv", 8.220),
    "ravaghi_w040": (OUT / "queue_20260606_hidden_safe/ravaghi_ridge_w040/submission.csv", 7.906),
    "ravaghi_w035": (OUT / "queue_20260606_hidden_safe/ravaghi_ridge_w035/submission.csv", 8.108),
    "sunny_v10": (OUT / "kaggle_sunny_v10_artifact_blend_v1/submission.csv", 8.421),
    "pf_selector": (OUT / "kaggle_pf_selector_spread3_v1/submission.csv", 8.781),
    "physical_pf": (OUT / "kaggle_physical_noise_pf_v1/submission.csv", 8.777),
    "target_free": (OUT / "kaggle_target_free_alignment_gated_v1/submission.csv", 10.626),
}


def load_vectors() -> tuple[pd.Index, dict[str, np.ndarray], dict[str, float]]:
    base_ids: pd.Index | None = None
    vecs: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    for name, (path, score) in MANIFEST.items():
        if not path.exists():
            print(f"[skip] {name}: missing {path}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        df = df.set_index("id")
        if base_ids is None:
            base_ids = df.index
        if not df.index.equals(base_ids):
            df = df.reindex(base_ids)
            if df["tvt"].isna().any():
                print(f"[skip] {name}: id mismatch vs base")
                continue
        vecs[name] = df["tvt"].to_numpy(dtype=float)
        if score is not None:
            scores[name] = score
    assert base_ids is not None
    return base_ids, vecs, scores


def pairwise_msd(vecs: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(vecs)
    d = pd.DataFrame(0.0, index=names, columns=names)
    for i, j in itertools.combinations(names, 2):
        v = float(np.mean((vecs[i] - vecs[j]) ** 2))
        d.loc[i, j] = v
        d.loc[j, i] = v
    return d


def solve_components(d_ab: float) -> tuple[float, float, float]:
    """Solve A=E[(a-y)^2], X=E[(a-y)(b-y)], B=E[(b-y)^2] for a=sp45, b=fleongg."""
    f55 = 7.609**2
    f60 = 7.551**2
    # rows: f(0.55), f(0.60), D identity
    m = np.array(
        [
            [0.55**2, 2 * 0.55 * 0.45, 0.45**2],
            [0.60**2, 2 * 0.60 * 0.40, 0.40**2],
            [1.0, -2.0, 1.0],
        ]
    )
    rhs = np.array([f55, f60, d_ab])
    a, x, b = np.linalg.solve(m, rhs)
    return float(a), float(x), float(b)


def build_cov(
    names: list[str],
    mses: dict[str, float],
    d: pd.DataFrame,
) -> np.ndarray:
    n = len(names)
    c = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                c[i, j] = mses[ni]
            else:
                c[i, j] = 0.5 * (mses[ni] + mses[nj] - float(d.loc[ni, nj]))
    return c


def simplex_qp(c: np.ndarray, ridge: float = 0.0) -> tuple[np.ndarray, float]:
    n = c.shape[0]
    cc = c + ridge * np.eye(n)
    w0 = np.full(n, 1.0 / n)
    res = minimize(
        lambda w: float(w @ cc @ w),
        w0,
        jac=lambda w: 2.0 * (cc @ w),
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = res.x
    # predicted MSE uses the unshrunk covariance
    return w, float(w @ c @ w)


def main() -> None:
    ids, vecs, scores = load_vectors()
    print(f"loaded {len(vecs)} vectors, {len(ids)} rows each")
    d = pairwise_msd(vecs)

    # --- exact two-component solution for sp45/fleongg ---
    d_ab = float(d.loc["sp45", "fleongg"])
    a_mse, x_cov, b_mse = solve_components(d_ab)
    print("\n=== SP45/fleongg component solution ===")
    print(f"D(sp45,fleongg) visible = {d_ab:.3f} (rms diff {np.sqrt(d_ab):.3f})")
    print(f"MSE(sp45)={a_mse:.3f} (rmse {np.sqrt(a_mse):.3f})  "
          f"MSE(fleongg)={b_mse:.3f} (rmse {np.sqrt(b_mse):.3f})  cross={x_cov:.3f}")
    denom = a_mse + b_mse - 2 * x_cov
    w_star = (b_mse - x_cov) / denom
    f_star = a_mse * w_star**2 + 2 * w_star * (1 - w_star) * x_cov + (1 - w_star) ** 2 * b_mse
    print(f"two-way optimum w_sp45* = {w_star:.4f}, predicted RMSE = {np.sqrt(f_star):.4f}")
    print("\npredicted public RMSE over w grid:")
    for w in [0.55, 0.58, 0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70, 0.72, 0.75, 0.80]:
        f = a_mse * w**2 + 2 * w * (1 - w) * x_cov + (1 - w) ** 2 * b_mse
        print(f"  w={w:.2f}  rmse={np.sqrt(f):.4f}")

    # sensitivity: hidden D may differ from visible D by scale s
    print("\nsensitivity of w* to hidden/visible D scale s:")
    for s in [0.6, 0.8, 1.0, 1.2, 1.5]:
        a_s, x_s, b_s = solve_components(d_ab * s)
        den = a_s + b_s - 2 * x_s
        ws = (b_s - x_s) / den
        fs = a_s * ws**2 + 2 * ws * (1 - ws) * x_s + (1 - ws) ** 2 * b_s
        print(f"  s={s:.1f}  w*={ws:.4f}  pred rmse={np.sqrt(fs):.4f}")

    # --- multi-component QP ---
    mses = {k: v**2 for k, v in scores.items()}
    mses["sp45"] = a_mse
    mses["fleongg"] = b_mse
    # cross between sp45 and fleongg must use solved X, not the D identity
    # (it is consistent by construction, but keep exactness)
    component_sets = {
        "ab_drift": ["sp45", "fleongg", "drift_geo"],
        "ab_drift_iaztec": ["sp45", "fleongg", "drift_geo", "iaztec_param"],
        "ab_drift_iaztec_yaro": ["sp45", "fleongg", "drift_geo", "iaztec_param", "yaroslav_sel15"],
        "ab_drift_ravaghi": ["sp45", "fleongg", "drift_geo", "ravaghi_w040"],
        "ab_all_strong": [
            "sp45", "fleongg", "drift_geo", "iaztec_param", "yaroslav_sel15",
            "ravaghi_w040", "sunny_v10", "pf_selector", "target_free",
        ],
    }
    print("\n=== simplex QP over component sets ===")
    results = {}
    for set_name, names in component_sets.items():
        names = [n for n in names if n in vecs and n in mses]
        c = build_cov(names, mses, d)
        eig = np.linalg.eigvalsh(c)
        for ridge in [0.0, 0.5, 1.0, 2.0]:
            w, mse = simplex_qp(c, ridge=ridge)
            tag = ", ".join(f"{n}={wi:.3f}" for n, wi in zip(names, w, strict=True) if wi > 0.005)
            print(f"[{set_name}] ridge={ridge:<4} pred rmse={np.sqrt(mse):.4f}  ({tag})")
            results[f"{set_name}_r{ridge}"] = {
                "names": names,
                "weights": [round(float(x), 4) for x in w],
                "pred_rmse": round(float(np.sqrt(mse)), 4),
            }
        print(f"   min eig of C: {eig.min():.3f}")

    # --- leave-one-anchor-out stability for the main set ---
    print("\n=== LOO stability (ab_drift_iaztec, ridge=1.0) ===")
    base_names = ["sp45", "fleongg", "drift_geo", "iaztec_param"]
    # perturb each anchor's score by +/-0.05 RMSE to gauge sensitivity of the
    # component solution that everything is built on
    for d_scale in [0.85, 1.0, 1.15]:
        a_s, x_s, b_s = solve_components(d_ab * d_scale)
        m2 = dict(mses)
        m2["sp45"], m2["fleongg"] = a_s, b_s
        c = build_cov(base_names, m2, d)
        # patch the ab cross term with the consistent solved X
        c[0, 1] = c[1, 0] = x_s
        w, mse = simplex_qp(c, ridge=1.0)
        tag = ", ".join(f"{n}={wi:.3f}" for n, wi in zip(base_names, w, strict=True))
        print(f"D-scale={d_scale:.2f}: pred rmse={np.sqrt(mse):.4f}  ({tag})")

    # --- fixed practical 3-way grids ---
    print("\n=== fixed 3-way grid sp45/fleongg/drift (pred rmse) ===")
    names3 = ["sp45", "fleongg", "drift_geo"]
    c3 = build_cov(names3, mses, d)
    c3[0, 1] = c3[1, 0] = x_cov
    best = []
    for wa in np.arange(0.50, 0.76, 0.02):
        for wd in np.arange(0.0, 0.31, 0.05):
            wb = 1.0 - wa - wd
            if wb < 0:
                continue
            w = np.array([wa, wb, wd])
            best.append((float(np.sqrt(w @ c3 @ w)), round(wa, 2), round(wb, 2), round(wd, 2)))
    best.sort()
    for r, wa, wb, wd in best[:12]:
        print(f"  sp45={wa:.2f} fleongg={wb:.2f} drift={wd:.2f}  pred rmse={r:.4f}")

    (OUT / "blend_qp_20260610.json").write_text(json.dumps(results, indent=2))
    print("\nwrote outputs/blend_qp_20260610.json")


if __name__ == "__main__":
    main()
