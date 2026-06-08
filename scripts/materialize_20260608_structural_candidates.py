from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "kaggle" / "kernels" / "ravaghi_ridge_w040"
BASE_CODE = BASE_DIR / "ravaghi_ridge_w040.py"
OUT_ROOT = ROOT / "kaggle" / "kernels"


@dataclass(frozen=True)
class RidgeSpCandidate:
    dirname: str
    kernel_id: str
    title: str
    code_file: str
    profile: str
    ridge_weight: float
    selector_weight: float
    tune_pp: bool
    projection_degree: int | None = None
    projection_weight: float = 1.0
    feature_particles: int = 600
    final_particles: int = 500
    final_seeds: int = 128
    use_all_beams: bool = True


RIDGE_CANDIDATES = [
    RidgeSpCandidate(
        dirname="ridge_sp7776_exact",
        kernel_id="surajranganath17/rogii-ridge-sp7776-exact",
        title="ROGII Ridge SP7776 Exact",
        code_file="ridge_sp7776_exact.py",
        profile="sp7776_exact",
        ridge_weight=0.30,
        selector_weight=0.70,
        tune_pp=True,
    ),
    RidgeSpCandidate(
        dirname="ridge_sp45_proj",
        kernel_id="surajranganath17/rogii-ridge-sp45-projection",
        title="ROGII Ridge SP45 Projection",
        code_file="ridge_sp45_proj.py",
        profile="sp45_proj",
        ridge_weight=0.30,
        selector_weight=0.70,
        tune_pp=False,
        projection_degree=5,
    ),
    RidgeSpCandidate(
        dirname="ridge_sp7776_proj",
        kernel_id="surajranganath17/rogii-ridge-sp7776-projection",
        title="ROGII Ridge SP7776 Projection",
        code_file="ridge_sp7776_proj.py",
        profile="sp7776_proj",
        ridge_weight=0.30,
        selector_weight=0.70,
        tune_pp=True,
        projection_degree=5,
    ),
    RidgeSpCandidate(
        dirname="ridge_sp7776_projdeg3",
        kernel_id="surajranganath17/rogii-ridge-sp7776-projection-deg3",
        title="ROGII Ridge SP7776 Projection Deg3",
        code_file="ridge_sp7776_projdeg3.py",
        profile="sp7776_projdeg3",
        ridge_weight=0.30,
        selector_weight=0.70,
        tune_pp=True,
        projection_degree=3,
    ),
]


ORIGINAL_FINAL_BLOCK = """sub = (
    sub_1.merge(sub_2, on='id', suffixes=('_1', '_2'))
       .assign(tvt=lambda x: 0.40 * x['tvt_1'] + 0.60 * x['tvt_2'])
       [['id', 'tvt']]
)
sub.to_csv("submission.csv", index=False)
sub
"""

ORIGINAL_PP_SCORE_BLOCK = """d = apply_pp(train_df, ridge_oof_preds, pf_oof, **pp_params)
ridge_score = root_mean_squared_error(ytrue, base + d)

overall_scores["ridge (pp)"] = ridge_score
fold_scores["ridge (pp)"] = [ridge_score] * CFG.n_splits
"""

TUNED_PP_SCORE_BLOCK = """baseline_d = apply_pp(train_df, ridge_oof_preds, pf_oof, **pp_params)
baseline_pred = base + baseline_d
baseline_score = root_mean_squared_error(ytrue, baseline_pred)

PP_SELECTED_PARAMS = pp_params.copy()
PP_SELECTED_SCORE = baseline_score
pp_grid = [
    {'alpha': alpha, 'tau': tau, 'w_pf': w_pf}
    for alpha in [0.98, 0.99, 1.0, 1.01, 1.02]
    for tau in [35, 50, 65, 85, 105, 130, 170, 220]
    for w_pf in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16]
]
for params in pp_grid:
    d = apply_pp(train_df, ridge_oof_preds, pf_oof, **params)
    score = root_mean_squared_error(ytrue, base + d)
    if score < PP_SELECTED_SCORE - 1e-9:
        PP_SELECTED_SCORE = score
        PP_SELECTED_PARAMS = params

overall_scores["ridge (pp)"] = baseline_score
fold_scores["ridge (pp)"] = [baseline_score] * CFG.n_splits
overall_scores["ridge (pp tuned)"] = PP_SELECTED_SCORE
fold_scores["ridge (pp tuned)"] = [PP_SELECTED_SCORE] * CFG.n_splits
print(f"Baseline pp score: {baseline_score:.5f} params={pp_params}")
print(f"Selected pp score: {PP_SELECTED_SCORE:.5f} params={PP_SELECTED_PARAMS}")
"""


def projection_block(candidate: RidgeSpCandidate) -> str:
    if candidate.projection_degree is None:
        return ""
    degree = int(candidate.projection_degree)
    weight = float(candidate.projection_weight)
    return f"""

# Domain structural projection: smooth U = TVT + Z - anchor per well.
PROJECTION_DEGREE = {degree}
PROJECTION_WEIGHT = {weight:.8f}

def _robust_polyfit_projection(_s, _y, _deg):
    if len(_s) < _deg + 2:
        return _y.copy()
    _coef = np.polyfit(_s, _y, _deg)
    for _ in range(4):
        _res = _y - np.polyval(_coef, _s)
        _scale = np.median(np.abs(_res)) * 1.4826 + 1e-6
        _weights = 1.0 / (1.0 + (_res / (2.0 * _scale)) ** 2)
        _coef = np.polyfit(_s, _y, _deg, w=_weights)
    return np.polyval(_coef, _s)

_proj_base = sub.copy()
_proj_base['well'] = _proj_base['id'].str[:8]
_proj_base['row_idx'] = _proj_base['id'].str[9:].astype(int)
_proj_out = dict(zip(_proj_base['id'].to_numpy(), _proj_base['tvt'].astype(float).to_numpy()))
_proj_ok = 0
for _wid, _group in _proj_base.groupby('well', sort=False):
    try:
        _hw = pd.read_csv(CFG.dataset_path / 'test' / f'{{_wid}}__horizontal_well.csv')
        _known = _hw[_hw['TVT_input'].notna()]
        if len(_known) < 5:
            continue
        _last = _known.iloc[-1]
        _anchor = float(_last['TVT_input']) + float(_last['Z'])
        _ps = float(_last['MD'])
        _end = float(_hw['MD'].iloc[-1])
        _ordered = _group.sort_values('row_idx')
        _row_idx = _ordered['row_idx'].to_numpy(int)
        _z = _hw['Z'].to_numpy(float)[_row_idx]
        _md = _hw['MD'].to_numpy(float)[_row_idx]
        _s = (_md - _ps) / max(_end - _ps, 1e-6)
        _raw = _ordered['tvt'].to_numpy(float)
        _u = (_raw + _z) - _anchor
        _u_fit = _robust_polyfit_projection(_s, _u, PROJECTION_DEGREE)
        _projected = (_anchor + _u_fit) - _z
        if not np.all(np.isfinite(_projected)):
            continue
        _final = (1.0 - PROJECTION_WEIGHT) * _raw + PROJECTION_WEIGHT * _projected
        for _rid, _value in zip(_ordered['id'].to_numpy(), _final):
            _proj_out[_rid] = float(_value)
        _proj_ok += 1
    except Exception as _exc:
        print(f"{{CANDIDATE_NAME}} projection fallback {{_wid}}: {{_exc}}", flush=True)

sub = _proj_base[['id']].copy()
sub['tvt'] = sub['id'].map(_proj_out).astype(float)
print(
    f"{{CANDIDATE_NAME}} projection degree={{PROJECTION_DEGREE}} "
    f"weight={{PROJECTION_WEIGHT:.3f}} wells={{_proj_ok}}",
    flush=True,
)
"""


def final_block(candidate: RidgeSpCandidate) -> str:
    return f"""CANDIDATE_NAME = {candidate.profile!r}
FINAL_RIDGE_WEIGHT = {candidate.ridge_weight:.8f}
FINAL_SELECTOR_WEIGHT = {candidate.selector_weight:.8f}

_blend = sub_1.merge(sub_2, on='id', suffixes=('_1', '_2'))
_blend['tvt'] = (
    FINAL_RIDGE_WEIGHT * _blend['tvt_1'].astype(float).to_numpy()
    + FINAL_SELECTOR_WEIGHT * _blend['tvt_2'].astype(float).to_numpy()
)
sub = _blend[['id', 'tvt']].copy()
{projection_block(candidate)}
if len(sub) != len(sample):
    raise RuntimeError(f"{{CANDIDATE_NAME}} final row mismatch: {{len(sub)}} != {{len(sample)}}")
if not sub['id'].equals(sample['id']):
    raise RuntimeError(f"{{CANDIDATE_NAME}} final ids do not match sample order")
if sub['tvt'].isna().any() or not np.isfinite(sub['tvt'].astype(float)).all():
    raise RuntimeError(f"{{CANDIDATE_NAME}} final submission contains non-finite tvt values")
print(
    f"{{CANDIDATE_NAME}} final ridge={{FINAL_RIDGE_WEIGHT:.3f}} "
    f"selector={{FINAL_SELECTOR_WEIGHT:.3f}} "
    f"range={{float(sub['tvt'].min()):.3f}}..{{float(sub['tvt'].max()):.3f}}",
    flush=True,
)
sub.to_csv("submission.csv", index=False)
sub
"""


def materialize_ridge_candidate(candidate: RidgeSpCandidate) -> None:
    target_dir = OUT_ROOT / candidate.dirname
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    for wheel in BASE_DIR.glob("*.whl"):
        shutil.copy2(wheel, target_dir / wheel.name)

    code = BASE_CODE.read_text()
    code = code.replace(
        'ROGII_HIDDEN_SAFE_PROFILE = "w040"',
        f'ROGII_HIDDEN_SAFE_PROFILE = "{candidate.profile}"',
    )
    code = code.replace("PF_N=256; ANCC_N=256", f"PF_N={candidate.feature_particles}; ANCC_N={candidate.feature_particles}")
    code = code.replace("FINAL_SELECTOR_PF_PARTICLES=256", f"FINAL_SELECTOR_PF_PARTICLES={candidate.final_particles}")
    code = code.replace("FINAL_SELECTOR_PF_SEEDS=32", f"FINAL_SELECTOR_PF_SEEDS={candidate.final_seeds}")
    if candidate.use_all_beams:
        code = code.replace("ACTIVE_BEAM_CONFIGS = BEAM_CONFIGS[:7]", "ACTIVE_BEAM_CONFIGS = BEAM_CONFIGS")
    code = code.replace("warnings.filterwarnings(\"ignore\")\n", "warnings.filterwarnings(\"ignore\")\nnp.random.seed(20260608)\n", 1)
    code = code.replace("**pp_params\n)", "**PP_SELECTED_PARAMS\n)")
    code = code.replace("pp_params = {\n    'alpha': 1.0,\n    'tau': 85,\n    'w_pf': 0.09\n}\n", "pp_params = {\n    'alpha': 1.0,\n    'tau': 85,\n    'w_pf': 0.09\n}\nPP_SELECTED_PARAMS = pp_params.copy()\n")
    if candidate.tune_pp:
        if ORIGINAL_PP_SCORE_BLOCK not in code:
            raise RuntimeError("Could not find original PP scoring block")
        code = code.replace(ORIGINAL_PP_SCORE_BLOCK, TUNED_PP_SCORE_BLOCK)
    if ORIGINAL_FINAL_BLOCK not in code:
        raise RuntimeError("Could not find original final block")
    code = code.replace(ORIGINAL_FINAL_BLOCK, final_block(candidate))
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
        "keywords": ["ridge-sp", "projection"],
        "dataset_sources": ["ravaghi/wellbore-geology-prediction-artifacts"],
        "kernel_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "model_sources": [],
    }
    (target_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def materialize_drift_candidate() -> None:
    target_dir = OUT_ROOT / "drift_geosteering_infer"
    source_candidates = [
        Path("/tmp/rogii_kernels_research/drift_geosteering_meta/rogii-drift-geosteering-infer.ipynb"),
        target_dir / "drift_geosteering_infer.ipynb",
    ]
    source_nb = next((path for path in source_candidates if path.is_file()), None)
    if source_nb is None:
        raise FileNotFoundError("No drift geosteering source notebook found")
    notebook_bytes = source_nb.read_bytes()
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    (target_dir / "drift_geosteering_infer.ipynb").write_bytes(notebook_bytes)
    metadata = {
        "id": "surajranganath17/rogii-drift-geosteering-infer",
        "title": "ROGII Drift Geosteering Infer",
        "code_file": "drift_geosteering_infer.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["drift", "geosteering"],
        "dataset_sources": ["fleongg/rogii-claude-models-pub"],
        "kernel_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "model_sources": [],
    }
    (target_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    for candidate in RIDGE_CANDIDATES:
        materialize_ridge_candidate(candidate)
        print(f"materialized {candidate.dirname}: {candidate.kernel_id}", flush=True)
    materialize_drift_candidate()
    print("materialized drift_geosteering_infer: surajranganath17/rogii-drift-geosteering-infer", flush=True)


if __name__ == "__main__":
    main()
