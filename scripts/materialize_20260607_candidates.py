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
class Candidate:
    dirname: str
    kernel_id: str
    title: str
    code_file: str
    profile: str
    ridge_weight: float
    selector_weight: float
    final_seeds: int
    postprocess_mode: str = "none"
    prefix_max_weight: float = 0.0
    formation_max_weight: float = 0.0


CANDIDATES = [
    Candidate(
        dirname="ridge_w040_pf40",
        kernel_id="surajranganath17/rogii-ridge-w040-pf40",
        title="ROGII Ridge W040 PF40",
        code_file="ridge_w040_pf40.py",
        profile="w040_pf40",
        ridge_weight=0.40,
        selector_weight=0.60,
        final_seeds=40,
    ),
    Candidate(
        dirname="ridge_w040_selector070",
        kernel_id="surajranganath17/rogii-ridge-w040-selector070",
        title="ROGII Ridge W040 Selector070",
        code_file="ridge_w040_selector070.py",
        profile="w040_selector070",
        ridge_weight=0.30,
        selector_weight=0.70,
        final_seeds=32,
    ),
    Candidate(
        dirname="ridge_w040_selector080",
        kernel_id="surajranganath17/rogii-ridge-w040-selector080",
        title="ROGII Ridge W040 Selector080",
        code_file="ridge_w040_selector080.py",
        profile="w040_selector080",
        ridge_weight=0.20,
        selector_weight=0.80,
        final_seeds=32,
    ),
    Candidate(
        dirname="ridge_w040_prefix_gate",
        kernel_id="surajranganath17/rogii-ridge-w040-prefix-gate",
        title="ROGII Ridge W040 Prefix Gate",
        code_file="ridge_w040_prefix_gate.py",
        profile="w040_prefix_gate",
        ridge_weight=0.32,
        selector_weight=0.68,
        final_seeds=32,
        postprocess_mode="prefix",
        prefix_max_weight=0.12,
    ),
    Candidate(
        dirname="ridge_w040_formprefix_gate",
        kernel_id="surajranganath17/rogii-ridge-w040-formprefix-gate",
        title="ROGII Ridge W040 FormPrefix Gate",
        code_file="ridge_w040_formprefix_gate.py",
        profile="w040_formprefix_gate",
        ridge_weight=0.32,
        selector_weight=0.68,
        final_seeds=32,
        postprocess_mode="formprefix",
        prefix_max_weight=0.08,
        formation_max_weight=0.08,
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


def final_block(c: Candidate) -> str:
    return f"""CANDIDATE_NAME = {c.profile!r}
FINAL_RIDGE_WEIGHT = {c.ridge_weight:.8f}
FINAL_SELECTOR_WEIGHT = {c.selector_weight:.8f}
POSTPROCESS_MODE = {c.postprocess_mode!r}
PREFIX_MAX_WEIGHT = {c.prefix_max_weight:.8f}
FORMATION_MAX_WEIGHT = {c.formation_max_weight:.8f}

_blend = sub_1.merge(sub_2, on='id', suffixes=('_1', '_2'))
_blend['tvt_base'] = (
    FINAL_RIDGE_WEIGHT * _blend['tvt_1'].astype(float)
    + FINAL_SELECTOR_WEIGHT * _blend['tvt_2'].astype(float)
)
_blend['tvt'] = _blend['tvt_base'].astype(float)

if POSTPROCESS_MODE in ('prefix', 'formprefix') and PREFIX_MAX_WEIGHT > 0:
    _prefix_cols = ['id', 'last_known_tvt', 'sc_ens_d', 'sc8_sc', 'sc15_sc', 'sc25_sc', 'known_len', 'sig_std']
    _prefix = test_df[[c for c in _prefix_cols if c in test_df.columns]].copy()
    if set(_prefix_cols).issubset(_prefix.columns):
        _prefix['tvt_prefix'] = _prefix['last_known_tvt'].astype(float) + _prefix['sc_ens_d'].astype(float)
        _prefix['prefix_score'] = _prefix[['sc8_sc', 'sc15_sc', 'sc25_sc']].max(axis=1).astype(float)
        _prefix_conf = np.clip((_prefix['prefix_score'].to_numpy(float) - 0.58) / 0.22, 0.0, 1.0)
        _prefix_conf *= np.clip(_prefix['known_len'].to_numpy(float) / 260.0, 0.0, 1.0)
        _prefix_conf *= np.clip(18.0 / (np.abs(_prefix['sig_std'].to_numpy(float)) + 18.0), 0.0, 1.0)
        _prefix['prefix_weight'] = PREFIX_MAX_WEIGHT * _prefix_conf
        _blend = _blend.merge(_prefix[['id', 'tvt_prefix', 'prefix_weight']], on='id', how='left')
        _w = _blend['prefix_weight'].fillna(0.0).to_numpy(float)
        _p = _blend['tvt_prefix'].fillna(_blend['tvt']).to_numpy(float)
        _blend['tvt'] = (1.0 - _w) * _blend['tvt'].to_numpy(float) + _w * _p
        print(
            f"{{CANDIDATE_NAME}} prefix_gate rows={{int((_w > 0).sum())}} "
            f"mean_w={{float(np.mean(_w)):.5f}} max_w={{float(np.max(_w)):.5f}}",
            flush=True,
        )
    else:
        print(f"{{CANDIDATE_NAME}} prefix_gate skipped; missing prefix columns", flush=True)

if POSTPROCESS_MODE == 'formprefix' and FORMATION_MAX_WEIGHT > 0:
    _form_cols = ['id', 'last_known_tvt', 'tvt_dense_d', 'dense_std', 'form_std_d', 'dense_dist', 'spatial_knn_dist']
    _form = test_df[[c for c in _form_cols if c in test_df.columns]].copy()
    if set(_form_cols).issubset(_form.columns):
        _form['tvt_form'] = _form['last_known_tvt'].astype(float) + _form['tvt_dense_d'].astype(float)
        _form_conf = np.clip(1.0 - np.abs(_form['dense_std'].to_numpy(float)) / 90.0, 0.0, 1.0)
        _form_conf *= np.clip(1.0 - np.abs(_form['form_std_d'].to_numpy(float)) / 140.0, 0.0, 1.0)
        _form_conf *= np.clip(1.0 / (1.0 + np.abs(_form['dense_dist'].to_numpy(float)) / 6.0), 0.0, 1.0)
        _form['formation_weight'] = FORMATION_MAX_WEIGHT * _form_conf
        _blend = _blend.merge(_form[['id', 'tvt_form', 'formation_weight']], on='id', how='left')
        _w = _blend['formation_weight'].fillna(0.0).to_numpy(float)
        _p = _blend['tvt_form'].fillna(_blend['tvt']).to_numpy(float)
        _blend['tvt'] = (1.0 - _w) * _blend['tvt'].to_numpy(float) + _w * _p
        print(
            f"{{CANDIDATE_NAME}} formation_gate rows={{int((_w > 0).sum())}} "
            f"mean_w={{float(np.mean(_w)):.5f}} max_w={{float(np.max(_w)):.5f}}",
            flush=True,
        )
    else:
        print(f"{{CANDIDATE_NAME}} formation_gate skipped; missing formation columns", flush=True)

sub = _blend[['id', 'tvt']].copy()
if len(sub) != len(sample):
    raise RuntimeError(f"{{CANDIDATE_NAME}} final row mismatch: {{len(sub)}} != {{len(sample)}}")
if not sub['id'].equals(sample['id']):
    raise RuntimeError(f"{{CANDIDATE_NAME}} final ids do not match sample order")
if sub['tvt'].isna().any() or not np.isfinite(sub['tvt'].astype(float)).all():
    raise RuntimeError(f"{{CANDIDATE_NAME}} final submission contains non-finite tvt values")
print(
    f"{{CANDIDATE_NAME}} final blend ridge={{FINAL_RIDGE_WEIGHT:.3f}} "
    f"selector={{FINAL_SELECTOR_WEIGHT:.3f}} mode={{POSTPROCESS_MODE}} "
    f"range={{float(sub['tvt'].min()):.3f}}..{{float(sub['tvt'].max()):.3f}}",
    flush=True,
)
sub.to_csv("submission.csv", index=False)
sub
"""


def materialize_candidate(c: Candidate) -> None:
    target_dir = OUT_ROOT / c.dirname
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    for wheel in BASE_DIR.glob("*.whl"):
        shutil.copy2(wheel, target_dir / wheel.name)

    code = BASE_CODE.read_text()
    code = code.replace('ROGII_HIDDEN_SAFE_PROFILE = "w040"', f'ROGII_HIDDEN_SAFE_PROFILE = "{c.profile}"')
    code = code.replace("FINAL_SELECTOR_PF_SEEDS=32", f"FINAL_SELECTOR_PF_SEEDS={c.final_seeds}")
    code = code.replace(
        "warnings.filterwarnings(\"ignore\")\n",
        "warnings.filterwarnings(\"ignore\")\nnp.random.seed(20260607)\n",
        1,
    )
    if ORIGINAL_FINAL_BLOCK not in code:
        raise RuntimeError("Base final block not found; aborting materialization")
    code = code.replace(ORIGINAL_FINAL_BLOCK, final_block(c))
    (target_dir / c.code_file).write_text(code)

    metadata = {
        "id": c.kernel_id,
        "title": c.title,
        "code_file": c.code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["ridge", "hidden-safe", "candidate"],
        "dataset_sources": ["ravaghi/wellbore-geology-prediction-artifacts"],
        "kernel_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "model_sources": [],
    }
    (target_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    for candidate in CANDIDATES:
        materialize_candidate(candidate)
        print(f"materialized {candidate.dirname}: {candidate.kernel_id}")


if __name__ == "__main__":
    main()
