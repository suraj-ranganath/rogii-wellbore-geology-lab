# ruff: noqa
#!/usr/bin/env python
# coding: utf-8

# * 🚀modified:
#     1. Using 128 seeds in particle filtering is excessive, leading to overfitting noise. It would be better to reduce the number of seeds to 64 
#     2. a larger initial sampling range 2.0->3.0 (https://www.kaggle.com/code/needless090/lb8-781-rogii-sel15-spread3)

# In[ ]:


import os, glob, warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)


def find_input_dir():
    for c in [
        '/kaggle/input/rogii-wellbore-geology-prediction',
        '/kaggle/input/competitions/rogii-wellbore-geology-prediction',
        'data/raw/rogii-wellbore-geology-prediction',
        '../data/raw/rogii-wellbore-geology-prediction',
        '../../data/raw/rogii-wellbore-geology-prediction',
        '../../../data/raw/rogii-wellbore-geology-prediction',
    ]:
        if os.path.isdir(c):
            print(f'INPUT_DIR={c}')
            return c
    hits = glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True)
    if hits:
        d = os.path.dirname(hits[0])
        print(f'Discovered INPUT_DIR={d}')
        return d
    raise FileNotFoundError('Cannot locate competition data')


INPUT_DIR = find_input_dir()
TRAIN_DIR = os.path.join(INPUT_DIR, 'train')
TEST_DIR  = os.path.join(INPUT_DIR, 'test')

_hw_files  = sorted(glob.glob(os.path.join(TEST_DIR, '*__horizontal_well.csv')))
TEST_WELLS = [os.path.basename(f).split('__')[0] for f in _hw_files]
print(f'Test wells: {TEST_WELLS}')


def load_well(wid, split='train'):
    base = TRAIN_DIR if split == 'train' else TEST_DIR
    hw = pd.read_csv(os.path.join(base, f'{wid}__horizontal_well.csv'))
    tw = pd.read_csv(os.path.join(base, f'{wid}__typewell.csv'))
    return hw, tw


def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset


def run_particle_filter(hw, tw, n_particles=500, seed=42):
    """Conservative PF. Returns (predictions_array, total_log_likelihood)."""
    tw_s   = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    ls   = last_tvt + last_Z
    pos  = ls + 3.0 * rng.standard_normal(N)  # wider init spread helps wells with abrupt TVT shift at PS
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    # Interpolate GR gaps before tracking â€” critical for wells with high NaN fraction
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos   = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d  = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos  = pos[idx]  + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=128, scale=5.0):
    """
    128-seed lik-weighted PF ensemble.
    More seeds â†’ better coverage of the TVT exploration space.
    """
    preds = []
    liks  = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)

    liks   = np.array(liks)
    liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale)
    weights /= weights.sum()

    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128):
    preds = []
    liks = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f"pf_scale_{scale:g}"] = (weights[:, None] * pred_arr).sum(0)
    out["pf_mean"] = pred_arr.mean(0)
    return out


# 14 beam configs: original 7 + 7 new ones exploring broader parameter space
BEAM_CONFIGS = [
    # Original 7 configs (from ajayrao43)
    (10, 20.0, 144.0, 2),
    (10,  8.0,  64.0, 2),
    ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5),
    (20,  4.0,  36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    # 7 new configs: wider beam, different motion/error scales
    (20, 30.0, 200.0, 2),
    (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2),
    (10, 50.0, 400.0, 0),
]


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    """Vectorized beam search for TVT tracking via GR matching."""
    n  = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC    = mc * np.array([2., 1., 0., 1., 2.])

    bidx  = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci])**2 / es
        tot  = bcost[:bn, None] + gr_e + MC[None, :]
        tot  = np.where(valid, tot, np.inf)

        ni_f  = ni.flatten()
        tot_f = tot.flatten()
        vf    = valid.flatten()
        ni_f  = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s  = ni_f[order]
        tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u  = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top  = top[np.argsort(tot_u[top])]

        bidx[:kept]  = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:]  = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept

        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw, tw):
    """Average 14 beam-search configs."""
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s  = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr    = gr_all[ev.index]

    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]

    beam_mean = np.stack(beam_results, 0).mean(0)

    out = hw['TVT_input'].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def selector_well_code(hw):
    eval_mask = hw['TVT_input'].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, 'Z'].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split('_')
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if 'beam' in parts:
        beam_weight = float(parts[parts.index('beam') + 1])
    if 'hold' in parts:
        hold_weight = float(parts[parts.index('hold') + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}")
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split('_beam_')[0].split('_hold_')[0]]
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


sample = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
parsed_ids = sample['id'].astype(str).str.rsplit('_', n=1, expand=True)
sample['well'] = parsed_ids[0]
sample['row_idx'] = parsed_ids[1].astype(int)

train_wids = set(
    os.path.basename(f).split('__')[0]
    for f in glob.glob(os.path.join(TRAIN_DIR, '*__horizontal_well.csv'))
)
print(f'Training wells available: {len(train_wids)}')

rows = []
for wid in TEST_WELLS:
    print(f'\nProcessing {wid}...')
    hw_te, tw_te = load_well(wid, 'test')

    tvt_phys = None
    hw_tr    = None
    tw_tr    = None

    # Physical model for visible wells
    if wid in train_wids:
        try:
            hw_tr, tw_tr = load_well(wid, 'train')
            hw_te['TVT_input'] = hw_tr['TVT_input'].values
            tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
            print(f'  Physical model OK')
        except Exception as e:
            print(f'  Physical model failed: {e}')
            tvt_phys = None

    selector_code, selector_variant, selector_n_eval, selector_z_span = selector_well_code(hw_te)

    # 128-seed likelihood-weighted PF ensemble, cached across selector scales.
    try:
        tw_ref = tw_tr if tw_tr is not None else tw_te
        pf_by_scale = run_pf_lik_ensemble_scales(hw_te, tw_ref, n_particles=500, n_seeds=64)
        tvt_pf = pf_by_scale["pf_scale_8"]
        print(f'  PF 64-seed lik-ensemble OK scales={SELECTOR_SCALES}')
    except Exception as e:
        print(f'  PF failed: {e}')
        last_known = hw_te['TVT_input'].dropna()
        last_val   = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
        tvt_pf = hw_te['TVT_input'].fillna(last_val).values.astype(float)
        pf_by_scale = {f"pf_scale_{scale:g}": tvt_pf.copy() for scale in SELECTOR_SCALES}

    try:
        tw_ref = tw_tr if tw_tr is not None else tw_te
        tvt_beam = run_beam_ensemble(hw_te, tw_ref)
        print(f'  Beam 14-config ensemble OK')
    except Exception as e:
        print(f'  Beam failed: {e}')
        tvt_beam = tvt_pf.copy()

    last_known = hw_te['TVT_input'].dropna()
    last_known_tvt = float(last_known.iloc[-1]) if len(last_known) > 0 else float(np.nanmean(tvt_pf))
    tvt_selector = apply_selector_variant(selector_variant, pf_by_scale, tvt_beam, last_known_tvt)
    print(
        f'  Selector code={selector_code} variant={selector_variant} '
        f'n_eval={selector_n_eval:.0f} z_span={selector_z_span:.3f}'
    )

    ws = sample[sample['well'] == wid]
    for _, row in ws.iterrows():
        ridx = int(row['row_idx'])
        if tvt_phys is not None:
            # Visible well: physical model is primary (RMSE ~0.007 ft)
            tvt_val = float(tvt_phys.iloc[ridx])
        else:
            # Hidden well: targetless selector from whole-well CV.
            tvt_val = float(tvt_selector[ridx])
        rows.append({'id': row['id'], 'tvt': tvt_val})
    print(f'  Added {len(ws)} rows')

submission = sample[['id']].merge(pd.DataFrame(rows), on='id', how='left')
if submission['tvt'].isna().any():
    missing = int(submission['tvt'].isna().sum())
    raise RuntimeError(f'Missing predictions for {missing} sample rows')
submission.to_csv('submission.csv', index=False)
print(f'\nDone: {len(submission)} rows')
print(submission.head())
