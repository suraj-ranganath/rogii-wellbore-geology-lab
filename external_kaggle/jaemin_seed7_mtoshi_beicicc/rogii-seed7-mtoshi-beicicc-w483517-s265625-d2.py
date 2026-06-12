
# %% markdown 1: # 🛢️ ROGII Wellbore Geology Prediction — Dual-Pipeline Blend + a Self-Verifying Physical Override **TL;DR** — This notebook blends **two independently-built geosteering pipelines** (decorrelated errors → free accuracy), then finishes with a


# %% markdown 2: --- # Part 1 · Pipeline A — "ridge-sp45" A full geosteering pipeline: **physics trackers** (particle filter + beam search) generate TVT candidates, a **LightGBM/CatBoost stack** fuses them with spatial & GR features, a **Ridge meta-model** 


# %% cell 3
import sys, os, glob, subprocess
# koolbox setup: wheel install or sys.path fallback
_install_scripts = sorted(glob.glob('/kaggle/input/**/install_requirements.sh', recursive=True))
if _install_scripts:
    _install_script = _install_scripts[0]
    print('install package-manager requirements', _install_script)
    subprocess.run(['bash', _install_script], cwd=os.path.dirname(_install_script), check=False)
kb_dir = '/kaggle/input/koolbox-offline'
if not os.path.isdir(kb_dir):
    # alt path under datasets/
    cand = glob.glob('/kaggle/input/**/koolbox*', recursive=True)
    print('koolbox candidates:', cand[:5])
    if cand: kb_dir = cand[0]
print('using koolbox dir:', kb_dir)
if os.path.isfile(kb_dir) and kb_dir.endswith('.whl'):
    print('install wheel file', kb_dir)
    subprocess.run(['pip', 'install', '--no-deps', kb_dir], check=False)
elif os.path.isdir(kb_dir):
    print('listing:', os.listdir(kb_dir)[:20])
    whls = glob.glob(f'{kb_dir}/**/*.whl', recursive=True)
    if whls:
        for w in whls:
            print('install', w)
            subprocess.run(['pip', 'install', '--no-deps', w], check=False)
    else:
        sys.path.insert(0, kb_dir)
        # also try subdirs
        for sub in os.listdir(kb_dir):
            sub_path = os.path.join(kb_dir, sub)
            if os.path.isdir(sub_path):
                sys.path.insert(0, sub_path)
import koolbox
print('koolbox OK:', koolbox.__file__)

# %% markdown 4: ### 1.2 · Imports & configuration `CFG` points at the competition data and the artifacts dataset. When the artifacts are mounted, pre-computed features and pre-trained boosters are loaded → the notebook runs as fast inference; when absent, 


# %% cell 5
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from koolbox import Trainer
from pathlib import Path
from numba import njit
import matplotlib.pyplot as plt
import multiprocessing
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import hashlib
import time
import glob
import os

warnings.filterwarnings("ignore")

def _stable_seed_from_wid(wid, base=42):
    digest = hashlib.md5(str(wid).encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base)) % (2**31 - 1)

# %% cell 6
class CFG:
    dataset_path = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")
    
    seed = 7
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)
    
    metric = root_mean_squared_error

# %% markdown 7: ### 1.3 · Physics toolbox - `tvt_from_contacts` — exact TVT reconstruction from formation-contact columns (also reused by the guarded override in Part 4) - `run_particle_filter` — sequential Monte-Carlo tracker: particles carry (TVT, TVT-ra


# %% cell 8
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)

SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05',
    5: 'pf_scale_12_beam_0.2_hold_0.05',
}

SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

FORMATION_COLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10,  8.0,  64.0, 2),
    ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5),
    (20,  4.0,  36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2),
    (10, 50.0, 400.0, 0),
]


def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset


def load_well(wid, split='train'):
    base = CFG.dataset_path / split
    hw = pd.read_csv(base / f'{wid}__horizontal_well.csv')
    tw = pd.read_csv(base / f'{wid}__typewell.csv')
    return hw, tw


def run_particle_filter(hw, tw, n_particles=500, seed=42):
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
    pos  = ls + 4.5 * rng.standard_normal(N)  # sp45 patch (sel15 vb best)
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    # Interpolate GR gaps before tracking
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
        out[f'pf_scale_{scale:g}'] = (weights[:, None] * pred_arr).sum(0)
    out['pf_mean'] = pred_arr.mean(0)
    return out


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
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
    base = pf_by_scale.get(f'pf_scale_{scale:g}')
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split('_beam_')[0].split('_hold_')[0]]
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred

# %% markdown 9: ### 1.4 · Numba kernels + per-well feature builder JIT-compiled PF/beam inner loops, spatial classes (`FormationPlaneKNN`, `DenseANCCImputer`), and `build_well` / `build_dataset` which assemble the full feature table.


# %% cell 10
SEED=7
NCPU=1

FORMATIONS=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
PLANE_K=10; DENSE_SPW=60; DENSE_K=20; N_SPLITS=5

BEAMS=[
    (10,20.0,144.0,2,"cons"),
    (10, 8.0, 64.0,2,"loose"),
    ( 8,35.0,220.0,1,"vcons"),
    (10,14.0, 90.0,5,"sm5"),
    (20, 4.0, 36.0,3,"vloose"),
    (12,12.0,100.0,3,"mid"),
    (15,25.0,180.0,2,"stiff"),
]

PF_N=600; ANCC_N=600
PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
PF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5
PF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3
ANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IR=0.01; ANCC_IS=0.3; ANCC_RP=0.1; ANCC_RR=0.001

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    u0=np.random.uniform(0.,1./N)
    np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*np.random.randn()
        na[j] =aux[ci]+rv*np.random.randn()
    return np2,na

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    """Beam search Â±2 delta, Numba JIT."""
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):            # Â±2: TVT can go down
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)
                fnd=np.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]
                cC[i],cC[mi]=cC[mi],cC[i]
                cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=np.zeros(n,np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

@njit(cache=True)
def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IS,RP,RR,RESAMP):
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts,std_

@njit(cache=True)
def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP):
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+0.5*np.random.randn()
        vel[j]=iv+0.02*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=(gr_v[i]-ep)/gs
                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=(gr_sm_v[i]-es)/(gs*1.5)
                    ls=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*ls
                else: lk=lp
                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ws2=0.
        for j in range(N):
            dv=(vel[j]-ve)/max(zsig*2.,0.005)
            lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)
            w[j]*=lz; ws2+=w[j]
        if ws2>0.:
            for j in range(N): w[j]/=ws2
        else:
            for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
        wm=0.
        for j in range(N): wm+=w[j]*pos[j]
        pts[i]=wm; va=0.
        for j in range(N): va+=w[j]*(pos[j]-wm)**2
        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]
    return pts,std_

# Dense grid for O(1) typewell lookup
def _grid(tw_tvt,tw_gr,step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)

def _gr_sig(hw,tw_tvt,tw_gr):
    kn=hw[hw['TVT_input'].notna()&hw['GR'].notna()]
    if len(kn)<20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values-np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)),
                          PF_GR_SIG_MIN,PF_GR_SIG_MAX))

def _nn(arr,v):
    i=int(np.searchsorted(arr,v,'left'))
    if i>=len(arr): return len(arr)-1
    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1
    return i

def _smooth(vals,fb,r):
    s=pd.Series(vals,dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)

def beam_search(gr_h,tw_tvt,tw_gr,start_tvt,bs,mc,es,r):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))
    return tw_tvt[path].astype(np.float32)

def run_pf_ancc(hw,tw_tvt,tw_gr,N=ANCC_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    pts,std=_pf_ancc(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                      ev['GR'].values.astype(np.float64),gg,gmin,gst,
                      gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)

def run_pf_z(hw,tw_tvt,tw_gr,N=PF_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    tw_s=pd.Series(tw_gr).rolling(PF_GR_WIN,center=True,min_periods=1).mean().values.astype(np.float32)
    kna=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    dz_k=np.diff(kna['Z'].values); dvt=np.diff(kna['TVT_input'].values)
    dmd_k=np.diff(kna['MD'].values); m2=dmd_k>0
    if m2.sum()>=10:
        vz=dz_k[m2]/dmd_k[m2]; vt=dvt[m2]/dmd_k[m2]
        A=np.column_stack([vz,np.ones_like(vz)]); c,_,_,_=np.linalg.lstsq(A,vt,rcond=None)
        beta,icpt,zsig=float(c[0]),float(c[1]),max(float(np.std(vt-(c[0]*vz+c[1]))),0.001)
    else: beta,icpt,zsig=-1.,0.,0.1
    t2=kna.tail(20); dvt2=np.diff(t2['TVT_input'].values); dmd2=np.diff(t2['MD'].values); m3=dmd2>0
    iv=float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    gs2,_,_=_grid(tw_tvt,tw_s)
    gr_sm=hw['GR'].rolling(PF_GR_WIN,center=True,min_periods=1).mean()
    pts,std=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                   ev['GR'].values.astype(np.float64),
                   gr_sm.loc[ev.index].values.astype(np.float64),
                   gg,gs2,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                   beta,icpt,zsig,N,
                   PF_MOM,PF_VN,PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)


_md=np.linspace(1,50,20,np.float64); _z=np.zeros(20,np.float64); _gr=np.full(20,50.,np.float64)
_gg=np.linspace(45,55,100,np.float64)
_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5)
_pf_z(_md,_z,_gr,_gr,_gg,_gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,0.993,0.005,0.01,0.3,0.2,0.003,0.5)
_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)

def robust_slope(x,y,w=None):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def affine_cal(kgr,tw_at_k,min_pts=20):
    v=np.isfinite(kgr)&np.isfinite(tw_at_k)
    if v.sum()<min_pts or np.std(tw_at_k[v])<1e-6:
        return 1.,float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a,b=np.polyfit(tw_at_k[v],kgr[v],1); return float(a),float(b)

def seg_b_well(ktvt,kz,form_col):
    """Segment b_well: early/mid/late thirds + full prefix.
    Returns (b_full, b_early, b_mid, b_late, b_wls) for feature richness."""
    bv=ktvt+kz-form_col; n=len(bv)
    b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    t1,t2=n//3, 2*n//3
    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full
    b_mid  =float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full
    # WLS (tail-upweighted)
    w=np.exp(0.02*np.arange(n)); w/=w.sum()
    b_wls=float(np.dot(w,bv))
    return b_full,b_early,b_mid,b_late,b_wls

def multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3):
    """Multi-scale NCC. Returns score-weighted ensemble + per-scale signals."""
    out=[]
    for hw in hws:
        win=2*hw+1; nk=len(kgr); nh=len(hgr)
        if nk<win+1 or nh==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        kg=pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        hg=pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        sts=np.arange(0,nk-win+1,stride,dtype=np.int32); M=len(sts)
        if M==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        C=kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]].astype(np.float32)
        Cn=(C-C.mean(1,keepdims=True))/(C.std(1,keepdims=True)+1e-6)
        hp=np.pad(hg,hw,mode='edge')
        H=hp[np.arange(nh)[:,None]+np.arange(win)[None,:]].astype(np.float32)
        Hn=(H-H.mean(1,keepdims=True))/(H.std(1,keepdims=True)+1e-6)
        ncc=Hn@Cn.T/win; best=ncc.argmax(1); score=ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw,0,nk-1)].astype(np.float32),score))
    # Score-weighted ensemble (NEW: softmax-weighted combination)
    tvts=np.stack([o[0] for o in out],1); scores=np.stack([o[1] for o in out],1)
    sw=np.exp(3.*scores); sw/=sw.sum(1,keepdims=True)+1e-9
    sc_ens=(tvts*sw).sum(1).astype(np.float32)
    return out, sc_ens   # [(tvt8,sc8),(tvt15,sc15),(tvt25,sc25)], ensemble

class FormationPlaneKNN:
    def __init__(self,well_ids,data_dir):
        rows=[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y']+FORMATIONS).dropna()
            except: continue
            if len(df)==0: continue
            row={'wid':wid,'x':float(df['X'].median()),'y':float(df['Y'].median())}
            for c in FORMATIONS: row[f'{c}_m']=float(df[c].median())
            rows.append(row)
        self.df=pd.DataFrame(rows); self.wmap={w:i for i,w in enumerate(self.df['wid'])}
        xy=self.df[['x','y']].to_numpy(); self.scale=np.where(xy.std(0)<1e-3,1.,xy.std(0))
        self.tree=cKDTree(xy/self.scale)
        self.xa=self.df['x'].to_numpy(); self.ya=self.df['y'].to_numpy()
        self.fa=self.df[[f'{c}_m' for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self,xy_q,self_wid=None,k=PLANE_K):
        q=xy_q/self.scale; nf=min(k+5,len(self.df))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid in self.wmap: dist=np.where(idx==self.wmap[self_wid],np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.).astype(np.float64)
        xn=self.xa[ik]; yn=self.ya[ik]; fn=self.fa[ik]; wx=w*xn; wy=w*yn
        A=np.zeros((len(q),3,3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs=np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),(w[:,:,None]*fn).sum(1)],1)
        try: coef=np.linalg.solve(A,rhs)
        except:
            coef=np.zeros((len(q),3,6))
            for r in range(len(q)):
                try: coef[r]=np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq=xy_q[:,0]; Yq=xy_q[:,1]
        pred=(Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)]=self.fa.mean(0)
        return pred,np.where(vk,dk,np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self,well_ids,data_dir,spw=DENSE_SPW):
        xs,ys,anccs,wids=[],[],[],[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y','ANCC']).dropna()
            except: continue
            if len(df)==0: continue
            ix=np.linspace(0,len(df)-1,min(spw,len(df)),dtype=int); s=df.iloc[ix]
            xs.append(s['X'].values); ys.append(s['Y'].values)
            anccs.append(s['ANCC'].values); wids.extend([wid]*len(s))
        self.xy=np.column_stack([np.concatenate(xs),np.concatenate(ys)])
        self.ancc=np.concatenate(anccs).astype(np.float32); self.wids=np.array(wids)
        self.scale=np.where(self.xy.std(0)<1e-3,1.,self.xy.std(0))
        self.tree=cKDTree(self.xy/self.scale)

    def impute(self,xy_q,self_wid=None,k=DENSE_K,nfetch=5000):
        xy_q=np.atleast_2d(xy_q); q=xy_q/self.scale; nf=min(nfetch,len(self.ancc))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid: dist=np.where(self.wids[idx]==self_wid,np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.)
        sw=w.sum(1); safe=np.where(sw<1e-9,1.,sw); an=self.ancc[ik]
        ap=(an*w).sum(1)/safe; ap=np.where(sw<1e-9,float(self.ancc.mean()),ap)
        var=((an-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32),np.sqrt(np.maximum(var,0.)).astype(np.float32),np.where(vk,dk,np.inf).min(1).astype(np.float32)

hw_paths=sorted((CFG.dataset_path / "train").glob('*__horizontal_well.csv'))
train_wids=[p.stem.replace('__horizontal_well','') for p in hw_paths]
FI=FormationPlaneKNN(train_wids,CFG.dataset_path / "train")
DI=DenseANCCImputer(train_wids,CFG.dataset_path / "train")

_FI=FI; _DI=DI
ANCH_OFFS=np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80],np.float32)
BEAM_OFFS=np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40],np.float32)
SC_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)
PF_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)

def build_well(hw_path,tw_path,is_train):
    global _FI,_DI
    wid=Path(hw_path).stem.replace('__horizontal_well','')
    np.random.seed(_stable_seed_from_wid(wid, CFG.seed))
    try:
        hw=pd.read_csv(hw_path); tw=pd.read_csv(tw_path).sort_values('TVT')
    except: return None
    if is_train and 'TVT' not in hw.columns: return None
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0 or len(kn)<10: return None
    if is_train and hw['TVT'].isna().all(): return None
    tw_tvt=tw['TVT'].to_numpy(np.float32); tw_gr=tw['GR'].to_numpy(np.float32)
    if len(tw_tvt)<3: return None

    pf_a,std_a=run_pf_ancc(hw,tw_tvt,tw_gr)
    if len(pf_a)==0: return None
    pf_z,std_z=run_pf_z(hw,tw_tvt,tw_gr)
    pf_use=pf_a.astype(np.float32); std_use=std_a.astype(np.float32)
    has_z=len(pf_z)==len(pf_a) and not np.any(np.isnan(pf_z))

    lk=kn.iloc[-1]; last_tvt=float(lk['TVT_input'])
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    hgr=gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr=gr_full.iloc[:len(kn)].to_numpy(np.float32)

    # 7 beams (Numba JIT Â±2)
    bpaths={}
    for (bs,mc,es,r,tag) in BEAMS:
        bpaths[tag]=beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r)
    beam_ref=(bpaths['cons']+bpaths['sm5'])/2.

    # Multi-scale NCC â†’ score-weighted ensemble
    ktvt=kn['TVT_input'].to_numpy(np.float32)
    sc_res,sc_ens=multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3)
    sc8,sc8s=sc_res[0]; sc15,sc15s=sc_res[1]; sc25,sc25s=sc_res[2]
    sc_cons=(sc8+sc15+sc25)/3.
    sc_trust=float(np.clip(len(kn)/200.,0.,0.6))
    hyb_ref=(1-sc_trust)*beam_ref+sc_trust*sc_ens  # use ensemble not single

    tw_at_k=np.interp(ktvt,tw_tvt,tw_gr).astype(np.float32)
    a_cal,b_cal=affine_cal(kgr,tw_at_k)
    kmd=kn['MD'].to_numpy(np.float32); kz=kn['Z'].to_numpy(np.float32)
    pfx_rmse=float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all=robust_slope(kmd,ktvt); slp_50=robust_slope(kmd[-50:],ktvt[-50:])
    slp_z=robust_slope(kz,ktvt)

    swid=wid if is_train else None
    xy_ev=ev[['X','Y']].to_numpy(np.float64); xy_kn=kn[['X','Y']].to_numpy(np.float64)
    form_ev,knn_d=_FI.impute(xy_ev,self_wid=swid)
    form_kn,_   =_FI.impute(xy_kn,self_wid=swid)
    z_kn=kn['Z'].to_numpy(np.float32); z_ev=ev['Z'].to_numpy(np.float32)

    # Per-formation: segment b_well (early/mid/late/wls) + TVT + known-zone RMSE
    tvt_fs={}; form_rmse={}; form_list=[]
    for fi2,fn in enumerate(FORMATIONS):
        b_full,b_early,b_mid,b_late,b_wls=seg_b_well(ktvt,z_kn,form_kn[:,fi2])
        tvt_f  =(-z_ev+form_ev[:,fi2]+b_full ).astype(np.float32)
        tvt_fw =(-z_ev+form_ev[:,fi2]+b_wls  ).astype(np.float32)
        tvt_f50=(-z_ev+form_ev[:,fi2]+b_late ).astype(np.float32)
        tvt_fs[f'tvtF_{fn}']=tvt_f; tvt_fs[f'tvtFw_{fn}']=tvt_fw
        tvt_fs[f'tvtF50_{fn}']=tvt_f50
        tvt_fs[f'bw_{fn}']=np.float32(b_full); tvt_fs[f'bww_{fn}']=np.float32(b_wls)
        tvt_fs[f'bw50_{fn}']=np.float32(b_late)
        tvt_fs[f'bw_early_{fn}']=np.float32(b_early)   # NEW: early segment
        tvt_fs[f'bw_mid_{fn}']=np.float32(b_mid)       # NEW: mid segment
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2)))
        form_list.append(tvt_f)

    fs=np.stack(form_list,1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32)
    form_std_d =fs.std(1).astype(np.float32)
    form_rng_d =(fs.max(1)-fs.min(1)).astype(np.float32)

    d_ancc,d_std,d_dist=_DI.impute(xy_ev,self_wid=swid)
    d_kn,d_std_kn,_=_DI.impute(xy_kn,self_wid=swid)
    b_vd=ktvt+z_kn-d_kn
    _,b_de,b_dm,b_dl,b_dw=seg_b_well(ktvt,z_kn,d_kn)
    b_d=float(np.median(b_vd))
    tvt_dense  =(-z_ev+d_ancc+b_d  ).astype(np.float32)
    tvt_densew =(-z_ev+d_ancc+b_dw ).astype(np.float32)
    tvt_dense50=(-z_ev+d_ancc+b_dl ).astype(np.float32)
    res_kn=ktvt+z_kn-d_kn
    d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))

    all_sigs=[pf_use]+[p for p in bpaths.values()]+[sc8,sc15,sc25,sc_ens,tvt_fs['tvtF_ANCC'],tvt_dense]
    sig_mat=np.stack(all_sigs,1)
    sig_std=sig_mat.std(1).astype(np.float32)
    sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)

    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1)
        rolls[f'grm{w}']=r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f'grs{w}']=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f'glag{lag}']=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f'glead{lag}']=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)
                   ).iloc[ev.index].values.astype(np.float32)

    hmd=ev['MD'].to_numpy(np.float32); md_since=hmd-float(lk['MD'])
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32)
    slp_b_50 =(last_tvt+slp_50 *md_since).astype(np.float32)

    mdd=hw['MD'].diff().replace(0,np.nan)
    dzdmd=(hw['Z'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd=(hw['X'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dydmd=(hw['Y'].diff()/mdd).iloc[ev.index].values.astype(np.float32)

    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh,np.float32(v),np.float32)

    feats={
        'well':wid,'id':[f'{wid}_{i}' for i in ev.index],
        'last_known_tvt':sc(last_tvt),
        'pf_ancc':pf_use,'pf_ancc_std':std_use,
        'pf_ancc_delta':(pf_use-last_tvt).astype(np.float32),
        'pf_z':(pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        'pf_z_delta':((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.)),
        'pf_vs_z':((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f'beam_{t}_d':(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        'beam_mean_d':np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        'beam_std_d': np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        'beam_med_d': np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        'sc8_d':(sc8-np.float32(last_tvt)).astype(np.float32),'sc8_sc':sc8s,
        'sc15_d':(sc15-np.float32(last_tvt)).astype(np.float32),'sc15_sc':sc15s,
        'sc25_d':(sc25-np.float32(last_tvt)).astype(np.float32),'sc25_sc':sc25s,
        'sc_cons_d':(sc_cons-np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d':(sc_ens-np.float32(last_tvt)).astype(np.float32),  # score-weighted ensemble
        'sc_trust':sc(sc_trust),'hyb_d':(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        'sig_std':sig_std,'sig_mean_d':sig_mean,
        **tvt_fs,
        **{f'frm_rmse_{fn}':sc(form_rmse[fn]) for fn in FORMATIONS},
        'form_mean_d':form_mean_d,'form_std_d':form_std_d,'form_rng_d':form_rng_d,
        'spatial_ancc_d':(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),
        'spatial_knn_dist':knn_d,
        'dense_ancc':d_ancc,'dense_std':d_std,'dense_dist':d_dist,
        'tvt_dense_d' :(tvt_dense -last_tvt).astype(np.float32),
        'tvt_densew_d':(tvt_densew-last_tvt).astype(np.float32),
        'tvt_dense50_d':(tvt_dense50-last_tvt).astype(np.float32),
        'dense_rmse':sc(d_rmse),'dense_bias':sc(d_bias),'dense_nb_std':sc(d_nb_std),
        'pf_vs_spatial':(pf_use-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'pf_vs_dense':(pf_use-tvt_dense).astype(np.float32),
        'spatial_vs_dense':(tvt_fs['tvtF_ANCC']-tvt_dense).astype(np.float32),
        'beam_vs_spatial':(bpaths['cons']-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'sc_vs_beam':(sc_ens-bpaths['cons']).astype(np.float32),
        'cal_a':sc(a_cal),'cal_b':sc(b_cal),
        'pfx_rmse':sc(pfx_rmse),'known_len':sc(len(kn)),'eval_len':sc(nh),
        'slp_all':sc(slp_all),'slp_50':sc(slp_50),'slp_z':sc(slp_z),
        'slp_b_d_all':(slp_b_all-last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50 -last_tvt).astype(np.float32),
        'ktvt_range':sc(float(np.ptp(ktvt))),'ktvt_std':sc(float(ktvt.std())),
        'md_since':md_since,'frac':frac,'frac2':frac**2,'sqrt_frac':np.sqrt(frac),
        'z':z_ev,
        'dx':(ev['X']-float(lk['X'])).to_numpy(np.float32),
        'dy':(ev['Y']-float(lk['Y'])).to_numpy(np.float32),
        'dz':(z_ev-float(lk['Z'])).astype(np.float32),
        'dxy':np.sqrt((ev['X']-float(lk['X']))**2+(ev['Y']-float(lk['Y']))**2).to_numpy(np.float32),
        'dzdmd':dzdmd,'dxdmd':dxdmd,'dydmd':dydmd,
        'gr':hgr,'gr_d1':gr_d1,'gr_d2':gr_d2,'gr_env':gr_env,'gr_nrg':gr_nrg,
        'gr_vs_tw_anc':hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),
        'gr_vs_slp_all':hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f'tda{int(o)}' :hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f'tdbc{int(o)}':hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f'tdsc{int(o)}':hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f'tdpf{int(o)}':hgr-np.interp(pf_use+o,tw_tvt,tw_gr).astype(np.float32) for o in PF_OFFS},
        'tw_range':sc(float(np.ptp(tw_tvt))),'tw_gr_mean':sc(float(tw_gr.mean())),
    }
    for k,v in rolls.items(): feats[k]=v
    result=pd.DataFrame(feats)
    if is_train:
        if 'TVT' not in ev.columns or ev['TVT'].isna().all(): return None
        result['target']=(ev['TVT'].to_numpy(np.float32)-np.float32(last_tvt))
    return result

def build_dataset(paths,is_train,label):
    args=[(str(p),str(p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv'),is_train)
          for p in paths
          if (p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv').exists()]
    t0=time.time()
    res=Parallel(n_jobs=NCPU,prefer='threads',verbose=3)(
        delayed(build_well)(hp,tp,it) for hp,tp,it in args)
    parts=[r for r in res if r is not None]
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

# %% markdown 11: ### 1.5 · Build (or fast-load) the feature table If the artifacts dataset is mounted, the pre-computed `train.csv` is loaded directly — minutes instead of hours.


# %% cell 12
if (CFG.artifacts_path / "data" / "train.csv").exists():
    train_df = pd.read_csv(CFG.artifacts_path / "data" / "train.csv", low_memory=False)
else:
    train_paths = sorted((CFG.dataset_path / "train").glob('*__horizontal_well.csv'))
    train_df = build_dataset(train_paths, is_train=True, label="train")    

test_paths = sorted((CFG.dataset_path / "test").glob('*__horizontal_well.csv'))
test_df = build_dataset(test_paths, is_train=False, label="test")

features = [c for c in train_df.columns if c not in {'well','id','target'}]

X = train_df[features]
y = train_df['target']
g = train_df['well']

X_test = test_df[features]

# %% markdown 13: ### 1.6 · Training setup — booster configs & CV splitter GroupKFold **by well** (no leakage across a well's rows). LightGBM configs use `device_type='gpu'` — set **Accelerator = GPU**.


# %% cell 14
lgb_params = [
    dict(
        boosting_type="gbdt", 
        num_leaves=255, 
        min_child_samples=15,
        subsample=0.8, 
        subsample_freq=1, 
        colsample_bytree=0.8,
        reg_lambda=3.0, 
        reg_alpha=0.05, 
        objective="regression",
        verbose=-1, 
        n_jobs=-1, 
        max_bin=255,
        learning_rate=0.030, 
        n_estimators=5000, 
        seed=123
    ),
    dict(
        n_jobs=-1, 
        verbose=-1, 
        reg_alpha=10.788188919840913, 
        subsample=0.47437582748953966, 
        num_leaves=64, 
        reg_lambda=95.75401894533888, 
        n_estimators=10000,
        random_state=0,
        boosting_type='gbdt', 
        learning_rate=0.00934485794382918,
        colsample_bytree=0.39283351290380497,
        min_child_weight=0.24081152127177283, 
        min_child_samples=40,
    ),
    dict(
        n_jobs=-1, 
        verbose=-1, 
        reg_alpha=10.788188919840913, 
        subsample=0.47437582748953966, 
        num_leaves=64, 
        reg_lambda=95.75401894533888, 
        n_estimators=10000,
        random_state=29,
        boosting_type='gbdt', 
        learning_rate=0.00934485794382918,
        colsample_bytree=0.39283351290380497,
        min_child_weight=0.24081152127177283, 
        min_child_samples=40,
    ),
]

cb_params = [
    dict(
        iterations=8000, 
        depth=7, 
        l2_leaf_reg=2.0,
        min_data_in_leaf=15, 
        border_count=254,
        loss_function="RMSE", 
        task_type="CPU", 
        od_type="Iter", 
        od_wait=300, 
        verbose=0,
        learning_rate=0.020, 
        random_seed=7
    ),
    dict(
        iterations=8000, 
        depth=7, 
        l2_leaf_reg=2.0,
        min_data_in_leaf=15, 
        border_count=254,
        loss_function="RMSE", 
        task_type="CPU", 
        od_type="Iter", 
        od_wait=300, 
        verbose=0,
        learning_rate=0.030, 
        random_seed=123
    ),
]

ridge_params = {
    "random_state": 42,
    "alpha": 1.6602834637650032,
    "tol": 0.0005030247295617308,
    "positive": True,
    "fit_intercept": True
}

pp_params = {
    'alpha': 1.0,
    'tau': 85,
    'w_pf': 0.09
}

# %% cell 15
oof_preds = {}
test_preds = {}

overall_scores = {}
fold_scores = {}

# %% markdown 16: ### 1.7 · LightGBM (pre-trained fast-path when the artifacts are mounted)


# %% cell 17
for i, params in enumerate(lgb_params):   
    save_path = f"models/lightgbm-{i+1}"
    
    if (CFG.artifacts_path / save_path).exists():
        print(f"Loading lightgbm-{i+1} from disk...")
        
        trainer_paths = (CFG.artifacts_path / save_path).glob('*.pkl')
        trainer = joblib.load(list(trainer_paths)[0])
        
        print(f"Loaded lightgbm-{i+1} with overall RMSE: {trainer.overall_score:.4f}\n")
    else:
     
        trainer = Trainer(
            estimator=LGBMRegressor(**params),
            task="regression",
            metric=CFG.metric,
            cv=CFG.cv,
            cv_args={"groups": g},
            use_early_stopping=True,
            verbose=True,
            save=True,
            save_path=save_path
        )
        
        trainer.fit(
            X, 
            y,
            fit_args={
                "eval_metric": "rmse",
                "callbacks": [
                    log_evaluation(period=250), 
                    early_stopping(stopping_rounds=250)
                ]
            }
        )
        print("\n\n")

    oof_preds[f"lightgbm-{i+1}"] = trainer.oof_preds
    test_preds[f"lightgbm-{i+1}"] = trainer.predict(X_test)
    overall_scores[f"lightgbm-{i+1}"] = trainer.overall_score
    fold_scores[f"lightgbm-{i+1}"] = trainer.fold_scores

# %% markdown 18: ### 1.8 · CatBoost


# %% cell 19
for i, params in enumerate(cb_params):    
    save_path = f"models/catboost-{i+1}"
    if (CFG.artifacts_path / save_path).exists():
        print(f"Loading catboost-{i+1} from disk...")
        
        trainer_paths = (CFG.artifacts_path / save_path).glob('*.pkl')
        trainer = joblib.load(list(trainer_paths)[0])
        
        print(f"Loaded catboost-{i+1} with overall RMSE: {trainer.overall_score:.4f}\n")
    else:
        trainer = Trainer(
            estimator=CatBoostRegressor(**params),
            task="regression",
            metric=CFG.metric,
            cv=CFG.cv,
            cv_args={"groups": g},
            use_early_stopping=True,
            verbose=True,
            save=True,
            save_path=save_path
        )
        
        trainer.fit(
            X, 
            y,
            fit_args={
                "verbose": 250,
                "early_stopping_rounds": 250,
                "use_best_model": True
            }
        )
        print("\n\n")

    oof_preds[f"catboost-{i+1}"] = trainer.oof_preds
    test_preds[f"catboost-{i+1}"] = trainer.predict(X_test)
    overall_scores[f"catboost-{i+1}"] = trainer.overall_score
    fold_scores[f"catboost-{i+1}"] = trainer.fold_scores

# %% markdown 20: ### 1.9 · Ridge meta-ensemble over the boosters' out-of-fold predictions


# %% cell 21
oof_preds = pd.DataFrame(oof_preds)
test_preds = pd.DataFrame(test_preds)

# %% cell 22
ridge_trainer = Trainer(
    Ridge(**ridge_params),
    task="regression",
    metric=CFG.metric,
    cv=CFG.cv,
    cv_args={"groups": g},
    verbose=True
)

ridge_trainer.fit(oof_preds, y)

ridge_oof_preds = ridge_trainer.oof_preds
ridge_test_preds = ridge_trainer.predict(test_preds)

overall_scores["ridge"] = ridge_trainer.overall_score
fold_scores["ridge"] = ridge_trainer.fold_scores

# %% markdown 23: ### 1.10 · Post-processing — warm-up damping (τ) + Savitzky–Golay smoothing Right after the prediction-start point the geology has barely moved — damping the predicted delta there is free accuracy.


# %% cell 24
def apply_pp(df, md, pd_, alpha, tau, w_pf):
    d = md * (1-w_pf) + pd_ * w_pf
    if tau: 
        d *= (1.-np.exp(-np.maximum(df['md_since'].values,0.) / tau))
        
    return d * alpha

def sg_smooth(df, col, sg_w=17, sg_p=3):
    df = df.copy()
    
    for _, g in df.groupby('well', sort=False):
        v = g[col].values
        n = len(v)
        wl = min(sg_w, n)
        
        if wl % 2 == 0: 
            wl -= 1
            
        if wl >= sg_p + 2: 
            v = savgol_filter(v, wl, sg_p)
            
        df.loc[g.index,col] = v
        
    return df

# %% cell 25
base = train_df['last_known_tvt'].values
ytrue = y.values + base

pf_oof = (train_df['pf_ancc'].values - base)

d = apply_pp(train_df, ridge_oof_preds, pf_oof, **pp_params)
ridge_score = root_mean_squared_error(ytrue, base + d)

overall_scores["ridge (pp)"] = ridge_score
fold_scores["ridge (pp)"] = [ridge_score] * CFG.n_splits

# %% markdown 26: ### 1.11 · Inference — Ridge (tabular) predictions


# %% cell 27
test_df2 = test_df.copy()
pf_test = test_df2['pf_ancc'].values - test_df2['last_known_tvt'].values

test_df2['pred'] = test_df2['last_known_tvt'].values + apply_pp(
    test_df2, 
    ridge_test_preds,
    pf_test, 
    **pp_params
)
test_df2 = sg_smooth(test_df2, 'pred')

# %% cell 28
sample_sub = pd.read_csv(CFG.dataset_path / "sample_submission.csv")
sub_1 = (sample_sub[['id']].merge(
    test_df2[['id', 'pred']].rename(columns={'pred':'tvt'}),
    on='id', 
    how='left'
))

sub_1['tvt']=sub_1['tvt'].fillna(float(train_df['last_known_tvt'].mean()+train_df['target'].mean()))
sub_1

# %% markdown 29: ### 1.12 · Inference — heuristic physical model (per-well PF selector; exact contacts where available)


# %% cell 30
sample = pd.read_csv(CFG.dataset_path / 'sample_submission.csv')
sample['well']    = sample['id'].str[:8]
sample['row_idx'] = sample['id'].str[9:].astype(int)

train_hw_files = sorted(glob.glob(str(CFG.dataset_path / 'train' / '*__horizontal_well.csv')))
train_wells = [os.path.basename(f).split('__')[0] for f in train_hw_files]

test_hw_files = sorted(glob.glob(str(CFG.dataset_path / 'test' / '*__horizontal_well.csv')))
test_wells = [os.path.basename(f).split('__')[0] for f in test_hw_files]

rows = []
for i, wid in enumerate(test_wells):
    print(f'\nProcessing {i + 1}/{len(test_wells)}: {wid}...')
    hw_te, tw_te = load_well(wid, 'test')

    tvt_phys = None
    hw_tr    = None
    tw_tr    = None

    # Physical model for visible wells
    if wid in train_wells:
        try:
            hw_tr, tw_tr = load_well(wid, 'train')
            hw_te['TVT_input'] = hw_tr['TVT_input'].values
            tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
            print(f'  Physical model OK')
        except Exception as e:
            print(f'  Physical model failed: {e}')
            tvt_phys = None

    selector_code, selector_variant, selector_n_eval, selector_z_span = selector_well_code(hw_te)

    # 128-seed likelihood-weighted PF ensemble
    try:
        tw_ref = tw_tr if tw_tr is not None else tw_te
        pf_by_scale = run_pf_lik_ensemble_scales(hw_te, tw_ref, n_particles=500, n_seeds=128)
        tvt_pf = pf_by_scale['pf_scale_8']
        print(f'  PF 128-seed lik-ensemble OK scales={SELECTOR_SCALES}')
    except Exception as e:
        print(f'  PF failed: {e}')
        last_known = hw_te['TVT_input'].dropna()
        last_val   = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
        tvt_pf = hw_te['TVT_input'].fillna(last_val).values.astype(float)
        pf_by_scale = {f'pf_scale_{scale:g}': tvt_pf.copy() for scale in SELECTOR_SCALES}

    # Beam search ensemble
    try:
        tw_ref = tw_tr if tw_tr is not None else tw_te
        tvt_beam = run_beam_ensemble(hw_te, tw_ref)
        print(f'  Beam 14-config ensemble OK')
    except Exception as e:
        print(f'  Beam failed: {e}')
        tvt_beam = tvt_pf.copy()

    # Selector blending
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
            tvt_val = float(tvt_phys.iloc[ridx])
        else:
            tvt_val = float(tvt_selector[ridx])
        rows.append({'id': row['id'], 'tvt': tvt_val})
    print(f'  Added {len(ws)} rows')

# %% cell 31
sub_2 = pd.DataFrame(rows)

# %% markdown 32: ### 1.13 · Blend + robust projection `0.3·tabular + 0.7·physics`, then a per-well **robust degree-4 polynomial fit** (IRLS outlier down-weighting) of `U = tvt + Z − anchor` versus normalized MD — a trajectory-level de-noiser. The `tvt+Z` fr


# %% cell 33
sub = (
    sub_1.merge(sub_2, on='id', suffixes=('_1', '_2'))
       .assign(tvt=lambda x: 0.3 * x['tvt_1'] + 0.7 * x['tvt_2'])
       [['id', 'tvt']]
)
sub.to_csv("submission.csv", index=False)
sub

# %% cell 34
# === robust low-order PROJECTION post-processing (patched degree=4, blend=0.75) (CV-validated: raw PF -0.54, deployed components -0.33) ===
# Runs AFTER the 0.3*ridge+0.7*selector blend writes submission.csv; OVERWRITES it with the projected
# version. Per-well robust deg-5 fit of dU = tvt + Z - anchor vs normalized MD -> denoise jitter +
# down-weight wrong-branch outliers. Deterministic; defensive per-well fallback to raw.
import numpy as _np, pandas as _pd
def _robfit(s, y, deg=5):
    if len(s) < deg + 2:
        return y.copy()
    c = _np.polyfit(s, y, deg)
    for _ in range(4):
        r = y - _np.polyval(c, s)
        sc = _np.median(_np.abs(r)) * 1.4826 + 1e-6
        c = _np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return _np.polyval(c, s)
try:
    _base = _pd.read_csv("submission.csv")   # the just-written blended submission
    assert set(['id','tvt']).issubset(_base.columns)
    _base['well'] = _base['id'].str[:8]
    _base['row_idx'] = _base['id'].str[9:].astype(int)
    _out = dict(zip(_base['id'].values, _base['tvt'].astype(float).values))
    _n_ok = 0
    for _wid, _g in _base.groupby('well'):
        try:
            _hw = _pd.read_csv(CFG.dataset_path / 'test' / (_wid + '__horizontal_well.csv'))
            _kn = _hw[_hw['TVT_input'].notna()]
            if len(_kn) < 5:
                continue
            _last = _kn.iloc[-1]
            _anchor = float(_last['TVT_input']) + float(_last['Z'])
            _ps = float(_last['MD']); _end = float(_hw['MD'].iloc[-1])
            _gi = _g.sort_values('row_idx')
            _ri = _gi['row_idx'].values
            _Z = _hw['Z'].values[_ri].astype(float)
            _md = _hw['MD'].values[_ri].astype(float)
            _s = (_md - _ps) / max(_end - _ps, 1e-6)
            _tvt = _gi['tvt'].values.astype(float)
            _fit = _robfit(_s, (_tvt + _Z) - _anchor, 4)
            _tvt_fit_full = (_anchor + _fit) - _Z
            _tvt_fit = 0.25 * _tvt + 0.75 * _tvt_fit_full
            if not _np.all(_np.isfinite(_tvt_fit)):
                continue
            for _rid, _val in zip(_gi['id'].values, _tvt_fit):
                _out[_rid] = float(_val)
            _n_ok += 1
        except Exception as _e:
            print('proj fallback', _wid, _e)
    print('projection applied to', _n_ok, 'wells')
    _final = _base[['id']].copy()
    _final['tvt'] = _final['id'].map(_out).astype(float)
    _final[['id','tvt']].to_csv("submission.csv", index=False)
    print('wrote projected submission.csv', _final.shape)
except Exception as _e:
    print('PROJECTION SKIPPED (kept blended submission):', _e)

# %% markdown 35: ### 1.14 · Pipeline A results — and preserve its output for the final blend


# %% cell 36
fold_scores_df = pd.DataFrame(fold_scores)
overall_scores_df = pd.DataFrame({k: [v] for k, v in overall_scores.items()}).transpose().sort_values(by=0, ascending=True)
order = overall_scores_df.index.tolist()

min_score = overall_scores_df.values.flatten().min()
max_score = overall_scores_df.values.flatten().max()
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, fold_scores_df.shape[1] * 0.5))

boxplot = sns.boxplot(data=fold_scores_df, order=order, ax=axs[0], orient="h", color="grey")
axs[0].set_title(f"Fold RMSE")
axs[0].set_xlabel("")
axs[0].set_ylabel("")

barplot = sns.barplot(x=overall_scores_df.values.flatten(), y=overall_scores_df.index, ax=axs[1], color="grey")
axs[1].set_title(f"Overall RMSE")
axs[1].set_xlabel("")
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel("")

for i, (score, model) in enumerate(zip(overall_scores_df.values.flatten(), overall_scores_df.index)):
    color = "cyan" if "ridge" in model.lower() else "grey"
    barplot.patches[i].set_facecolor(color)
    boxplot.patches[i].set_facecolor(color)
    barplot.text(score, i, round(score, 3), va="center")

plt.tight_layout()
plt.show()

# %% cell 37
from pathlib import Path as _BlendPath
import pandas as _blend_pd
_sp45_path = _BlendPath('/kaggle/working/submission.csv') if _BlendPath('/kaggle/working').exists() else _BlendPath('submission.csv')
_sp45_df = _blend_pd.read_csv(_sp45_path)
_sp45_df.to_csv((_BlendPath('/kaggle/working') if _BlendPath('/kaggle/working').exists() else _BlendPath('.')) / 'sp45_projection_submission.csv', index=False)
print('saved sp45_projection_submission.csv', _sp45_df.shape, flush=True)


# === fleongg pretrained inference section ===

# %% markdown 38: --- # Part 2 · Pipeline B — "fleongg" (likelihood-PF + GBM stack) An **independent implementation** of the same physics: likelihood-weighted multi-scale particle filter, offset-well spatial priors, and a GroupKFold GBM stack. Independent im


# %% cell 39
import os, sys, glob, time, warnings, multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
os.environ.setdefault("SHOW_FIGS", "0")

# ---- environment / paths (Kaggle or local) -------------------------------------
def _find_data():
    for c in ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",
              "/kaggle/input/rogii-wellbore-geology-prediction"]:
        if Path(c).exists() and (Path(c)/"train").exists():
            return Path(c)
    # fallback: find any mounted folder that contains a train/ directory
    for p in glob.glob("/kaggle/input/**/train", recursive=True):
        return Path(p).parent
    return Path(os.environ.get("ROGII_DATA", "."))   # local override for development

class CFG:
    DATA = _find_data()
    OUT  = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    seed = 7
    n_splits = 5
    n_jobs = 1
    # lik-PF
    PF_SEEDS = 128
    PF_PARTICLES = 500
    PF_SCALES = (3., 5., 8., 12.)
    # FAST dev (local smoke test): limit train wells & trees
    FAST = bool(int(os.environ.get("FAST", "0")))
    N_TRAIN_WELLS = int(os.environ.get("N_TRAIN_WELLS", "0"))  # 0 = all
    USE_GPU = os.environ.get("USE_GPU", "auto")
    SHOW_FIGS = os.environ.get("SHOW_FIGS", "1") == "1"   # EDA plots (on in the notebook)

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
def _demo_well():
    """A train well with TVT + a sizable eval zone, for the EDA plots."""
    for w in sorted(p.stem.replace("__horizontal_well", "")
                    for p in (CFG.DATA/"train").glob("*__horizontal_well.csv")):
        try:
            d = pd.read_csv(CFG.DATA/"train"/f"{w}__horizontal_well.csv", usecols=["TVT", "TVT_input"])
        except Exception:
            continue
        if "TVT" in d and d.TVT.notna().any() and d.TVT_input.isna().sum() > 2000:
            return w
    return None
print("DATA:", CFG.DATA, "| OUT:", CFG.OUT, "| cores:", CFG.n_jobs, "| FAST:", CFG.FAST)

def load_well(wid, split="train"):
    base = CFG.DATA / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return hw, tw

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float))**2)))

# %% markdown 40: ### 2.2 · EDA — the problem, visually *(plots are off by default; set `SHOW_FIGS=1` to see them)*


# %% cell 41
def fig_overview(wid):
    import matplotlib.pyplot as plt
    hw, tw = load_well(wid)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]; ps = kn.MD.iloc[-1]
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    ax[0].plot(hw.MD, hw.Z, lw=1.2, color="#333"); ax[0].axvline(ps, color="crimson", ls="--", label="PS")
    ax[0].set_ylabel("Z / TVD (ft)"); ax[0].legend(loc="upper right")
    ax[0].set_title(f"Well {wid}: trajectory · gamma-ray · TVT target")
    ax[1].plot(kn.MD, kn.GR, lw=.7, color="steelblue", label="GR known")
    ax[1].plot(ev.MD, ev.GR, lw=.7, color="darkorange", label="GR eval"); ax[1].axvline(ps, color="crimson", ls="--")
    ax[1].set_ylabel("GR (API)"); ax[1].legend(loc="upper right")
    ax[2].plot(kn.MD, kn.TVT, lw=1.6, color="seagreen", label="TVT known (=input)")
    ax[2].plot(ev.MD, ev.TVT, lw=1.6, color="crimson", label="TVT to predict"); ax[2].axvline(ps, color="crimson", ls="--")
    ax[2].set_ylabel("TVT (ft)"); ax[2].set_xlabel("MD (ft)"); ax[2].invert_yaxis(); ax[2].legend(loc="upper right")
    for a in ax: a.grid(alpha=.25)
    plt.tight_layout(); plt.show()

def fig_correlation(wid):
    import matplotlib.pyplot as plt
    hw, tw = load_well(wid); ev = hw[hw.TVT_input.isna()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    ax[0].plot(tw.GR, tw.TVT, lw=1.0, color="black")
    ax[0].set_xlabel("GR (API)"); ax[0].set_ylabel("TVT (ft)"); ax[0].invert_yaxis()
    ax[0].set_title("Typewell signature: GR vs TVT")
    sc = ax[1].scatter(ev.GR, ev.TVT, s=4, c=ev.MD, cmap="viridis")
    ax[1].set_xlabel("GR (API)"); ax[1].set_ylabel("TVT (ft)"); ax[1].invert_yaxis()
    ax[1].set_title("Horizontal GR at its true TVT\nmatches the typewell signature")
    plt.colorbar(sc, ax=ax[1], label="MD (ft)")
    for a in ax: a.grid(alpha=.25)
    plt.tight_layout(); plt.show()

def fig_drift_tail(n_wells=250):
    import matplotlib.pyplot as plt
    wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"train").glob("*__horizontal_well.csv"))
    rng = np.random.default_rng(1); samp = sorted(rng.choice(wids, min(n_wells, len(wids)), replace=False).tolist())
    per = []
    for wid in samp:
        try: hw = pd.read_csv(CFG.DATA/"train"/f"{wid}__horizontal_well.csv", usecols=["TVT_input", "TVT"])
        except: continue
        ev = hw[hw.TVT_input.isna()]; kn = hw[hw.TVT_input.notna()]
        if len(ev) == 0 or len(kn) < 10 or hw.TVT.isna().all(): continue
        t = ev.TVT.values
        if np.isnan(t).any(): continue
        per.append(np.sqrt(np.mean((t-kn.TVT_input.iloc[-1])**2)))
    per = np.array(per); srt = np.sort(per)[::-1]; cum = np.cumsum(srt**2)/np.sum(srt**2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(per, bins=40, color="indianred", alpha=.85)
    ax[0].axvline(np.median(per), color="k", ls="--", label=f"median={np.median(per):.1f}")
    ax[0].axvline(per.mean(), color="b", ls="--", label=f"mean={per.mean():.1f}")
    ax[0].set_xlabel("per-well last-known-baseline RMSE (ft)"); ax[0].set_ylabel("wells"); ax[0].legend()
    ax[0].set_title("Per-well error is heavily right-skewed")
    ax[1].plot(np.arange(1, len(srt)+1)/len(srt)*100, cum*100, color="purple"); ax[1].axhline(80, color="gray", ls=":")
    ax[1].set_xlabel("% of wells (worst first)"); ax[1].set_ylabel("% of pooled squared error")
    ax[1].set_title("A few drift wells dominate the metric")
    for a in ax: a.grid(alpha=.25)
    plt.tight_layout(); plt.show()

# %% cell 42
DEMO = "00bbac68" if (CFG.DATA/"train"/"00bbac68__horizontal_well.csv").exists() else _demo_well()
if CFG.SHOW_FIGS:
    print("demo well:", DEMO)
    if DEMO:
        fig_overview(DEMO)
        fig_correlation(DEMO)
    fig_drift_tail()

# %% markdown 43: ### 2.3 · Trackers — likelihood-weighted PF (128 seeds × 4 scales) + beam search


# %% cell 44
# ---- single particle filters (ANCC-anchored & Z-velocity-coupled), numba ---------
PF_N = 600; ANCC_N = 600
PF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.
PF_GR_WIN = 5; PF_GR_WT = 0.3; PF_RESAMP = 0.5; PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001

BEAMS = [(10,20.,144.,2,"cons"),(10,8.,64.,2,"loose"),(8,35.,220.,1,"vcons"),
         (10,14.,90.,5,"sm5"),(20,4.,36.,3,"vloose"),(12,12.,100.,3,"mid"),(15,25.,180.,2,"stiff")]

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j]+w[j]
    u0 = np.random.uniform(0., 1./N); np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0+j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci]+rp*np.random.randn(); na[j] = aux[ci]+rv*np.random.randn()
    return np2, na

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS*6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx+d
                if ni < 0 or ni >= nt: continue
                tot = cost+(gv-tw_gr[ni])**2/es+mc*(d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i+1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]; cC[i], cC[mi] = cC[mi], cC[i]; cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n-1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls+IS*np.random.randn(); rate[j] = ir+0.01*np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = ALPHA*rate[j]+RN*np.random.randn(); pos[j] += rate[j]*dm+PN*np.random.randn()
            tvt_j = pos[j]-z_v[i]; tvt_j = max(tvt_j, vmin-50.); tvt_j = min(tvt_j, vmin+len(gg)*step+50.)
            pos[j] = tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs
                lk = max(np.exp(-0.5*d*d) if d*d < 600. else 0., 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1./N
        tv = 0.
        for j in range(N): tv += w[j]*(pos[j]-z_v[i])
        pts[i] = tv; va = 0.
        for j in range(N): va += w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i] = va**0.5; pm = md_v[i]
    return pts, std_

@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
         MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ip+0.5*np.random.randn(); vel[j] = iv+0.02*np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0]-1.; pz = z_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.); dzd = (z_v[i]-pz)/dm; ve = beta*dzd+icpt
        for j in range(N):
            vel[j] = MOM*vel[j]+VN*np.random.randn(); pos[j] += vel[j]*dm+PN*np.random.randn()
            pos[j] = max(pos[j], vmin-50.); pos[j] = min(pos[j], vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step); dp = (gr_v[i]-ep)/gs
                lp = max(np.exp(-0.5*dp*dp) if dp*dp < 600. else 0., 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step); ds = (gr_sm_v[i]-es)/(gs*1.5)
                    lsm = max(np.exp(-0.5*ds*ds) if ds*ds < 600. else 0., 1e-300); lk = (1.-GR_WT)*lp+GR_WT*lsm
                else: lk = lp
                lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ws2 = 0.
        for j in range(N):
            dv = (vel[j]-ve)/max(zsig*2., 0.005); lz = max(np.exp(-0.5*dv*dv) if dv*dv < 600. else 0., 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.:
            for j in range(N): w[j] /= ws2
        else:
            for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1./N
        wm = 0.
        for j in range(N): wm += w[j]*pos[j]
        pts[i] = wm; va = 0.
        for j in range(N): va += w[j]*(pos[j]-wm)**2
        std_[i] = va**0.5; pm = md_v[i]; pz = z_v[i]
    return pts, std_

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn.GR.values-np.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))

def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr): return len(arr)-1
    if i > 0 and abs(arr[i-1]-v) <= abs(arr[i]-v): return i-1
    return i

def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r*2+1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt); sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    return tw_tvt[_beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))].astype(np.float32)

def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, std = _pf_ancc(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), ev.GR.values.astype(np.float64),
                        gg, gmin, gst, gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr); tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    dz_k = np.diff(kna.Z.values); dvt = np.diff(kna.TVT_input.values); dmd_k = np.diff(kna.MD.values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2]/dmd_k[m2]; vt = dvt[m2]/dmd_k[m2]; A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt-(c[0]*vz+c[1]))), 0.001)
    else: beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20); dvt2 = np.diff(t2.TVT_input.values); dmd2 = np.diff(t2.MD.values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr); gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw.GR.rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), ev.GR.values.astype(np.float64),
                     gr_sm.loc[ev.index].values.astype(np.float64), gg, gs2, gmin, gst, gs,
                     float(kna.TVT_input.iloc[-1]), iv, beta, icpt, zsig, N,
                     PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2*hw+1; nk = len(kgr); nh = len(hgr)
        if nk < win+1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk-win+1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None]+np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C-C.mean(1, keepdims=True))/(C.std(1, keepdims=True)+1e-6)
        hp = np.pad(hg, hw, mode="edge"); H = hp[np.arange(nh)[:, None]+np.arange(win)[None, :]].astype(np.float32)
        Hn = (H-H.mean(1, keepdims=True))/(H.std(1, keepdims=True)+1e-6)
        ncc = Hn@Cn.T/win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw, 0, nk-1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.*scores); sw /= sw.sum(1, keepdims=True)+1e-9
    return out, (tvts*sw).sum(1).astype(np.float32)

# %% cell 45
# ---- 128-seed likelihood-weighted particle filter (the workhorse), numba ---------
@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                     MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds); tmax = vmin + len(gg)*step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
        for j in range(N):
            pos[j] = ls + init_spr*np.random.randn(); rate[j] = ir + 0.01*np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM*rate[j] + VN*np.random.randn(); pos[j] += rate[j]*dm + PN*np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin-100.: tvt_j = vmin-100.
                if tvt_j > tmax+100.: tvt_j = tmax+100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs; dd = d*d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j]*lk; w[j] = w[j]*lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
            neff = 0.0
            for j in range(N): neff += w[j]*w[j]
            neff = 1.0/neff
            if neff < RESAMP*N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1./N); newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j/N
                    while ci < N-1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP*np.random.randn(); newrate[j] = rate[ci] + RR*np.random.randn()
                for j in range(N): pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1./N
            est = 0.0
            for j in range(N): est += w[j]*(pos[j]-z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks

def lik_pf(hw, tw, n_particles=CFG.PF_PARTICLES, n_seeds=CFG.PF_SEEDS, scales=CFG.PF_SCALES,
           init_spr=4.5, seed_base=0, with_quality=False):
    """Likelihood-weighted PF ensemble. Returns ({pf_scale_X: pred_eval}, ev_index[, quality])."""
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return {}, np.array([]), {}
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = _pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                   gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, seed_base,
                                   0.998, 0.002, 0.005, 0.1, 0.001, 0.5, init_spr)
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln/float(sc)); wts /= wts.sum(); out[f"pf_scale_{sc:g}"] = (wts[:, None]*preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    q = {}
    if with_quality:
        q = {"pf_best_ll": float(liks.max())/len(ev), "pf_ll_spread": float(liks.std()),
             "pf_pt_std": preds.std(0).astype(np.float32), "pf_gr_sig": gs}
    return out, ev.index.values, q

# JIT warm-up so timings below are representative
_m = np.linspace(1, 50, 20); _z = np.zeros(20); _g = np.full(20, 50.); _gg = np.linspace(45, 55, 100)
_pf_ancc(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 8, .998, .002, .005, .3, .1, .001, .5)
_pf_z(_m, _z, _g, _g, _gg, _gg, 45., .1, 20., 50., 0., -1., 0., .1, 8, .993, .005, .01, .3, .2, .003, .5)
_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)
_pf_lik_allseeds(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 64, 4, 0, .998, .002, .005, .1, .001, .5, 4.5)
print("trackers compiled.")

def fig_tracker_vs_truth(wid):
    import matplotlib.pyplot as plt
    hw, tw = load_well(wid); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    tw_tvt = tw.TVT.to_numpy(np.float32); tw_gr = tw.GR.to_numpy(np.float32); last = float(kn.TVT_input.iloc[-1])
    pf, _ = run_pf_ancc(hw, tw_tvt, tw_gr); out, _, _ = lik_pf(hw, tw, scales=(3.,))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ev.MD, ev.TVT, lw=2.2, color="black", label="True TVT", zorder=5)
    ax.plot(ev.MD, np.full(len(ev), last), lw=1.1, color="gray", ls=":", label="last-known baseline")
    ax.plot(ev.MD, pf, lw=1.0, color="tab:blue", alpha=.8, label="single particle filter")
    ax.plot(ev.MD, out["pf_scale_3"], lw=1.5, color="crimson", alpha=.9, label="128-seed lik-weighted PF")
    ax.set_xlabel("MD (ft)"); ax.set_ylabel("TVT (ft)"); ax.invert_yaxis(); ax.grid(alpha=.25)
    ax.set_title(f"Well {wid}: trackers vs ground truth — the lik-PF resists drift"); ax.legend(loc="best")
    plt.tight_layout(); plt.show()

# %% cell 46
if CFG.SHOW_FIGS and DEMO:
    fig_tracker_vs_truth(DEMO)

# %% markdown 47: ### 2.4 · Offset-well spatial priors — plane-KNN through formation tops + dense ANCC surface


# %% cell 48
PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20

def robust_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6: return 0.
    return float(np.polyfit(x[m], y[m], 1)[0])

def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1., float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1); return float(a), float(b)

def seg_b_well(ktvt, kz, form_col):
    bv = ktvt+kz-form_col; n = len(bv); b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n-50):])) if n >= 5 else b_full
    t1, t2 = n//3, 2*n//3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1+1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02*np.arange(n)); w /= w.sum()
    return b_full, b_early, b_mid, b_late, float(np.dot(w, bv))

class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            try: df = pd.read_csv(data_dir/f"{wid}__horizontal_well.csv", usecols=["X","Y"]+FORMATIONS).dropna()
            except: continue
            if len(df) == 0: continue
            row = {"wid": wid, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in FORMATIONS: row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df.wid)}
        xy = self.df[["x","y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy/self.scale); self.xa = self.df.x.to_numpy(); self.ya = self.df.y.to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)
    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q/self.scale; nf = min(k+5, len(self.df)); dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap: dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1./(dk+1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w*xn; wy = w*yn
        A = np.zeros((len(q), 3, 3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs = np.stack([(wx[:,:,None]*fn).sum(1), (wy[:,:,None]*fn).sum(1), (w[:,:,None]*fn).sum(1)], 1)
        try: coef = np.linalg.solve(A, rhs)
        except:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq = xy_q[:,0]; Yq = xy_q[:,1]
        pred = (Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, an, wd = [], [], [], []
        for wid in well_ids:
            try: df = pd.read_csv(data_dir/f"{wid}__horizontal_well.csv", usecols=["X","Y","ANCC"]).dropna()
            except: continue
            if len(df) == 0: continue
            ix = np.linspace(0, len(df)-1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s.X.values); ys.append(s.Y.values); an.append(s.ANCC.values); wd.extend([wid]*len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(an).astype(np.float32); self.wids = np.array(wd)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0)); self.tree = cKDTree(self.xy/self.scale)
    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q); q = xy_q/self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid: dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordr = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1./(dk+1e-3), 0.); sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw)
        a = self.ancc[ik]; ap = (a*w).sum(1)/safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((a-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var, 0.)).astype(np.float32), np.where(vk, dk, np.inf).min(1).astype(np.float32)

_FI = None; _DI = None
ANCH_OFFS = np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80], np.float32)
BEAM_OFFS = np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40], np.float32)
SC_OFFS = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)
PF_OFFS = SC_OFFS.copy()

# %% markdown 49: ### 2.5 · Feature table — tracker deltas, agreement/uncertainty, GR statistics, geometry, spatial anchors


# %% cell 50
def build_well(hw_path, tw_path, is_train, likpf_map=None):
    global _FI, _DI
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    np.random.seed(_stable_seed_from_wid(wid, CFG.seed))
    try: hw = pd.read_csv(hw_path); tw = pd.read_csv(tw_path).sort_values("TVT")
    except: return None
    if is_train and "TVT" not in hw.columns: return None
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10: return None
    if is_train and hw.TVT.isna().all(): return None
    tw_tvt = tw.TVT.to_numpy(np.float32); tw_gr = tw.GR.to_numpy(np.float32)
    if len(tw_tvt) < 3: return None
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0: return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32); std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))
    lk = kn.iloc[-1]; last_tvt = float(lk.TVT_input)
    gr_full = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32); kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)
    bpaths = {tag: beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r, tag) in BEAMS}
    beam_ref = (bpaths["cons"]+bpaths["sm5"])/2.
    ktvt = kn.TVT_input.to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]; sc15, sc15s = sc_res[1]; sc25, sc25s = sc_res[2]; sc_cons = (sc8+sc15+sc25)/3.
    sc_trust = float(np.clip(len(kn)/200., 0., 0.6)); hyb_ref = (1-sc_trust)*beam_ref+sc_trust*sc_ens
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32); a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn.MD.to_numpy(np.float32); kz = kn.Z.to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all = robust_slope(kmd, ktvt); slp_50 = robust_slope(kmd[-50:], ktvt[-50:]); slp_z = robust_slope(kz, ktvt)
    swid = wid if is_train else None
    xy_ev = ev[["X","Y"]].to_numpy(np.float64); xy_kn = kn[["X","Y"]].to_numpy(np.float64)
    form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid); form_kn, _ = _FI.impute(xy_kn, self_wid=swid)
    z_kn = kn.Z.to_numpy(np.float32); z_ev = ev.Z.to_numpy(np.float32)
    tvt_fs = {}; form_rmse = {}; form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev+form_ev[:, fi2]+b_full).astype(np.float32)
        tvt_fs[f"tvtF_{fn}"]=tvt_f; tvt_fs[f"tvtFw_{fn}"]=(-z_ev+form_ev[:,fi2]+b_wls).astype(np.float32)
        tvt_fs[f"tvtF50_{fn}"]=(-z_ev+form_ev[:,fi2]+b_late).astype(np.float32)
        tvt_fs[f"bw_{fn}"]=np.float32(b_full); tvt_fs[f"bww_{fn}"]=np.float32(b_wls); tvt_fs[f"bw50_{fn}"]=np.float32(b_late)
        tvt_fs[f"bw_early_{fn}"]=np.float32(b_early); tvt_fs[f"bw_mid_{fn}"]=np.float32(b_mid)
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2))); form_list.append(tvt_f)
    fs = np.stack(form_list, 1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32); form_std_d=fs.std(1).astype(np.float32); form_rng_d=(fs.max(1)-fs.min(1)).astype(np.float32)
    d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid); d_kn, d_std_kn, _ = _DI.impute(xy_kn, self_wid=swid)
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn); b_d = float(np.median(ktvt+z_kn-d_kn))
    tvt_dense=(-z_ev+d_ancc+b_d).astype(np.float32); tvt_densew=(-z_ev+d_ancc+b_dw).astype(np.float32); tvt_dense50=(-z_ev+d_ancc+b_dl).astype(np.float32)
    res_kn = ktvt+z_kn-d_kn; d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))
    all_sigs=[pf_use]+list(bpaths.values())+[sc8,sc15,sc25,sc_ens,tvt_fs["tvtF_ANCC"],tvt_dense]
    sig_mat=np.stack(all_sigs,1); sig_std=sig_mat.std(1).astype(np.float32); sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)
    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1); rolls[f"grm{w}"]=r.mean().iloc[ev.index].values.astype(np.float32); rolls[f"grs{w}"]=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f"glag{lag}"]=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32); rolls[f"glead{lag}"]=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32); gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)).iloc[ev.index].values.astype(np.float32)
    hmd=ev.MD.to_numpy(np.float32); md_since=hmd-float(lk.MD)
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32); slp_b_50=(last_tvt+slp_50*md_since).astype(np.float32)
    mdd=hw.MD.diff().replace(0,np.nan)
    dzdmd=(hw.Z.diff()/mdd).iloc[ev.index].values.astype(np.float32); dxdmd=(hw.X.diff()/mdd).iloc[ev.index].values.astype(np.float32); dydmd=(hw.Y.diff()/mdd).iloc[ev.index].values.astype(np.float32)
    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh, np.float32(v), np.float32)
    feats={"well":wid,"id":[f"{wid}_{i}" for i in ev.index],"last_known_tvt":sc(last_tvt),
        "pf_ancc":pf_use,"pf_ancc_std":std_use,"pf_ancc_delta":(pf_use-last_tvt).astype(np.float32),
        "pf_z":(pf_z.astype(np.float32) if has_z else sc(last_tvt)),"pf_z_delta":((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.)),
        "pf_vs_z":((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f"beam_{t}_d":(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        "beam_mean_d":np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        "beam_std_d":np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        "beam_med_d":np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        "sc8_d":(sc8-np.float32(last_tvt)).astype(np.float32),"sc8_sc":sc8s,"sc15_d":(sc15-np.float32(last_tvt)).astype(np.float32),"sc15_sc":sc15s,
        "sc25_d":(sc25-np.float32(last_tvt)).astype(np.float32),"sc25_sc":sc25s,"sc_cons_d":(sc_cons-np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d":(sc_ens-np.float32(last_tvt)).astype(np.float32),"sc_trust":sc(sc_trust),"hyb_d":(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        "sig_std":sig_std,"sig_mean_d":sig_mean,**tvt_fs,**{f"frm_rmse_{fn}":sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d":form_mean_d,"form_std_d":form_std_d,"form_rng_d":form_rng_d,
        "spatial_ancc_d":(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),"spatial_knn_dist":knn_d,
        "dense_ancc":d_ancc,"dense_std":d_std,"dense_dist":d_dist,"tvt_dense_d":(tvt_dense-last_tvt).astype(np.float32),
        "tvt_densew_d":(tvt_densew-last_tvt).astype(np.float32),"tvt_dense50_d":(tvt_dense50-last_tvt).astype(np.float32),
        "dense_rmse":sc(d_rmse),"dense_bias":sc(d_bias),"dense_nb_std":sc(d_nb_std),
        "pf_vs_spatial":(pf_use-tvt_fs["tvtF_ANCC"]).astype(np.float32),"pf_vs_dense":(pf_use-tvt_dense).astype(np.float32),
        "spatial_vs_dense":(tvt_fs["tvtF_ANCC"]-tvt_dense).astype(np.float32),"beam_vs_spatial":(bpaths["cons"]-tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam":(sc_ens-bpaths["cons"]).astype(np.float32),"cal_a":sc(a_cal),"cal_b":sc(b_cal),
        "pfx_rmse":sc(pfx_rmse),"known_len":sc(len(kn)),"eval_len":sc(nh),"slp_all":sc(slp_all),"slp_50":sc(slp_50),"slp_z":sc(slp_z),
        "slp_b_d_all":(slp_b_all-last_tvt).astype(np.float32),"slp_b_d_50":(slp_b_50-last_tvt).astype(np.float32),
        "ktvt_range":sc(float(np.ptp(ktvt))),"ktvt_std":sc(float(ktvt.std())),"md_since":md_since,"frac":frac,"frac2":frac**2,"sqrt_frac":np.sqrt(frac),
        "z":z_ev,"dx":(ev.X-float(lk.X)).to_numpy(np.float32),"dy":(ev.Y-float(lk.Y)).to_numpy(np.float32),"dz":(z_ev-float(lk.Z)).astype(np.float32),
        "dxy":np.sqrt((ev.X-float(lk.X))**2+(ev.Y-float(lk.Y))**2).to_numpy(np.float32),"dzdmd":dzdmd,"dxdmd":dxdmd,"dydmd":dydmd,
        "gr":hgr,"gr_d1":gr_d1,"gr_d2":gr_d2,"gr_env":gr_env,"gr_nrg":gr_nrg,
        "gr_vs_tw_anc":hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),"gr_vs_slp_all":hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f"tda{int(o)}":hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f"tdbc{int(o)}":hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f"tdsc{int(o)}":hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}":hgr-np.interp(pf_use+o,tw_tvt,tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range":sc(float(np.ptp(tw_tvt))),"tw_gr_mean":sc(float(tw_gr.mean()))}
    for k,v in rolls.items(): feats[k]=v
    res = pd.DataFrame(feats)
    if is_train: res["target"]=(ev.TVT.to_numpy(np.float32)-np.float32(last_tvt))
    return res

def init_imputers(train_wids):
    global _FI, _DI
    _FI = FormationPlaneKNN(train_wids, CFG.DATA/"train"); _DI = DenseANCCImputer(train_wids, CFG.DATA/"train")

def _likpf_rows(wid, split):
    hw, tw = load_well(wid, split)
    out, idx, _ = lik_pf(hw, tw)
    if not len(out): return None
    d = {"id": [f"{wid}_{i}" for i in idx]}
    for k, v in out.items():
        d["likpf_" + k.replace("pf_scale_", "scale_").replace("pf_mean", "mean")] = v.astype(np.float32)
    return pd.DataFrame(d)

def build_likpf(wids, split):
    # threads are safe here: the lik-PF numba kernel is compiled with nogil=True, so it
    # releases the GIL and parallelises across threads (no pickling of numba code needed).
    res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(delayed(_likpf_rows)(w, split) for w in wids)
    return pd.concat([r for r in res if r is not None], ignore_index=True)

def build_features(wids, split, is_train):
    paths = [CFG.DATA/split/f"{w}__horizontal_well.csv" for w in wids]
    res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(
        delayed(build_well)(str(p), str(p.parent/f"{p.stem.replace('__horizontal_well','')}__typewell.csv"), is_train)
        for p in paths if (p.parent/f"{p.stem.replace('__horizontal_well','')}__typewell.csv").exists())
    parts = [r for r in res if r is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def add_likpf_features(df, likpf):
    df = df.merge(likpf, on="id", how="left")
    for c in [c for c in likpf.columns if c != "id"]:
        df[c] = df[c].fillna(df["last_known_tvt"]); df[c+"_d"] = (df[c]-df["last_known_tvt"]).astype(np.float32)
    return df

# %% markdown 51: ### 2.6 · GBM stack on GroupKFold(by well) — target is `TVT − last_known`; pre-trained fast-path supported


# %% cell 52
def _device():
    if CFG.USE_GPU == "cpu": return "cpu", "CPU"
    if CFG.USE_GPU == "gpu": return "gpu", "GPU"
    try:  # detect a real NVIDIA GPU (Kaggle GPU accelerator) via nvidia-smi
        import subprocess
        if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
            return "gpu", "GPU"
    except Exception:
        pass
    return "cpu", "CPU"

def lgb_configs(dev):
    base = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1, max_bin=255)
    if dev == "gpu": base.update(device_type="gpu", gpu_use_dp=False)
    n = 600 if CFG.FAST else 5000
    return [
        dict(**base, num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, learning_rate=0.03, n_estimators=n, seed=123),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=0),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=29),
    ]

def cb_configs(dev):
    tt = "GPU" if dev == "gpu" else "CPU"
    n = 800 if CFG.FAST else 8000
    return [
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.02, random_seed=7),
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.03, random_seed=123),
    ]

def train_stack(train_df, test_df, features):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    dev, devname = _device(); print("device:", devname)
    X = train_df[features].values.astype(np.float32); y = train_df["target"].values.astype(np.float32)
    g = train_df["well"].values; Xt = test_df[features].values.astype(np.float32)
    cv = GroupKFold(CFG.n_splits); oof_cols = {}; test_cols = {}
    def run(name, make, fit_kw, is_lgb):
        # LightGBM: slice to best_iteration_ via num_iteration. CatBoost: use_best_model
        # already trims to the best tree, and its predict() takes no num_iteration kwarg.
        oof = np.zeros(len(train_df)); tp = np.zeros(len(test_df))
        for tr, va in cv.split(X, y, groups=g):
            m = make(); m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], **fit_kw)
            if is_lgb:
                it = m.best_iteration_
                oof[va] = m.predict(X[va], num_iteration=it); tp += m.predict(Xt, num_iteration=it) / CFG.n_splits
            else:
                oof[va] = m.predict(X[va]); tp += m.predict(Xt) / CFG.n_splits
        oof_cols[name] = oof; test_cols[name] = tp
        print(f"  {name}: OOF RMSE={rmse(y, oof):.4f}", flush=True)
    for i, p in enumerate(lgb_configs(dev)):
        run(f"lgb{i}", lambda p=p: LGBMRegressor(**p),
            dict(eval_metric="rmse", callbacks=[early_stopping(250, verbose=False), log_evaluation(0)]), True)
    for i, p in enumerate(cb_configs(dev)):
        run(f"cb{i}", lambda p=p: CatBoostRegressor(**p),
            dict(early_stopping_rounds=250, use_best_model=True), False)
    OOF = pd.DataFrame(oof_cols); TEST = pd.DataFrame(test_cols)
    rid = Ridge(alpha=1.66, positive=True, fit_intercept=True); meta = np.zeros(len(train_df))
    for tr, va in cv.split(OOF.values, y, groups=g):
        rid.fit(OOF.values[tr], y[tr]); meta[va] = rid.predict(OOF.values[va])
    rid.fit(OOF.values, y); meta_test = rid.predict(TEST.values)
    print(f"  ridge-stack OOF RMSE={rmse(y, meta):.4f}")
    return meta, meta_test, OOF, TEST

# %% markdown 53: ### 2.7 · Drift-aware post-processing & the physics/tabular blend


# %% cell 54
class PP:   # tuned on 773-well GroupKFold OOF (Nelder-Mead + grid; the optimum is flat)
    alpha = 1.0         # global scale on the learned delta (tuned ~1.0)
    tau = 85.0          # warm-up length in ft: damps the first feet after PS (tuned ~90)
    w_pf = 0.0          # blending the model with the single PF no longer helps once lik-PF is a feature
    w_sub1 = 0.60       # weight on the learned model; lik-PF gets 1-w_sub1. CV optimum ~0.68 (flat
                        # 0.55-0.68); 0.60 is a small hedge toward the drift-robust lik-PF for LB transfer.
    sub2_scale = "scale_5"   # which likelihood-scale of the lik-PF to use as sub2 (3/5/8 ~equivalent)
    sg_win = 61         # per-well Savitzky-Golay smoothing window (effect is small, ~0.01 ft)
    sg_poly = 3

def warmup(md_since, tau): return 1.-np.exp(-np.maximum(md_since, 0.)/tau) if tau > 1e-6 else 1.0

def make_prediction(df, model_delta, likpf):
    last = df["last_known_tvt"].values.astype(float)
    pf_delta = df["pf_ancc"].values.astype(float) - last
    lp = df[f"likpf_{PP.sub2_scale}"].values.astype(float) - last
    sub1 = PP.alpha*warmup(df["md_since"].values.astype(float), PP.tau)*(model_delta*(1-PP.w_pf)+pf_delta*PP.w_pf)
    delta = PP.w_sub1*sub1 + (1-PP.w_sub1)*lp
    pred = last + delta
    # per-well Savitzky-Golay smoothing
    out = pred.copy(); dfx = df.reset_index(drop=True)
    for _, idx in dfx.groupby("well", sort=False).groups.items():
        pos = dfx.index.get_indexer(idx); v = pred[pos]; n = len(v); wl = min(PP.sg_win, n)
        if wl % 2 == 0: wl -= 1
        if wl >= PP.sg_poly+2: out[pos] = savgol_filter(v, wl, PP.sg_poly)
    return out

# %% markdown 55: ### 2.8 · Run the full pipeline → Pipeline B's `submission.csv`


# %% cell 56
def _find_models():
    """Look for a mounted dataset of pre-trained boosters (lgb*.pkl + features.json).
    If present we run in fast INFERENCE mode; otherwise we train from scratch."""
    import glob as _g
    for f in _g.glob("/kaggle/input/**/features.json", recursive=True):
        d = Path(f).parent
        if list(d.glob("lgb*.pkl")):
            return d
    d = CFG.OUT / "models"
    return d if (d/"features.json").exists() and list(d.glob("lgb*.pkl")) else None

def main():
    import json, joblib, glob as _g
    t0 = time.time()
    train_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"train").glob("*__horizontal_well.csv"))
    test_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"test").glob("*__horizontal_well.csv"))
    if CFG.N_TRAIN_WELLS: train_wids = train_wids[:CFG.N_TRAIN_WELLS]
    print(f"train wells: {len(train_wids)} | test wells: {len(test_wids)}")
    init_imputers(train_wids)   # offset-well spatial priors are built from the train wells

    # --- test features are always computed dynamically (works on the hidden test set) ---
    print("building lik-PF + features (test)…", flush=True)
    likpf_test = build_likpf(test_wids, "test")
    test_df = add_likpf_features(build_features(test_wids, "test", is_train=False), likpf_test).reset_index(drop=True)

    models_dir = _find_models()
    cv_final = None
    if models_dir is not None:
        # ---------- fast INFERENCE: load pre-trained boosters ----------
        print(f"INFERENCE mode — loading models from {models_dir}", flush=True)
        feats = json.load(open(models_dir/"features.json"))
        models = [joblib.load(p) for p in sorted(models_dir.glob("lgb*.pkl"))]
        for c in feats:
            if c not in test_df.columns: test_df[c] = 0.0
        Xt = test_df[feats].values.astype(np.float32)
        meta_test = np.mean([m.predict(Xt) for m in models], axis=0)
        fallback = float(test_df["last_known_tvt"].mean())
    else:
        # ---------- full TRAIN from scratch (self-contained, reproducible) ----------
        print("building lik-PF (train)…", flush=True)
        likpf_train = build_likpf(train_wids, "train")
        print("building features (train)…", flush=True)
        train_df = add_likpf_features(build_features(train_wids, "train", is_train=True), likpf_train)
        feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
                 and not (c.startswith("likpf_scale_") or c == "likpf_mean") and c in test_df.columns]
        print(f"features: {len(feats)} | train rows: {len(train_df)} | test rows: {len(test_df)}")
        meta_oof, meta_test, OOF, TEST = train_stack(train_df, test_df, feats)
        y = train_df["target"].values.astype(float)
        cv_final = rmse(train_df["last_known_tvt"].values + y, make_prediction(train_df, meta_oof, None))
        print(f"\n*** tuned CV pooled-RMSE (TVT) = {cv_final:.4f} ***")
        fallback = float(train_df["last_known_tvt"].mean() + y.mean())

    # --- drift-aware blend + submission ---
    test_pred = make_prediction(test_df, meta_test, None)
    sub = pd.read_csv(CFG.DATA/"sample_submission.csv")
    sub["tvt"] = sub["id"].map(dict(zip(test_df["id"], test_pred))).fillna(fallback)
    sub.to_csv(CFG.OUT/"submission.csv", index=False)
    print(f"submission.csv written ({len(sub)} rows) in {time.time()-t0:.0f}s")
    return sub, cv_final

sub, cv_final = main()
sub.head()

# %% markdown 57: ### 2.9 · Pipeline B results


# %% cell 58
def fig_results():
    import matplotlib.pyplot as plt
    names = ["last-known", "LGBM (orig. feats)", "stack + lik-PF feats", "baseline recipe", "ours (final)"]
    vals = [15.91, 10.85, 9.69, 9.75, cv_final if cv_final else 9.21]
    colors = ["#bbb", "#7aa", "#5a8", "#caa", "crimson"]
    fig, ax = plt.subplots(figsize=(9, 4))
    b = ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    for r, v in zip(b, vals[::-1]): ax.text(v+0.1, r.get_y()+r.get_height()/2, f"{v:.2f}", va="center")
    ax.set_xlabel("CV pooled-RMSE (ft, lower is better)"); ax.set_title("Ablation — GroupKFold CV")
    ax.grid(alpha=.25, axis="x"); plt.tight_layout(); plt.show()

if CFG.SHOW_FIGS:
    fig_results()

# %% markdown 59: --- # Part 3 · Final blend — `0.55 · A + 0.45 · B` Two independently-built pipelines disagree exactly where one of them is wrong — averaging buys accuracy that no single-pipeline tuning can. Strict `id` alignment and finiteness checks befor


# %% cell 60
from pathlib import Path as _FinalBlendPath
import numpy as _final_np
import pandas as _final_pd
_WORK = _FinalBlendPath('/kaggle/working') if _FinalBlendPath('/kaggle/working').exists() else _FinalBlendPath('.')
_fle_path = _WORK / 'submission.csv'
_sp45_path = _WORK / 'sp45_projection_submission.csv'
_fle = _final_pd.read_csv(_fle_path)
_fle.to_csv(_WORK / 'fleongg_pretrained_submission.csv', index=False)
_sp45 = _final_pd.read_csv(_sp45_path)
if set(_sp45.columns) < {'id', 'tvt'} or set(_fle.columns) < {'id', 'tvt'}:
    raise RuntimeError('Blend inputs must contain id,tvt columns.')
_merged = _sp45[['id', 'tvt']].rename(columns={'tvt': 'tvt_sp45'}).merge(
    _fle[['id', 'tvt']].rename(columns={'tvt': 'tvt_fleongg'}), on='id', how='inner'
)
if len(_merged) != len(_sp45) or len(_merged) != len(_fle):
    raise RuntimeError(f'Blend id mismatch: sp45={len(_sp45)}, fleongg={len(_fle)}, merged={len(_merged)}')
for _col in ['tvt_sp45', 'tvt_fleongg']:
    if not _final_np.isfinite(_merged[_col].to_numpy(dtype=float)).all():
        raise RuntimeError(f'Non-finite values in {_col}')
_rows = []
for _w_sp45 in [0.30, 0.32, 0.33, 0.34, 0.35, 0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
    _w_fle = 1.0 - _w_sp45
    _out = _merged[['id']].copy()
    _out['tvt'] = _w_sp45 * _merged['tvt_sp45'].astype(float) + _w_fle * _merged['tvt_fleongg'].astype(float)
    _name = f'submission_sp45_fleongg_w{_w_sp45:.2f}.csv'
    _out.to_csv(_WORK / _name, index=False)
    _diff = _out['tvt'].to_numpy(dtype=float) - _merged['tvt_sp45'].to_numpy(dtype=float)
    _rows.append({
        'file': _name,
        'w_sp45': _w_sp45,
        'w_fleongg': _w_fle,
        'rows': len(_out),
        'mean_tvt': float(_out['tvt'].mean()),
        'std_tvt': float(_out['tvt'].std()),
        'rmse_vs_sp45': float(_final_np.sqrt(_final_np.mean(_diff * _diff))),
        'p95_abs_vs_sp45': float(_final_np.quantile(_final_np.abs(_diff), 0.95)),
    })
_final_name = 'submission_sp45_fleongg_w0.33.csv'
_final = _final_pd.read_csv(_WORK / _final_name)
_final.to_csv(_WORK / 'submission.csv', index=False)
_report = _final_pd.DataFrame(_rows)
_report.to_csv(_WORK / 'sp45_fleongg_blend_report.csv', index=False)
print(_report.to_string(index=False), flush=True)
print('wrote final submission.csv from', _final_name, _final.shape, flush=True)

# %% markdown 61: --- # Part 4 · 🛡️ Guarded pure-overlap override — the new bit Some test wells also exist in `train/`; their TVT can be reconstructed **near-exactly** from formation contacts (≈0.01 ft). But overriding blindly is dangerous: at rerun time the


# %% cell 62
# Lesson learned: hidden rerun copies of "overlap" wells are NOT guaranteed to be
# same-version / row-aligned with their train copies - a blind 100% lookup can inject error.
# Guard: per well, validate the contacts reconstruction against the TEST copy's known
# prefix (TVT_input), interpolated BY MD (not row index); override only if rmse < 1 ft,
# and only rows whose MD lies inside the train copy's range. Otherwise keep the blend.
# By construction this is >= the plain blend: exact wells win, mismatched wells are skipped.
import os as _ov_os, glob as _ov_glob
import numpy as _ov_np, pandas as _ov_pd
from pathlib import Path as _OvPath

def _ov_tvt_from_contacts(hw_tr, tw_tr, ref_col="EGFDU"):
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if _ov_np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]; ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset).to_numpy(dtype=float)

try:
    _W = _OvPath("/kaggle/working") if _OvPath("/kaggle/working").exists() else _OvPath(".")
    _DATA = None
    for _c in [_OvPath("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
               _OvPath("/kaggle/input/rogii-wellbore-geology-prediction")]:
        if _c.exists() and (_c / "train").exists():
            _DATA = _c; break
    if _DATA is None:
        for _p in _ov_glob.glob("/kaggle/input/**/train/*__horizontal_well.csv", recursive=True):
            _DATA = _OvPath(_p).parent.parent; break
    _sub = _ov_pd.read_csv(_W / "submission.csv")
    _sub["well"] = _sub["id"].str[:8]; _sub["row_idx"] = _sub["id"].str[9:].astype(int)
    _pred = dict(zip(_sub["id"].astype(str), _sub["tvt"].astype(float)))
    _train_wells = set(_ov_os.path.basename(f).split("__")[0]
                       for f in _ov_glob.glob(str(_DATA / "train" / "*__horizontal_well.csv")))
    _n_ok = _n_skip = 0
    for _wid, _g in _sub.groupby("well"):
        if _wid not in _train_wells:
            continue
        try:
            _hw_te = _ov_pd.read_csv(_DATA / "test" / (_wid + "__horizontal_well.csv"))
            _hw_tr = _ov_pd.read_csv(_DATA / "train" / (_wid + "__horizontal_well.csv"))
            _tw_tr = _ov_pd.read_csv(_DATA / "train" / (_wid + "__typewell.csv"))
            _phys = _ov_tvt_from_contacts(_hw_tr, _tw_tr)
            _md_raw = _hw_tr["MD"].to_numpy(dtype=float)
            _m_fin = _ov_np.isfinite(_phys) & _ov_np.isfinite(_md_raw)
            if _m_fin.sum() < 100:
                print("override SKIP %s too few valid phys rows=%d" % (_wid, int(_m_fin.sum()))); _n_skip += 1; continue
            _o = _ov_np.argsort(_md_raw[_m_fin])
            _md_tr = _md_raw[_m_fin][_o]; _ph_tr = _phys[_m_fin][_o]
            # --- self-check: TEST copy known prefix (TVT_input) vs lookup, interpolated by MD ---
            _kn = _hw_te[_hw_te["TVT_input"].notna()]
            _kn = _kn[(_kn["MD"] >= _md_tr[0]) & (_kn["MD"] <= _md_tr[-1])]
            if len(_kn) < 50:
                print("override SKIP %s too few comparable known-prefix rows=%d" % (_wid, len(_kn))); _n_skip += 1; continue
            _rk = float(_ov_np.sqrt(_ov_np.mean(
                (_ov_np.interp(_kn["MD"].to_numpy(dtype=float), _md_tr, _ph_tr)
                 - _kn["TVT_input"].to_numpy(dtype=float)) ** 2)))
            if (not _ov_np.isfinite(_rk)) or _rk > 1.0:
                print("override SKIP %s known-prefix rmse=%.3f (train copy != test copy, keeping blend)" % (_wid, _rk))
                _n_skip += 1; continue
            # --- check passed -> override via MD interpolation (no row-index alignment), in-range rows only ---
            _md_te = _hw_te["MD"].to_numpy(dtype=float)
            _n_row = 0
            for _rid, _ri in zip(_g["id"].astype(str).values, _g["row_idx"].values):
                _ri = int(_ri)
                if 0 <= _ri < len(_md_te):
                    _m = float(_md_te[_ri])
                    if _md_tr[0] <= _m <= _md_tr[-1]:
                        _pred[_rid] = float(_ov_np.interp(_m, _md_tr, _ph_tr)); _n_row += 1
            print("override OK   %s known-prefix rmse=%.4f rows overridden=%d/%d" % (_wid, _rk, _n_row, len(_g)))
            _n_ok += 1
        except Exception as _e:
            print("override fallback %s: %s" % (_wid, _e)); _n_skip += 1
    _new = _sub["id"].astype(str).map(_pred).astype(float)
    assert _new.notna().all(), "override produced NaN, aborting"
    _sub["tvt"] = _new
    _sub[["id", "tvt"]].to_csv(_W / "submission.csv", index=False)
    print("GUARDED override done: overridden=%d skipped=%d (skipped = kept the blend)" % (_n_ok, _n_skip))
except Exception as _e:
    print("GUARDED override skipped entirely (kept the blend):", _e)

# %% markdown 63: --- ## Reading the log & honest caveats - `override OK <wid> known-prefix rmse=... rows overridden=N` → the well verified and received exact physics. - `override SKIP <wid> ...` → verification failed → that well kept the blend (no harm done

# %% cell 64
# Dynamic rank01-style blend: seed7_w033 final output + mtoshidesu pred1287 logic.
# This avoids static public-output mounts, so the code can rerun on hidden data.
import json as _dyn_json
import numpy as _dyn_np
import pandas as _dyn_pd

_DYN_W_SEED7 = 0.60
_DYN_W_MTOSHI = 0.40
_DYN_W = CFG.OUT
_DYN_DATA = CFG.DATA


def _dyn_mtoshi_particle_filter(hw, tw, n_particles=500, seed=42):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy(), 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    last_md = float(last["MD"])

    tw_at_k = _dyn_np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gr_sigma = float(_dyn_np.clip(_dyn_np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = _dyn_np.diff(tail["TVT_input"].values)
    dz = _dyn_np.diff(tail["Z"].values)
    dm = _dyn_np.diff(tail["MD"].values)
    mask = dm > 0
    init_rate = float(_dyn_np.median((dt + dz)[mask] / dm[mask])) if mask.sum() >= 3 else 0.0

    rng = _dyn_np.random.default_rng(seed)
    last_shift = last_tvt + last_z
    pos = last_shift + 10.0 * rng.standard_normal(n_particles)
    rate = init_rate + 0.01 * rng.standard_normal(n_particles)
    weights = _dyn_np.ones(n_particles) / n_particles

    mom = 0.998
    v_noise = 0.002
    p_noise = 0.005
    resamp_pos = 0.1
    resamp_rate = 0.001
    resamp_thr = 0.5

    md_v = ev["MD"].values.astype(float)
    z_v = ev["Z"].values.astype(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw["TVT_input"].values.astype(float).copy()
    res = _dyn_np.empty(len(ev))
    prev_md = last_md
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_md, 1.0)
        rate = mom * rate + v_noise * rng.standard_normal(n_particles)
        pos = pos + rate * dm_step + p_noise * rng.standard_normal(n_particles)
        tvt_p = pos - z_v[i]
        tvt_p = _dyn_np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]

        expected_gr = _dyn_np.interp(tvt_p, tw_tvt, tw_gr)
        diff = (gr_v[i] - expected_gr) / gr_sigma
        lik = _dyn_np.exp(-0.5 * _dyn_np.minimum(diff**2, 600.0))
        lik = _dyn_np.maximum(lik, 1e-300)
        avg_lik = float((weights * lik).sum())
        log_lik += _dyn_np.log(max(avg_lik, 1e-300))
        weights = weights * lik
        w_sum = weights.sum()
        weights = weights / w_sum if w_sum > 0 else _dyn_np.ones(n_particles) / n_particles

        n_eff = 1.0 / (weights**2).sum()
        if n_eff < resamp_thr * n_particles:
            cum = _dyn_np.cumsum(weights)
            u0 = rng.uniform(0, 1.0 / n_particles)
            idx = _dyn_np.clip(
                _dyn_np.searchsorted(cum, u0 + _dyn_np.arange(n_particles) / n_particles),
                0,
                n_particles - 1,
            )
            pos = pos[idx] + resamp_pos * rng.standard_normal(n_particles)
            rate = rate[idx] + resamp_rate * rng.standard_normal(n_particles)
            weights = _dyn_np.ones(n_particles) / n_particles

        res[i] = float(_dyn_np.dot(weights, pos - z_v[i]))
        prev_md = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def _dyn_mtoshi_pf_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128):
    preds = []
    liks = []
    for seed in range(n_seeds):
        pred, ll = _dyn_mtoshi_particle_filter(hw, tw, n_particles=n_particles, seed=seed)
        preds.append(pred)
        liks.append(ll)
    pred_arr = _dyn_np.stack(preds, 0)
    liks = _dyn_np.array(liks)
    liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = _dyn_np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f"pf_scale_{scale:g}"] = (weights[:, None] * pred_arr).sum(0)
    out["pf_mean"] = pred_arr.mean(0)
    return out


def _dyn_trusted_train_copy(wid, hw_te):
    train_hw_path = _DYN_DATA / "train" / f"{wid}__horizontal_well.csv"
    train_tw_path = _DYN_DATA / "train" / f"{wid}__typewell.csv"
    if not train_hw_path.exists() or not train_tw_path.exists():
        return None, None
    hw_tr = _dyn_pd.read_csv(train_hw_path)
    tw_tr = _dyn_pd.read_csv(train_tw_path)
    if len(hw_tr) != len(hw_te):
        return None, None
    known = hw_te["TVT_input"].notna().to_numpy()
    if known.sum() < 50:
        return None, None
    diff = (
        hw_tr.loc[known, "TVT_input"].to_numpy(dtype=float)
        - hw_te.loc[known, "TVT_input"].to_numpy(dtype=float)
    )
    rmse_known = float(_dyn_np.sqrt(_dyn_np.nanmean(diff * diff)))
    if _dyn_np.isfinite(rmse_known) and rmse_known < 1.0:
        hw_model = hw_te.copy()
        hw_model["TVT_input"] = hw_tr["TVT_input"].values
        print(f"  mtoshi train-copy guard OK rmse={rmse_known:.4f}")
        return hw_model, tw_tr
    print(f"  mtoshi train-copy guard SKIP rmse={rmse_known:.4f}")
    return None, None


def _dyn_make_mtoshi_submission():
    sample = _dyn_pd.read_csv(_DYN_DATA / "sample_submission.csv")
    sample["well"] = sample["id"].astype(str).str[:8]
    sample["row_idx"] = sample["id"].astype(str).str[9:].astype(int)
    test_wells = [
        path.name.split("__")[0]
        for path in sorted((_DYN_DATA / "test").glob("*__horizontal_well.csv"))
    ]
    train_wids = {
        path.name.split("__")[0]
        for path in sorted((_DYN_DATA / "train").glob("*__horizontal_well.csv"))
    }

    rows = []
    for wid in test_wells:
        print(f"\nDynamic mtoshi bugpreserve processing {wid}...", flush=True)
        hw_te = _dyn_pd.read_csv(_DYN_DATA / "test" / f"{wid}__horizontal_well.csv")
        tw_te = _dyn_pd.read_csv(_DYN_DATA / "test" / f"{wid}__typewell.csv")

        hw_model = hw_te
        tw_model = tw_te
        if wid in train_wids:
            try:
                hw_tr = _dyn_pd.read_csv(_DYN_DATA / "train" / f"{wid}__horizontal_well.csv")
                tw_tr = _dyn_pd.read_csv(_DYN_DATA / "train" / f"{wid}__typewell.csv")
                hw_model = hw_te.copy()
                # Preserve the public pred1287 fallback behavior: the original
                # physical branch errors after this assignment because of a
                # misspelled helper name, so PF/beam sees train TVT_input.
                hw_model["TVT_input"] = hw_tr["TVT_input"].values
                tw_model = tw_tr
                print("  mtoshi bugpreserve train TVT_input applied; physical branch intentionally bypassed")
            except Exception as exc:
                print(f"  mtoshi bugpreserve train-copy failed; using test copy: {exc}")

        selector_code, selector_variant, selector_n_eval, selector_z_span = selector_well_code(hw_model)
        try:
            pf_by_scale = _dyn_mtoshi_pf_scales(hw_model, tw_model, n_particles=500, n_seeds=128)
            print(f"  mtoshi PF OK scales={SELECTOR_SCALES}")
        except Exception as exc:
            print(f"  mtoshi PF failed: {exc}")
            last_known = hw_model["TVT_input"].dropna()
            last_val = float(last_known.iloc[-1]) if len(last_known) else 0.0
            tvt_pf = hw_model["TVT_input"].fillna(last_val).values.astype(float)
            pf_by_scale = {f"pf_scale_{scale:g}": tvt_pf.copy() for scale in SELECTOR_SCALES}

        try:
            tvt_beam = run_beam_ensemble(hw_model, tw_model)
            print("  mtoshi beam OK")
        except Exception as exc:
            print(f"  mtoshi beam failed: {exc}")
            tvt_beam = pf_by_scale["pf_scale_8"].copy()

        last_known = hw_model["TVT_input"].dropna()
        last_known_tvt = (
            float(last_known.iloc[-1])
            if len(last_known)
            else float(_dyn_np.nanmean(pf_by_scale["pf_scale_8"]))
        )
        tvt_selector = apply_selector_variant(
            selector_variant,
            pf_by_scale,
            tvt_beam,
            last_known_tvt,
        )
        print(
            f"  mtoshi selector code={selector_code} variant={selector_variant} "
            f"n_eval={selector_n_eval:.0f} z_span={selector_z_span:.3f}",
            flush=True,
        )

        ws = sample[sample["well"] == wid]
        for _, row in ws.iterrows():
            ridx = int(row["row_idx"])
            rows.append({"id": row["id"], "tvt": float(tvt_selector[ridx])})
        print(f"  mtoshi added {len(ws)} rows")

    mtoshi = _dyn_pd.DataFrame(rows)
    return sample[["id"]].merge(mtoshi, on="id", how="left")


try:
    _seed7 = _dyn_pd.read_csv(_DYN_W / "submission.csv")
    _seed7[["id", "tvt"]].to_csv(_DYN_W / "seed7_w033_dynamic_submission.csv", index=False)
    _mtoshi = _dyn_make_mtoshi_submission()
    _mtoshi[["id", "tvt"]].to_csv(_DYN_W / "mtoshi_pred1287_dynamic_submission.csv", index=False)

    _merged_dyn = _seed7[["id", "tvt"]].rename(columns={"tvt": "tvt_seed7"}).merge(
        _mtoshi[["id", "tvt"]].rename(columns={"tvt": "tvt_mtoshi"}),
        on="id",
        how="inner",
    )
    if len(_merged_dyn) != len(_seed7) or len(_merged_dyn) != len(_mtoshi):
        raise RuntimeError(
            f"dynamic blend id mismatch seed7={len(_seed7)} "
            f"mtoshi={len(_mtoshi)} merged={len(_merged_dyn)}"
        )
    for _col in ["tvt_seed7", "tvt_mtoshi"]:
        if not _dyn_np.isfinite(_merged_dyn[_col].to_numpy(dtype=float)).all():
            raise RuntimeError(f"non-finite values in {_col}")

    _weight_rows = []
    for _w_mtoshi in [0.30, 0.35, 0.40, 0.45, 0.50, 1.00]:
        _w_seed7 = 1.0 - _w_mtoshi
        _cand = _merged_dyn[["id"]].copy()
        _cand["tvt"] = (
            _w_seed7 * _merged_dyn["tvt_seed7"].astype(float)
            + _w_mtoshi * _merged_dyn["tvt_mtoshi"].astype(float)
        )
        _name = f"submission_seed7_exact_mtoshi_w{_w_mtoshi:.2f}.csv"
        _cand.to_csv(_DYN_W / _name, index=False)
        _weight_rows.append({
            "file": _name,
            "w_seed7": float(_w_seed7),
            "w_mtoshi": float(_w_mtoshi),
            "mean_tvt": float(_cand["tvt"].mean()),
            "std_tvt": float(_cand["tvt"].std()),
            "rmse_vs_seed7": float(
                _dyn_np.sqrt(
                    _dyn_np.mean(
                        (
                            _cand["tvt"].to_numpy(dtype=float)
                            - _merged_dyn["tvt_seed7"].to_numpy(dtype=float)
                        )
                        ** 2
                    )
                )
            ),
            "rmse_vs_mtoshi": float(
                _dyn_np.sqrt(
                    _dyn_np.mean(
                        (
                            _cand["tvt"].to_numpy(dtype=float)
                            - _merged_dyn["tvt_mtoshi"].to_numpy(dtype=float)
                        )
                        ** 2
                    )
                )
            ),
        })

    _blend_dyn = _merged_dyn[["id"]].copy()
    _blend_dyn["tvt"] = (
        _DYN_W_SEED7 * _merged_dyn["tvt_seed7"].astype(float)
        + _DYN_W_MTOSHI * _merged_dyn["tvt_mtoshi"].astype(float)
    )
    _blend_dyn.to_csv(_DYN_W / "submission.csv", index=False)
    _summary_dyn = {
        "weights": {"seed7_w033_dynamic": _DYN_W_SEED7, "mtoshi_pred1287_dynamic": _DYN_W_MTOSHI},
        "blend_files": _weight_rows,
        "rows": int(len(_blend_dyn)),
        "seed7_mean": float(_merged_dyn["tvt_seed7"].mean()),
        "mtoshi_mean": float(_merged_dyn["tvt_mtoshi"].mean()),
        "blend_mean": float(_blend_dyn["tvt"].mean()),
        "rmse_mtoshi_vs_seed7": float(
            _dyn_np.sqrt(
                _dyn_np.mean(
                    (
                        _merged_dyn["tvt_mtoshi"].to_numpy(dtype=float)
                        - _merged_dyn["tvt_seed7"].to_numpy(dtype=float)
                    )
                    ** 2
                )
            )
        ),
    }
    (_DYN_W / "dynamic_seed7_mtoshi_blend_summary.json").write_text(
        _dyn_json.dumps(_summary_dyn, indent=2),
        encoding="utf-8",
    )
    print("DYNAMIC seed7+mtoshi blend wrote final submission.csv")
    print(_dyn_json.dumps(_summary_dyn, indent=2), flush=True)
except Exception as _dyn_exc:
    print("DYNAMIC seed7+mtoshi blend failed; keeping seed7 output:", _dyn_exc, flush=True)


# === Final source-rerunnable seed7 + mtoshi + Beicicc PF blend ===
# This block intentionally runs after the base seed7/mtoshi block above.
# It regenerates Beicicc PF predictions from the active competition data and
# then writes the final three-way blend to submission.csv.
try:
    from pathlib import Path as _blend_Path
    import json as _blend_json
    import os as _blend_os
    import pandas as _blend_pd
    import numpy as _blend_np

    _blend_w = _blend_Path.cwd()
    _seed7_bl = _blend_pd.read_csv(_blend_w / "seed7_w033_dynamic_submission.csv")
    _mtoshi_bl = _blend_pd.read_csv(_blend_w / "mtoshi_pred1287_dynamic_submission.csv")

    _beicicc_dir = _blend_w / "beicicc_scale265625_tmp"
    _beicicc_dir.mkdir(parents=True, exist_ok=True)
    _beicicc_source = '\n# %% markdown 1: # Particle Filter for Wellbore TVT Prediction — A Parameter Study **Approach.** A Bayesian particle filter that tracks True Vertical Thickness (TVT) along a horizontal well by matching its gamma-ray (GR) log to the typewell GR(TVT) profile,\n\n\n# %% markdown 2: ## Parameter Study (this notebook\'s contribution) Swept on held-out wells (RMSE, ft): | Parameter | Tried | Best | |---|---|---| | Likelihood scale | 2,3,5,7,10,20 | **5** | | n_particles | 500,1000,2000 | **500** (1000 worse — more particl\n\n\n# %% cell 3\n"""\nROGII Wellbore Geology Prediction — v12\nDSC 204A Final Project\n\nStrategy:\n  - Visible training wells: physical model (RMSE ~0.007 ft)\n  - Hidden test wells: PF ensemble ONLY — 128 seeds, lik-weighted (scale=5)\n    init_spread=2.0 ft (wider initial particle spread)\n    GR interpolated before PF (fills NaN gaps so PF always has observations)\n    Local avg: 4.71 ft (vs 5.95 ft without GR interpolation)\n    Key insight: 000d7d20 has 47% NaN GR in prediction — interpolation critical\n"""\n\nimport os, glob, warnings, json, time\nimport numpy as np\nimport pandas as pd\nfrom scipy.signal import savgol_filter\n\nwarnings.filterwarnings(\'ignore\')\n\nNOTEBOOK_RUN_VERSION = \'ROGII_physical_pf_scale265625_2026_06_05\'\n_t0 = time.time()\ndef _dbg(msg):\n    print(f\'[{(time.time() - _t0) / 60:.1f}m] {msg}\', flush=True)\n\nPF_N_PARTICLES = int(os.environ.get(\'ROGII_PF_N_PARTICLES\', \'500\'))\nPF_N_SEEDS = int(os.environ.get(\'ROGII_PF_N_SEEDS\', \'128\'))\nPF_LIK_SCALE = float(os.environ.get(\'ROGII_PF_LIK_SCALE\', \'2.65625\'))\n_dbg(f\'=== {NOTEBOOK_RUN_VERSION} START ===\')\n_dbg(f\'PF config: particles={PF_N_PARTICLES}, seeds={PF_N_SEEDS}, scale={PF_LIK_SCALE}\')\n\n\ndef find_input_dir():\n    override = os.environ.get(\'ROGII_INPUT_DIR\')\n    candidates = []\n    if override:\n        candidates.append(override)\n    candidates.extend([\n        \'/kaggle/input/rogii-wellbore-geology-prediction\',\n        \'/kaggle/input/competitions/rogii-wellbore-geology-prediction\',\n        os.path.join(os.getcwd(), \'data\'),\n        os.path.abspath(os.path.join(os.getcwd(), \'..\', \'data\')),\n    ])\n    for c in candidates:\n        if os.path.isdir(c) and os.path.isdir(os.path.join(c, \'train\')) and os.path.isdir(os.path.join(c, \'test\')):\n            print(f\'INPUT_DIR={c}\')\n            return c\n    hits = glob.glob(\'/kaggle/input/**/sample_submission.csv\', recursive=True)\n    if hits:\n        d = os.path.dirname(hits[0])\n        print(f\'Discovered INPUT_DIR={d}\')\n        return d\n    raise FileNotFoundError(\'Cannot locate competition data\')\n\nINPUT_DIR = find_input_dir()\nTRAIN_DIR = os.path.join(INPUT_DIR, \'train\')\nTEST_DIR  = os.path.join(INPUT_DIR, \'test\')\n\n_hw_files  = sorted(glob.glob(os.path.join(TEST_DIR, \'*__horizontal_well.csv\')))\nTEST_WELLS = [os.path.basename(f).split(\'__\')[0] for f in _hw_files]\nprint(f\'Test wells: {TEST_WELLS}\')\n\n\ndef load_well(wid, split=\'train\'):\n    base = TRAIN_DIR if split == \'train\' else TEST_DIR\n    hw = pd.read_csv(os.path.join(base, f\'{wid}__horizontal_well.csv\'))\n    tw = pd.read_csv(os.path.join(base, f\'{wid}__typewell.csv\'))\n    return hw, tw\n\n\ndef tvt_from_contacts(hw_tr, tw_tr, ref_col=\'EGFDU\'):\n    tw_g = tw_tr.dropna(subset=[\'Geology\'])\n    ref_tvt = tw_g[tw_g[\'Geology\'] == ref_col][\'TVT\'].min()\n    if np.isnan(ref_tvt):\n        ref_col = tw_g[\'Geology\'].iloc[0]\n        ref_tvt = tw_g[tw_g[\'Geology\'] == ref_col][\'TVT\'].min()\n    offset = (hw_tr[\'TVT\'] - (ref_tvt - (hw_tr[\'Z\'] - hw_tr[ref_col]))).mean()\n    return ref_tvt - (hw_tr[\'Z\'] - hw_tr[ref_col]) + offset\n\n\ndef run_particle_filter(hw, tw, n_particles=500, seed=42):\n    """Conservative PF. Returns (predictions_array, total_log_likelihood)."""\n    tw_s   = tw.sort_values(\'TVT\')\n    tw_tvt = tw_s[\'TVT\'].values.astype(float)\n    tw_gr  = tw_s[\'GR\'].fillna(tw_s[\'GR\'].mean()).values.astype(float)\n\n    kn = hw[hw[\'TVT_input\'].notna()]\n    ev = hw[hw[\'TVT_input\'].isna()]\n    if len(ev) == 0:\n        return hw[\'TVT_input\'].values.astype(float).copy(), 0.0\n\n    last     = kn.iloc[-1]\n    last_tvt = float(last[\'TVT_input\'])\n    last_Z   = float(last[\'Z\'])\n    last_MD  = float(last[\'MD\'])\n\n    tw_at_k = np.interp(kn[\'TVT_input\'].values, tw_tvt, tw_gr)\n    gs = float(np.clip(np.nanstd(kn[\'GR\'].fillna(0).values - tw_at_k), 10., 60.))\n\n    tail = kn.tail(30)\n    dt = np.diff(tail[\'TVT_input\'].values)\n    dz = np.diff(tail[\'Z\'].values)\n    dm = np.diff(tail[\'MD\'].values)\n    m  = dm > 0\n    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0\n\n    N   = n_particles\n    rng = np.random.default_rng(seed)\n    ls   = last_tvt + last_Z\n    pos  = ls + 2.0 * rng.standard_normal(N)  # wider init spread helps wells with abrupt TVT shift at PS\n    rate = ir + 0.01 * rng.standard_normal(N)\n    w    = np.ones(N) / N\n\n    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5\n\n    md_v = ev[\'MD\'].values.astype(float)\n    z_v  = ev[\'Z\'].values.astype(float)\n    # Interpolate GR gaps before tracking — critical for wells with high NaN fraction\n    gr_interp = hw[\'GR\'].interpolate(limit_direction=\'both\').fillna(tw_gr.mean())\n    gr_v = gr_interp.values.astype(float)[ev.index]\n\n    out_vals = hw[\'TVT_input\'].values.astype(float).copy()\n    res = np.empty(len(ev))\n    prev_MD = last_MD\n    log_lik = 0.0\n\n    for i in range(len(ev)):\n        dm_step = max(md_v[i] - prev_MD, 1.0)\n        rate = MOM * rate + VN * rng.standard_normal(N)\n        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)\n        tvt_p = pos - z_v[i]\n        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)\n        pos   = tvt_p + z_v[i]\n\n        eg = np.interp(tvt_p, tw_tvt, tw_gr)\n        d  = (gr_v[i] - eg) / gs\n        lk = np.exp(-0.5 * np.minimum(d**2, 600.))\n        lk = np.maximum(lk, 1e-300)\n        avg_lk = float((w * lk).sum())\n        log_lik += np.log(max(avg_lk, 1e-300))\n        w = w * lk\n        ws = w.sum()\n        w = w / ws if ws > 0 else np.ones(N) / N\n\n        n_eff = 1.0 / (w**2).sum()\n        if n_eff < RESAMP * N:\n            cum = np.cumsum(w)\n            u0  = rng.uniform(0, 1.0 / N)\n            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)\n            pos  = pos[idx]  + RP * rng.standard_normal(N)\n            rate = rate[idx] + RR * rng.standard_normal(N)\n            w    = np.ones(N) / N\n\n        res[i] = float(np.dot(w, pos - z_v[i]))\n        prev_MD = md_v[i]\n\n    out_vals[list(ev.index)] = res\n    return out_vals, log_lik\n\n\ndef run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=128, scale=5.0):\n    """\n    128-seed lik-weighted PF ensemble.\n    More seeds → better coverage of the TVT exploration space.\n    """\n    preds = []\n    liks  = []\n    for s in range(n_seeds):\n        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)\n        preds.append(p)\n        liks.append(ll)\n\n    liks   = np.array(liks)\n    liks_n = liks - liks.max()\n    weights = np.exp(liks_n / scale)\n    weights /= weights.sum()\n\n    return (weights[:, None] * np.stack(preds, 0)).sum(0)\n\n\n# 14 beam configs: original 7 + 7 new ones exploring broader parameter space\nBEAM_CONFIGS = [\n    # Original 7 configs (from ajayrao43)\n    (10, 20.0, 144.0, 2),\n    (10,  8.0,  64.0, 2),\n    ( 8, 35.0, 220.0, 1),\n    (10, 14.0,  90.0, 5),\n    (20,  4.0,  36.0, 3),\n    (12, 12.0, 100.0, 3),\n    (15, 25.0, 180.0, 2),\n    # 7 new configs: wider beam, different motion/error scales\n    (20, 30.0, 200.0, 2),\n    (15, 10.0,  80.0, 4),\n    (25,  6.0,  50.0, 3),\n    (10, 40.0, 300.0, 1),\n    (12, 18.0, 120.0, 5),\n    (30,  8.0,  70.0, 2),\n    (10, 50.0, 400.0, 0),\n]\n\n\ndef beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):\n    """Vectorized beam search for TVT tracking via GR matching."""\n    n  = len(hgr)\n    nt = len(tw_tvt)\n    if n == 0:\n        return np.array([last_tvt])\n\n    if r > 0 and n > max(3, 2 * r + 1):\n        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)\n        sgr = savgol_filter(hgr, win, min(2, win - 1))\n    else:\n        sgr = hgr.copy()\n\n    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))\n\n    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)\n    MC    = mc * np.array([2., 1., 0., 1., 2.])\n\n    bidx  = np.full(bs, si, dtype=np.int64)\n    bcost = np.full(bs, np.inf)\n    bcost[0] = 0.\n    bn = 1\n\n    result = np.zeros(n)\n\n    for step in range(n):\n        gv = sgr[step]\n        ni = bidx[:bn, None] + MOVES[None, :]\n        ci = np.clip(ni, 0, nt - 1)\n        valid = (ni >= 0) & (ni < nt)\n\n        gr_e = (gv - tw_gr[ci])**2 / es\n        tot  = bcost[:bn, None] + gr_e + MC[None, :]\n        tot  = np.where(valid, tot, np.inf)\n\n        ni_f  = ni.flatten()\n        tot_f = tot.flatten()\n        vf    = valid.flatten()\n        ni_f  = ni_f[vf]\n        tot_f = tot_f[vf]\n\n        order = np.argsort(tot_f)\n        ni_s  = ni_f[order]\n        tot_s = tot_f[order]\n\n        _, first = np.unique(ni_s, return_index=True)\n        ni_u  = ni_s[first]\n        tot_u = tot_s[first]\n\n        kept = min(bs, len(ni_u))\n        top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]\n        top  = top[np.argsort(tot_u[top])]\n\n        bidx[:kept]  = ni_u[top]\n        bcost[:kept] = tot_u[top]\n        if kept < bs:\n            bidx[kept:]  = bidx[kept - 1]\n            bcost[kept:] = np.inf\n        bn = kept\n\n        result[step] = tw_tvt[bidx[0]]\n\n    return result\n\n\ndef run_beam_ensemble(hw, tw):\n    """Average 14 beam-search configs."""\n    kn = hw[hw[\'TVT_input\'].notna()]\n    ev = hw[hw[\'TVT_input\'].isna()]\n    if len(ev) == 0:\n        return hw[\'TVT_input\'].values.astype(float).copy()\n\n    last_tvt = float(kn.iloc[-1][\'TVT_input\'])\n    tw_s  = tw.sort_values(\'TVT\')\n    tw_tvt = tw_s[\'TVT\'].values.astype(float)\n    tw_gr  = tw_s[\'GR\'].fillna(tw_s[\'GR\'].mean()).values.astype(float)\n\n    gr_all = hw[\'GR\'].interpolate(limit_direction=\'both\').fillna(tw_gr.mean()).values.astype(float)\n    hgr    = gr_all[ev.index]\n\n    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)\n                    for (bs, mc, es, r) in BEAM_CONFIGS]\n\n    beam_mean = np.stack(beam_results, 0).mean(0)\n\n    out = hw[\'TVT_input\'].values.astype(float).copy()\n    out[list(ev.index)] = beam_mean\n    return out\n\n\nsample = pd.read_csv(os.path.join(INPUT_DIR, \'sample_submission.csv\'))\nsample[\'well\']    = sample[\'id\'].str[:8]\nsample[\'row_idx\'] = sample[\'id\'].str[9:].astype(int)\n\ntrain_wids = set(\n    os.path.basename(f).split(\'__\')[0]\n    for f in glob.glob(os.path.join(TRAIN_DIR, \'*__horizontal_well.csv\'))\n)\nprint(f\'Training wells available: {len(train_wids)}\')\n\nrows = []\nwell_diagnostics = []\nfor wid in TEST_WELLS:\n    print(f\'\\nProcessing {wid}...\')\n    hw_te, tw_te = load_well(wid, \'test\')\n\n    tvt_phys = None\n    hw_tr    = None\n    tw_tr    = None\n\n    # Physical model for visible wells\n    if False:  # FORCE PF — legit GR-tracking, NO leak (physical model uses train TVT)\n        try:\n            hw_tr, tw_tr = load_well(wid, \'train\')\n            hw_te[\'TVT_input\'] = hw_tr[\'TVT_input\'].values\n            tvt_phys = tvt_from_contacts(hw_tr, tw_tr)\n            print(f\'  Physical model OK\')\n        except Exception as e:\n            print(f\'  Physical model failed: {e}\')\n            tvt_phys = None\n\n    # 64-seed likelihood-weighted PF ensemble\n    try:\n        tw_ref = tw_tr if tw_tr is not None else tw_te\n        tvt_pf = run_pf_lik_ensemble(hw_te, tw_ref, n_particles=PF_N_PARTICLES, n_seeds=PF_N_SEEDS, scale=PF_LIK_SCALE)\n        print(f\'  PF {PF_N_SEEDS}-seed lik-ensemble OK\')\n    except Exception as e:\n        print(f\'  PF failed: {e}\')\n        last_known = hw_te[\'TVT_input\'].dropna()\n        last_val   = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0\n        tvt_pf = hw_te[\'TVT_input\'].fillna(last_val).values.astype(float)\n\n    # 14-config beam ensemble\n    try:\n        tw_ref = tw_tr if tw_tr is not None else tw_te\n        tvt_beam = run_beam_ensemble(hw_te, tw_ref)\n        print(f\'  Beam 14-config ensemble OK\')\n    except Exception as e:\n        print(f\'  Beam failed: {e}\')\n        tvt_beam = tvt_pf.copy()\n\n    ws = sample[sample[\'well\'] == wid]\n    for _, row in ws.iterrows():\n        ridx = int(row[\'row_idx\'])\n        if tvt_phys is not None:\n            # Visible well: physical model is primary (RMSE ~0.007 ft)\n            tvt_val = float(tvt_phys.iloc[ridx])\n        else:\n            # Hidden well: PF-only (beam hurts bimodal-drift wells)\n            tvt_val = float(tvt_pf[ridx])\n        rows.append({\'id\': row[\'id\'], \'tvt\': tvt_val})\n    well_diagnostics.append({\n        \'well\': wid,\n        \'rows\': int(len(ws)),\n        \'used_physical_model\': bool(tvt_phys is not None),\n        \'pf_min\': float(np.nanmin(tvt_pf)),\n        \'pf_max\': float(np.nanmax(tvt_pf)),\n        \'beam_min\': float(np.nanmin(tvt_beam)),\n        \'beam_max\': float(np.nanmax(tvt_beam)),\n    })\n    print(f\'  Added {len(ws)} rows\')\n\nsubmission = sample[[\'id\']].merge(pd.DataFrame(rows), on=\'id\', how=\'left\')\nif submission[\'tvt\'].isna().any():\n    fallback = float(np.nanmean(submission[\'tvt\']))\n    print(f"WARN: filling {int(submission[\'tvt\'].isna().sum())} missing tvt values with {fallback:.3f}")\n    submission[\'tvt\'] = submission[\'tvt\'].fillna(fallback)\nsubmission.to_csv(\'submission.csv\', index=False)\n\ndiag = {\n    \'run_version\': NOTEBOOK_RUN_VERSION,\n    \'input_dir\': INPUT_DIR,\n    \'test_wells\': TEST_WELLS,\n    \'pf_n_particles\': int(PF_N_PARTICLES),\n    \'pf_n_seeds\': int(PF_N_SEEDS),\n    \'pf_lik_scale\': float(PF_LIK_SCALE),\n    \'submission_rows\': int(len(submission)),\n    \'submission_tvt_min\': float(submission[\'tvt\'].min()),\n    \'submission_tvt_max\': float(submission[\'tvt\'].max()),\n    \'submission_tvt_mean\': float(submission[\'tvt\'].mean()),\n    \'well_diagnostics\': well_diagnostics,\n}\nwith open(\'run_diagnostics_physical_pf_9150_repeat.json\', \'w\') as f:\n    json.dump(diag, f, indent=2, sort_keys=True)\n\nprint(f\'\\nDone: {len(submission)} rows\')\nprint(f"TVT range: [{submission[\'tvt\'].min():.3f}, {submission[\'tvt\'].max():.3f}]")\nprint(\'Diagnostics: run_diagnostics_physical_pf_9150_repeat.json\')\nprint(submission.head())\n'
    _old_cwd = _blend_os.getcwd()
    try:
        _blend_os.chdir(_beicicc_dir)
        exec(compile(_beicicc_source, "embedded_beicicc_scale265625.py", "exec"), {"__name__": "__main__"})
    finally:
        _blend_os.chdir(_old_cwd)

    _beicicc_bl = _blend_pd.read_csv(_beicicc_dir / "submission.csv")
    _merged_bl = (
        _seed7_bl[["id", "tvt"]].rename(columns={"tvt": "tvt_seed7"})
        .merge(_mtoshi_bl[["id", "tvt"]].rename(columns={"tvt": "tvt_mtoshi"}), on="id", how="inner")
        .merge(_beicicc_bl[["id", "tvt"]].rename(columns={"tvt": "tvt_beicicc"}), on="id", how="inner")
    )
    if len(_merged_bl) != len(_seed7_bl) or len(_merged_bl) != len(_mtoshi_bl) or len(_merged_bl) != len(_beicicc_bl):
        raise RuntimeError(
            "three-way blend id mismatch "
            f"seed7={len(_seed7_bl)} mtoshi={len(_mtoshi_bl)} "
            f"beicicc={len(_beicicc_bl)} merged={len(_merged_bl)}"
        )
    for _col in ["tvt_seed7", "tvt_mtoshi", "tvt_beicicc"]:
        if not _blend_np.isfinite(_merged_bl[_col].to_numpy(dtype=float)).all():
            raise RuntimeError(f"non-finite values in {_col}")

    _W_SEED7 = 0.48
    _W_MTOSHI = 0.35
    _W_BEICICC = 0.17
    _final_bl = _merged_bl[["id"]].copy()
    _final_bl["tvt"] = (
        _W_SEED7 * _merged_bl["tvt_seed7"].astype(float)
        + _W_MTOSHI * _merged_bl["tvt_mtoshi"].astype(float)
        + _W_BEICICC * _merged_bl["tvt_beicicc"].astype(float)
    )
    _final_bl.to_csv(_blend_w / "submission.csv", index=False)
    _summary_bl = {
        "weights": {
            "seed7_w033_dynamic": _W_SEED7,
            "mtoshi_pred1287_dynamic": _W_MTOSHI,
            "beicicc_pf_scale265625": _W_BEICICC,
        },
        "rows": int(len(_final_bl)),
        "seed7_mean": float(_merged_bl["tvt_seed7"].mean()),
        "mtoshi_mean": float(_merged_bl["tvt_mtoshi"].mean()),
        "beicicc_mean": float(_merged_bl["tvt_beicicc"].mean()),
        "blend_mean": float(_final_bl["tvt"].mean()),
        "rmse_mtoshi_vs_seed7": float(_blend_np.sqrt(_blend_np.mean((_merged_bl["tvt_mtoshi"].to_numpy(float) - _merged_bl["tvt_seed7"].to_numpy(float)) ** 2))),
        "rmse_beicicc_vs_seed7": float(_blend_np.sqrt(_blend_np.mean((_merged_bl["tvt_beicicc"].to_numpy(float) - _merged_bl["tvt_seed7"].to_numpy(float)) ** 2))),
        "rmse_beicicc_vs_mtoshi": float(_blend_np.sqrt(_blend_np.mean((_merged_bl["tvt_beicicc"].to_numpy(float) - _merged_bl["tvt_mtoshi"].to_numpy(float)) ** 2))),
    }
    (_blend_w / "seed7_mtoshi_beicicc_blend_summary.json").write_text(
        _blend_json.dumps(_summary_bl, indent=2),
        encoding="utf-8",
    )
    print("FINAL seed7+mtoshi+beicicc blend wrote submission.csv")
    print(_blend_json.dumps(_summary_bl, indent=2), flush=True)
except Exception as _blend_exc:
    print("FINAL seed7+mtoshi+beicicc blend failed; keeping previous submission.csv:", _blend_exc, flush=True)
