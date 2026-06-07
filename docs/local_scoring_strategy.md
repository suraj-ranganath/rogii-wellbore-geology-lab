# Local Scoring Strategy

The downloadable `test/` split is not a public-leaderboard emulator. It contains
three visible plumbing wells that overlap `train/`, so local scoring on
`sample_submission.csv` can be extremely optimistic and rank methods incorrectly.

## Scores We Use

| Score | Command | Use |
| --- | --- | --- |
| Visible-overlap diagnostic | `uv run python scripts/score_visible_overlap.py` | Check row order, value scale, and obvious breakage only. Do not use for ranking. |
| Fast local proxy | `uv run python scripts/local_tail_cv.py --max-wells 80 --folds 5 --repeats 3 --splitter stratified --include-lgbm --lgbm-estimators 80` | Quick method-family ranking before packaging a Kaggle kernel. |
| Ridge final-blend proxy | `uv run python scripts/local_ridge_final_blend_cv.py --max-wells 80 --selector-particles 128 --selector-seeds 8` | Tune final Ridge/PF selector weights and gates inside the current strongest Ridge family. |
| Serious local proxy | `bash scripts/run_ds_serv6_tail_cv.sh tailcv_full_lgbm --max-wells 773 --folds 5 --repeats 3 --splitter stratified --include-lgbm --lgbm-estimators 300` | Full train-tail CV on ds-serv6 before spending submissions on a new family. |
| GPU model proxy | `bash scripts/run_ds_serv6_tail_cv.sh tailcv_gpu_catboost --max-wells 773 --folds 5 --repeats 3 --splitter stratified --include-catboost --catboost-task-type GPU --catboost-devices 0 --catboost-iterations 500` | CatBoost/GPU sweeps on ds-serv6. |

## Current Calibration

The current Kaggle-best `ridge_w040` submission has:

- visible-overlap RMSE: `2.579`
- Kaggle public score: `7.906`

Visible-overlap ranking is not reliable. Known examples:

- `pf_selector_spread3`: visible `0.005`, Kaggle public `8.781`
- `physical_noise_pf`: visible `0.005`, Kaggle public `8.777`
- `ridge_w040`: visible `2.579`, Kaggle public `7.906`

Across known scored submissions, visible-overlap Spearman rank correlation with
Kaggle public is only about `0.260`.

The train-tail proxy is more useful because it simulates the hidden target
pattern on hundreds of train wells. On the 80-well repeated stratified proxy:

- `lgbm_last_residual`: `14.679`
- `ridge_last_residual`: `14.874`
- `last_known`: `15.079`
- `ridge_z_residual`: `15.522`
- `linear_md`: `69.701`
- `z_anchor`: `129.110`

This confirms the main rule: residualize around `last_known`; naive MD/dZ
extrapolation is not viable as a standalone method.

## Promotion Rule

Before using one of the 5/day Kaggle submissions, a new method should satisfy
at least one of these:

- Improve the repeated train-tail proxy by at least `0.20` RMSE against the
  closest comparable local baseline.
- Improve worst-well or tail-risk metrics without degrading global RMSE by more
  than `0.05`.
- Add a genuinely diverse prediction family whose local proxy is close to best
  and whose Kaggle-side output smoke/diversity checks pass.

For the current strong Ridge/PF/selector family, use the final-blend proxy
instead of the generic tail-CV proxy. It reconstructs the Ravaghi base-model OOF
predictions from the public artifact trainers, reproduces the Ridge meta-model
score close to the Kaggle notebook log (`10.417` local reconstruction versus
`10.434` logged), then scores the final Ridge/PF selector blend on train hidden
tails.

On the submitted Ridge weights, the 80-well final-blend proxy ranks:

1. `w040`: `9.319`
2. `w035`: `9.478`
3. `w030`: `9.657`
4. `w025`: `9.857`
5. `w020`: `10.075`

That matches the most important Kaggle result: `w040` was best public. Spearman
rank correlation across `w020`-`w040` is `0.90`; the only miss is the small
`w020`/`w025` swap. The same proxy suggests testing heavier Ridge weights:
`0.70` scored `8.866`, ahead of `0.60`/`0.80` at `8.918`, so the next queue
should cover conservative `0.42/0.45/0.50` and aggressive `0.60/0.70`.

## Calibration Against Successful Submissions

See `docs/proxy_calibration_20260607.md`.

The short version: family-level local proxy scores correlate well with known
Kaggle public scores after adding PF selector replay. The proxy correctly ranks:

1. Ridge/PF/selector `w040`
2. PF selector / physical-noise PF
3. Last-known

The final-blend proxy can now rank fine variants inside the Ridge family. Gated
prefix/form variants currently score worse locally than plain heavier Ridge
weights and should be deprioritized unless a later larger proxy contradicts
this.
