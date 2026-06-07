# Local Scoring Strategy

The downloadable `test/` split is not a public-leaderboard emulator. It contains
three visible plumbing wells that overlap `train/`, so local scoring on
`sample_submission.csv` can be extremely optimistic and rank methods incorrectly.

## Scores We Use

| Score | Command | Use |
| --- | --- | --- |
| Visible-overlap diagnostic | `uv run python scripts/score_visible_overlap.py` | Check row order, value scale, and obvious breakage only. Do not use for ranking. |
| Fast local proxy | `uv run python scripts/local_tail_cv.py --max-wells 80 --folds 5 --repeats 3 --splitter stratified --include-lgbm --lgbm-estimators 80` | Quick method-family ranking before packaging a Kaggle kernel. |
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

For the current strong Ridge/PF/selector family, the next step is to wire the
final PF/selector blend itself into the local train-tail proxy. The notebook logs
already report an internal Ridge OOF around `10.434`, but the existing logs do
not score the final selector blend variants locally, which is why all `w020` to
`w040` variants share the same internal OOF line.

## Calibration Against Successful Submissions

See `docs/proxy_calibration_20260607.md`.

The short version: family-level local proxy scores correlate well with known
Kaggle public scores after adding PF selector replay. The proxy correctly ranks:

1. Ridge/PF/selector `w040`
2. PF selector / physical-noise PF
3. Last-known

It still cannot rank fine variants inside the Ridge family until the final
blend/postprocess stage is scored locally.
