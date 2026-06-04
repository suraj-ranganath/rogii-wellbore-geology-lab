# Leaderboard Strategy

The competition is not a generic tabular regression problem. The task brief describes paired horizontal/typewell CSVs, known horizontal `TVT_input` until the prediction-start point, and hidden `TVT` afterward. Quality is RMSE over the predicted TVT points. The strongest work should therefore model a partially observed geosteering trajectory, not independent rows.

## Phase 0: Reproducibility

- Keep data, outputs, models, and submissions out of git.
- Use `uv` for every command.
- Validate by held-out horizontal well, never random row splits.
- Track experiment config, code revision, CV score, and local artifact path for every run.
- No Kaggle submission until the local validation harness is trusted.

## Phase 1: Baseline Ladder

1. Last-known TVT and linear extrapolation from the pre-PS segment.
2. Gradient boosting over trajectory, GR, pre-PS TVT state, and typewell interpolation features.
3. Per-well postprocessing: smoothness, physically plausible slope limits, and endpoint consistency.
4. GR-to-GR correlation features between horizontal segments and typewell TVT windows.
5. Spatial offset features from nearby training wells: distance, azimuth, local dip, residual transfer.

The visible test files overlap three train wells with known target values. Treat any copy-from-train result as a public-slice diagnostic only, not as model quality. The default baseline also excludes train-only formation-top columns because visible test does not provide them.

Initial no-training benchmark over all 773 train wells: last-known TVT RMSE is 15.91, while naive linear TVT extrapolation is 113.63. A 30-well smoke CV with residual LightGBM scored 17.22, so the first real modeling target is to beat 15.91 under held-out-well validation before tuning.

## Phase 2: Serious Models

- LightGBM/CatBoost/XGBoost ensembles with GroupKFold by well.
- Dynamic time warping or constrained sequence alignment against typewell GR.
- State-space or sequence model over `MD` with known-prefix conditioning.
- Residual model on top of geology-aware deterministic alignment.
- Per-area or per-typewell specialization if validation shows heterogeneous fields.

## Phase 3: Leaderboard Discipline

- Maintain a private experiment log before public leaderboard probing.
- Use leaderboard submissions only to calibrate CV-to-LB gap.
- Prefer ensemble diversity from different assumptions over minor hyperparameter churn.
- Audit every strong jump for leakage, sample-format mistakes, and overfitting to public LB.
