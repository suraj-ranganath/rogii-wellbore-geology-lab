# Experiment Log

All commands use `uv` from the project root. Generated JSON artifacts stay under
`outputs/` and are ignored by git; this log records the interpretable result.

## 2026-06-04

| Run | Command | Scope | RMSE | Notes |
| --- | --- | --- | ---: | --- |
| Last-known prior | `uv run rogii eval-priors` | all 773 train wells | 15.91 | Strong null; predicts the last known `TVT_input` across the hidden tail. |
| Linear prior | `uv run rogii eval-priors` | all 773 train wells | 113.63 | Not viable; tail TVT is not a simple MD extrapolation. |
| Residual LGBM smoke | `uv run rogii cv-baseline --config configs/smoke.yaml --output outputs/smoke_cv.json` | first 30 train wells, 3 GroupKFold splits | 17.22 | Baseline residual target before alignment features. |
| + prefix-NCC smoke | `uv run rogii cv-baseline --config configs/smoke.yaml --output outputs/smoke_cv_prefix_ncc.json` | first 30 train wells, 3 GroupKFold splits | 17.07 | Adds target-free GR self-correlation against known prefix. |
| + prefix-NCC + beam smoke | `uv run rogii cv-baseline --config configs/smoke.yaml --output outputs/smoke_cv_prefix_ncc_beam.json` | first 30 train wells, 3 GroupKFold splits | 16.65 | Adds deterministic typewell GR beam paths and residual features. |
| Kaggle calibration candidate | `python kaggle/kernels/last_known_baseline/last_known_baseline.py` | hidden rerun target, local visible smoke only | 15.91 local CV prior | Self-contained code-submission candidate to validate Kaggle notebook plumbing. |

Interpretation:

- Alignment features are directionally useful in the smoke harness.
- A single residual LightGBM is still weaker than the last-known prior on this small split.
- Next serious push should build fold-safe candidate estimators and blend them, including the null prior, beam paths, PF paths, and gated formation priors.
