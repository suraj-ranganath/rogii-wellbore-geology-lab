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
| Kaggle submission 53350074 | `uv run kaggle competitions submit rogii-wellbore-geology-prediction -k surajranganath17/rogii-last-known-baseline -v 1 -f submission.csv -m "last-known TVT calibration baseline"` | Kaggle code submission | pending | Kernel version 1 completed and wrote `submission.csv`; scorer status was `PENDING` at submission time. |
| PF selector spread3 | `uv run kaggle competitions submit rogii-wellbore-geology-prediction -k surajranganath17/rogii-pf-selector-spread3 -v 1 -f submission.csv -m "pf selector spread3 public reference"` | Kaggle code submission | pending | Submission ref `53350555`; adapted public `needless090/lb8-781-rogii-sel15-spread3`, kernel version 1 completed in about 3.2 minutes and wrote a valid `submission.csv`. |
| Physical-noise PF | `uv run kaggle competitions submit rogii-wellbore-geology-prediction -k surajranganath17/rogii-physical-noise-pf -v 1 -f submission.csv -m "physical noise pf 64 seed spread3"` | Kaggle code submission | pending | Submission ref `53353647`; adapted public `aiwody/physical-model-less-overfitting-noise`, kernel version 1 completed in about 2 minutes and wrote a valid `submission.csv`. |
| Super-solution top3 | `uv run kaggle kernels push -p kaggle/kernels/super_solution_top3 -t 32400 --accelerator gpu` | Kaggle GPU kernel | running | Distinct tree/physics stack adapted from public `romantamrazov/rogii-super-solution-lb-top-3`; not submitted yet because kernel version 1 is still running. |
| Sunny+v10 artifact blend | `uv run kaggle kernels push -p kaggle/kernels/sunny_v10_artifact_blend` | Kaggle GPU kernel | running | Reconstructs the public `kojimar/rogii-physical-pf-signal-meets-artifact-stack` 0.80 Sunny physical/PF + 0.20 v10 artifact-stack blend without using its inaccessible helper dataset; not submitted yet because kernel version 1 is still running. |

Interpretation:

- Alignment features are directionally useful in the smoke harness.
- A single residual LightGBM is still weaker than the last-known prior on this small split.
- Next serious push should build fold-safe candidate estimators and blend them, including the null prior, beam paths, PF paths, and gated formation priors.
