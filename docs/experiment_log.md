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
| Kaggle submission 53350074 | `uv run kaggle competitions submit rogii-wellbore-geology-prediction -k surajranganath17/rogii-last-known-baseline -v 1 -f submission.csv -m "last-known TVT calibration baseline"` | Kaggle code submission | 15.883 public | Kernel version 1 completed and wrote `submission.csv`; calibration confirmed expected weak baseline. |
| PF selector spread3 | `uv run kaggle competitions submit rogii-wellbore-geology-prediction -k surajranganath17/rogii-pf-selector-spread3 -v 1 -f submission.csv -m "pf selector spread3 public reference"` | Kaggle code submission | 8.781 public | Submission ref `53350555`; adapted public `needless090/lb8-781-rogii-sel15-spread3`, kernel version 1 completed in about 3.2 minutes and wrote a valid `submission.csv`. |
| Physical-noise PF | `uv run kaggle competitions submit rogii-wellbore-geology-prediction -k surajranganath17/rogii-physical-noise-pf -v 1 -f submission.csv -m "physical noise pf 64 seed spread3"` | Kaggle code submission | 8.777 public | Submission ref `53353647`; adapted public `aiwody/physical-model-less-overfitting-noise`, kernel version 1 completed in about 2 minutes and wrote a valid `submission.csv`. |
| Super-solution top3 | `uv run kaggle kernels push -p kaggle/kernels/super_solution_top3 -t 32400 --accelerator gpu` | Kaggle GPU kernel | error | Distinct tree/physics stack adapted from public `romantamrazov/rogii-super-solution-lb-top-3`; failed after ~2.0h when CatBoost rejected `subsample` with default Bayesian bootstrap. No submission made. |
| Sunny+v10 artifact blend | `uv run kaggle competitions submit rogii-wellbore-geology-prediction -k surajranganath17/rogii-sunny-v10-artifact-blend -v 1 -f submission.csv -m "sunny v10 artifact blend"` | Kaggle code submission | 8.421 public | Submission ref `53356985`; reconstructs the public `kojimar/rogii-physical-pf-signal-meets-artifact-stack` 0.80 Sunny physical/PF + 0.20 v10 artifact-stack blend without using its inaccessible helper dataset; best completed submission so far. |
| Target-free alignment gated | `uv run kaggle competitions submit rogii-wellbore-geology-prediction -k surajranganath17/rogii-strat-align-sidecar-v2 -v 1 -f submission.csv -m "target free alignment gated"` | Kaggle code submission | 10.626 public | Submission ref `53359947`; private script-kernel copy of public `pilkwang/rogii-eda-target-free-alignment-for-tvt`, using `full_stack_sel15_gated_model_gated` with same-well physical shortcut disabled. Worse than PF/Sunny on the public split, but its prediction vector may still be useful as an anti-correlated extrapolation direction. |
| Public train-TVT extrapolate | `uv run python kaggle/kernels/public_train_tvt_extrapolate/public_train_tvt_extrapolate.py` | local candidate generation | not submitted | Public-only calibrated candidate: `train_tvt + 0.1579362539101862 * (train_tvt - last_known_tvt)`. The coefficient is solved from completed public scores for physical-noise PF/train-TVT 8.777 and last-known 15.883; predicted public RMSE from that two-point model is about 8.586. |
| Public LB blend optimizer | `uv run python scripts/public_lb_blend_optimizer.py` | local candidate generation | not submitted | Writes score-calibrated pairwise blend/extrapolation candidates under `outputs/public_lb_blend_candidates/`; ready to rerun when Sunny+v10 and target-free public scores resolve. |
| Public well-shift probes | `uv run python scripts/public_lb_probe_tool.py make --base-score 8.781 --shift 10` | local probe generation | not submitted | Writes one plus-shift probe per public well so tomorrow's public submissions can infer mean per-well residuals and build a corrected public-only candidate. |
| Next-window 5-candidate queue | `uv run python scripts/queue_next_window_candidates.py --timeout-minutes 720 --poll-seconds 180` | queued Kaggle code submissions | pending | Pushes fixed super-solution v2 plus four public-calibrated candidate kernels, waits for the 2026-06-05 00:00 UTC reset, then submits up to five completed candidates. |
| Static public-kernel v2 fix | `uv run python scripts/materialize_public_static_kernels.py && uv run kaggle kernels push -p <static_kernel_dir>` | queued Kaggle code submissions | not submitted | Static public candidates now embed their compressed CSV payload directly in the code file because Kaggle did not stage adjacent payload files for script kernels. Static v2 outputs were downloaded from Kaggle and verified to exactly match the intended local blend CSVs. |

Interpretation:

- Alignment features are directionally useful in the smoke harness.
- A single residual LightGBM is still weaker than the last-known prior on this small split.
- Next serious push should build fold-safe candidate estimators and blend them, including the null prior, beam paths, PF paths, and gated formation priors.
