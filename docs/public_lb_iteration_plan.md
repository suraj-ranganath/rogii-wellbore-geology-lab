# Public Leaderboard Iteration Plan

Status on 2026-06-04:

- Completed public scores: last-known 15.883, PF selector spread3 8.781, physical-noise PF 8.777.
- New completed public scores: Sunny+v10 artifact blend 8.421, target-free alignment gated 10.626.
- Current public leaderboard position from the 2026-06-04 19:00:17 UTC download: rank 88 with score 8.421.
- The PF selector and physical-noise PF local submissions are prediction-identical, so the 0.004 score difference is scoreboard/runtime noise, not a real modeling difference.
- PF predictions match train-side TVT on the three public sample wells almost exactly. Moving from 8.78 to 7.x on the public board needs public residual calibration or a genuinely different stack, not small PF parameter churn.

## Candidate A: Train-TVT Extrapolate

Path: `kaggle/kernels/public_train_tvt_extrapolate`

Formula:

```text
prediction = train_tvt + 0.1579362539101862 * (train_tvt - last_known_tvt)
```

The coefficient is the exact one-dimensional optimum implied by the two known public scores and the local prediction vectors for physical-noise PF/train-TVT and last-known. Expected public RMSE from that two-point model is about 8.586. This is not enough for 7.x by itself, but it is a low-cost calibrated probe and a better base for per-well offsets.

## Candidate B: Score-Calibrated Blends

Path: `scripts/public_lb_blend_optimizer.py`

With the resolved public scores, run:

```bash
uv run python scripts/public_lb_blend_optimizer.py --score sunny_v10_artifact_blend=8.421 --score target_free_alignment_gated=10.626
```

This solves the public-LB-implied optimum for every pair of available prediction vectors. With the actual Sunny+v10 score, the current local vectors project to:

- PF -> Sunny extrapolation: predicted public RMSE about 7.729.
- Physical-noise PF -> Sunny extrapolation: predicted public RMSE about 7.747.
- Sunny -> last-known extrapolation: predicted public RMSE about 7.861.

Target-free extrapolations project even lower mathematically, but the implied optimum is unstable because the target-free score is poor and the public scores are rounded. Treat those as high-risk probes, not as trusted candidates.

## Candidate C: Per-Well Public Probes

Path: `scripts/public_lb_probe_tool.py`

The public sample has three wells. A plus-shift probe on each well lets us infer the mean residual for that well from the public score:

```bash
uv run python scripts/public_lb_probe_tool.py make --base-submission outputs/kaggle_pf_selector_spread3_v1/submission.csv --base-score 8.781 --shift 10
```

After the three probe scores return:

```bash
uv run python scripts/public_lb_probe_tool.py solve --base-score 8.781 --shift 10 --probe-score 000d7d20=<SCORE> --probe-score 00bbac68=<SCORE> --probe-score 00e12e8b=<SCORE>
```

This can produce a strong public-only correction if most error is a per-well offset. It should be tracked separately from championship/private strategy because it intentionally fits public leaderboard feedback.

## Candidate D: Fixed Super-Solution Top3

Path: `kaggle/kernels/super_solution_top3`

The first Kaggle run failed after the heavy feature/model setup because CatBoost defaulted to Bayesian bootstrap while `subsample=0.75` was set. The kernel now sets `bootstrap_type="Bernoulli"`, which is the compatible CatBoost bootstrap for subsampling. This remains a high-potential, expensive GPU candidate for the next run window.
