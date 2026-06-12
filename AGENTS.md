# Agent Guide: ROGII Wellbore Geology Prediction

This repo is an active Kaggle competition workspace for
`rogii-wellbore-geology-prediction`. The objective is to maximize final
competition performance, not just public leaderboard position. Be rigorous with
validation, keep changes reproducible, and never spend Kaggle submissions unless
the user explicitly asks.

## Current State

- Current best public score: `7.541` from `final queue sp45 fleongg w060 override`.
- Leaderboard top is `5.986`; ~96 teams sit below our `7.551`.
- Strongest current family: SP45 projected Ridge/PF output blended with
  fleongg pretrained geosteering inference.
- 2026-06-10 analytic evidence (`scripts/blend_qp_20260610.py`,
  `docs/strategy_20260610.md`): the blend optimum is well above `w=0.60`,
  plausibly at or beyond pure SP45; pure SP45 public RMSE is bounded
  `>= 7.25`. A `0.83 sp45 + 0.17 drift_geo` mix is the simplex-QP optimum.
- Cross-run nondeterminism on Kaggle is large (component RMS `0.46` sp45,
  `1.32` fleongg between identical-code runs), so close-variant public scores
  carry run noise.
- The pixiux guarded train-overlap override (vendored at
  `kaggle/kernels/shared/pixiux_overlap_override.py`) is LB-validated at
  `~-0.05` on this family (`7.572 -> 7.519`) and hidden-rerun safe. It is
  appended to runtime-safe SP45/fleongg candidates.
- The 2026-06-11 00:00 UTC batch failed operationally: bagged SP45-heavy
  notebooks exceeded hidden submission runtime; do not resubmit bag3 kernels
  without major runtime reduction. The medali CNN-MTP exact candidate scored
  `14.298` and is not competitive standalone.
- The 2026-06-12 00:00 UTC queue improved the best score only marginally:
  `w060s` scored `7.541`, `fle3n_v5_w060_h0455` `7.565`,
  `fle3n_v5_exact_h050` `7.585`, `fle3n_v5f_exact_h050` `7.637`, and
  `w100s` `7.766`.
- Active queue for the 2026-06-13 00:00 UTC reset
  (`scripts/queue_20260612_improvement_candidates.py`, tmux session
  `rogii_improvequeue_20260612`): `fle3n_v5_exact_r2`,
  `fle3n_v5f_exact_r2`, `jaemin_seed7_mtoshi_beicicc`,
  `jaemin_affine_seed7_mtoshi`, and `jaemin_sp45_fleongg_w065s`.
  All five are completed version-1 Kaggle commits with validated
  `submission.csv` outputs. The queue is waiting at daily cap `5/5`.
- Weaker recent directions: standalone ridge-artifact/projection candidates
  (`7.822+`), Yaroslav D6 (`7.903`), and JY dynamic correction (`7.672`).
- Visible-overlap local scoring is not reliable for selecting close variants.
  Use it for row/order/value sanity checks only (exception: it is the right
  tool to verify the overlap override plumbing).

Read these before making competition decisions:

- [docs/strategy_history.md](docs/strategy_history.md): compact index of every
  major strategy tried, outcome, and verdict.
- `docs/experiment_log.md`: chronological experiment record and scores.
- `docs/strategy_20260612_improvement_queue.md`: active improvement queue
  after the `7.541` result.
- `docs/strategy_20260611_final_queue.md`: active final queue after the
  timed-out bagged batch.
- `docs/strategy_20260610.md`: previous SP45-heavy ladder, bagging, and
  overlap-override strategy; useful context, but the bagged queue timed out.
- `docs/local_scoring_strategy.md`: what local scores mean and do not mean.
- `docs/competition_constraints.md`: Kaggle rules and daily limits.
- `docs/proxy_calibration_20260607.md`: local-vs-Kaggle calibration notes.

## Hard Rules

- Do not submit to Kaggle unless the user explicitly asks for submissions.
- Do not launch long runs unless the user asks or clearly approves.
- Do not use public/static test ID leakage or embedded public-test CSVs. Hidden
  reruns replace `test/`, and static public-ID submissions failed before.
- Do not trust the downloadable `test/` overlap wells as leaderboard proxy.
- Do not commit credentials, cookies, Kaggle tokens, `kaggle.json`, `.env`, data
  dumps, model binaries, logs, or generated outputs.
- Work with the existing git state. Do not revert user changes.
- Use `uv` for Python commands.

## Setup And Checks

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run rogii --help
```

Kaggle authentication:

```bash
uv run kaggle auth login
uv run kaggle competitions submissions -c rogii-wellbore-geology-prediction
```

Safe data/setup commands:

```bash
uv run rogii inspect-data
uv run rogii audit-data
uv run rogii eval-priors
uv run rogii eval-formation-priors
uv run rogii eval-dense-formation-priors
```

## Project Layout

- `src/rogii_wellbore/`: reusable package code and Typer CLI.
- `scripts/`: experiment, materialization, queue, and scoring scripts.
- `kaggle/kernels/`: Kaggle notebook/script kernel packages.
- `configs/`: baseline experiment configs.
- `docs/`: strategy, scoring, constraints, and experiment history.
- `tests/`: fast unit tests.
- Ignored/generated: `data/raw/`, `models/`, `outputs/`, `submissions/`,
  `logs/`, `submission.csv`.

## Local Scoring

Use these in increasing cost/order:

```bash
uv run python scripts/score_visible_overlap.py
uv run python scripts/local_tail_cv.py --max-wells 80 --folds 5 --repeats 3 --splitter stratified --include-lgbm --lgbm-estimators 80
uv run python scripts/local_pf_selector_cv.py --max-wells 40 --n-seeds 16
uv run python scripts/local_ridge_final_blend_cv.py --max-wells 80 --selector-particles 128 --selector-seeds 8
```

Use visible-overlap only for:

- `submission.csv` schema and row count.
- sample ID ordering.
- finite, plausible TVT range.
- obvious broken-output detection.

Do not use visible-overlap to choose close blend weights. It preferred `w0.55`
over `w0.60`, while Kaggle public preferred `w0.60`.

For larger local runs on the UCSD server:

```bash
bash scripts/run_ds_serv6_tail_cv.sh tailcv_full_lgbm --max-wells 773 --include-lgbm --lgbm-estimators 300
bash scripts/run_ds_serv6_tail_cv.sh tailcv_gpu_catboost --max-wells 773 --include-catboost --catboost-task-type GPU --catboost-devices 0 --catboost-iterations 500
```

The server target is `suraj@ds-serv6.ucsd.edu`. Use it for heavier local
validation, not for Kaggle submissions.

## Kaggle Submission Constraints

From `docs/competition_constraints.md`:

- Kaggle Notebook submissions only.
- Daily limit: 5 submissions.
- Final selected submissions: up to 2.
- CPU/GPU notebook runtime limit: 9 hours.
- Internet disabled during committed submission runs.
- External data/models must be public and freely available.
- Required output: `submission.csv`.
- Final deadline: `2026-08-05 23:59 UTC`.

Before spending submissions:

1. Materialize the candidate kernel under `kaggle/kernels/...`.
2. Push the kernel, wait for `KernelWorkerStatus.COMPLETE`.
3. Download Kaggle output with `uv run kaggle kernels output ...`.
4. Validate `submission.csv` against sample order, finite values, row count, and
   plausible TVT range.
5. Only then queue or submit, and only with explicit user approval.

Queue scripts reuse `scripts/queue_20260607_heavy_ridge_candidates.py` helpers.
They validate output before submission. State files live under `outputs/` and
are generated/ignored.

## Recent Submission Results

Five-candidate batch submitted on `2026-06-10 00:01 UTC`:

| Description | Public RMSE |
| --- | ---: |
| `jaemin sp45 fleongg blend w060` | `7.551` |
| `jaemin sp45 fleongg blend exact` | `7.609` |
| `jaemin sp45 fleongg jy consensus` | `7.672` |
| `iaztec ridge artifact param shim` | `7.822` |
| `yaroslav sel15 forced selector reference` | `7.903` |

Operational implications:

- Continue from the clean JAEMIN blend family.
- Test SP45-heavy weights around `0.62`, `0.65`, `0.70`.
- Avoid more JY correction work until missing features are resolved.
- Avoid spending slots on ridge-artifact-only variants unless they add a new
  signal or improve a trustworthy local proxy.

## Kernel Materialization Notes

Useful current materializers:

```bash
uv run python scripts/materialize_20260609_jaemin_candidates.py
uv run python scripts/materialize_20260609_discussion_candidates.py
```

Notable fixes:

- `iaztec_ridge_artifact_param` version 1 failed due missing `koolbox`.
  Version 2 includes a Yaroslav-style `koolbox` resolver/shim and completed.
- `jaemin_sp45_fleongg_jy` uses a small consensus-gated JY correction; raw JY
  features were incomplete, with 68 missing feature columns.

## Development Conventions

- Prefer existing helper scripts and documented patterns over new abstractions.
- Keep generated Kaggle outputs out of git.
- Use `apply_patch` for manual file edits.
- Run at least:

```bash
uv run ruff check <changed files>
uv run python -m py_compile <changed scripts>
uv run pytest
```

- When changing competition strategy, update `docs/experiment_log.md`.
- When changing the active plan, update the relevant `docs/strategy_*.md`.
- When creating queued candidates, record kernel IDs, versions, messages, output
  validation ranges, and submission status.

## Quick Commands

```bash
# Show submissions and current scores
uv run kaggle competitions submissions -c rogii-wellbore-geology-prediction

# Check a kernel status
uv run kaggle kernels status <owner>/<kernel-slug>

# Download a kernel output for validation
uv run kaggle kernels output <owner>/<kernel-slug> -p outputs/<candidate-dir>

# Push a prepared kernel
uv run kaggle kernels push -p kaggle/kernels/<candidate-dir>
```

Remember: pushing a kernel is not a competition submission. Submitting with
`uv run kaggle competitions submit ...` spends one of the 5 daily slots.
