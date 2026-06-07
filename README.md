# ROGII Wellbore Geology Prediction

This repo is the working Kaggle project for `rogii-wellbore-geology-prediction`.

The target is to predict `TVT` for the hidden part of each horizontal well using:

- horizontal well trajectory and GR logs (`MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`)
- paired typewell logs (`TVT`, `GR`, `Geology`)
- spatial relationships between nearby wells
- PS-aware validation by held-out well

## Setup

```bash
uv sync --extra dev
uv run rogii --help
```

## Kaggle Authentication

The current Kaggle CLI needs authentication before data can be downloaded.

```bash
uv run kaggle auth login
```

Alternatively, generate an API token in Kaggle settings and export it before running commands:

```bash
export KAGGLE_API_TOKEN=...
```

Then download and inspect the competition data after auth succeeds:

```bash
uv run rogii download-data
uv run rogii inspect-data
```

## Fast Commands

These are intentionally lightweight and safe to run during setup:

```bash
uv run pytest
uv run ruff check .
uv run rogii inspect-data
uv run rogii audit-data
uv run rogii eval-priors
uv run rogii eval-formation-priors
uv run rogii eval-dense-formation-priors
uv run python scripts/smoke_20260607_candidates.py
uv run python scripts/local_tail_cv.py --max-wells 80
uv run python scripts/score_visible_overlap.py
```

Current competition constraints are tracked in `docs/competition_constraints.md`.
Fast experiment results are tracked in `docs/experiment_log.md`.

Training commands exist, but should be launched only when we are ready for an experiment:

```bash
uv run rogii cv-baseline --config configs/smoke.yaml --output outputs/smoke_cv.json
uv run rogii train-baseline --config configs/default.yaml
uv run rogii predict-submission --model models/baseline.joblib
```

The code never submits to Kaggle. Submission upload stays manual until explicitly requested.

For the current queued notebook candidates, run the local smoke harness before spending
submissions:

```bash
uv run python scripts/smoke_20260607_candidates.py
```

To refresh the notebook outputs from Kaggle first without submitting:

```bash
uv run python scripts/smoke_20260607_candidates.py --refresh-kaggle
```

For candidate-method development, use train wells as a local hidden-tail scoring
set:

```bash
uv run python scripts/local_tail_cv.py --max-wells 80
uv run python scripts/local_tail_cv.py --max-wells 200 --include-lgbm
uv run python scripts/local_tail_cv.py --max-wells 80 --include-catboost --catboost-iterations 80
```

For larger local-CV sweeps on `ds-serv6`:

```bash
bash scripts/run_ds_serv6_tail_cv.sh tailcv_full_lgbm --max-wells 773 --include-lgbm --lgbm-estimators 300
bash scripts/run_ds_serv6_tail_cv.sh tailcv_gpu_catboost --max-wells 773 --include-catboost --catboost-task-type GPU --catboost-devices 0 --catboost-iterations 500
```

To diagnose a produced `submission.csv` against the three downloadable
visible-overlap wells:

```bash
uv run python scripts/score_visible_overlap.py
uv run python scripts/score_visible_overlap.py candidate=submissions/candidate.csv
```

This visible-overlap score is not comparable to the Kaggle public score because
the downloadable wells overlap train and are only a plumbing test.

## Project Layout

- `configs/`: experiment configuration
- `docs/`: strategy, notes, and experiment protocol
- `src/rogii_wellbore/`: reusable package code
- `tests/`: fast unit tests
- `data/`, `models/`, `outputs/`, `submissions/`: local generated artifacts ignored by git
