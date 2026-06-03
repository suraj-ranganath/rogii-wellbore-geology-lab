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
```

Training commands exist, but should be launched only when we are ready for an experiment:

```bash
uv run rogii train-baseline --config configs/default.yaml
```

The code never submits to Kaggle. Submission upload stays manual until explicitly requested.

## Project Layout

- `configs/`: experiment configuration
- `docs/`: strategy, notes, and experiment protocol
- `src/rogii_wellbore/`: reusable package code
- `tests/`: fast unit tests
- `data/`, `models/`, `outputs/`, `submissions/`: local generated artifacts ignored by git
