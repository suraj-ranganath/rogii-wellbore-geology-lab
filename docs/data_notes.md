# Data Notes

Observed after authenticated Kaggle download on 2026-06-03.

## Files

- Raw data directory: `data/raw/rogii-wellbore-geology-prediction`
- Train horizontal wells: 773
- Visible test horizontal wells: 3
- Train horizontal rows: 5,092,255
- Visible test horizontal rows: 19,221
- Sample submission rows: 14,151

Every train and visible test horizontal well has a paired `__typewell.csv`.

## Columns

Train horizontal files include:

- `MD`, `X`, `Y`, `Z`
- formation-top columns such as `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`
- `TVT`, `GR`, `TVT_input`

Visible test horizontal files include:

- `MD`, `X`, `Y`, `Z`
- `GR`, `TVT_input`

Because the formation-top columns are absent from visible test, the default baseline excludes train-only numeric columns. They can still be explored explicitly, but they should not be used in the main CV score unless we have a matching test-time feature plan.

## Public Slice Leakage

The 3 visible test well IDs also appear in train:

- `000d7d20`
- `00bbac68`
- `00e12e8b`

For those wells, the shared visible test columns exactly match the corresponding train rows, and train `TVT` is available for every sample-submission row. This means a copy-from-train file could score perfectly on the visible/public slice, but it is not evidence of hidden-leaderboard generalization. We should track this separately from real model validation.

Official discussion clarification: the visible `test/` files and visible `sample_submission.csv` are example data drawn from train to help author Code Competition notebooks. On scoring, Kaggle replaces those files with the actual hidden test set, described on the data page as about 200 wells. The actual hidden horizontal files expose `MD`, `X`, `Y`, `Z`, `GR`, and `TVT_input`; train-only formation columns and `TVT` are not directly available for hidden test rows.

Implication: do not build a copy-from-train submission path. The championship path is a robust inference notebook that reads whatever hidden test files are mounted at submit time and predicts all hidden sample IDs.
