# Competition Constraints

Checked against Kaggle's competition API/pages on 2026-06-04.

- Competition: `rogii-wellbore-geology-prediction`
- Metric: RMSE over `submission.csv` rows.
- Submissions: Kaggle Notebooks only.
- Daily submission limit: 5.
- Final submissions for judging: up to 2.
- Team size: 5.
- Runtime: CPU notebook <= 9 hours; GPU notebook <= 9 hours.
- Internet: disabled during committed submission runs.
- External data/models: allowed only if freely and publicly available.
- Output file: must be named `submission.csv`.
- Entry/team merger deadline: 2026-07-29 23:59 UTC.
- Final submission deadline: 2026-08-05 23:59 UTC.

Operational implications:

- Use local experiments for iteration; spend Kaggle submissions only on calibrated checkpoints.
- Package every champion candidate as an offline notebook artifact with all code/model assets attached as public Kaggle inputs.
- Avoid any method that depends on the visible fake test overlap; hidden reruns replace `test/` with the actual test wells.
