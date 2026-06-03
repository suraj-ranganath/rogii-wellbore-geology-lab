# Sources Checked

- Kaggle competition page: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview
- Public task-brief extraction: https://github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md
- Kaggle LinkedIn announcement: https://www.linkedin.com/posts/kaggle_rogii-wellbore-geology-prediction-activity-7457463642897338368-FwOv

Working assumptions as of 2026-06-03:

- The competition is active, with an entry deadline reported as 2026-07-29 in Kaggle's LinkedIn announcement.
- Each well has one horizontal-well CSV and one paired typewell CSV.
- Horizontal files include `MD`, `X`, `Y`, `Z`, `GR`, `TVT`, and `TVT_input` in train data.
- Typewell files include `TVT`, `GR`, and `Geology`.
- The target metric is RMSE on predicted post-PS `TVT` points.
- Local Kaggle CLI auth currently expects OAuth via `kaggle auth login` or a `KAGGLE_API_TOKEN`.
