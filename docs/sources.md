# Sources Checked

- Kaggle competition page: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview
- Kaggle rules page: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules
- Kaggle code requirements page: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/code
- Public task-brief extraction: https://github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md
- Kaggle LinkedIn announcement: https://www.linkedin.com/posts/kaggle_rogii-wellbore-geology-prediction-activity-7457463642897338368-FwOv

Working assumptions as of 2026-06-04:

- The competition is active, with entry and team merger deadlines on 2026-07-29 23:59 UTC and final submission deadline on 2026-08-05 23:59 UTC.
- The Kaggle API reports `maxDailySubmissions=5`, `maxTeamSize=5`, and notebook-only submissions.
- The Code Requirements page requires CPU/GPU notebook runtime <= 9 hours, internet disabled, and a `submission.csv` output file.
- Each well has one horizontal-well CSV and one paired typewell CSV.
- Horizontal files include `MD`, `X`, `Y`, `Z`, `GR`, `TVT`, and `TVT_input` in train data.
- Typewell files include `TVT`, `GR`, and `Geology`.
- The target metric is RMSE on predicted post-PS `TVT` points.
- Local Kaggle CLI auth currently expects OAuth via `kaggle auth login` or a `KAGGLE_API_TOKEN`.
