# Proxy Calibration - 2026-06-07

Goal: check whether the local train-tail proxy agrees with Kaggle public scores
for methods we have already submitted successfully.

## Family-Level Comparison

| Method family | Local proxy | Kaggle public | Local source |
| --- | ---: | ---: | --- |
| Ridge/PF/selector `w040` | `10.434` | `7.906` | Internal Ridge OOF from successful `w040` notebook log. |
| PF selector | `11.448` | `8.781` | `scripts/local_pf_selector_cv.py --max-wells 80 --n-particles 128 --n-seeds 8`. |
| Physical-noise PF | `11.448` | `8.777` | Same PF family proxy; Kaggle candidate used 64 seeds. |
| Last-known | `15.079` | `15.883` | 80-well train-tail baseline. |

With these family-level points:

- Pearson correlation: `0.992`
- Spearman rank correlation: `0.949`
- If the duplicate PF-family row is collapsed, rank order is exactly aligned:
  `Ridge/PF/selector > PF selector > last-known`.

## Interpretation

The local proxy is useful for separating major method families. It correctly
prefers the strong Ridge/PF/selector family over PF selector, and PF selector
over last-known, matching Kaggle public.

It is not yet enough to rank small variants within the same family. The `w020`
through `w040` Ridge submissions all share the same internal Ridge OOF line
around `10.434`, but Kaggle distinguishes them from `8.439` to `7.906`.
That means we still need to wire the final PF/selector blend and postprocessing
weights into the local proxy before trusting it for fine-grained tuning.

## Current Commands

Fast generic proxy:

```bash
uv run python scripts/local_tail_cv.py --max-wells 80 --folds 5 --repeats 3 --splitter stratified --include-lgbm --lgbm-estimators 80
```

PF selector replay proxy:

```bash
uv run python scripts/local_pf_selector_cv.py --max-wells 80 --n-particles 128 --n-seeds 8 --output outputs/local_pf_selector_cv_80_s8p128.json
```

Heavier PF selector replay for ds-serv6, after the repo/data has been synced to
`/data/suraj/rogii-wellbore-geology-lab`:

```bash
ssh suraj@ds-serv6.ucsd.edu '
  cd /data/suraj/rogii-wellbore-geology-lab &&
  tmux new-session -d -s pf_selector_full "
    uv run python scripts/local_pf_selector_cv.py \
      --max-wells 773 \
      --n-particles 500 \
      --n-seeds 64 \
      --output outputs/pf_selector_full.json \
      2>&1 | tee logs/pf_selector_full.log
  "
'
```
