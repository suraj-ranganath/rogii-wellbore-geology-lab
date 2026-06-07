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

The final Ridge/PF blend proxy now adds fine-grained signal inside the current
best Ridge family. It reconstructs the Ridge meta OOF score at `10.417`, close
to the submitted notebook log line `10.434`, then scores the final blend on
train hidden tails.

On 80 wells with `128` selector particles and `8` seeds, submitted weights rank:

| Ridge weight | Local final-blend proxy | Kaggle public |
| ---: | ---: | ---: |
| `0.40` | `9.319` | `7.906` |
| `0.35` | `9.478` | `8.108` |
| `0.30` | `9.657` | `8.187` |
| `0.25` | `9.857` | `8.439` |
| `0.20` | `10.075` | `8.233` |

For those five points:

- Pearson correlation: `0.775`
- Spearman rank correlation: `0.900`

This is good enough to reject the lower Ridge-weight/gated candidates and to
test a heavier Ridge-weight queue. The proxy suggests `0.70` (`8.866`) as the
best local grid point, with `0.60`/`0.80` tied next (`8.918`). Because the
40-well proxy still preferred the known `0.40` neighborhood, the replacement
Kaggle queue covers `0.42`, `0.45`, `0.50`, `0.60`, and `0.70` rather than only
aggressive weights.

## Current Commands

Fast generic proxy:

```bash
uv run python scripts/local_tail_cv.py --max-wells 80 --folds 5 --repeats 3 --splitter stratified --include-lgbm --lgbm-estimators 80
```

PF selector replay proxy:

```bash
uv run python scripts/local_pf_selector_cv.py --max-wells 80 --n-particles 128 --n-seeds 8 --output outputs/local_pf_selector_cv_80_s8p128.json
```

Ridge final-blend proxy:

```bash
uv run python scripts/local_ridge_final_blend_cv.py --max-wells 80 --selector-particles 128 --selector-seeds 8 --output outputs/local_ridge_final_blend_cv_80_s8p128.json
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
