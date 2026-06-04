# Public Research Notes

Collected from Kaggle discussions and public notebooks on 2026-06-04.

## What To Adopt

- Predict residual drift, not absolute `TVT`. Public writeups repeatedly report absolute-target models worse than the last-known baseline.
- Build formation-surface priors from train-only columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) using spatial KNN or weighted plane fits. Hidden test does not expose these columns, but train-derived surfaces are allowed and physically meaningful.
- Calibrate a per-well `b_well = median(TVT_input + Z - formation_surface)` from the known prefix, then estimate `TVT = -Z + formation_surface + b_well`.
- Use normalized cross-correlation (NCC) over GR/typewell windows as a feature family, not as the sole prediction. Public reports say NCC is highly correlated but standalone RMSE is still weaker than the best ensembles.
- Use physics/path estimators such as particle filters and Viterbi/beam search as weak signals. The best public writeups use estimator disagreement and divergence features heavily.
- Use model diversity: LGBM/CatBoost/HistGradientBoosting or XGBoost with non-negative or simple robust blending.
- Validate by well, and monitor worst-well behavior. Row splits are invalid.

## What To Avoid

- Copying visible test targets: Kaggle replaces visible fake test data with hidden test data at scoring.
- Relying on train-only formation columns as direct test inputs. They must be imputed from train.
- Overweighting DTW: one strong public writeup says DTW features worsened OOF versus NCC.
- Online/test-time learning from artificial prefix splits without proof. Public reports show small or negative honest OOF gains.
- Coordinate-overlap postprocessing for hidden wells. It mainly exploits the fake visible test overlap and does not generalize.

## Public Score Anchors

- Last-known TVT baseline: about `15.91` RMSE, matching our local evaluation.
- Public tree/physics feature stacks report OOF near `9.85-10.05` and public LB around `8.9`.
- Current leading public LB is around `6.5`, so to contend for the championship we need to beat the public feature-stack family, not just reproduce it.
