# Public Research Notes

Collected from Kaggle discussions and public notebooks on 2026-06-04.

## What To Adopt

- Predict residual drift, not absolute `TVT`. Public writeups repeatedly report absolute-target models worse than the last-known baseline.
- Blend genuinely different path families. The strongest public notebook inspected on 2026-06-04 combines Sunny's physical/PF signal with the v10 artifact stack, and its author explicitly framed the gain as complementary error rather than a better single model.
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
- Depending on inaccessible public helper datasets. The public `kojimar/rogii-physical-pf-signal-meets-artifact-stack` helper dataset returns `403`/not-found, so our candidate embeds the accessible component sources instead of cloning that dependency.

## Public Score Anchors

- Last-known TVT baseline: about `15.91` RMSE, matching our local evaluation.
- Public tree/physics feature stacks report OOF near `9.85-10.05` and public LB around `8.8-9.5`.
- Newer public code page anchors inspected on 2026-06-04: `ROGII: Physical PF Signal Meets Artifact Stack` at `8.293`, `Target-Free Alignment for TVT` at `8.782`, `rogii-sel15-rerun` at `8.863`, and GR/NCC tree stacks around `8.905`.
- Current leading public LB is around `6.5`, so to contend for the championship we need to beat the public feature-stack family, not just reproduce it.

## 2026-06-05 Public Code Sweep

- `ravi20076/rogii2026-public-blend-v2` is dynamic and blends a PF selector with a tree/tortuosity stack. The public notebook had an argparse mismatch (`nargs=3` with two weights); our private copy fixes that to `nargs=2`.
- `beicicc/rogii-0605-aiden-pf-plateauavg` is a fresh dynamic PF-only notebook using 128 seeds and a likelihood-scale plateau around `2.6875,2.71875,2.75`; it completed on Kaggle and wrote a valid `submission.csv`.
- `sunning11/rogii-gold-v20` hard-coded the three visible example test wells, but its formation-datum invariant can be made hidden-compatible by dynamically iterating mounted test wells and preserving sample order. Our script version scored `3.55` RMSE on the visible overlap sanity check.
- `nina2025/rogii-h-blend-v2` failed on Kaggle's P100 path because TabICL raised `CUDA error: no kernel image is available for execution on the device`. The artifact manifest requires TabICL outputs, so this was excluded from the next submission batch.
