# Strategy History

This is the compact index of competition strategies tried so far. Use
`docs/experiment_log.md` for command-level detail and exact artifacts.

## Current Champion

| Strategy | Public RMSE | Status | Lesson |
| --- | ---: | --- | --- |
| SP45 projection + fleongg pretrained, `w_sp45=0.60` | `7.551` | Current best | Strongest family. Continue with SP45-heavy blend sweeps around `0.62`, `0.65`, `0.70`. |

## Tried Strategy Families

| Family | Best public RMSE | Representative runs | Verdict |
| --- | ---: | --- | --- |
| Last-known / simple priors | `15.883` | `last-known TVT calibration baseline` | Only a plumbing/calibration baseline. Last-known local hidden-tail RMSE is about `15.91`; linear MD extrapolation is much worse. |
| PF selector / physical model | `8.777` | `pf selector spread3`, `physical noise pf 64 seed spread3` | Strong early public reference, but plateaued around `8.78`. Good component family, not current leader. |
| Sunny/v10 artifact blend | `8.421` | `sunny v10 artifact blend` | Improved over PF selector but superseded by Ridge/PF and SP45/fleongg families. |
| Target-free alignment / sidecar | `10.626` | `target free alignment gated` | Not competitive standalone. Useful only as an anti-correlated reference; avoid spending slots directly. |
| Super-solution top3 / heavy GPU stack | `10.150` | `super solution top3 fixed catboost bootstrap` | Multiple runtime/CatBoost issues and poor public score. Do not revive without major code audit. |
| Public-score/static CSV extrapolations | `15.883` or no public score | public train-TVT/static public blends | Invalid or fragile for hidden reruns. Do not repeat static public-ID strategies. |
| Hidden-safe Ravaghi Ridge weights | `7.906` | `ravaghi ridge hidden-safe v3 w040` | First serious improvement. Ridge/PF selector blend works, but later SP45/fleongg is stronger. |
| Dynamic-Z exact-80 blends | `8.613` | `ridge w040 dynamic zq12 shrink1000 exact80` | Local exact-80 proxy overfit. Public scores worsened; avoid tiny proxy-driven gates. |
| Structural Ridge-SP / U projection | `8.173` | `ridge sp45 robust u projection`, `ridge sp7776` variants | Public Ridge-SP references underperformed. Projection alone is not enough. |
| Drift geosteering pretrained | `7.858` | `drift geosteering pretrained structural` | Good independent signal and prior best. It is close to fleongg/geosteering behavior but weaker than SP45/fleongg blend. |
| JAEMIN SP45/fleongg blend | `7.551` | `jaemin sp45 fleongg blend w060` | Best direction so far. `w0.60` beat public exact `w0.55` (`7.551` vs `7.609`). |
| JY dynamic correction on JAEMIN | `7.672` | `jaemin sp45 fleongg jy consensus` | Small gated correction still hurt vs clean `w0.60`. Missing 68 JY features is a warning; do not continue until feature frame is fixed. |
| Yaroslav SEL15 / Ridge artifact D6 | `7.903` | `yaroslav sel15 forced selector reference` | Valid discussion reference, but not competitive against SP45/fleongg. |
| iAztec Ridge artifact param | `7.822` | `iaztec ridge artifact param shim` | Fixed `koolbox` issue and completed, but still weaker than SP45/fleongg. Useful reference only. |

## Important Negative Results

- Visible-overlap scoring is not a model selector. It slightly preferred
  JAEMIN `w0.55` over `w0.60`, while Kaggle public preferred `w0.60`.
- Static public-test CSV submissions are not hidden-compatible in this code
  competition. Kaggle reruns notebooks against hidden data.
- Local exact-80/dynamic-Z gates overfit and did not transfer.
- JY model-package predictions are suspect until missing features are resolved.
- More ridge-artifact-only submissions are unlikely to beat SP45/fleongg unless
  they introduce a genuinely new signal.

## Recommended Next Moves

1. Build a controlled SP45/fleongg weight sweep around the current champion:
   `0.62`, `0.65`, `0.68`, `0.70`, plus one conservative fallback if needed.
2. If spending fewer than five submissions, prioritize the best two or three
   SP45-heavy variants over more ridge-artifact references.
3. Validate every pushed Kaggle kernel output before submission:
   row count, sample ID order, finite TVT, and plausible TVT range.
4. Use visible-overlap only for output sanity, not weight selection.
5. Consider blending `w0.60` with drift/fleongg-like independent signals only if
   there is a principled, hidden-compatible reason; public score alone is not
   enough.

## Reference Docs

- `docs/experiment_log.md`: full chronological log.
- `docs/strategy_20260609.md`: latest batch details and postmortem.
- `docs/local_scoring_strategy.md`: local proxy caveats.
- `docs/proxy_calibration_20260607.md`: local-vs-Kaggle calibration.
