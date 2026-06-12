# 2026-06-11 Final Queue Strategy

## Situation

- Current best public score is `7.551` from `jaemin sp45 fleongg blend w060`.
- The 2026-06-11 UTC batch spent five submissions. Four SP45-heavy bagged or
  blend notebooks timed out on the hidden submission run, and the standalone
  medali CNN-MTP exact candidate scored `14.298`.
- The next queue must therefore prioritize runtime-safe, already-completed
  version-1 commits. No bag3 orchestrator notebooks are used.

## Submitted Queue

The launchd watcher `com.suraj.rogii.finalqueue20260611` ran:

```bash
uv run python scripts/queue_20260611_final_candidates.py --timeout-minutes 1440 --poll-seconds 120
```

It submitted all five candidates immediately after the 2026-06-12 00:00 UTC
reset. State lives in `outputs/queue_20260611_final_state.json`.

| Priority | Candidate | Kernel | Rationale |
| ---: | --- | --- | --- |
| 1 | `jaemin_sp45_fleongg_w060s` | `surajranganath17/rogii-sp45-fleongg-w060s` | Best validated family, previous public `7.551`, with guarded overlap override. |
| 2 | `jaemin_sp45_fleongg_w100s` | `surajranganath17/rogii-sp45-fleongg-w100s-runtime-safe` | Pure SP45 upside from the analytic QP; runtime-safe single-run path. |
| 3 | `fle3n_v5_exact_h050` | `surajranganath17/rogii-fle3n-v5-exact-h050` | Public `fleongg/fle3n-rogii-v5` claims `7.528` using a dynamic interpretation hedge. |
| 4 | `fle3n_v5_w060_h0455` | `surajranganath17/rogii-fle3n-v5-w060-h0455` | Same v5 code with SP45-heavy final base and hedge transfer weight `0.455`, matching the public notebook's reported optimum region. |
| 5 | `fle3n_v5f_exact_h050` | `surajranganath17/rogii-fle3n-v5f-exact-h050` | Independent/probe variant from the same high-scoring fle3n v5 hedge family. |

All five were pushed, reached `KernelWorkerStatus.COMPLETE`, downloaded,
validated against sample order, row count, finite predictions, and plausible
TVT range, then submitted. Initial submission status was `PENDING`.

Submission refs:

| Candidate | Message | Ref |
| --- | --- | ---: |
| `jaemin_sp45_fleongg_w060s` | `final queue sp45 fleongg w060 override` | `53583069` |
| `jaemin_sp45_fleongg_w100s` | `final queue sp45 fleongg w100 override` | `53583076` |
| `fle3n_v5_exact_h050` | `final queue fle3n v5 exact h050` | `53583086` |
| `fle3n_v5_w060_h0455` | `final queue fle3n v5 w060 h0455` | `53583093` |
| `fle3n_v5f_exact_h050` | `final queue fle3n v5f exact h050` | `53583103` |

## Excluded Ideas

- `yooughtul/rogii-top-ensemble`: public runtime is too high for hidden rerun
  risk after the bagged timeout batch.
- `praxel/rogii-nn-fle3n-blend`: interesting NN blend, but the required model
  dataset was not discoverable from this account, so failure risk is too high.
- `pilkwang/rogii-ridge-pf-blend-model-package-experiments`: runtime is safe,
  but the notebook itself disables the model-package correction under strict
  guards and has less upside than fle3n v5.

## Guardrails

- These are code submissions, not static public-test CSVs.
- The overlap override is dynamic and guarded; it no-ops when hidden test wells
  do not overlap train wells.
- If scores are still pending, poll Kaggle submissions rather than resubmitting.
