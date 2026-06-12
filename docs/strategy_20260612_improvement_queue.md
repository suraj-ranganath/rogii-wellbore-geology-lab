# 2026-06-12 Improvement Queue Strategy

## Result Audit

The 2026-06-12 UTC queue completed without timeouts:

| Description | Public RMSE |
| --- | ---: |
| `final queue sp45 fleongg w060 override` | `7.541` |
| `final queue fle3n v5 w060 h0455` | `7.565` |
| `final queue fle3n v5 exact h050` | `7.585` |
| `final queue fle3n v5f exact h050` | `7.637` |
| `final queue sp45 fleongg w100 override` | `7.766` |

Main interpretation: the public fle3n v5/v5f sources are real but did not beat
our `w0.60 + override` rerun. Pure SP45 is too heavy. The next set should keep
two requested fle3n reruns for stochastic upside, add the new public JAEMIN
seed7/affine variants, and use one nearby SP45-weight probe (`w0.65`) rather
than another exact `w0.60` rerun.

## Active Queue

Runner:

```bash
tmux attach -t rogii_improvequeue_20260612
tail -f logs/queue_20260612_improvement_tmux.log
```

The queue script is:

```bash
uv run python scripts/queue_20260612_improvement_candidates.py --timeout-minutes 1440 --poll-seconds 120
```

It is waiting at the daily cap and should submit after the 2026-06-13
00:00 UTC reset.

| Priority | Candidate | Kernel | Rationale |
| ---: | --- | --- | --- |
| 1 | `fle3n_v5_exact_r2` | `surajranganath17/rogii-fle3n-v5-exact-r2` | Requested exact rerun of `fleongg/fle3n-rogii-v5`; public source is byte-identical to our earlier fork, so upside is stochastic/run-context variance. |
| 2 | `fle3n_v5f_exact_r2` | `surajranganath17/rogii-fle3n-v5f-exact-r2` | Requested exact rerun of `fleongg/fle3n-rogii-v5f-probe`; close to v5 but not identical. |
| 3 | `jaemin_seed7_mtoshi_beicicc` | `surajranganath17/rogii-jaemin-seed7-mtoshi-beicicc` | New public JAEMIN script combining seed7 SP45/fleongg with mtoshi and Beicicc PF sources. |
| 4 | `jaemin_affine_seed7_mtoshi` | `surajranganath17/rogii-jaemin-affine-seed7-mtoshi` | New JAEMIN affine/anchor variant with different blend weights from the seed7 script. |
| 5 | `jaemin_sp45_fleongg_w065s` | `surajranganath17/rogii-sp45-fleongg-w065s-runtime-safe` | Nearby SP45-heavy probe with guarded override; safer improvement attempt than pure SP45. |

All five reached `KernelWorkerStatus.COMPLETE`, were downloaded, and passed
sample order, row count, finite-value, and plausible TVT range validation.

## Operational Notes

- Daily cap is `5/5` on 2026-06-12 UTC, so no manual submission should be made
  before reset.
- A LaunchAgent was rejected with `EX_CONFIG` before wrapper startup, likely
  due macOS Documents/TCC access. The active runner is a detached `tmux`
  session instead.
- A temporary `caffeinate -dimsu -t 72000` process is running to prevent idle
  sleep until after reset. Closing the lid can still suspend the machine.
