#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/Caskroom/miniconda/base/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
cd /Users/suraj/Documents/dsc204a-final-project
{
  date -u '+queue wrapper start %Y-%m-%d %H:%M:%S UTC'
  pwd
  command -v uv
} >> logs/queue_20260612_improvement_wrapper.trace 2>&1
exec uv run python scripts/queue_20260612_improvement_candidates.py --timeout-minutes 1440 --poll-seconds 120
