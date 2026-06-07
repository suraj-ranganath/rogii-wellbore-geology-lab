#!/usr/bin/env bash
set -euo pipefail

REMOTE="${ROGII_REMOTE:-suraj@ds-serv6.ucsd.edu}"
REMOTE_DIR="${ROGII_REMOTE_DIR:-/data/suraj/rogii-wellbore-geology-lab}"
RUN_NAME="${1:-tailcv_full_lgbm_$(date -u +%Y%m%dT%H%M%SZ)}"
shift || true

if [ "$#" -gt 0 ]; then
  CV_ARGS=("$@")
else
  CV_ARGS=(
    --max-wells 773
    --folds 5
    --include-lgbm
    --lgbm-estimators 300
    --output "outputs/${RUN_NAME}.json"
  )
fi

GIT_COMMIT="$(git rev-parse HEAD)"

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR/data/raw'"
rsync -az \
  pyproject.toml \
  uv.lock \
  README.md \
  configs \
  src \
  scripts \
  "$REMOTE:$REMOTE_DIR/"
rsync -az \
  --include='*/' \
  --include='*.csv' \
  --exclude='*' \
  data/raw/ \
  "$REMOTE:$REMOTE_DIR/data/raw/"

printf -v REMOTE_ARGS '%q ' "${CV_ARGS[@]}"

ssh "$REMOTE" bash -lc "'
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd \"$REMOTE_DIR\"
mkdir -p logs outputs
uv sync --extra dev
tmux new-session -d -s \"$RUN_NAME\" \"cd '$REMOTE_DIR' && export PATH=\\\"\$HOME/.local/bin:\\\$PATH\\\" && echo git_commit=$GIT_COMMIT && nvidia-smi && uv run python scripts/local_tail_cv.py $REMOTE_ARGS 2>&1 | tee logs/${RUN_NAME}.log\"
tmux ls | grep \"$RUN_NAME\"
'"

echo "Started $RUN_NAME on $REMOTE"
echo "Attach: ssh $REMOTE 'tmux attach -t $RUN_NAME'"
echo "Log:    ssh $REMOTE 'tail -f $REMOTE_DIR/logs/${RUN_NAME}.log'"
