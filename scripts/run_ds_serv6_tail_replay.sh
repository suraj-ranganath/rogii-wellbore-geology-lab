#!/usr/bin/env bash
# Local hidden-tail replay of the JAEMIN SP45 + fleongg champion pipeline on the
# UCSD GPU server. Builds a synthetic competition dir on the remote, runs the
# unmodified kernel (via surgical path redirection) once per replicate seed in a
# tmux session, then scores each component + blend grid against held-out truth.
#
# Usage:
#   scripts/run_ds_serv6_tail_replay.sh RUN_NAME [options]
# Options (env or flags):
#   --n-eval-wells N     held-out eval wells (default 40)
#   --max-train-wells N  cap synthetic train wells (default 0 = all)
#   --replicates N       number of seed replicates (default 1)
#   --fast 0|1           kernel FAST mode (default 0)
#   --n-train-wells N    kernel N_TRAIN_WELLS cap (default 0 = all)
#   --use-gpu auto|gpu|cpu (default gpu)
#   --gpu-id ID          CUDA device (default: auto-pick freest)
set -euo pipefail

REMOTE="${ROGII_REMOTE:-suraj@ds-serv6.ucsd.edu}"
REMOTE_DIR="${ROGII_REMOTE_DIR:-/data/suraj/rogii-wellbore-geology-lab}"
RUN_NAME="${1:?usage: $0 RUN_NAME [options]}"
shift || true

N_EVAL_WELLS=40
MAX_TRAIN_WELLS=0
REPLICATES=1
FAST=0
N_TRAIN_WELLS=0
USE_GPU=gpu
GPU_ID=""
SEED=204

while [ "$#" -gt 0 ]; do
  case "$1" in
    --n-eval-wells) N_EVAL_WELLS="$2"; shift 2;;
    --max-train-wells) MAX_TRAIN_WELLS="$2"; shift 2;;
    --replicates) REPLICATES="$2"; shift 2;;
    --fast) FAST="$2"; shift 2;;
    --n-train-wells) N_TRAIN_WELLS="$2"; shift 2;;
    --use-gpu) USE_GPU="$2"; shift 2;;
    --gpu-id) GPU_ID="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

GIT_COMMIT="$(git rev-parse HEAD)"
KOOLBOX_LOCAL="kaggle/kernels/jaemin_sp45_fleongg_w060"
# fall back to any kernel dir that ships the koolbox wheel
if ! ls "$KOOLBOX_LOCAL"/koolbox-*.whl >/dev/null 2>&1; then
  KOOLBOX_LOCAL="kaggle/kernels/ridge_sp45_proj"
fi

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR/data/raw' '$REMOTE_DIR/data/koolbox' '$REMOTE_DIR/kaggle/kernels'"

# Sync code + competition CSVs (data/raw already synced from prior runs; refresh).
rsync -az pyproject.toml uv.lock README.md configs src scripts "$REMOTE:$REMOTE_DIR/"
rsync -az kaggle/kernels/jaemin_sp45_fleongg_w060 "$REMOTE:$REMOTE_DIR/kaggle/kernels/"
rsync -az --include='*/' --include='*.csv' --exclude='*' \
  data/raw/ "$REMOTE:$REMOTE_DIR/data/raw/"
# koolbox wheel (real dependency).
rsync -az "$KOOLBOX_LOCAL"/koolbox-*.whl "$REMOTE:$REMOTE_DIR/data/koolbox/"

SYNTH_DIR="$REMOTE_DIR/data/synth_${RUN_NAME}"
KOOLBOX_DIR="$REMOTE_DIR/data/koolbox"

# Build the remote driver script locally (fully expanded), then ship + run it.
# Remote-side runtime values ($GPU, loop vars) use a literal backslash so they
# expand on the server, not here.
DRIVER_LOCAL="$(mktemp -t rogii_replay_driver.XXXXXX.sh)"
trap 'rm -f "$DRIVER_LOCAL"' EXIT

cat > "$DRIVER_LOCAL" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd "$REMOTE_DIR"
mkdir -p logs outputs
uv sync --extra dev
# koolbox is a real kernel dependency shipped as a pure-python wheel. Install it
# into the uv venv so the kernel's ``import koolbox`` resolves (its own pip-based
# loader fails under PEP 668 externally-managed Python).
uv pip install --no-deps "$KOOLBOX_DIR"/koolbox-*.whl

GPU="$GPU_ID"
if [ -z "\$GPU" ]; then
  # Freest = lowest (utilization, memory) lexicographically: avoids GPUs that are
  # idle on memory but pegged at 100% compute by another user's small process.
  GPU=\$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \\
        | sort -t, -k2 -n -k3 -n | head -1 | cut -d, -f1 | tr -d ' ')
fi
echo "git_commit=$GIT_COMMIT chosen_gpu=\$GPU"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

uv run python scripts/build_tail_replay_dataset.py \\
  --out-dir "$SYNTH_DIR" \\
  --n-eval-wells $N_EVAL_WELLS \\
  --max-train-wells $MAX_TRAIN_WELLS \\
  --seed $SEED

RUN_DIRS=()
for i in \$(seq 1 $REPLICATES); do
  OFFSET=\$(( (i - 1) * 1000 ))
  RD="outputs/tail_replay_${RUN_NAME}/rep\${i}"
  mkdir -p "\$RD"
  echo "=== replicate \$i (seed_offset=\$OFFSET) -> \$RD ==="
  uv run python scripts/run_tail_replay_kernel.py \\
    --data-dir "$SYNTH_DIR" \\
    --run-dir "\$RD" \\
    --koolbox "$KOOLBOX_DIR" \\
    --use-gpu $USE_GPU \\
    --gpu-id "\$GPU" \\
    --fast $FAST \\
    --n-train-wells $N_TRAIN_WELLS \\
    --seed-offset \$OFFSET
  RUN_DIRS+=(--run-dir "\$RD")
done

uv run python scripts/score_tail_replay.py \\
  --truth "$SYNTH_DIR/truth.csv" \\
  "\${RUN_DIRS[@]}" \\
  --output "outputs/tail_replay_${RUN_NAME}/results.json" \\
  --runtime-notes "run=$RUN_NAME replicates=$REPLICATES n_eval=$N_EVAL_WELLS max_train=$MAX_TRAIN_WELLS fast=$FAST gpu=\$GPU"
echo "REPLAY_DONE $RUN_NAME"
EOF

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR/logs'"
rsync -az "$DRIVER_LOCAL" "$REMOTE:$REMOTE_DIR/logs/${RUN_NAME}_driver.sh"
ssh "$REMOTE" bash -lc "\"
set -euo pipefail
export PATH=\\\"\\\$HOME/.local/bin:\\\$PATH\\\"
cd '$REMOTE_DIR'
tmux kill-session -t '$RUN_NAME' 2>/dev/null || true
tmux new-session -d -s '$RUN_NAME' \\\"bash logs/${RUN_NAME}_driver.sh 2>&1 | tee logs/${RUN_NAME}.log\\\"
tmux ls | grep '$RUN_NAME'
\""

echo "Started tail-replay $RUN_NAME on $REMOTE"
echo "Attach: ssh $REMOTE 'tmux attach -t $RUN_NAME'"
echo "Log:    ssh $REMOTE 'tail -f $REMOTE_DIR/logs/${RUN_NAME}.log'"
echo "Result: $REMOTE_DIR/outputs/tail_replay_${RUN_NAME}/results.json"
