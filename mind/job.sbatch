#!/bin/bash
# One GPU, one node, short. MODE=smoke runs a 256-image, 1-epoch check; MODE=full trains for real.
# Submit from the project folder on the cluster:
#   sbatch --export=ALL,MODE=smoke --time=00:20:00 -J ns-smoke mind/job.sbatch
#   sbatch --export=ALL,MODE=full  --time=01:00:00 -J ns-full  mind/job.sbatch
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --exclude=mind-1-19,mind-1-11,mind-0-18,mind-0-20,mind-0-22,mind-0-24,mind-1-24
#SBATCH -o logs/%x-%j.out
set -euo pipefail

PROJ="${PROJ:-$PWD}"
SIF="$HOME/containers/lmlab.sif"
PY="$HOME/lmlab-env/bin/python"
OUT="$PROJ/runs/${MODE}-${SLURM_JOB_ID}"
mkdir -p "$OUT" "$PROJ/logs"
echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} mode=$MODE out=$OUT"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJ/pydeps"

run() { apptainer exec --nv -B /user_data "$SIF" "$@"; }

# Install the ONNX wheels (copied up from the laptop) once, skipping anything the env already has.
if [ ! -f "$PROJ/pydeps/.done" ]; then
  mkdir -p "$PROJ/pydeps"
  todo=()
  for w in "$PROJ"/wheels/*.whl; do
    name=$(basename "$w" | cut -d- -f1)
    if ! PYTHONPATH= run "$PY" -c "import importlib.metadata as m; m.version('${name}')" >/dev/null 2>&1; then
      todo+=("$w")
    fi
  done
  echo "installing: ${todo[*]##*/}"
  PYTHONPATH= run "$PY" -m pip install --quiet --no-deps --no-index --target "$PROJ/pydeps" "${todo[@]}"
  touch "$PROJ/pydeps/.done"
fi
run "$PY" -c "import sys, torch, numpy, onnx, onnxruntime; print(sys.version.split()[0], torch.__version__, numpy.__version__, onnx.__version__, onnxruntime.__version__, torch.cuda.is_available())"

ARGS=(--data "$PROJ/data/pneumoniamnist_128.npz" --weights "$PROJ/data/resnet18-f37072fd.pth" --out "$OUT")
[ "$MODE" = "smoke" ] && ARGS+=(--smoke)
run "$PY" "$PROJ/train/run.py" "${ARGS[@]}"
ls -la "$OUT"
