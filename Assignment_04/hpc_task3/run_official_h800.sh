#!/usr/bin/env bash
set -euo pipefail

BASE=/public/home/ba25001026/3DGS
TASK3_DIR="$BASE/task3_official"
REPO_DIR="$TASK3_DIR/gaussian-splatting"
DATA_DIR="$BASE/data/chair"
OUTPUT_DIR="$TASK3_DIR/output/chair_h8008"
LOG_DIR="$TASK3_DIR/logs"
ENV_DIR=/public/home/ba25001026/.conda/envs/gauss-sd
PYTHON="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"
CUDA_HOME=/public/software/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "task3_start=$(date '+%F %T')"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "repo_dir=$REPO_DIR"
echo "data_dir=$DATA_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "python=$PYTHON"
nvcc --version | tail -n 1
nvidia-smi

"$PYTHON" - <<'PY'
import sys
import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device0", torch.cuda.get_device_name(0))
PY

cd "$REPO_DIR"

if ! "$PYTHON" - <<'PY'
import plyfile
import diff_gaussian_rasterization
import simple_knn._C
try:
    import fused_ssim
except Exception:
    pass
PY
then
    echo "Installing official 3DGS Python/CUDA dependencies..."
    "$PIP" install -i https://pypi.tuna.tsinghua.edu.cn/simple plyfile tqdm
    "$PIP" install --no-build-isolation --no-cache-dir "$REPO_DIR/submodules/diff-gaussian-rasterization"
    "$PIP" install --no-build-isolation --no-cache-dir "$REPO_DIR/submodules/simple-knn"
    "$PIP" install --no-build-isolation --no-cache-dir "$REPO_DIR/submodules/fused-ssim" || true
fi

"$PYTHON" - <<'PY'
import plyfile
import diff_gaussian_rasterization
import simple_knn._C
print("official_deps_ok")
PY

MEM_LOG="$LOG_DIR/nvidia_smi_${SLURM_JOB_ID:-manual}.csv"
(
    while true; do
        nvidia-smi --query-gpu=timestamp,name,index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 5
    done
) > "$MEM_LOG" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

TRAIN_START=$(date +%s)
/usr/bin/time -v "$PYTHON" train.py \
    -s "$DATA_DIR" \
    -m "$OUTPUT_DIR" \
    --disable_viewer \
    --test_iterations -1
TRAIN_END=$(date +%s)
TRAIN_SECONDS=$((TRAIN_END - TRAIN_START))

"$PYTHON" render.py \
    -s "$DATA_DIR" \
    -m "$OUTPUT_DIR" \
    --skip_test

RENDER_DIR="$OUTPUT_DIR/train/ours_30000/renders"
VIDEO_PATH="$OUTPUT_DIR/official_train_renders.mp4"
RENDER_DIR="$RENDER_DIR" VIDEO_PATH="$VIDEO_PATH" "$PYTHON" - <<'PY'
import os
from pathlib import Path
import cv2

render_dir = Path(os.environ["RENDER_DIR"])
video_path = Path(os.environ["VIDEO_PATH"])
images = sorted(render_dir.glob("*.png"))
if not images:
    raise RuntimeError(f"No render images found in {render_dir}")

first = cv2.imread(str(images[0]))
if first is None:
    raise RuntimeError(f"Cannot read first render image: {images[0]}")

height, width = first.shape[:2]
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
for image_path in images:
    frame = cv2.imread(str(image_path))
    if frame is None:
        continue
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
    writer.write(frame)
writer.release()
print(f"video_saved={video_path}")
print(f"video_frames={len(images)}")
PY

kill "$MONITOR_PID" 2>/dev/null || true
trap - EXIT

PEAK_MEM_MB=$(awk -F, '{gsub(/ /, "", $4); if ($4+0 > max) max=$4+0} END {print max+0}' "$MEM_LOG")
SUMMARY="$TASK3_DIR/summary_${SLURM_JOB_ID:-manual}.txt"
cat > "$SUMMARY" <<EOF
Task 3 Official 3DGS Summary
job_id=${SLURM_JOB_ID:-manual}
partition=${SLURM_JOB_PARTITION:-GPU_H800_8}
host=$(hostname)
data_dir=$DATA_DIR
output_dir=$OUTPUT_DIR
train_seconds=$TRAIN_SECONDS
peak_memory_mb=$PEAK_MEM_MB
memory_log=$MEM_LOG
render_dir=$RENDER_DIR
video_path=$VIDEO_PATH
final_point_cloud=$OUTPUT_DIR/point_cloud/iteration_30000/point_cloud.ply
repo_commit=$(git rev-parse --short HEAD)
task3_end=$(date '+%F %T')
EOF

cat "$SUMMARY"
echo "task3_done=$(date '+%F %T')"
